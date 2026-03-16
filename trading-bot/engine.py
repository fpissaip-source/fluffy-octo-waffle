"""
engine.py — Drei-Schichten-Architektur:
  Schicht 1 (Perception):  7 quantitative Formeln scannen den Markt
  Schicht 2 (Reasoning):   GPT-4o entscheidet PFLICHTWEISE vor jeder Order
  Schicht 3 (Execution):   Alpaca fuehrt Order aus

24/7 Modus:
  - Marktzeiten: alle Symbole (Aktien + Crypto)
  - Nachts/Wochenende: nur Crypto (BTC, ETH, SOL etc.)
  - Extended Hours: Aktien mit extended_hours=True
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Optional

# Crypto-Symbole handeln 24/7
CRYPTO_SYMBOLS = {"BTCUSD", "ETHUSD", "SOLUSD", "AVAXUSD", "LINKUSD"}

def is_crypto(symbol: str) -> bool:
    return symbol.upper() in CRYPTO_SYMBOLS

from broker import AlpacaBroker
from config import Config
from risk_manager import RiskManager, compute_atr
from adaptive import AdaptiveLearner
from formulas import momentum, kelly, ev_gap, kl_divergence, bayesian, stoikov
from formulas import sentiment as sentiment_formula
from market_context import MarketContext

logger = logging.getLogger("bot.engine")


# ═══════════════════════════════════════════════════════
#  SCHICHT 2: REASONING LAYER (GPT-4o)
# ═══════════════════════════════════════════════════════

class ReasoningLayer:
    """
    Pflicht-Entscheidungsschicht vor jeder Kauforder.
    GPT-4o bekommt alle Perception-Daten und entscheidet:
      - BUY:  Handel erlaubt
      - HOLD: Handel blockiert
    Ohne gueltige GPT-4o-Bestaetigung wird KEINE Order ausgefuehrt.
    """

    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.model = Config.REASONING_MODEL
        self.min_confidence = Config.REASONING_MIN_CONFIDENCE
        self.market_ctx = MarketContext()
        logger.info(f"ReasoningLayer initialisiert: {self.model} (min_confidence={self.min_confidence})")

    def approve_trade(
        self,
        symbol: str,
        signal: "TradeSignal",
        price: float,
        equity: float,
        regime: str,
    ) -> dict:
        """
        Fragt GPT-4o ob der Trade ausgefuehrt werden soll.
        Returns: {"approved": bool, "confidence": float, "reason": str}
        """
        formula_summary = "\n".join(
            f"  - {name}: signal={r['signal']:.3f} {'PASS' if r['passed'] else 'FAIL'}"
            + (f" ({r['details'].get('regime','') or ''})" if r.get("details") else "")
            for name, r in signal.results.items()
        )

        sentiment_details = signal.results.get("Sentiment", {}).get("details", {})
        sentiment_score = sentiment_details.get("score", 0)
        sentiment_articles = sentiment_details.get("articles", 0)
        macro_score = sentiment_details.get("macro", 0)

        market_context_str = self.market_ctx.format_for_prompt(symbol)

        prompt = f"""Du bist ein erfahrener quantitativer Trader. Analysiere dieses Trading-Signal und entscheide ob ein Kauf sinnvoll ist.

SYMBOL: {symbol}
PREIS: ${price:.2f}
DEPOT: ${equity:,.2f}
MARKT-REGIME: {regime}

QUANTITATIVE SIGNALE (Perception Layer):
{formula_summary}

SENTIMENT-ANALYSE:
  - Symbol-Sentiment: {sentiment_score:+.3f}
  - Makro-Sentiment: {macro_score:+.3f}
  - Analysierte Artikel: {sentiment_articles}

ECHTZEIT MARKT-KONTEXT:
{market_context_str}

KONTEXT:
  - Position-Groesse: ~{signal.qty} Aktien (~${signal.qty * price:,.0f})
  - Risiko: {(signal.qty * price / equity * 100):.1f}% des Depots

Bewerte: Ist das ein gutes Chance/Risiko-Verhaeltnis fuer einen Kauf JETZT?
Beruecksichtige: VIX-Level, Sektor-Trend, Sentiment, Signalstaerke, Positionsgroesse.

Antworte NUR mit JSON:
{{"decision": "BUY" oder "HOLD", "confidence": 0.0-1.0, "reason": "ein Satz auf Deutsch", "risk_factors": ["Faktor1", "Faktor2"]}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=200,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}],
                timeout=Config.REASONING_TIMEOUT,
            )
            text = response.choices[0].message.content or ""

            import re
            match = re.search(r'\{[^{}]+\}', text, re.DOTALL)
            if match:
                result = json.loads(match.group())
                approved = (
                    result.get("decision", "HOLD") == "BUY"
                    and float(result.get("confidence", 0)) >= self.min_confidence
                )
                logger.info(
                    f"[REASONING] {symbol}: {result.get('decision')} "
                    f"confidence={result.get('confidence', 0):.2f} | {result.get('reason', '')}"
                )
                return {
                    "approved": approved,
                    "confidence": float(result.get("confidence", 0)),
                    "reason": result.get("reason", ""),
                    "risk_factors": result.get("risk_factors", []),
                    "raw": result,
                }

            logger.warning(f"[REASONING] {symbol}: Konnte JSON nicht parsen — Trade BLOCKIERT")
            return {"approved": False, "confidence": 0.0, "reason": "JSON parse error", "risk_factors": []}

        except Exception as e:
            logger.error(f"[REASONING] {symbol}: GPT-4o Fehler — Trade BLOCKIERT: {e}")
            return {"approved": False, "confidence": 0.0, "reason": f"API error: {e}", "risk_factors": []}


# ═══════════════════════════════════════════════════════
#  DYNAMISCHE WATCHLIST (GPT-4o findet neue Aktien)
# ═══════════════════════════════════════════════════════

class WatchlistDiscovery:
    """
    Nutzt GPT-4o um alle 4 Stunden neue handelbare Aktien zu finden.
    Kombiniert mit der Basis-Watchlist aus .env.
    Max 15 Symbole gesamt.
    """

    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.last_update = 0
        self.update_interval = 3600  # alle 1 Stunde
        self.dynamic_symbols: list[str] = []
        logger.info("[WATCHLIST] Dynamic discovery initialisiert")

    def should_update(self) -> bool:
        return time.time() - self.last_update > self.update_interval

    def discover(self, market_open: bool) -> list[str]:
        """Fragt GPT-4o nach den besten Symbolen fuer die naechsten Stunden."""
        if not self.should_update():
            return self.dynamic_symbols

        context = "US Aktienmarkt ist gerade geoeffnet." if market_open else \
                  "US Aktienmarkt ist geschlossen, nur Crypto handelbar."

        prompt = f"""Du bist ein konservativer quantitativer Trader mit Fokus auf kapitalerhalt. {context}

Welche 8 Symbole bieten gerade ein sicheres 1-5% Gewinnpotenzial mit minimalem Risiko (nächste 4 Stunden)?

Kriterien:
- NUR S&P 500 oder Nasdaq 100 Schwergewichte (Market Cap > 50 Mrd USD)
- Sehr hohe Liquidität (min. 10M Tagesvolumen)
- Klarer, ruhiger Aufwärtstrend — KEIN spekulativer Breakout
- Niedrige Volatilität bevorzugt (Beta < 1.5)
- Bevorzuge: AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, JPM, V, MA, UNH, JNJ
- Keine Penny Stocks, keine Micro Caps, kein High-Beta-Zockerpapiere
- Bei geschlossenem Markt: nur etablierte Crypto (BTC, ETH — keine Altcoins)
- Ziel: 1-5% sicherer Gewinn, nicht 20% Lotterie

Antworte NUR mit JSON:
{{"symbols": ["SYM1", "SYM2", "SYM3", "SYM4", "SYM5", "SYM6", "SYM7", "SYM8"], "reasoning": "ein Satz"}}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                max_tokens=200,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.choices[0].message.content or ""
            import re
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                result = json.loads(match.group())
                symbols = [s.upper().strip() for s in result.get("symbols", [])]
                self.dynamic_symbols = symbols[:8]
                self.last_update = time.time()
                logger.info(f"[WATCHLIST] Neue Symbole: {self.dynamic_symbols} | {result.get('reasoning', '')}")
                return self.dynamic_symbols
        except Exception as e:
            logger.warning(f"[WATCHLIST] Discovery fehlgeschlagen: {e}")

        return self.dynamic_symbols

    def get_active_watchlist(self, market_open: bool) -> list[str]:
        """
        Kombiniert Basis-Watchlist (.env) mit GPT-4o Vorschlaegen.
        Nachts/Wochenende: nur Crypto-Symbole.
        """
        dynamic = self.discover(market_open)

        if not market_open:
            # Nur Crypto wenn Markt zu
            crypto_base = [s for s in Config.WATCHLIST if is_crypto(s)]
            crypto_dynamic = [s for s in dynamic if is_crypto(s)]
            combined = list(dict.fromkeys(crypto_base + crypto_dynamic))
            if not combined:
                combined = ["BTCUSD", "ETHUSD", "SOLUSD"]
            return combined[:10]

        # Markt offen: Basis + dynamisch, max 15
        combined = list(dict.fromkeys(Config.WATCHLIST + dynamic))
        return combined[:15]


class TradeSignal:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.timestamp = datetime.now()
        self.results: dict = {}
        self.all_passed = False
        self.action: Optional[str] = None
        self.qty: int = 0
        self.reason: str = ""

    def add_result(self, result: dict):
        self.results[result["name"]] = result

    def evaluate(self):
        if len(self.results) < 5:
            self.all_passed = False
            self.action = "HOLD"
            self.reason = f"Only {len(self.results)}/7 formulas ran"
            return

        # Pflicht-Filter: Kelly + Bayesian muessen immer passen (Risiko-Schutz)
        mandatory = ["Kelly", "Bayesian"]
        for m in mandatory:
            if m in self.results and not self.results[m]["passed"]:
                self.all_passed = False
                self.action = "HOLD"
                self.reason = f"Mandatory filter failed: {m}"
                return

        # Mindestens 5 von 7 Filtern muessen passen
        passed_count = sum(1 for r in self.results.values() if r["passed"])
        total = len(self.results)
        min_pass = max(5, total - 2)  # Bei 7 Filtern: mind. 5

        self.all_passed = passed_count >= min_pass

        if self.all_passed:
            self.action = "BUY"
            failed = [n for n, r in self.results.items() if not r["passed"]]
            self.reason = f"{passed_count}/{total} filters passed" + (f" (ignored: {', '.join(failed)})" if failed else "")
        else:
            failed = [n for n, r in self.results.items() if not r["passed"]]
            self.action = "HOLD"
            self.reason = f"Only {passed_count}/{total} passed — need {min_pass}. Failed: {', '.join(failed)}"

    def summary(self) -> str:
        lines = [
            f"\n{'=' * 60}",
            f"  {self.symbol}  |  {self.timestamp.strftime('%H:%M:%S')}",
            f"{'=' * 60}",
            f"  {'LAYER 1: PERCEPTION':}",
        ]
        for name, r in self.results.items():
            status = "PASS" if r["passed"] else "FAIL"
            lines.append(f"  {name:<16} {status:<8} signal={r['signal']}")
        lines.append(f"{'-' * 60}")
        if self.all_passed:
            lines.append(f"  LAYER 2: REASONING  -> GPT-4o entscheidet...")
            lines.append(f"  LAYER 3: EXECUTION  -> Qty: {self.qty}")
        else:
            lines.append(f"  > HOLD (Perception Layer blockiert)")
        lines.append(f"  > REASON: {self.reason}")
        lines.append(f"{'=' * 60}\n")
        return "\n".join(lines)


class Engine:
    def __init__(self):
        logger.info("Initializing engine...")
        self.broker = AlpacaBroker()
        self.risk = RiskManager()
        self.learner = AdaptiveLearner()
        self.reasoning = ReasoningLayer()
        self.watchlist = WatchlistDiscovery()
        self.trade_log: list[TradeSignal] = []
        self.position_highs: dict[str, float] = {}

    def analyze_symbol(self, symbol: str) -> TradeSignal:
        signal = TradeSignal(symbol)

        bars = self.broker.get_bars(symbol, timeframe="5Min", limit=Config.LOOKBACK_BARS)
        if bars.empty or len(bars) < 50:
            logger.warning(f"{symbol}: Not enough data ({len(bars) if not bars.empty else 0} bars)")
            signal.reason = "Insufficient data"
            return signal

        snapshot = self.broker.get_snapshot(symbol)
        equity = self.broker.get_equity()
        positions = self.broker.get_positions()
        inventory_skew = 0.3 if symbol in positions else 0.0

        # ── F1: Momentum ──
        try:
            r1 = momentum.evaluate(bars, threshold=Config.MIN_MOMENTUM_SCORE)
            signal.add_result(r1)
        except Exception as e:
            signal.add_result({"name": "Momentum", "signal": 0, "passed": False, "details": {"error": str(e)}})

        # ── F2: Kelly ──
        try:
            r2 = kelly.evaluate(bars, equity=equity)
            signal.add_result(r2)
        except Exception as e:
            signal.add_result({"name": "Kelly", "signal": 0, "passed": False, "details": {"error": str(e)}})

        # ── F3: EV-Gap ──
        try:
            r3 = ev_gap.evaluate(bars, win_prob=0.55)
            signal.add_result(r3)
        except Exception as e:
            signal.add_result({"name": "EV-Gap", "signal": 0, "passed": False, "details": {"error": str(e)}})

        # ── F4: KL-Divergence ──
        try:
            r4 = kl_divergence.evaluate(bars, threshold=Config.KL_DIVERGENCE_THRESHOLD)
            signal.add_result(r4)
        except Exception as e:
            signal.add_result({"name": "KL-Divergence", "signal": 0, "passed": False, "details": {"error": str(e)}})

        # ── F5: Bayesian ──
        try:
            r5 = bayesian.evaluate(bars, prior=0.50, threshold=Config.MIN_BAYESIAN_POSTERIOR)
            signal.add_result(r5)
        except Exception as e:
            signal.add_result({"name": "Bayesian", "signal": 0.5, "passed": False, "details": {"error": str(e)}})

        # ── F6: Stoikov ──
        try:
            r6 = stoikov.evaluate(bars, snapshot=snapshot, inventory_skew=inventory_skew,
                                  time_remaining=0.5)
            signal.add_result(r6)
        except Exception as e:
            signal.add_result({"name": "Stoikov", "signal": 0, "passed": False, "details": {"error": str(e)}})

        # ── F7: Sentiment ──
        try:
            r7 = sentiment_formula.evaluate(bars, broker=self.broker, symbol=symbol)
            signal.add_result(r7)
        except Exception as e:
            signal.add_result({"name": "Sentiment", "signal": 0, "passed": True, "details": {"error": str(e)}})

        # ── Regime Update ──
        self.risk.update_regime(bars)

        # ── Decision ──
        signal.evaluate()

        # ── Adaptive Override Check ──
        if not signal.all_passed:
            override = self.learner.should_override_entry(
                self.risk.regime.value, signal.results
            )
            if override["override"]:
                signal.all_passed = True
                signal.action = "BUY"
                signal.reason = override["reason"]
                logger.info(f"{symbol}: ADAPTIVE OVERRIDE — {override['reason']}")

        # ── Weighted Score (fuer Logging) ──
        w_score = self.learner.weighted_score(self.risk.regime.value, signal.results)
        signal.reason += f" | score={w_score:.2f}"

        if signal.all_passed and "Kelly" in signal.results:
            bet_size = signal.results["Kelly"]["details"].get("bet_size_usd", 0)
            price = bars["close"].iloc[-1]
            if price > 0:
                signal.qty = max(1, int(bet_size / price))

        return signal

    def execute_signal(self, signal: TradeSignal) -> Optional[str]:
        if not signal.all_passed or signal.action != "BUY":
            return None
        if signal.qty <= 0:
            logger.warning(f"{signal.symbol}: All passed but qty=0")
            return None
        if self.broker.has_position(signal.symbol):
            logger.info(f"{signal.symbol}: Already have position, skipping")
            return None

        # Risk Manager checks
        positions = self.broker.get_positions()
        if not self.risk.can_open_position(len(positions)):
            logger.warning(
                f"{signal.symbol}: Max positions reached ({len(positions)}) "
                f"for regime {self.risk.regime.value}"
            )
            return None

        # Cap qty by risk manager
        equity = self.broker.get_equity()
        price = self.broker.get_latest_price(signal.symbol) or 1
        max_qty = self.risk.max_position_size(equity, price)
        signal.qty = min(signal.qty, max_qty)

        # ── SCHICHT 2: Reasoning Layer (GPT-4o Pflicht-Check) ──
        reasoning = self.reasoning.approve_trade(
            symbol=signal.symbol,
            signal=signal,
            price=price,
            equity=equity,
            regime=self.risk.regime.value,
        )
        signal.reason += f" | GPT4o={reasoning['confidence']:.0%}: {reasoning['reason']}"

        if not reasoning["approved"]:
            logger.warning(
                f"[REASONING BLOCKED] {signal.symbol}: "
                f"confidence={reasoning['confidence']:.0%} — {reasoning['reason']}"
            )
            if reasoning["risk_factors"]:
                logger.warning(f"  Risiken: {', '.join(reasoning['risk_factors'])}")
            return None

        logger.info(f"{'=' * 40}")
        logger.info(f"EXECUTING: BUY {signal.qty}x {signal.symbol}")
        logger.info(f"Regime: {self.risk.regime.value} | {self.risk.params['description']}")
        logger.info(f"GPT-4o: {reasoning['confidence']:.0%} confident — {reasoning['reason']}")
        logger.info(f"{'=' * 40}")

        order_id = self.broker.market_buy(signal.symbol, signal.qty)
        if order_id:
            signal.reason += f" -> Order {order_id}"
            self.trade_log.append(signal)
            self.position_highs[signal.symbol] = price

            # ── Adaptive Learning: Record Entry ──
            formula_scores = {
                name: r.get("signal", 0) for name, r in signal.results.items()
            }
            sentiment_score = signal.results.get("Sentiment", {}).get("signal", 0)
            self.learner.record_entry(
                symbol=signal.symbol,
                regime=self.risk.regime.value,
                formula_scores=formula_scores,
                sentiment_score=sentiment_score,
                entry_price=price,
                qty=signal.qty,
            )

        return order_id

    def check_exit_conditions(self):
        """Adaptive Exit-Checks basierend auf Markt-Regime."""
        positions = self.broker.get_positions()
        equity = self.broker.get_equity()

        # Kill-Switch Check
        if self.risk.check_kill_switch(equity):
            logger.critical("KILL SWITCH ACTIVE — closing ALL positions")
            for symbol in positions:
                self.broker.close_position(symbol)
            return

        for symbol, pos in positions.items():
            try:
                bars = self.broker.get_bars(symbol, timeframe="5Min", limit=50)
                if bars.empty:
                    continue

                # Regime updaten
                self.risk.update_regime(bars)

                # ATR berechnen
                atr = compute_atr(bars)
                entry_price = pos["avg_entry"]
                current_price = bars["close"].iloc[-1]

                # Highest price tracken (fuer Trailing Stop)
                if symbol not in self.position_highs:
                    self.position_highs[symbol] = entry_price
                self.position_highs[symbol] = max(
                    self.position_highs[symbol], current_price
                )

                # Bayesian posterior holen
                bay = bayesian.evaluate(bars, prior=0.50)
                posterior = bay.get("signal", 0.5)

                # Adaptive Exit-Entscheidung
                exit_decision = self.risk.should_exit(
                    entry_price=entry_price,
                    current_price=current_price,
                    highest_price=self.position_highs[symbol],
                    atr=atr,
                    bayesian_posterior=posterior,
                )

                if exit_decision["should_exit"]:
                    plpc = pos["unrealized_plpc"]
                    logger.info(
                        f"EXIT {symbol}: {exit_decision['reason']} "
                        f"| P/L: {plpc:+.1%} | Regime: {self.risk.regime.value}"
                    )
                    self.broker.close_position(symbol)
                    self.position_highs.pop(symbol, None)

                    # ── Adaptive Learning: Record Exit ──
                    self.learner.record_exit(
                        symbol, current_price, exit_decision["reason"]
                    )

            except Exception as e:
                logger.error(f"Exit check error {symbol}: {e}")

    def scan_once(self, market_open: bool):
        active_watchlist = self.watchlist.get_active_watchlist(market_open)

        logger.info(f"\n{'=' * 60}")
        logger.info(f"  SCAN @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"  Markt: {'OFFEN' if market_open else 'GESCHLOSSEN (nur Crypto)'}")
        logger.info(f"  Watchlist ({len(active_watchlist)}): {', '.join(active_watchlist)}")
        logger.info(f"  Regime: {self.risk.regime.value}")
        logger.info(f"{'=' * 60}")

        self.check_exit_conditions()

        for symbol in active_watchlist:
            try:
                signal = self.analyze_symbol(symbol)
                print(signal.summary())
                if signal.all_passed:
                    self.execute_signal(signal)
            except Exception as e:
                logger.error(f"Error {symbol}: {e}")

        equity = self.broker.get_equity()
        positions = self.broker.get_positions()
        logger.info(f"Equity: ${equity:,.2f}  |  Positions: {len(positions)}  |  Trades: {len(self.trade_log)}")

    def run(self):
        logger.info("=" * 60)
        logger.info("  7 FILTERS. GPT-4o REASONING. 24/7.")
        logger.info(f"  Mode: {'PAPER' if Config.is_paper() else '!! LIVE !!'}")
        logger.info(f"  Base Watchlist: {Config.WATCHLIST}")
        logger.info(f"  Dynamic Discovery: alle 1h via GPT-4o")
        logger.info("=" * 60)

        while True:
            try:
                market_open = self.broker.is_market_open()

                if not market_open:
                    # Markt zu: nur Crypto scannen (echte 24/7 Assets)
                    logger.info("Boerse geschlossen — scanne nur Crypto...")

                self.scan_once(market_open)

            except KeyboardInterrupt:
                logger.info("\nShutting down...")
                break
            except Exception as e:
                logger.error(f"Scan error: {e}")

            interval = Config.SCAN_INTERVAL if self.broker.is_market_open() else 120
            logger.info(f"Next scan in {interval}s...")
            time.sleep(interval)
