"""
engine.py — Drei-Schichten-Architektur:
  Schicht 1 (Perception):  7 quantitative Formeln scannen den Markt
  Schicht 2 (Reasoning):   Gemini entscheidet PFLICHTWEISE vor jeder Order
  Schicht 3 (Execution):   Alpaca fuehrt Order aus

24/7 Modus:
  - Marktzeiten: alle Symbole (Aktien + Crypto)
  - Nachts/Wochenende: nur Crypto (BTC, ETH, SOL etc.)
  - Extended Hours: Aktien mit extended_hours=True
"""

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from screener import SpikeSensor

logger = logging.getLogger("bot.engine")


# ═══════════════════════════════════════════════════════
#  SCHICHT 2: REASONING LAYER (Gemini)
# ═══════════════════════════════════════════════════════

class ReasoningLayer:
    """
    Pflicht-Entscheidungsschicht vor jeder Kauforder.
    Gemini bekommt alle Perception-Daten und entscheidet:
      - BUY:  Handel erlaubt
      - HOLD: Handel blockiert
    Ohne gueltige Gemini-Bestaetigung wird KEINE Order ausgefuehrt.
    """

    def __init__(self):
        from google import genai
        self.client = genai.Client(api_key=Config.GEMINI_API_KEY)
        self.model = Config.REASONING_MODEL
        self.min_confidence = Config.REASONING_MIN_CONFIDENCE
        self.market_ctx = MarketContext()
        logger.info(f"ReasoningLayer initialisiert: {self.model} (min_confidence={self.min_confidence})")

    def _cascade_fallback(self, symbol: str, cascade_level: int, reason: str) -> dict:
        """
        Fallback wenn Gemini nicht antwortet (Timeout / API-Fehler).
        6/7 oder 7/7 → Auto-Approve (starkes Signal auch ohne LLM).
        4/7 oder 5/7 → Blockiert (zu schwach fuer Blind-Trade).
        """
        if cascade_level >= 6:
            logger.warning(
                f"[REASONING] {symbol}: {reason} → AUTO-APPROVE "
                f"wegen Kaskade {cascade_level}/7"
            )
            return {
                "approved": True,
                "confidence": 0.70,
                "probability_pct": 70,
                "reason": f"Gemini-Fallback: Auto-Approve wegen {cascade_level}/7 Kaskade",
                "risk_factors": ["gemini_timeout"],
                "raw": {},
                "prompt": "",
                "raw_response": "FALLBACK",
            }
        logger.warning(
            f"[REASONING] {symbol}: {reason} → BLOCKIERT "
            f"wegen Kaskade {cascade_level}/7 (zu schwach)"
        )
        return {
            "approved": False,
            "confidence": 0.0,
            "probability_pct": 0,
            "reason": f"Gemini-Fallback: Blockiert wegen {cascade_level}/7 Kaskade",
            "risk_factors": ["gemini_timeout"],
            "raw": {},
            "prompt": "",
            "raw_response": "FALLBACK",
        }

    def approve_trade(
        self,
        symbol: str,
        signal: "TradeSignal",
        price: float,
        equity: float,
        regime: str,
    ) -> dict:
        """
        Fragt Gemini ob der Trade ausgefuehrt werden soll.
        Timeout: 5 Sekunden. Bei Timeout/Fehler greift Kaskaden-Fallback.
        Returns: {"approved": bool, "confidence": float, "reason": str, "prompt": str, "raw_response": str}
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
        cascade_info = f"{signal.cascade_label}" if signal.cascade_level else "unbekannt"

        prompt = f"""Du bist ein erfahrener quantitativer Trader. Analysiere dieses Trading-Signal und triff eine finale Kaufentscheidung.

SYMBOL: {symbol}
PREIS: ${price:.2f}
DEPOT: ${equity:,.2f}
MARKT-REGIME: {regime}

FILTER-KASKADE: {cascade_info}
(Hinweis: 7/7 = perfektes Setup, 4/7 = minimales Setup — berücksichtige das in der Wahrscheinlichkeit)

QUANTITATIVE SIGNALE (alle 7 Filter):
{formula_summary}

SENTIMENT-ANALYSE:
  - Symbol-Sentiment: {sentiment_score:+.3f}
  - Makro-Sentiment: {macro_score:+.3f}
  - Analysierte Artikel: {sentiment_articles}

ECHTZEIT MARKT-KONTEXT:
{market_context_str}

POSITION:
  - Groesse: ~{signal.qty} Aktien (~${signal.qty * price:,.0f})
  - Risiko: {(signal.qty * price / equity * 100):.1f}% des Depots

Deine Aufgabe: Gib eine Gewinnwahrscheinlichkeit in % (0-100) und entscheide BUY oder HOLD.
Bei 7/7 Filtern: hohe Wahrscheinlichkeit erwartet. Bei 4/7: konservativ sein.
Beruecksichtige: Kaskaden-Level, VIX, Sektor-Trend, Sentiment, Positionsgroesse.

Antworte NUR mit JSON:
{{"decision": "BUY" oder "HOLD", "probability_pct": 0-100, "confidence": 0.0-1.0, "reason": "ein Satz auf Deutsch", "risk_factors": ["Faktor1", "Faktor2"]}}"""

        def _do_call() -> dict:
            try:
                from google.genai import types as genai_types

                # ── Structured Output Schema — kein JSON-Parsing nötig ──
                response_schema = genai_types.Schema(
                    type=genai_types.Type.OBJECT,
                    properties={
                        "decision":        genai_types.Schema(type=genai_types.Type.STRING, enum=["BUY", "HOLD"]),
                        "probability_pct": genai_types.Schema(type=genai_types.Type.INTEGER),
                        "confidence":      genai_types.Schema(type=genai_types.Type.NUMBER),
                        "reason":          genai_types.Schema(type=genai_types.Type.STRING),
                        "risk_factors":    genai_types.Schema(
                            type=genai_types.Type.ARRAY,
                            items=genai_types.Schema(type=genai_types.Type.STRING),
                        ),
                    },
                    required=["decision", "probability_pct", "confidence", "reason", "risk_factors"],
                )

                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=response_schema,
                        temperature=0.1,
                        max_output_tokens=300,
                        thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
                    ),
                )

                raw_text = response.text or ""
                result = json.loads(raw_text)

                approved = (
                    result.get("decision", "HOLD") == "BUY"
                    and float(result.get("confidence", 0)) >= self.min_confidence
                )
                prob_pct = int(result.get("probability_pct", round(float(result.get("confidence", 0)) * 100)))
                logger.info(
                    f"[REASONING] {symbol}: {result.get('decision')} "
                    f"Wahrscheinlichkeit={prob_pct}% | {result.get('reason', '')}"
                )
                return {
                    "approved": approved,
                    "confidence": float(result.get("confidence", 0)),
                    "probability_pct": prob_pct,
                    "reason": result.get("reason", ""),
                    "risk_factors": result.get("risk_factors", []),
                    "raw": result,
                    "prompt": prompt,
                    "raw_response": raw_text,
                }

            except Exception as e:
                logger.error(f"[REASONING] {symbol}: Gemini Fehler: {e}")
                return {"approved": False, "confidence": 0.0, "reason": f"API error: {e}",
                        "risk_factors": [], "raw": {}, "prompt": prompt, "raw_response": str(e)}

        # ── 5-Sekunden Timeout — blockiert nicht bei hängendem Gemini ──
        from concurrent.futures import TimeoutError as FuturesTimeoutError
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_do_call)
            try:
                return future.result(timeout=5)
            except FuturesTimeoutError:
                return self._cascade_fallback(symbol, signal.cascade_level, "Gemini-Timeout (5s)")
            except Exception as e:
                return self._cascade_fallback(symbol, signal.cascade_level, f"Gemini-Exception: {e}")


# ═══════════════════════════════════════════════════════
#  DYNAMISCHE WATCHLIST (Gemini findet neue Aktien)
# ═══════════════════════════════════════════════════════

class WatchlistDiscovery:
    """
    Nutzt Gemini um alle 4 Stunden neue handelbare Aktien zu finden.
    Kombiniert mit der Basis-Watchlist aus .env.
    Max 15 Symbole gesamt.
    """

    def __init__(self, broker=None):
        from google import genai
        self.client = genai.Client(api_key=Config.GEMINI_API_KEY)
        self.last_update = 0
        self.update_interval = 900   # alle 15 Minuten (war 1h)
        self.dynamic_symbols: list[str] = []
        self.broker = broker
        logger.info("[WATCHLIST] Dynamic discovery initialisiert")

    def should_update(self) -> bool:
        return time.time() - self.last_update > self.update_interval

    def _get_top_candidates(self, market_status: str) -> list[str]:
        """
        Holt Top-50 handelbare Kandidaten direkt via Alpaca-Batch-Snapshots.
        Sortiert nach pct_move * volume (kombinierter Momentum-Score).
        Kein Halluzinieren: Nur Ticker die WIRKLICH handeln kommen durch.
        """
        if not self.broker:
            return []

        if market_status == "closed":
            from engine import CRYPTO_SYMBOLS
            return list(CRYPTO_SYMBOLS)

        try:
            from screener import SpikeSensor
            universe = SpikeSensor.UNIVERSE
            snaps = self.broker.get_snapshots_batch(universe)

            candidates = []
            for symbol, snap in snaps.items():
                try:
                    daily = snap.daily_bar
                    if not daily:
                        continue
                    open_p = float(daily.open)
                    close_p = float(daily.close)
                    volume = int(daily.volume)
                    if open_p <= 0 or close_p <= 0 or volume == 0:
                        continue
                    pct_move = abs((close_p - open_p) / open_p)
                    candidates.append((symbol, pct_move, volume))
                except Exception:
                    continue

            # Kombinierter Score: relative Bewegung * Volumen
            candidates.sort(key=lambda x: x[1] * x[2], reverse=True)
            top50 = [s for s, _, _ in candidates[:50]]
            logger.info(f"[WATCHLIST] Alpaca Top-50 nach Momentum×Vol: {top50[:10]}...")
            return top50

        except Exception as e:
            logger.warning(f"[WATCHLIST] Top-Kandidaten Fehler: {e}")
            return []

    def _verify_symbols(self, symbols: list[str]) -> list[str]:
        """Prueft ob Gemini-Symbole wirklich auf Alpaca handelbar sind (Volumen > 0)."""
        if not self.broker:
            return symbols
        verified = []
        for sym in symbols:
            try:
                snap = self.broker.get_snapshot(sym)
                if snap and snap.get("volume", 0) > 0:
                    verified.append(sym)
                else:
                    logger.warning(f"[WATCHLIST] {sym} verworfen: kein Volumen auf Alpaca")
            except Exception as e:
                logger.warning(f"[WATCHLIST] {sym} verworfen: Snapshot fehlgeschlagen ({e})")
        return verified

    def discover(self, market_status: str) -> list[str]:
        """
        Invertierte Logik: Alpaca liefert Top-50 (echte Daten),
        Gemini waehlt daraus die 8 besten basierend auf News/Katalysatoren.
        Kein Halluzinieren — nur Ticker die wirklich handeln.
        """
        if not self.should_update():
            return self.dynamic_symbols

        # Schritt 1: Echte Kandidaten von Alpaca holen
        candidates = self._get_top_candidates(market_status)

        if not candidates:
            logger.warning("[WATCHLIST] Keine Alpaca-Kandidaten — behalte letzte Symbole")
            return self.dynamic_symbols

        if market_status == "closed":
            self.dynamic_symbols = candidates[:8]
            self.last_update = time.time()
            return self.dynamic_symbols

        candidates_str = ", ".join(candidates[:50])
        context_map = {
            "open": "US Aktienmarkt ist GERADE OFFEN (9:30–16:00 ET).",
            "extended": "US Aktienmarkt ist in VOR-/NACHBÖRSENHANDEL.",
        }
        context = context_map.get(market_status, "")

        # Schritt 2: Gemini waehlt aus echten Kandidaten (kein Erfinden)
        prompt = f"""Du bist ein erfahrener Day-Trader. {context}

Hier sind die 50 aktivsten US-Aktien der letzten Stunde nach Volumen und Kurs-Bewegung (echte Alpaca-Daten):
{candidates_str}

Wähle die 8 besten aus DIESER LISTE basierend auf deinem Wissen über:
- Aktuelle News und Katalysatoren (Earnings, FDA, M&A, Short Squeeze)
- Momentum und Trendstärke
- Liquidität und Handelbarkeit

WICHTIG: Nur Symbole aus der obigen Liste verwenden — keine anderen erfinden.

Antworte NUR mit JSON:
{{"symbols": ["SYM1", "SYM2", "SYM3", "SYM4", "SYM5", "SYM6", "SYM7", "SYM8"], "reasoning": "ein Satz"}}"""

        try:
            from google.genai import types as genai_types
            response = self.client.models.generate_content(
                model=self.model if hasattr(self, "model") else Config.REASONING_MODEL,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.3,
                    max_output_tokens=200,
                ),
            )
            text = response.text or ""
            result = None
            depth, start = 0, None
            for i, c in enumerate(text):
                if c == '{':
                    if depth == 0:
                        start = i
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0 and start is not None:
                        try:
                            result = json.loads(text[start:i + 1])
                        except json.JSONDecodeError:
                            pass
                        break

            if result:
                # Nur Symbole aus der Kandidaten-Liste akzeptieren (kein Halluzinieren)
                candidate_set = set(candidates)
                raw_symbols = [s.upper().strip() for s in result.get("symbols", [])]
                validated = [s for s in raw_symbols if s in candidate_set]
                hallucinated = [s for s in raw_symbols if s not in candidate_set]
                if hallucinated:
                    logger.warning(f"[WATCHLIST] Gemini halluzinierte {len(hallucinated)} Symbole → verworfen: {hallucinated}")
                self.dynamic_symbols = validated[:8]
                self.last_update = time.time()
                logger.info(f"[WATCHLIST] Neue Symbole ({len(self.dynamic_symbols)}): {self.dynamic_symbols} | {result.get('reasoning', '')}")
                return self.dynamic_symbols

        except Exception as e:
            logger.warning(f"[WATCHLIST] Discovery fehlgeschlagen: {e}")

        # Fallback: direkt Top-8 aus Alpaca-Daten nehmen
        self.dynamic_symbols = candidates[:8]
        self.last_update = time.time()
        return self.dynamic_symbols

    def get_active_watchlist(self, market_status: str) -> list[str]:
        """
        Kombiniert Basis-Watchlist (.env) mit Gemini Vorschlaegen.
        'closed' (Nacht/Wochenende): nur Crypto.
        'open' + 'extended': Aktien + Crypto.
        """
        dynamic = self.discover(market_status)

        if market_status == "closed":
            # Nur Crypto handeln wenn Markt + Extended geschlossen
            crypto = [s for s in (Config.WATCHLIST + dynamic)
                      if any(s.endswith(x) for x in ("USD", "BTC", "ETH", "SOL"))]
            return list(dict.fromkeys(crypto))[:10]

        # open + extended: volles Universum
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
        self.cascade_level: int = 0
        self.cascade_label: str = ""

    def add_result(self, result: dict):
        self.results[result["name"]] = result

    def evaluate(self):
        if len(self.results) < 5:
            self.all_passed = False
            self.action = "HOLD"
            self.reason = f"Only {len(self.results)}/7 formulas ran"
            return

        # Pflicht-Filter: Kelly muss immer passen (Edge-Schutz)
        if "Kelly" in self.results and not self.results["Kelly"]["passed"]:
            self.all_passed = False
            self.action = "HOLD"
            self.reason = "Mandatory filter failed: Kelly"
            return

        # Stoikov ist informational — bestimmt Order-Typ, zaehlt nicht in Kaskade
        # Kaskade: pruefe 7→6→5→4 (nur die 6 nicht-Stoikov-Filter), Gemini entscheidet am Ende
        passed_count = sum(1 for name, r in self.results.items() if r["passed"] and name != "Stoikov")
        total = len(self.results)

        if passed_count >= 7:
            self.cascade_level = 7
            self.cascade_label = "PERFEKT (7/7) — Maximales Signal"
        elif passed_count >= 6:
            self.cascade_level = 6
            self.cascade_label = "STARK (6/7) — Hohes Signal"
        elif passed_count >= 5:
            self.cascade_level = 5
            self.cascade_label = "GUT (5/7) — Solides Signal"
        elif passed_count >= 4:
            self.cascade_level = 4
            self.cascade_label = "MINIMAL (4/7) — Schwaches Signal"
        else:
            self.all_passed = False
            self.action = "HOLD"
            failed = [n for n, r in self.results.items() if not r["passed"]]
            self.reason = f"Zu schwach: nur {passed_count}/{total} — Gemini nicht befragt. Failed: {', '.join(failed)}"
            return

        # Ab 4/7 + Kelly ✓ → Gemini entscheidet
        self.all_passed = True
        self.action = "BUY"
        failed = [n for n, r in self.results.items() if not r["passed"]]
        self.reason = f"{self.cascade_label} → Gemini befragt" + (f" | Offen: {', '.join(failed)}" if failed else "")

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
            lines.append(f"  LAYER 2: REASONING  -> Gemini entscheidet...")
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
        self.watchlist = WatchlistDiscovery(self.broker)
        self.spike_sensor = SpikeSensor(self.broker)
        self.trade_log: list[TradeSignal] = []
        self.position_highs: dict[str, float] = {}
        self._async_threads: list[threading.Thread] = []

    def _async_gemini_autopsy(
        self,
        signal: "TradeSignal",
        price: float,
        equity: float,
        order_id: str,
        vix_value,
    ):
        """
        Läuft im Hintergrund-Thread NACH der Order-Ausführung.
        Befragt Gemini retrospektiv — ohne den Trade zu blockieren.
        Ergebnis wird als Autopsy-JSON gespeichert (inkl. ob Gemini zugestimmt hätte).
        """
        def _run():
            try:
                reasoning = self.reasoning.approve_trade(
                    symbol=signal.symbol,
                    signal=signal,
                    price=price,
                    equity=equity,
                    regime=self.risk.regime.value,
                )
                reasoning["cascade_level"] = signal.cascade_level
                reasoning["express_lane"] = True  # Markierung: Trade war bereits ausgeführt
                self.learner.save_autopsy(
                    symbol=signal.symbol,
                    regime=self.risk.regime.value,
                    formula_results=signal.results,
                    reasoning=reasoning,
                    price=price,
                    vix=vix_value,
                    order_id=order_id,
                )
                verdict = "HÄTTE APPROVED" if reasoning.get("approved") else "HÄTTE GEBLOCKT"
                logger.info(
                    f"[EXPRESS LANE AUTOPSY] {signal.symbol}: Gemini {verdict} "
                    f"({reasoning.get('probability_pct', 0)}%) — {reasoning.get('reason', '')}"
                )
            except Exception as e:
                logger.error(f"[EXPRESS LANE AUTOPSY] {signal.symbol}: Fehler: {e}")

        t = threading.Thread(target=_run, daemon=True, name=f"autopsy-{signal.symbol}")
        t.start()
        self._async_threads.append(t)

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

        # Spread fuer dynamische Slippage-Berechnung
        spread = snapshot.get("spread", 0.0) if snapshot else 0.0

        # ── F1: Momentum ──
        try:
            r1 = momentum.evaluate(bars, threshold=Config.MIN_MOMENTUM_SCORE)
            signal.add_result(r1)
        except Exception as e:
            signal.add_result({"name": "Momentum", "signal": 0, "passed": False, "details": {"error": str(e)}})

        # ── F2: Kelly ──
        try:
            r2 = kelly.evaluate(bars, equity=equity, spread=spread)
            signal.add_result(r2)
        except Exception as e:
            signal.add_result({"name": "Kelly", "signal": 0, "passed": False, "details": {"error": str(e)}})

        # ── F3: EV-Gap ──
        try:
            r3 = ev_gap.evaluate(bars, win_prob=0.55, spread=spread)
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

        # ── SCHICHT 2: Reasoning Layer ──
        # EXPRESS LANE: 7/7 oder 6/7 → sofort handeln, Gemini läuft async im Hintergrund
        if signal.cascade_level >= 6:
            express_confidence = 0.85 if signal.cascade_level >= 7 else 0.75
            express_prob = 85 if signal.cascade_level >= 7 else 75
            reasoning = {
                "approved": True,
                "confidence": express_confidence,
                "probability_pct": express_prob,
                "reason": f"Express Lane: {signal.cascade_level}/7 Kaskade — Gemini analysiert async",
                "risk_factors": [],
                "raw": {},
                "prompt": "",
                "raw_response": "EXPRESS_LANE",
            }
            logger.info(
                f"[EXPRESS LANE] {signal.symbol}: {signal.cascade_level}/7 → "
                f"Direkt-Execution ohne Gemini-Wartezeit"
            )
        else:
            # 4/7 oder 5/7 → normaler Gemini-Check (blockierend)
            reasoning = self.reasoning.approve_trade(
                symbol=signal.symbol,
                signal=signal,
                price=price,
                equity=equity,
                regime=self.risk.regime.value,
            )

        signal.reason += f" | Gemini={reasoning.get('probability_pct', round(reasoning['confidence']*100))}%: {reasoning['reason']}"

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
        logger.info(f"Gemini: {reasoning.get('probability_pct', round(reasoning['confidence']*100))}% — {reasoning['reason']}")
        logger.info(f"{'=' * 40}")

        # Stoikov self-deciding: Limit-Order wenn Reservation Price verfuegbar
        stoikov_result = signal.results.get("Stoikov", {})
        stoikov_passed = stoikov_result.get("passed", False)
        reservation_price = stoikov_result.get("details", {}).get("reservation_price")
        market_status = self.broker.get_market_status()

        if stoikov_passed and reservation_price and reservation_price > 0 and market_status == "open":
            logger.info(
                f"[STOIKOV] {signal.symbol}: Limit-Order @ ${reservation_price:.2f} "
                f"(Reservation Price — besser als Market)"
            )
            order_id = self.broker.limit_buy(signal.symbol, signal.qty, reservation_price)
        else:
            # Market-Order: Stoikov nicht relevant (Extended Hours haben eigene Limit-Logik in broker)
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

            # ── Trade Autopsy ──
            try:
                vix_data = self.reasoning.market_ctx.vix.get()
                vix_value = vix_data.get("vix")
            except Exception:
                vix_value = None

            if reasoning.get("raw_response") == "EXPRESS_LANE":
                # Express Lane: Gemini-Analyse läuft async im Hintergrund
                self._async_gemini_autopsy(signal, price, equity, order_id, vix_value)
            else:
                # Normaler Trade: Autopsy sofort (Gemini hat bereits geantwortet)
                reasoning["cascade_level"] = signal.cascade_level
                self.learner.save_autopsy(
                    symbol=signal.symbol,
                    regime=self.risk.regime.value,
                    formula_results=signal.results,
                    reasoning=reasoning,
                    price=price,
                    vix=vix_value,
                    order_id=order_id,
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

    def scan_once(self, market_status: str) -> list[str]:
        """
        Scannt aktive Watchlist + Spike-Sensor Universum.
        Gibt Liste der erkannten Spike-Symbole zurück (für Telegram-Alerts).
        """
        active_watchlist = self.watchlist.get_active_watchlist(market_status)

        # Spike-Sensor: breiter Markt (nur wenn Markt offen/extended)
        spike_symbols: list[str] = []
        if market_status in ("open", "extended"):
            spike_symbols = self.spike_sensor.scan()
            # Spike-Symbole zur Watchlist hinzufügen (keine Duplikate)
            extra = [s for s in spike_symbols if s not in active_watchlist]
            if extra:
                logger.info(f"[SPIKE] {len(extra)} neue Symbole zur Analyse: {', '.join(extra)}")
            active_watchlist = list(dict.fromkeys(active_watchlist + extra))

        logger.info(f"\n{'=' * 60}")
        logger.info(f"  SCAN @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        status_labels = {"open": "REGULÄR 9:30–16:00", "extended": "VOR-/NACHBÖRSE", "closed": "GESCHLOSSEN (Crypto)"}
        logger.info(f"  Markt: {status_labels.get(market_status, market_status)}")
        logger.info(f"  Watchlist ({len(active_watchlist)}): {', '.join(active_watchlist)}")
        logger.info(f"  Regime: {self.risk.regime.value}")
        logger.info(f"{'=' * 60}")

        self.check_exit_conditions()

        # Phase 1 — Parallel: Formel-Analyse + Alpaca-Daten für alle Symbole gleichzeitig
        signals: list[TradeSignal] = [None] * len(active_watchlist)

        def _analyze(idx_sym):
            idx, sym = idx_sym
            try:
                return idx, self.analyze_symbol(sym)
            except Exception as e:
                logger.error(f"Error analyzing {sym}: {e}")
                return idx, None

        max_workers = min(len(active_watchlist), 8)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_analyze, (i, sym)): sym
                       for i, sym in enumerate(active_watchlist)}
            for future in as_completed(futures):
                idx, signal = future.result()
                if signal is not None:
                    signals[idx] = signal

        # Phase 2 — Sequenziell: Gemini Reasoning + Order-Ausführung
        for signal in signals:
            if signal is None:
                continue
            print(signal.summary())
            if signal.all_passed:
                self.execute_signal(signal)

        equity = self.broker.get_equity()
        positions = self.broker.get_positions()
        logger.info(f"Equity: ${equity:,.2f}  |  Positions: {len(positions)}  |  Trades: {len(self.trade_log)}")
        return spike_symbols

    def run(self):
        logger.info("=" * 60)
        logger.info("  7 FILTERS. Gemini REASONING. 24/7.")
        logger.info(f"  Mode: {'PAPER' if Config.is_paper() else '!! LIVE !!'}")
        logger.info(f"  Base Watchlist: {Config.WATCHLIST}")
        logger.info(f"  Dynamic Discovery: alle 1h via Gemini")
        logger.info("=" * 60)

        while True:
            try:
                market_status = self.broker.get_market_status()

                if market_status == "closed":
                    logger.info("Boerse + Extended Hours geschlossen (Nacht/Wochenende) — warte 5min...")
                    time.sleep(300)
                    continue

                self.scan_once(market_status)  # spike_symbols ignoriert in standalone run()

            except KeyboardInterrupt:
                logger.info("\nShutting down...")
                break
            except Exception as e:
                logger.error(f"Scan error: {e}")

            interval = Config.SCAN_INTERVAL
            logger.info(f"Next scan in {interval}s...")
            time.sleep(interval)
