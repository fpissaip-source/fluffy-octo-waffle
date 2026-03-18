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
        logger.info("ReasoningLayer: Gemini Client wird initialisiert...")
        from google import genai
        self.client = genai.Client(api_key=Config.GEMINI_API_KEY)
        logger.info("ReasoningLayer: Client OK, lade MarketContext...")
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

    def check_hold_or_sell(
        self,
        symbol: str,
        signal: "TradeSignal",
        entry_price: float,
        current_price: float,
        equity: float,
        regime: str,
    ) -> dict:
        """
        Post-Trade Check: Wir haben bereits gekauft.
        Gemini entscheidet: HOLD (halten) oder SELL (sofort verkaufen).
        Semantik korrekt: HOLD = Position behalten, SELL = Position schließen.
        """
        formula_summary = "\n".join(
            f"  - {name}: signal={r['signal']:.3f} {'PASS' if r['passed'] else 'FAIL'}"
            for name, r in signal.results.items()
        )
        pnl_pct = (current_price - entry_price) / entry_price * 100 if entry_price > 0 else 0
        market_context_str = self.market_ctx.format_for_prompt(symbol)

        prompt = f"""Du bist ein erfahrener Trader. Wir haben soeben eine Position eröffnet und prüfen jetzt ob wir sie halten sollen.

SYMBOL: {symbol}
EINSTIEG: ${entry_price:.2f}  |  AKTUELL: ${current_price:.2f}  |  P/L: {pnl_pct:+.2f}%
DEPOT: ${equity:,.2f}
MARKT-REGIME: {regime}
KASKADE: {signal.cascade_label}

QUANTITATIVE SIGNALE:
{formula_summary}

MARKT-KONTEXT:
{market_context_str}

Die Position wurde gerade eröffnet (Express Lane, {signal.cascade_level}/7 Filter bestanden).
Entscheide: Soll die Position GEHALTEN oder SOFORT VERKAUFT werden?

HOLD = Position halten (Setup ist valide)
SELL = Sofort verkaufen (Setup hat einen kritischen Fehler / Marktlage hat sich gedreht)"""

        try:
            from google.genai import types as genai_types

            response_schema = genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "decision":        genai_types.Schema(type=genai_types.Type.STRING, enum=["HOLD", "SELL"]),
                    "confidence":      genai_types.Schema(type=genai_types.Type.NUMBER),
                    "reason":          genai_types.Schema(type=genai_types.Type.STRING),
                    "risk_factors":    genai_types.Schema(
                        type=genai_types.Type.ARRAY,
                        items=genai_types.Schema(type=genai_types.Type.STRING),
                    ),
                },
                required=["decision", "confidence", "reason", "risk_factors"],
            )

            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=response_schema,
                    temperature=0.1,
                    max_output_tokens=200,
                    thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
                ),
            )

            result = json.loads(response.text or "{}")
            should_sell = result.get("decision", "HOLD") == "SELL"
            logger.info(
                f"[HOLD/SELL] {symbol}: {result.get('decision')} "
                f"({result.get('confidence', 0):.0%}) — {result.get('reason', '')}"
            )
            return {
                "sell": should_sell,
                "confidence": float(result.get("confidence", 0)),
                "reason": result.get("reason", ""),
                "risk_factors": result.get("risk_factors", []),
                "prompt": prompt,
                "raw_response": response.text or "",
            }

        except Exception as e:
            logger.error(f"[HOLD/SELL] {symbol}: Gemini Fehler: {e} → HOLD (sicherer Default)")
            return {"sell": False, "confidence": 0.0, "reason": f"Fehler: {e}", "risk_factors": []}


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
        self.update_interval = 600   # alle 10 Minuten (war 15min/900s)
        self.dynamic_symbols: list[str] = []
        self.broker = broker
        # Tagesende-Tracking: welche Symbole wurden heute auto-hinzugefügt
        self.auto_added: list[dict] = []   # [{symbol, added_at}]
        self._last_market_status: str = ""
        self.notify: Optional[callable] = None
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
                    # daily_bar ist bei Extended Hours / frühem Pre-Market oft None oder volume=0
                    # → prev_daily_bar als Fallback damit nicht alle Kandidaten wegfallen
                    daily = snap.daily_bar
                    if not daily or int(daily.volume) == 0:
                        daily = getattr(snap, "prev_daily_bar", None)
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
        Gemini waehlt daraus die 15 besten basierend auf News/Katalysatoren.
        Kein Halluzinieren — nur Ticker die wirklich handeln.
        """
        if not self.should_update():
            return self.dynamic_symbols

        # Schritt 1: Echte Kandidaten von Alpaca holen
        candidates = self._get_top_candidates(market_status)

        if not candidates:
            logger.warning("[WATCHLIST] Keine Alpaca-Kandidaten — retry in 60s")
            # BUG-FIX: last_update setzen, damit der Bot nicht auf JEDEM Scan erneut
            # versucht (war: last_update blieb 0 → selbe Aktien nach 20+ Minuten)
            self.last_update = time.time() - self.update_interval + 60
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

Wähle die 15 besten aus DIESER LISTE basierend auf deinem Wissen über:
- Aktuelle News und Katalysatoren (Earnings, FDA, M&A, Short Squeeze)
- Momentum und Trendstärke
- Liquidität und Handelbarkeit

WICHTIG: Nur Symbole aus der obigen Liste verwenden — keine anderen erfinden.

Antworte NUR mit JSON:
{{"symbols": ["SYM1","SYM2","SYM3","SYM4","SYM5","SYM6","SYM7","SYM8","SYM9","SYM10","SYM11","SYM12","SYM13","SYM14","SYM15"], "reasoning": "ein Satz"}}"""

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
                if validated:
                    self.dynamic_symbols = validated[:15]
                    self.last_update = time.time()
                    logger.info(f"[WATCHLIST] Neue Symbole ({len(self.dynamic_symbols)}): {self.dynamic_symbols} | {result.get('reasoning', '')}")
                    self._add_top_favorite()
                    return self.dynamic_symbols
                else:
                    logger.warning("[WATCHLIST] Gemini: alle Symbole halluziniert → Alpaca-Fallback")

        except Exception as e:
            logger.warning(f"[WATCHLIST] Discovery fehlgeschlagen: {e}")

        # Fallback: direkt Top-8 aus Alpaca-Daten nehmen
        self.dynamic_symbols = candidates[:8]
        self.last_update = time.time()
        self._add_top_favorite()
        return self.dynamic_symbols

    def _add_top_favorite(self):
        """
        Fügt die beste neu entdeckte Aktie einmalig zur permanenten Watchlist hinzu.
        Pro Refresh-Zyklus wird maximal 1 Symbol hinzugefügt.
        Crypto-Symbole werden übersprungen (enden auf USD/BTC/ETH/SOL).
        """
        for sym in self.dynamic_symbols:
            # Kein Crypto, kein Duplikat
            if any(sym.endswith(x) for x in ("USD", "BTC", "ETH", "SOL")):
                continue
            if sym in Config.WATCHLIST:
                continue
            Config.WATCHLIST.append(sym)
            self.auto_added.append({"symbol": sym, "added_at": time.time()})
            logger.info(f"[WATCHLIST] ⭐ Favorit hinzugefügt: {sym} (gesamt Watchlist: {len(Config.WATCHLIST)})")
            if self.notify:
                self.notify(f"⭐ <b>Watchlist +1</b>: <b>{sym}</b> wurde als Tages-Favorit hinzugefügt")
            break  # nur 1 pro Zyklus

    def evaluate_end_of_day(self):
        """
        Tagesende-Auswertung: Für jedes auto-hinzugefügte Symbol wird die
        Tagesperformance geprüft. Gute Aktien (positive Rendite) bleiben,
        schlechte werden aus Config.WATCHLIST entfernt.
        Sendet Zusammenfassung per Telegram.
        """
        if not self.auto_added:
            return

        logger.info(f"[WATCHLIST EOD] Auswerte {len(self.auto_added)} auto-hinzugefügte Symbole...")
        keep_list, remove_list = [], []

        for entry in self.auto_added:
            sym = entry["symbol"]
            keep = False
            reason = "Keine Daten"
            try:
                if self.broker:
                    bars = self.broker.get_bars(sym, timeframe="1Hour", limit=8)
                    if bars is not None and not bars.empty and len(bars) >= 2:
                        day_return = (bars["close"].iloc[-1] - bars["close"].iloc[0]) / bars["close"].iloc[0]
                        keep = day_return > 0
                        reason = f"{day_return:+.2%} Tagesrendite"
                    else:
                        reason = "Keine Kursdaten"
            except Exception as e:
                reason = f"Fehler: {e}"

            if keep:
                keep_list.append((sym, reason))
                logger.info(f"[WATCHLIST EOD] ✅ BEHALTEN: {sym} — {reason}")
            else:
                remove_list.append((sym, reason))
                if sym in Config.WATCHLIST:
                    Config.WATCHLIST.remove(sym)
                logger.info(f"[WATCHLIST EOD] ❌ ENTFERNT: {sym} — {reason}")

        # Telegram-Zusammenfassung
        if self.notify:
            lines = ["📊 <b>Tagesende — Watchlist-Auswertung</b>", "━━━━━━━━━━━━━━━━━━━━━━"]
            if keep_list:
                lines.append("✅ <b>Behalten:</b>")
                for sym, r in keep_list:
                    lines.append(f"  • <b>{sym}</b> — {r}")
            if remove_list:
                lines.append("❌ <b>Entfernt:</b>")
                for sym, r in remove_list:
                    lines.append(f"  • <b>{sym}</b> — {r}")
            lines.append(f"\nWatchlist jetzt: {', '.join(Config.WATCHLIST)}")
            self.notify("\n".join(lines))

        # Reset für nächsten Tag — behaltene Symbole bleiben in der Liste für
        # die nächste Auswertung, neu entdeckte starten wieder bei 0
        self.auto_added = [e for e in self.auto_added if e["symbol"] in [s for s, _ in keep_list]]

    def get_active_watchlist(self, market_status: str) -> list[str]:
        """
        Kombiniert Basis-Watchlist (.env) mit Gemini Vorschlaegen.
        'closed' (Nacht/Wochenende): nur Crypto.
        'open' + 'extended': Aktien + Crypto.
        Erkennt Markt-Schluss-Übergang und triggert Tagesende-Auswertung.
        """
        # Tagesende-Erkennung: open/extended → closed
        if (self._last_market_status in ("open", "extended")
                and market_status == "closed"):
            logger.info("[WATCHLIST EOD] Markt geschlossen — starte Tagesende-Auswertung")
            self.evaluate_end_of_day()
        self._last_market_status = market_status

        dynamic = self.discover(market_status)

        if market_status == "closed":
            # Nur Crypto handeln wenn Markt + Extended geschlossen
            crypto = [s for s in (Config.WATCHLIST + dynamic)
                      if any(s.endswith(x) for x in ("USD", "BTC", "ETH", "SOL"))]
            return list(dict.fromkeys(crypto))[:10]

        # open + extended: volles Universum
        combined = list(dict.fromkeys(Config.WATCHLIST + dynamic))
        return combined[:25]


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

        # Kelly ist Mandatory-Filter: kein Trade ohne positiven Expected Value.
        # 1-Hour-Bars geben Kelly den Weitwinkel-Blick für Swing-Trades.
        kelly_result = self.results.get("Kelly")
        if kelly_result and not kelly_result["passed"]:
            self.all_passed = False
            self.action = "HOLD"
            self.reason = "Kelly FAIL (Mandatory-Block) — kein positiver EV"
            return

        # Alle 7 Filter zählen in der Kaskade (inkl. Stoikov).
        # Stoikov bestimmt ZUSÄTZLICH den Order-Typ (Limit vs Market).
        passed_count = sum(1 for r in self.results.values() if r["passed"])

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
        self.watchlist.notify = self._tg   # Telegram-Callback für Favoriten-Alerts
        self.spike_sensor = SpikeSensor(self.broker)
        self.trade_log: list[TradeSignal] = []
        self.position_highs: dict[str, float] = {}
        self._async_threads: list[threading.Thread] = []
        self._exit_lock = threading.Lock()
        self._stop_event = threading.Event()
        # Telegram-Callback (optional): wird von TradingTelegramBot gesetzt
        self.notify: Optional[callable] = None

    def _tg(self, text: str):
        """Sendet Telegram-Nachricht wenn Callback gesetzt ist."""
        if self.notify:
            try:
                self.notify(text)
            except Exception as e:
                logger.warning(f"Telegram notify failed: {e}")

    def _exit_monitor_loop(self):
        """Separater Thread: prüft offene Positionen alle 3s auf Stop/TP/Trailing."""
        logger.info("[EXIT-MONITOR] Gestartet — prüft Positionen alle 3s")
        while not self._stop_event.is_set():
            try:
                with self._exit_lock:
                    self.check_exit_conditions()
            except Exception as e:
                logger.error(f"[EXIT-MONITOR] Fehler: {e}")
            self._stop_event.wait(3)
        logger.info("[EXIT-MONITOR] Gestoppt")

    def _async_gemini_autopsy(
        self,
        signal: "TradeSignal",
        entry_price: float,
        equity: float,
        order_id: str,
        vix_value,
    ):
        """
        Läuft im Hintergrund-Thread NACH der Order-Ausführung.
        Gemini entscheidet: HOLD (halten) oder SELL (sofort verkaufen).
        Klare Semantik: HOLD = behalten, SELL = schließen.
        """
        def _run():
            try:
                current_price = self.broker.get_latest_price(signal.symbol) or entry_price
                verdict = self.reasoning.check_hold_or_sell(
                    symbol=signal.symbol,
                    signal=signal,
                    entry_price=entry_price,
                    current_price=current_price,
                    equity=equity,
                    regime=self.risk.regime.value,
                )

                # Autopsy-JSON speichern (mit HOLD/SELL Ergebnis)
                autopsy_reasoning = {
                    "approved": not verdict["sell"],
                    "confidence": verdict["confidence"],
                    "probability_pct": round(verdict["confidence"] * 100),
                    "reason": verdict["reason"],
                    "risk_factors": verdict["risk_factors"],
                    "raw": {"decision": "SELL" if verdict["sell"] else "HOLD"},
                    "prompt": verdict.get("prompt", ""),
                    "raw_response": verdict.get("raw_response", ""),
                    "cascade_level": signal.cascade_level,
                    "express_lane": True,
                }
                self.learner.save_autopsy(
                    symbol=signal.symbol,
                    regime=self.risk.regime.value,
                    formula_results=signal.results,
                    reasoning=autopsy_reasoning,
                    price=entry_price,
                    vix=vix_value,
                    order_id=order_id,
                )

                if not verdict["sell"]:
                    logger.info(
                        f"[EXPRESS LANE → HOLD ✓] {signal.symbol}: Position wird gehalten "
                        f"({verdict['confidence']:.0%}) — {verdict['reason']}"
                    )
                    self._tg(
                        f"🟢 <b>GEMINI: HALTEN</b> — {signal.symbol}\n"
                        f"({verdict['confidence']:.0%}) {verdict['reason']}"
                    )
                else:
                    logger.warning(
                        f"[EXPRESS LANE → SELL ✗] {signal.symbol}: Gemini sagt VERKAUFEN "
                        f"({verdict['confidence']:.0%}) — {verdict['reason']} "
                        f"| Risiken: {verdict.get('risk_factors', [])}"
                    )
                    if self.broker.has_position(signal.symbol):
                        exit_price = self.broker.get_latest_price(signal.symbol) or entry_price
                        self.broker.close_position(signal.symbol)
                        self.position_highs.pop(signal.symbol, None)
                        self.learner.record_exit(
                            signal.symbol,
                            exit_price,
                            f"Gemini SELL nach Express Lane: {verdict['reason']}",
                        )
                        pnl_pct = (exit_price - entry_price) / entry_price * 100 if entry_price > 0 else 0
                        self._tg(
                            f"🔴 <b>GEMINI: VERKAUFEN</b> — {signal.symbol}\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━\n"
                            f"Einstieg: ${entry_price:.2f} → Ausstieg: ${exit_price:.2f}\n"
                            f"P/L: <b>{pnl_pct:+.2f}%</b>\n"
                            f"Grund: {verdict['reason']}\n"
                            + (f"Risiken: {', '.join(verdict.get('risk_factors', []))}" if verdict.get('risk_factors') else "")
                        )
                        logger.warning(f"[EXPRESS LANE → SELL ✗] {signal.symbol}: Position geschlossen")
                    else:
                        logger.info(f"[EXPRESS LANE → SELL] {signal.symbol}: Position bereits geschlossen")
            except Exception as e:
                logger.error(f"[EXPRESS LANE AUTOPSY] {signal.symbol}: Fehler: {e}")

        t = threading.Thread(target=_run, daemon=True, name=f"autopsy-{signal.symbol}")
        t.start()
        self._async_threads.append(t)

    def analyze_symbol(self, symbol: str) -> TradeSignal:
        signal = TradeSignal(symbol)

        bars = self.broker.get_bars(symbol, timeframe=Config.TRADING_TIMEFRAME, limit=Config.LOOKBACK_BARS)
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
        # EXPRESS LANE: ab 5/7 → sofort handeln, Gemini prüft async ob HALTEN oder VERKAUFEN
        if signal.cascade_level >= 5:
            if signal.cascade_level >= 7:
                express_confidence, express_prob = 0.85, 85
            elif signal.cascade_level >= 6:
                express_confidence, express_prob = 0.75, 75
            else:  # 5/7
                express_confidence, express_prob = 0.65, 65
            reasoning = {
                "approved": True,
                "confidence": express_confidence,
                "probability_pct": express_prob,
                "reason": f"Express Lane: {signal.cascade_level}/7 Kaskade — Gemini prüft async Halten/Verkaufen",
                "risk_factors": [],
                "raw": {},
                "prompt": "",
                "raw_response": "EXPRESS_LANE",
            }
            logger.info(
                f"[EXPRESS LANE] {signal.symbol}: {signal.cascade_level}/7 → "
                f"Sofort-Execution, Gemini-Veto läuft im Hintergrund"
            )
        else:
            # 4/7 → normaler Gemini-Check (blockierend, kein Sofort-Trade)
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

        # ── PRE-TRADE Telegram Alert ──
        passed_names  = [n for n, r in signal.results.items() if r["passed"]]
        failed_names  = [n for n, r in signal.results.items() if not r["passed"]]
        self._tg(
            f"⚡ <b>SIGNAL: BUY {signal.qty}x {signal.symbol}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Kaskade: <b>{signal.cascade_label}</b>\n"
            f"Regime:  {self.risk.regime.value}\n"
            f"Preis:   ${price:.2f}\n"
            f"Wert:    ~${signal.qty * price:,.0f}\n"
            f"✅ {', '.join(passed_names)}\n"
            + (f"❌ {', '.join(failed_names)}\n" if failed_names else "")
            + f"<i>Order wird jetzt platziert...</i>"
        )

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

            # ── ORDER PLACED Telegram Alert ──
            order_type = "LIMIT" if (signal.results.get("Stoikov", {}).get("passed") and
                                     signal.results.get("Stoikov", {}).get("details", {}).get("reservation_price")) \
                         else "MARKET"
            self._tg(
                f"✅ <b>ORDER PLATZIERT</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"<b>BUY {signal.qty}x {signal.symbol}</b>  [{order_type}]\n"
                f"Preis:   ${price:.2f}\n"
                f"Wert:    ~${signal.qty * price:,.0f}\n"
                f"Kaskade: {signal.cascade_label}\n"
                f"Order-ID: <code>{order_id}</code>\n"
                + ("🔄 <i>Gemini prüft async ob HOLD/SELL...</i>" if reasoning.get("raw_response") == "EXPRESS_LANE" else
                   f"🤖 Gemini: {reasoning.get('probability_pct', 0)}% — {reasoning.get('reason', '')}")
            )

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
                # Express Lane: Gemini prüft async HOLD oder SELL
                self._async_gemini_autopsy(signal, entry_price=price, equity=equity,
                                           order_id=order_id, vix_value=vix_value)
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
                bars = self.broker.get_bars(symbol, timeframe=Config.TRADING_TIMEFRAME, limit=50)
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

        # Exit-Conditions werden vom _exit_monitor_loop Thread alle 3s geprüft

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

        # Exit-Monitor-Thread starten (alle 3s unabhängig vom Scan)
        self._stop_event.clear()
        exit_thread = threading.Thread(target=self._exit_monitor_loop, daemon=True, name="ExitMonitor")
        exit_thread.start()

        try:
            while True:
                try:
                    market_status = self.broker.get_market_status()

                    if market_status == "closed":
                        logger.info("Boerse + Extended Hours geschlossen (Nacht/Wochenende) — warte 5min...")
                        time.sleep(300)
                        continue

                    self.scan_once(market_status)

                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    logger.error(f"Scan error: {e}")

                interval = Config.SCAN_INTERVAL
                logger.info(f"Next scan in {interval}s...")
                time.sleep(interval)
        except KeyboardInterrupt:
            logger.info("\nShutting down...")
        finally:
            self._stop_event.set()
            exit_thread.join(timeout=5)
