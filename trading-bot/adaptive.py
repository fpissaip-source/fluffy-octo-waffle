"""
adaptive.py — Reinforcement Learning / Adaptives Lernen

Aus dem PDF: "Erkennung von Marktphasen mittels Reinforcement Learning
ohne Overfitting" und "Reinforcement Learning (stetig)"

Ansatz: Contextual Bandit
- Jeder Trade wird mit seinem Kontext gespeichert
  (Regime, Formel-Scores, Sentiment, Ergebnis)
- Das System lernt welche Formel-Kombinationen
  in welchem Regime am besten funktionieren
- Passt Gewichtungen und Thresholds automatisch an

KEIN Deep RL (wuerde Overfitting erzeugen bei wenig Daten).
Stattdessen: Bayesian Weight Updates — robust, interpretierbar.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

from config import Config

logger = logging.getLogger("bot.adaptive")

# Pfad fuer persistente Daten
DATA_DIR = Path(__file__).parent / "data"
TRADE_LOG_FILE = DATA_DIR / "trade_history.json"
WEIGHTS_FILE = DATA_DIR / "formula_weights.json"
AUTOPSY_DIR = Path(__file__).parent / "autopsy"
BLACKLIST_FILE = DATA_DIR / "blacklist.json"
SL_EVENTS_FILE = DATA_DIR / "sl_events.json"
LEARNING_SUMMARY_FILE = DATA_DIR / "learning_summary.json"


# ═══════════════════════════════════════════════════════
#  TRADE RECORD
# ═══════════════════════════════════════════════════════

class TradeRecord:
    """Einzelner Trade mit Kontext."""

    def __init__(
        self,
        symbol: str,
        regime: str,
        formula_scores: dict[str, float],
        sentiment_score: float,
        entry_price: float,
        qty: int,
    ):
        self.symbol = symbol
        self.regime = regime
        self.formula_scores = formula_scores
        self.sentiment_score = sentiment_score
        self.entry_price = entry_price
        self.qty = qty
        self.entry_time = datetime.now().isoformat()
        self.exit_price: Optional[float] = None
        self.exit_time: Optional[str] = None
        self.pnl: float = 0.0
        self.pnl_pct: float = 0.0
        self.exit_reason: str = ""

    def close(self, exit_price: float, reason: str = ""):
        self.exit_price = exit_price
        self.exit_time = datetime.now().isoformat()
        self.pnl = (exit_price - self.entry_price) * self.qty
        self.pnl_pct = (exit_price - self.entry_price) / self.entry_price
        self.exit_reason = reason

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "regime": self.regime,
            "formula_scores": self.formula_scores,
            "sentiment": self.sentiment_score,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "qty": self.qty,
            "pnl": round(self.pnl, 2),
            "pnl_pct": round(self.pnl_pct, 4),
            "entry_time": self.entry_time,
            "exit_time": self.exit_time,
            "exit_reason": self.exit_reason,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TradeRecord":
        rec = cls(
            symbol=d["symbol"],
            regime=d["regime"],
            formula_scores=d["formula_scores"],
            sentiment_score=d.get("sentiment", 0),
            entry_price=d["entry_price"],
            qty=d["qty"],
        )
        rec.exit_price = d.get("exit_price")
        rec.exit_time = d.get("exit_time")
        rec.pnl = d.get("pnl", 0)
        rec.pnl_pct = d.get("pnl_pct", 0)
        rec.entry_time = d.get("entry_time", "")
        rec.exit_reason = d.get("exit_reason", "")
        return rec


# ═══════════════════════════════════════════════════════
#  FORMULA WEIGHTS (pro Regime)
# ═══════════════════════════════════════════════════════

DEFAULT_WEIGHTS = {
    "CALM":     {"Momentum": 1.0, "Kelly": 1.0, "EV-Gap": 1.0, "KL-Divergence": 1.0, "Bayesian": 1.0, "Stoikov": 1.0, "Sentiment": 0.8},
    "NORMAL":   {"Momentum": 1.0, "Kelly": 1.0, "EV-Gap": 1.0, "KL-Divergence": 1.0, "Bayesian": 1.0, "Stoikov": 1.0, "Sentiment": 1.0},
    "VOLATILE": {"Momentum": 0.8, "Kelly": 1.2, "EV-Gap": 1.0, "KL-Divergence": 1.2, "Bayesian": 1.0, "Stoikov": 1.2, "Sentiment": 1.3},
    "CRISIS":   {"Momentum": 0.6, "Kelly": 1.3, "EV-Gap": 0.8, "KL-Divergence": 1.0, "Bayesian": 1.2, "Stoikov": 1.5, "Sentiment": 1.5},
}


# ═══════════════════════════════════════════════════════
#  ADAPTIVE LEARNER
# ═══════════════════════════════════════════════════════

class AdaptiveLearner:
    """
    Lernt aus vergangenen Trades und passt Formel-Gewichte an.

    Algorithmus:
    1. Speichert jeden Trade mit allen Formel-Scores
    2. Gruppiert nach Regime
    3. Fuer jede Formel: Korrelation zwischen Score und P/L
    4. Formeln die mit P/L korrelieren bekommen mehr Gewicht
    5. Formeln die nicht korrelieren verlieren Gewicht

    Update-Regel (Bayesian):
    new_weight = old_weight * (1 + learning_rate * correlation)

    Clamped auf [0.3, 2.0] um extreme Anpassungen zu vermeiden.
    """

    def __init__(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.trade_history: list[TradeRecord] = []
        self.weights: dict[str, dict[str, float]] = {}
        self.learning_rate = 0.15  # Angepasst für 15Min-Takt
        self.min_trades_to_learn = 10  # Mindestens 10 Trades bevor angepasst wird
        self.max_history = 500  # Max Trades im Speicher

        # ── Auto-Blacklist ──────────────────────────────
        # {symbol: [unix_timestamp, ...]} — Stop-Loss Ereignisse
        self.sl_events: dict[str, list[float]] = {}
        # {symbol: unix_timestamp_expiry} — geblockte Symbole
        self.blacklist: dict[str, float] = {}
        self.sl_blacklist_window_h = 48    # Fenster in dem 2x SL zählt
        self.sl_blacklist_duration_d = 7   # Sperre in Tagen

        # ── Learning Summary (Gemini Autopsy-Feedback) ──
        self._learning_summary: str = ""
        self._learning_summary_updated: float = 0
        self.learning_summary_ttl_h = 2    # Alle 2h neu generieren

        self._load()
        self._load_blacklist()

    # ── Persistenz ──────────────────────────────────────

    def _load(self):
        """Laedt Trade-History und Weights von Disk."""
        # Weights
        if WEIGHTS_FILE.exists():
            try:
                with open(WEIGHTS_FILE) as f:
                    self.weights = json.load(f)
                logger.info(f"Loaded weights from {WEIGHTS_FILE}")
            except Exception:
                self.weights = {k: dict(v) for k, v in DEFAULT_WEIGHTS.items()}
        else:
            self.weights = {k: dict(v) for k, v in DEFAULT_WEIGHTS.items()}

        # Trade History
        if TRADE_LOG_FILE.exists():
            try:
                with open(TRADE_LOG_FILE) as f:
                    data = json.load(f)
                self.trade_history = [TradeRecord.from_dict(d) for d in data]
                logger.info(f"Loaded {len(self.trade_history)} trade records")
            except Exception:
                self.trade_history = []

    def _save(self):
        """Speichert Trade-History und Weights auf Disk."""
        try:
            with open(WEIGHTS_FILE, "w") as f:
                json.dump(self.weights, f, indent=2)

            # Nur die letzten N Trades speichern
            records = self.trade_history[-self.max_history:]
            with open(TRADE_LOG_FILE, "w") as f:
                json.dump([r.to_dict() for r in records], f, indent=2)

        except Exception as e:
            logger.error(f"Save failed: {e}")

    # ── Blacklist Persistenz ─────────────────────────────

    def _load_blacklist(self):
        """Laedt Blacklist und SL-Events von Disk."""
        try:
            if BLACKLIST_FILE.exists():
                with open(BLACKLIST_FILE) as f:
                    data = json.load(f)
                self.blacklist = data.get("blacklist", {})
                self.sl_events = data.get("sl_events", {})
                # Abgelaufene Eintraege bereinigen
                now = datetime.now().timestamp()
                self.blacklist = {s: exp for s, exp in self.blacklist.items() if exp > now}
                logger.info(f"[BLACKLIST] Geladen: {len(self.blacklist)} gesperrte Symbole")
        except Exception as e:
            logger.warning(f"[BLACKLIST] Laden fehlgeschlagen: {e}")

    def _save_blacklist(self):
        """Speichert Blacklist und SL-Events auf Disk."""
        try:
            with open(BLACKLIST_FILE, "w") as f:
                json.dump({"blacklist": self.blacklist, "sl_events": self.sl_events}, f, indent=2)
        except Exception as e:
            logger.error(f"[BLACKLIST] Speichern fehlgeschlagen: {e}")

    # ── Auto-Blacklist Logik ─────────────────────────────

    def record_stop_loss(self, symbol: str):
        """
        Registriert ein Stop-Loss Ereignis fuer ein Symbol.
        Wenn 2x Stop-Loss in 48h → Symbol fuer 7 Tage gesperrt.
        """
        now = datetime.now().timestamp()
        window = self.sl_blacklist_window_h * 3600

        # SL-Events speichern (nur die letzten 48h behalten)
        events = self.sl_events.get(symbol, [])
        events = [t for t in events if now - t < window]
        events.append(now)
        self.sl_events[symbol] = events

        logger.info(f"[BLACKLIST] {symbol}: Stop-Loss #{len(events)} in {self.sl_blacklist_window_h}h")

        if len(events) >= 2:
            expiry = now + self.sl_blacklist_duration_d * 86400
            self.blacklist[symbol] = expiry
            expiry_dt = datetime.fromtimestamp(expiry).strftime("%Y-%m-%d %H:%M")
            logger.warning(
                f"[BLACKLIST] {symbol}: 2x Stop-Loss in {self.sl_blacklist_window_h}h "
                f"→ GESPERRT bis {expiry_dt}"
            )
            self._save_blacklist()
            return True  # Neu gesperrt

        self._save_blacklist()
        return False

    def is_blacklisted(self, symbol: str) -> bool:
        """Prueft ob ein Symbol gesperrt ist (auto-expire beachten)."""
        if symbol not in self.blacklist:
            return False
        now = datetime.now().timestamp()
        if self.blacklist[symbol] <= now:
            del self.blacklist[symbol]
            self._save_blacklist()
            logger.info(f"[BLACKLIST] {symbol}: Sperre abgelaufen — wieder handelbar")
            return False
        return True

    def temp_blacklist(self, symbol: str, minutes: int = 15):
        """Sperrt ein Symbol temporaer fuer X Minuten (z.B. nach manuellem closeall)."""
        expiry = datetime.now().timestamp() + minutes * 60
        self.blacklist[symbol] = expiry
        expiry_dt = datetime.fromtimestamp(expiry).strftime("%H:%M")
        logger.info(f"[BLACKLIST] {symbol}: Temporaer gesperrt fuer {minutes}min (bis {expiry_dt})")
        self._save_blacklist()

    def get_blacklist_status(self) -> dict:
        """Gibt aktuelle Blacklist fuer Dashboard/Telegram zurueck."""
        now = datetime.now().timestamp()
        active = {}
        for sym, expiry in self.blacklist.items():
            if expiry > now:
                remaining_h = (expiry - now) / 3600
                active[sym] = {
                    "expiry": datetime.fromtimestamp(expiry).strftime("%Y-%m-%d %H:%M"),
                    "remaining_h": round(remaining_h, 1),
                }
        return active

    # ── Learning Summary (Gemini liest Autopsies) ────────

    def get_learning_summary(self) -> str:
        """
        Gibt die gecachte Lern-Zusammenfassung zurueck.
        Wird von ReasoningLayer in den approve_trade Prompt injiziert.
        Leer wenn noch keine Zusammenfassung generiert wurde.
        """
        if LEARNING_SUMMARY_FILE.exists() and not self._learning_summary:
            try:
                with open(LEARNING_SUMMARY_FILE) as f:
                    data = json.load(f)
                self._learning_summary = data.get("summary", "")
                self._learning_summary_updated = data.get("updated", 0)
            except Exception:
                pass
        return self._learning_summary

    def should_refresh_learning_summary(self) -> bool:
        """Prueft ob die Lern-Zusammenfassung aktualisiert werden soll."""
        import time
        ttl = self.learning_summary_ttl_h * 3600
        return (time.time() - self._learning_summary_updated) > ttl

    def generate_learning_summary(self, gemini_client) -> str:
        """
        Gemini analysiert die letzten Autopsy-JSONs + Trade-History
        und erstellt eine kompakte Zusammenfassung der Lernmuster.

        Fokus:
        - Welche Setups haben verloren? (insbesondere Latenz-Probleme)
        - Welche Chart-Muster sollten vermieden werden (Yahoo-Fallback)?
        - Welche Formeln haben in welchem Regime gut/schlecht performt?

        Wird alle 2h neu generiert und gecacht.
        Ergebnis wird in approve_trade-Prompt injiziert.
        """
        import time

        closed_trades = [t for t in self.trade_history if t.exit_price is not None]
        if len(closed_trades) < 3:
            logger.info("[LEARNING] Zu wenig abgeschlossene Trades fuer Zusammenfassung")
            return ""

        # Letzte 30 Trades fuer Analyse
        recent = closed_trades[-30:]

        # Autopsy-Dateien der letzten Trades einlesen
        autopsy_data = []
        try:
            autopsy_files = sorted(AUTOPSY_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
            for f in autopsy_files[:20]:
                try:
                    with open(f) as af:
                        autopsy_data.append(json.load(af))
                except Exception:
                    continue
        except Exception as e:
            logger.warning(f"[LEARNING] Autopsy-Dateien lesen fehlgeschlagen: {e}")

        # Trade-Zusammenfassung erstellen
        trade_lines = []
        for t in recent:
            pnl_str = f"{t.pnl_pct:+.2%}"
            outcome = "GEWINN" if t.pnl_pct > 0 else "VERLUST"
            trade_lines.append(
                f"- {t.symbol} [{t.regime}] {pnl_str} ({outcome}) "
                f"Grund: {t.exit_reason or 'unbekannt'}"
            )

        trades_text = "\n".join(trade_lines) if trade_lines else "Keine abgeschlossenen Trades"

        # Autopsy-Zusammenfassung (nur Verluste mit Details)
        autopsy_lines = []
        for a in autopsy_data[:10]:
            sym = a.get("symbol", "?")
            regime = a.get("regime", "?")
            formulas = a.get("formula_results", {})
            failed = [n for n, r in formulas.items() if not r.get("passed")]
            # Passenden Trade in History finden
            matching = [t for t in recent if t.symbol == sym and t.pnl_pct < 0]
            if matching:
                worst = min(matching, key=lambda t: t.pnl_pct)
                autopsy_lines.append(
                    f"- {sym} [{regime}]: Verlust {worst.pnl_pct:+.2%} | "
                    f"Fehlgeschlagen: {', '.join(failed) or 'keine'} | "
                    f"Grund: {worst.exit_reason or '?'}"
                )

        autopsy_text = "\n".join(autopsy_lines) if autopsy_lines else "Keine Verlust-Autopsien"

        # Statistiken
        wins = [t for t in recent if t.pnl_pct > 0]
        losses = [t for t in recent if t.pnl_pct <= 0]
        sl_losses = [t for t in losses if "STOP LOSS" in (t.exit_reason or "")]

        prompt = f"""Du bist ein quantitativer Trading-Analyst. Analysiere die folgenden letzten Trades und erstelle eine KOMPAKTE Lern-Zusammenfassung (max. 8 Sätze auf Deutsch).

TRADING-STATISTIK (letzte {len(recent)} Trades):
- Gewinne: {len(wins)} | Verluste: {len(losses)}
- Avg Gewinn: {sum(t.pnl_pct for t in wins)/len(wins):.2%} | Avg Verlust: {sum(t.pnl_pct for t in losses)/len(losses):.2%}
- Stop-Loss-Auslösungen: {len(sl_losses)}

LETZTE TRADES:
{trades_text}

VERLUST-AUTOPSIEN:
{autopsy_text}

ANALYSE-AUFGABEN:
1. Welche Muster führen zu Verlusten? (besonders: Momentum-Trades bei Penny-Stocks mit yfinance-Verzögerung)
2. Welche Symbole/Setups sollten in Zukunft gemieden werden?
3. In welchen Regimen funktioniert was gut/schlecht?
4. Gibt es Anzeichen dass Verluste durch 15-Minuten Datenverzögerung entstanden sind?

Antworte mit einer kompakten Zusammenfassung die direkt in zukünftige Trading-Entscheidungen einfließen kann. Kein JSON — nur klarer Text."""

        try:
            from google.genai import types as genai_types
            response = gemini_client.models.generate_content(
                model=Config.REASONING_MODEL,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=500,
                    thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
                ),
            )
            summary = response.text or ""
            self._learning_summary = summary
            self._learning_summary_updated = time.time()

            # Auf Disk speichern
            with open(LEARNING_SUMMARY_FILE, "w") as f:
                json.dump({
                    "summary": summary,
                    "updated": self._learning_summary_updated,
                    "updated_at": datetime.now().isoformat(),
                    "trades_analyzed": len(recent),
                }, f, indent=2, ensure_ascii=False)

            logger.info(f"[LEARNING] Neue Zusammenfassung generiert ({len(recent)} Trades analysiert)")
            return summary

        except Exception as e:
            logger.error(f"[LEARNING] Gemini-Zusammenfassung fehlgeschlagen: {e}")
            return self._learning_summary  # Alten Cache zurueckgeben

    # ── Trade Recording ─────────────────────────────────

    def record_entry(
        self,
        symbol: str,
        regime: str,
        formula_scores: dict[str, float],
        sentiment_score: float,
        entry_price: float,
        qty: int,
    ) -> TradeRecord:
        """Speichert einen neuen Trade-Entry."""
        record = TradeRecord(
            symbol=symbol,
            regime=regime,
            formula_scores=formula_scores,
            sentiment_score=sentiment_score,
            entry_price=entry_price,
            qty=qty,
        )
        self.trade_history.append(record)
        self._save()
        logger.info(f"Recorded entry: {symbol} @ ${entry_price} (regime={regime})")
        return record

    def record_exit(self, symbol: str, exit_price: float, reason: str = ""):
        """Schliesst den letzten offenen Trade fuer ein Symbol."""
        # Finde den letzten offenen Trade
        for record in reversed(self.trade_history):
            if record.symbol == symbol and record.exit_price is None:
                record.close(exit_price, reason)
                self._save()
                logger.info(
                    f"Recorded exit: {symbol} @ ${exit_price} "
                    f"P/L: {record.pnl_pct:+.2%} ({reason})"
                )
                # ── Auto-Blacklist: Stop-Loss tracken ──
                if "STOP LOSS" in reason.upper():
                    newly_blocked = self.record_stop_loss(symbol)
                    if newly_blocked:
                        logger.warning(
                            f"[BLACKLIST] {symbol}: Automatisch gesperrt nach 2x Stop-Loss"
                        )
                # ── LERNEN nach jedem abgeschlossenen Trade ──
                self._update_weights()
                return

    # ── Lernalgorithmus ─────────────────────────────────

    # ── Regime Decay (Verhindert Overfitting auf veraltete Muster) ─────

    # Nach dieser Anzahl Tage ohne Trade in einem Regime beginnt der Zerfall
    DECAY_START_DAYS: int = 3
    # Wie stark pro Tag zurück zu DEFAULT_WEIGHTS (5% pro Tag)
    DECAY_RATE_PER_DAY: float = 0.05

    def _apply_decay(self):
        """
        Zerfällt Gewichte inaktiver Regime langsam zurück zu DEFAULT_WEIGHTS.

        Beispiel: CRISIS-Regime war 14 Tage nicht aktiv (kein Trade):
          → 7 Tage über Schwelle × 5%/Tag = 35% Zerfall Richtung Default
          → Ein Gewicht von 1.5 (CRISIS Stoikov) zerfällt auf ~1.33

        Verhindert, dass veraltete Muster aus vergangenen Crashes
        das aktuelle Trading beeinflussen.
        """
        now = datetime.now()
        changed = False

        for regime in ["CALM", "NORMAL", "VOLATILE", "CRISIS"]:
            if regime not in self.weights:
                continue

            # Letzten Trade in diesem Regime finden
            regime_trades = [
                t for t in self.trade_history
                if t.regime == regime and t.entry_time
            ]
            if not regime_trades:
                continue

            try:
                last_entry = max(regime_trades, key=lambda t: t.entry_time)
                last_time = datetime.fromisoformat(last_entry.entry_time)
                days_inactive = (now - last_time).days
            except Exception:
                continue

            if days_inactive < self.DECAY_START_DAYS:
                continue

            # Zerfall berechnen: linear mit Tagen über dem Schwellwert
            days_over = days_inactive - self.DECAY_START_DAYS + 1
            decay_factor = min(1.0, self.DECAY_RATE_PER_DAY * days_over)

            default = DEFAULT_WEIGHTS.get(regime, DEFAULT_WEIGHTS["NORMAL"])
            for formula_name in self.weights[regime]:
                current = self.weights[regime][formula_name]
                target = default.get(formula_name, 1.0)
                if abs(current - target) < 0.01:
                    continue
                new_weight = current + (target - current) * decay_factor
                new_weight = round(max(0.3, min(2.0, new_weight)), 3)
                if abs(new_weight - current) > 0.01:
                    logger.info(
                        f"[DECAY] [{regime}] {formula_name}: "
                        f"{current:.2f} → {new_weight:.2f} "
                        f"({days_inactive}d inaktiv, factor={decay_factor:.2f})"
                    )
                    self.weights[regime][formula_name] = new_weight
                    changed = True

        if changed:
            logger.info("[DECAY] Gewichte nach Inaktivitäts-Zerfall gespeichert")

    def _update_weights(self):
        """
        Aktualisiert Formel-Gewichte basierend auf Trade-Outcomes.

        Fuer jedes Regime:
        1. Sammle alle abgeschlossenen Trades
        2. Berechne Korrelation zwischen jedem Formel-Score und P/L
        3. Passe Gewichte an: Mehr Gewicht fuer Formeln die P/L vorhersagen
        4. Wende Regime-Decay an (inaktive Regime → zurück zu Default)
        """
        closed_trades = [t for t in self.trade_history if t.exit_price is not None]

        if len(closed_trades) < self.min_trades_to_learn:
            # Auch bei wenig Daten: Decay anwenden
            self._apply_decay()
            self._save()
            return

        for regime in ["CALM", "NORMAL", "VOLATILE", "CRISIS"]:
            regime_trades = [t for t in closed_trades if t.regime == regime]

            if len(regime_trades) < 5:
                continue  # Zu wenig Trades in diesem Regime

            if regime not in self.weights:
                self.weights[regime] = dict(DEFAULT_WEIGHTS.get(regime, DEFAULT_WEIGHTS["NORMAL"]))

            # Fuer jede Formel: Korrelation mit P/L
            pnl_array = np.array([t.pnl_pct for t in regime_trades])

            for formula_name in self.weights[regime]:
                scores = []
                for t in regime_trades:
                    if formula_name in t.formula_scores:
                        scores.append(t.formula_scores[formula_name])
                    elif formula_name == "Sentiment":
                        scores.append(t.sentiment_score)
                    else:
                        scores.append(0.0)

                score_array = np.array(scores)

                # Korrelation berechnen
                if np.std(score_array) > 0 and np.std(pnl_array) > 0:
                    correlation = np.corrcoef(score_array, pnl_array)[0, 1]

                    if np.isnan(correlation):
                        continue

                    # Weight Update
                    old_weight = self.weights[regime][formula_name]
                    new_weight = old_weight * (1.0 + self.learning_rate * correlation)

                    # Clamping: [0.3, 2.0]
                    new_weight = max(0.3, min(2.0, new_weight))

                    if abs(new_weight - old_weight) > 0.01:
                        logger.info(
                            f"Weight update [{regime}] {formula_name}: "
                            f"{old_weight:.2f} → {new_weight:.2f} "
                            f"(corr={correlation:+.3f})"
                        )

                    self.weights[regime][formula_name] = round(new_weight, 3)

        # Decay nach dem Korrelations-Update anwenden
        self._apply_decay()
        self._save()

    # ── Gewichteten Score berechnen ─────────────────────

    def weighted_score(self, regime: str, formula_results: dict) -> float:
        """
        Berechnet einen gewichteten Gesamtscore basierend auf
        gelernten Formel-Gewichten.

        Returns: 0.0 bis 1.0 (hoeher = staerkeres Signal)
        """
        if regime not in self.weights:
            regime = "NORMAL"

        w = self.weights[regime]
        total_weight = 0.0
        weighted_sum = 0.0

        for name, result in formula_results.items():
            if name in w:
                weight = w[name]
                # Signal normalisieren auf 0-1
                signal = result.get("signal", 0)
                passed = 1.0 if result.get("passed", False) else 0.0

                # Kombination aus Signal-Staerke und Pass/Fail
                value = 0.7 * passed + 0.3 * max(0, min(1, (signal + 1) / 2))

                weighted_sum += value * weight
                total_weight += weight

        if total_weight == 0:
            return 0.0

        return round(weighted_sum / total_weight, 4)

    def should_override_entry(self, regime: str, formula_results: dict) -> dict:
        """
        Prueft ob das adaptive System einen Trade empfiehlt,
        auch wenn nicht alle 6 Formeln bestanden haben.

        Wenn der gewichtete Score sehr hoch ist (>0.85) UND
        genug Trades zum Lernen da waren, kann der Bot auch
        handeln wenn 1-2 schwach gewichtete Formeln nicht bestanden haben.
        """
        score = self.weighted_score(regime, formula_results)

        passed_count = sum(1 for r in formula_results.values() if r.get("passed"))
        total_count = len(formula_results)

        # Standard: Alle muessen bestehen
        if passed_count == total_count:
            return {"override": False, "reason": "All passed normally", "score": score}

        # Override: Mindestens 5/7 bestanden UND gewichteter Score > 0.85
        # UND genug Trades zum Lernen
        closed_trades = [t for t in self.trade_history if t.exit_price is not None]
        enough_data = len(closed_trades) >= self.min_trades_to_learn

        if enough_data and passed_count >= total_count - 1 and score > 0.85:
            failed = [n for n, r in formula_results.items() if not r.get("passed")]
            return {
                "override": True,
                "reason": f"Adaptive override (score={score:.2f}, failed={failed})",
                "score": score,
            }

        return {
            "override": False,
            "reason": f"Score too low ({score:.2f}) or not enough data",
            "score": score,
        }

    # ── Trade Autopsy ───────────────────────────────────

    def save_autopsy(
        self,
        symbol: str,
        regime: str,
        formula_results: dict,
        reasoning: dict,
        price: float = 0.0,
        vix: float = None,
        order_id: str = "",
    ):
        """
        Speichert vollstaendigen State-Dump pro ausgefuehrtem Trade als JSON.
        Datei: autopsy/YYYYMMDD_HHMMSS_SYMBOL.json

        Enthaelt: Gemini-Prompt, Gemini-Antwort, VIX, alle 7 Formel-Rohergebnisse,
        Kaskaden-Level, Order-ID. Ermoeglicht Post-Mortem Debugging.
        """
        try:
            AUTOPSY_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = AUTOPSY_DIR / f"{ts}_{symbol}.json"

            data = {
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "regime": regime,
                "price": price,
                "vix": vix,
                "order_id": order_id,
                "cascade_level": reasoning.get("cascade_level"),
                "gemini_approved": reasoning.get("approved"),
                "gemini_confidence": reasoning.get("confidence"),
                "gemini_probability_pct": reasoning.get("probability_pct"),
                "gemini_reason": reasoning.get("reason", ""),
                "gemini_risk_factors": reasoning.get("risk_factors", []),
                "gemini_prompt": reasoning.get("prompt", ""),
                "gemini_raw_response": reasoning.get("raw_response", ""),
                "gemini_parsed_result": reasoning.get("raw", {}),
                "formula_results": {
                    name: {
                        "signal": r.get("signal"),
                        "passed": r.get("passed"),
                        "details": r.get("details", {}),
                    }
                    for name, r in formula_results.items()
                },
            }

            with open(filename, "w") as f:
                json.dump(data, f, indent=2, default=str)

            logger.info(f"[AUTOPSY] Gespeichert: {filename.name}")

        except Exception as e:
            logger.error(f"[AUTOPSY] Speichern fehlgeschlagen fuer {symbol}: {e}")

    # ── Performance Stats ───────────────────────────────

    def get_stats(self) -> dict:
        """Performance-Statistiken fuer Telegram/Dashboard."""
        closed = [t for t in self.trade_history if t.exit_price is not None]
        if not closed:
            return {"total_trades": 0, "message": "Keine abgeschlossenen Trades"}

        pnls = [t.pnl_pct for t in closed]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        total_pnl = sum(t.pnl for t in closed)

        stats = {
            "total_trades": len(closed),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / len(closed), 3) if closed else 0,
            "avg_win": round(np.mean(wins), 4) if wins else 0,
            "avg_loss": round(np.mean(losses), 4) if losses else 0,
            "best_trade": round(max(pnls), 4),
            "worst_trade": round(min(pnls), 4),
            "total_pnl": round(total_pnl, 2),
            "sharpe": round(np.mean(pnls) / (np.std(pnls) + 1e-10) * np.sqrt(252), 2),
        }

        # Per-Regime Stats
        regime_stats = {}
        for regime in ["CALM", "NORMAL", "VOLATILE", "CRISIS"]:
            rt = [t for t in closed if t.regime == regime]
            if rt:
                rpnls = [t.pnl_pct for t in rt]
                regime_stats[regime] = {
                    "trades": len(rt),
                    "win_rate": round(sum(1 for p in rpnls if p > 0) / len(rt), 3),
                    "avg_pnl": round(np.mean(rpnls), 4),
                }
        stats["per_regime"] = regime_stats

        return stats

    def get_weights_summary(self) -> str:
        """Formatierte Gewichte fuer Anzeige."""
        lines = []
        for regime, w in self.weights.items():
            lines.append(f"\n{regime}:")
            for formula, weight in sorted(w.items(), key=lambda x: -x[1]):
                bar = "█" * int(weight * 5) + "░" * (10 - int(weight * 5))
                lines.append(f"  {formula:<16} {bar} {weight:.2f}")
        return "\n".join(lines)
