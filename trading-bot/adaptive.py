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
from datetime import datetime
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
        self.learning_rate = 0.1  # Konservativ
        self.min_trades_to_learn = 10  # Mindestens 10 Trades bevor angepasst wird
        self.max_history = 500  # Max Trades im Speicher

        self._load()

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
                # ── LERNEN nach jedem abgeschlossenen Trade ──
                self._update_weights()
                return

    # ── Lernalgorithmus ─────────────────────────────────

    def _update_weights(self):
        """
        Aktualisiert Formel-Gewichte basierend auf Trade-Outcomes.

        Fuer jedes Regime:
        1. Sammle alle abgeschlossenen Trades
        2. Berechne Korrelation zwischen jedem Formel-Score und P/L
        3. Passe Gewichte an: Mehr Gewicht fuer Formeln die P/L vorhersagen
        """
        closed_trades = [t for t in self.trade_history if t.exit_price is not None]

        if len(closed_trades) < self.min_trades_to_learn:
            return  # Zu wenig Daten

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
