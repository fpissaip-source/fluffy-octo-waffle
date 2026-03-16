import logging
from datetime import datetime
from typing import Optional

import numpy as np

from config import Config
from database import get_session, TradeRecord as DBTradeRecord, FormulaWeight

logger = logging.getLogger("bot.adaptive")

DEFAULT_WEIGHTS = {
    "CALM":     {"Momentum": 1.0, "Kelly": 1.0, "EV-Gap": 1.0, "KL-Divergence": 1.0, "Bayesian": 1.0, "Z-Score": 0.8, "Sentiment": 0.8},
    "NORMAL":   {"Momentum": 1.0, "Kelly": 1.0, "EV-Gap": 1.0, "KL-Divergence": 1.0, "Bayesian": 1.0, "Z-Score": 1.0, "Sentiment": 1.0},
    "VOLATILE": {"Momentum": 0.8, "Kelly": 1.2, "EV-Gap": 1.0, "KL-Divergence": 1.2, "Bayesian": 1.0, "Z-Score": 1.2, "Sentiment": 1.3},
    "CRISIS":   {"Momentum": 0.6, "Kelly": 1.3, "EV-Gap": 0.8, "KL-Divergence": 1.0, "Bayesian": 1.2, "Z-Score": 1.5, "Sentiment": 1.5},
}


class AdaptiveLearner:
    def __init__(self):
        self.weights: dict[str, dict[str, float]] = {}
        self.learning_rate = 0.1
        self.min_trades_to_learn = 10
        self._load_weights()

    def _load_weights(self):
        try:
            session = get_session()
            rows = session.query(FormulaWeight).all()
            session.close()
            if rows:
                for row in rows:
                    if row.regime not in self.weights:
                        self.weights[row.regime] = {}
                    self.weights[row.regime][row.formula_name] = row.weight
                logger.info(f"Loaded weights from DB ({len(rows)} entries)")
            else:
                self.weights = {k: dict(v) for k, v in DEFAULT_WEIGHTS.items()}
        except Exception:
            self.weights = {k: dict(v) for k, v in DEFAULT_WEIGHTS.items()}

    def _save_weights(self):
        try:
            session = get_session()
            session.query(FormulaWeight).delete()
            for regime, formulas in self.weights.items():
                for formula_name, weight in formulas.items():
                    session.add(FormulaWeight(
                        regime=regime,
                        formula_name=formula_name,
                        weight=weight,
                    ))
            session.commit()
            session.close()
        except Exception as e:
            logger.error(f"Save weights failed: {e}")

    def record_entry(self, symbol: str, regime: str, formula_scores: dict[str, float],
                     sentiment_score: float, entry_price: float, qty: int,
                     weighted_score: float = 0.0):
        try:
            session = get_session()
            record = DBTradeRecord(
                symbol=symbol,
                regime=regime,
                action="BUY",
                qty=qty,
                entry_price=entry_price,
                formula_scores=formula_scores,
                sentiment_score=sentiment_score,
                weighted_score=weighted_score,
            )
            session.add(record)
            session.commit()
            session.close()
            logger.info(f"Recorded entry: {symbol} @ ${entry_price} (regime={regime})")
        except Exception as e:
            logger.error(f"Record entry failed: {e}")

    def record_exit(self, symbol: str, exit_price: float, reason: str = ""):
        try:
            session = get_session()
            record = session.query(DBTradeRecord).filter(
                DBTradeRecord.symbol == symbol,
                DBTradeRecord.exit_price.is_(None),
            ).order_by(DBTradeRecord.id.desc()).first()

            if record:
                record.exit_price = exit_price
                record.exit_time = datetime.now()
                record.pnl = (exit_price - record.entry_price) * record.qty
                record.pnl_pct = (exit_price - record.entry_price) / record.entry_price
                record.exit_reason = reason
                session.commit()
                logger.info(f"Recorded exit: {symbol} @ ${exit_price} P/L: {record.pnl_pct:+.2%}")
                session.close()
                self._update_weights()
            else:
                session.close()
        except Exception as e:
            logger.error(f"Record exit failed: {e}")

    def _update_weights(self):
        try:
            session = get_session()
            closed = session.query(DBTradeRecord).filter(
                DBTradeRecord.exit_price.isnot(None)
            ).all()
            session.close()

            if len(closed) < self.min_trades_to_learn:
                return

            for regime in ["CALM", "NORMAL", "VOLATILE", "CRISIS"]:
                regime_trades = [t for t in closed if t.regime == regime]
                if len(regime_trades) < 5:
                    continue

                if regime not in self.weights:
                    self.weights[regime] = dict(DEFAULT_WEIGHTS.get(regime, DEFAULT_WEIGHTS["NORMAL"]))

                pnl_array = np.array([t.pnl_pct for t in regime_trades])

                for formula_name in list(self.weights[regime].keys()):
                    scores = []
                    for t in regime_trades:
                        fs = t.formula_scores or {}
                        if formula_name in fs:
                            scores.append(fs[formula_name])
                        elif formula_name == "Sentiment":
                            scores.append(t.sentiment_score or 0)
                        else:
                            scores.append(0.0)

                    score_array = np.array(scores)
                    if np.std(score_array) > 0 and np.std(pnl_array) > 0:
                        correlation = np.corrcoef(score_array, pnl_array)[0, 1]
                        if np.isnan(correlation):
                            continue

                        old_weight = self.weights[regime][formula_name]
                        new_weight = old_weight * (1.0 + self.learning_rate * correlation)
                        new_weight = max(0.3, min(2.0, new_weight))

                        if abs(new_weight - old_weight) > 0.01:
                            logger.info(
                                f"Weight update [{regime}] {formula_name}: "
                                f"{old_weight:.2f} -> {new_weight:.2f} (corr={correlation:+.3f})"
                            )
                        self.weights[regime][formula_name] = round(new_weight, 3)

            self._save_weights()
        except Exception as e:
            logger.error(f"Weight update failed: {e}")

    def weighted_score(self, regime: str, formula_results: dict) -> float:
        if regime not in self.weights:
            regime = "NORMAL"
        w = self.weights.get(regime, DEFAULT_WEIGHTS["NORMAL"])
        total_weight = 0.0
        weighted_sum = 0.0

        for name, result in formula_results.items():
            if name in w:
                weight = w[name]
                passed = 1.0 if result.get("passed", False) else 0.0
                signal = result.get("signal", 0)
                value = 0.7 * passed + 0.3 * max(0, min(1, (signal + 1) / 2))
                weighted_sum += value * weight
                total_weight += weight

        if total_weight == 0:
            return 0.0
        return round(weighted_sum / total_weight, 4)

    def get_trade_history_stats(self) -> dict:
        try:
            session = get_session()
            closed = session.query(DBTradeRecord).filter(
                DBTradeRecord.exit_price.isnot(None)
            ).all()
            session.close()

            if not closed:
                return {"total_trades": 0}

            pnls = [t.pnl_pct for t in closed]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p <= 0]
            avg_win = np.mean(wins) if wins else 0.001
            avg_loss = abs(np.mean(losses)) if losses else 0.001

            return {
                "total_trades": len(closed),
                "wins": len(wins),
                "losses": len(losses),
                "win_rate": round(len(wins) / len(closed), 3),
                "avg_win": round(float(avg_win), 4),
                "avg_loss": round(float(avg_loss), 4),
                "payoff_ratio": round(float(avg_win / avg_loss), 3) if avg_loss > 0 else 1.5,
                "best_trade": round(max(pnls), 4),
                "worst_trade": round(min(pnls), 4),
                "total_pnl": round(sum(t.pnl for t in closed), 2),
                "sharpe": round(float(np.mean(pnls) / (np.std(pnls) + 1e-10) * np.sqrt(252)), 2),
            }
        except Exception as e:
            logger.error(f"Stats failed: {e}")
            return {"total_trades": 0}

    def get_weights_summary(self) -> str:
        lines = []
        for regime, w in self.weights.items():
            lines.append(f"\n{regime}:")
            for formula, weight in sorted(w.items(), key=lambda x: -x[1]):
                bar = "#" * int(weight * 5) + "." * (10 - int(weight * 5))
                lines.append(f"  {formula:<16} {bar} {weight:.2f}")
        return "\n".join(lines)

