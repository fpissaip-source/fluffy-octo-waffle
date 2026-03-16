import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from risk_manager import RiskManager, compute_atr, MarketRegime
from formulas import momentum, kelly, ev_gap, kl_divergence, bayesian, zscore, sentiment, regime
from adaptive import DEFAULT_WEIGHTS
from config import Config

logger = logging.getLogger("bot.backtest")


class BacktestEngine:
    def __init__(self):
        self.risk = RiskManager()
        self.initial_equity = 10000.0
        self.equity = self.initial_equity
        self.position: Optional[dict] = None
        self.trades: list[dict] = []
        self.equity_curve: list[dict] = []
        self.weights = DEFAULT_WEIGHTS["NORMAL"]
        self.required_filters = Config.REQUIRED_FILTERS
        self.score_threshold = Config.WEIGHTED_SCORE_THRESHOLD

    def _evaluate_bar_window(self, bars: pd.DataFrame) -> dict:
        results = {}
        for name, func, kwargs in [
            ("Momentum", momentum.evaluate, {"threshold": 0.15}),
            ("Kelly", kelly.evaluate, {"equity": self.equity}),
            ("EV-Gap", ev_gap.evaluate, {"win_prob": 0.55}),
            ("KL-Divergence", kl_divergence.evaluate, {"threshold": 0.15}),
            ("Bayesian", bayesian.evaluate, {"prior": 0.50, "threshold": 0.60}),
            ("Z-Score", zscore.evaluate, {"threshold": -1.0}),
            ("Sentiment", sentiment.evaluate, {}),
            ("Regime", regime.evaluate, {}),
        ]:
            try:
                results[name] = func(bars, **kwargs)
            except Exception:
                results[name] = {"name": name, "signal": 0, "passed": False}
        return results

    def _weighted_score(self, results: dict) -> float:
        total_weight = 0.0
        weighted_sum = 0.0
        for name, result in results.items():
            if name in self.weights:
                weight = self.weights[name]
                passed = 1.0 if result.get("passed", False) else 0.0
                signal = result.get("signal", 0)
                value = 0.7 * passed + 0.3 * max(0, min(1, (signal + 1) / 2))
                weighted_sum += value * weight
                total_weight += weight
        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _should_enter(self, results: dict) -> bool:
        required_ok = all(
            results.get(f, {}).get("passed", False) for f in self.required_filters
            if f in results
        )
        score = self._weighted_score(results)
        return required_ok and score >= self.score_threshold

    def run(self, bars: pd.DataFrame, symbol: str = "TEST") -> dict:
        if len(bars) < 60:
            return {"error": "Need at least 60 bars for backtest"}

        self.equity = self.initial_equity
        self.position = None
        self.trades = []
        self.equity_curve = []
        highest_price = 0.0

        for i in range(50, len(bars)):
            window = bars.iloc[max(0, i - 100):i + 1].copy()
            if len(window) < 30:
                continue

            current_price = window["close"].iloc[-1]
            timestamp = str(window.index[-1]) if hasattr(window.index[-1], 'isoformat') else str(i)

            if self.position is not None:
                highest_price = max(highest_price, current_price)
                atr = compute_atr(window)
                self.risk.update_regime(window)

                exit_decision = self.risk.should_exit(
                    entry_price=self.position["entry_price"],
                    current_price=current_price,
                    highest_price=highest_price,
                    atr=atr if atr > 0 else current_price * 0.01,
                )

                if exit_decision["should_exit"]:
                    pnl = (current_price - self.position["entry_price"]) * self.position["qty"]
                    pnl_pct = (current_price - self.position["entry_price"]) / self.position["entry_price"]
                    self.equity += pnl
                    self.trades.append({
                        "symbol": symbol,
                        "entry_price": self.position["entry_price"],
                        "exit_price": round(current_price, 2),
                        "qty": self.position["qty"],
                        "pnl": round(pnl, 2),
                        "pnl_pct": round(pnl_pct, 4),
                        "reason": exit_decision["reason"],
                        "entry_time": self.position["entry_time"],
                        "exit_time": timestamp,
                    })
                    self.position = None
            else:
                results = self._evaluate_bar_window(window)
                if self._should_enter(results):
                    max_usd = self.equity * 0.10
                    qty = max(1, int(max_usd / current_price))
                    self.position = {
                        "entry_price": current_price,
                        "qty": qty,
                        "entry_time": timestamp,
                    }
                    highest_price = current_price

            unrealized = 0.0
            if self.position:
                unrealized = (current_price - self.position["entry_price"]) * self.position["qty"]

            self.equity_curve.append({
                "timestamp": timestamp,
                "equity": round(self.equity + unrealized, 2),
            })

        if self.position:
            current_price = bars["close"].iloc[-1]
            pnl = (current_price - self.position["entry_price"]) * self.position["qty"]
            pnl_pct = (current_price - self.position["entry_price"]) / self.position["entry_price"]
            self.equity += pnl
            self.trades.append({
                "symbol": symbol,
                "entry_price": self.position["entry_price"],
                "exit_price": round(current_price, 2),
                "qty": self.position["qty"],
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 4),
                "reason": "BACKTEST_END",
                "entry_time": self.position["entry_time"],
                "exit_time": str(bars.index[-1]),
            })
            self.position = None

        return self._compute_stats(symbol)

    def _compute_stats(self, symbol: str) -> dict:
        if not self.trades:
            return {
                "symbol": symbol,
                "total_trades": 0,
                "message": "No trades generated",
                "pnl_curve": self.equity_curve[-50:] if self.equity_curve else [],
            }

        pnls = [t["pnl_pct"] for t in self.trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        equity_values = [e["equity"] for e in self.equity_curve]
        peak = equity_values[0]
        max_dd = 0.0
        for v in equity_values:
            if v > peak:
                peak = v
            dd = (v - peak) / peak
            if dd < max_dd:
                max_dd = dd

        total_pnl = sum(t["pnl"] for t in self.trades)
        total_return = (self.equity - self.initial_equity) / self.initial_equity

        return {
            "symbol": symbol,
            "total_trades": len(self.trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / len(self.trades), 3) if self.trades else 0,
            "total_pnl": round(total_pnl, 2),
            "total_return": f"{total_return:+.2%}",
            "sharpe_ratio": round(float(np.mean(pnls) / (np.std(pnls) + 1e-10) * np.sqrt(252)), 2),
            "max_drawdown": f"{max_dd:.2%}",
            "best_trade": f"{max(pnls):+.2%}" if pnls else "N/A",
            "worst_trade": f"{min(pnls):+.2%}" if pnls else "N/A",
            "avg_win": f"{np.mean(wins):+.2%}" if wins else "N/A",
            "avg_loss": f"{np.mean(losses):+.2%}" if losses else "N/A",
            "final_equity": round(self.equity, 2),
            "initial_equity": self.initial_equity,
            "trades": self.trades,
            "pnl_curve": self.equity_curve[::max(1, len(self.equity_curve) // 100)],
        }
