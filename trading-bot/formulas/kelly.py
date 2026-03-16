import numpy as np
import pandas as pd

from config import Config


def kelly_fraction(win_prob: float, payoff_ratio: float) -> float:
    q = 1.0 - win_prob
    f_star = (win_prob * payoff_ratio - q) / payoff_ratio
    return max(0.0, min(f_star, 1.0))


def evaluate(bars: pd.DataFrame, **kwargs) -> dict:
    if len(bars) < 20:
        return {"name": "Kelly", "signal": 0.0, "passed": False,
                "details": {"error": "Not enough data"}}

    equity = kwargs.get("equity", 10000.0)
    trade_history_stats = kwargs.get("trade_history_stats", None)

    if trade_history_stats and trade_history_stats.get("total_trades", 0) >= 5:
        win_prob = trade_history_stats["win_rate"]
        payoff = trade_history_stats.get("payoff_ratio", 1.5)
        source = "trade_history"
    else:
        win_prob = 0.50
        payoff = 1.5
        source = "default_conservative"

    full_kelly = kelly_fraction(win_prob, payoff)
    adjusted = full_kelly * Config.KELLY_FRACTION
    final_pct = min(adjusted, Config.MAX_POSITION_PCT)
    bet_size = equity * final_pct

    return {
        "name": "Kelly",
        "signal": round(adjusted, 4),
        "passed": adjusted > 0.005,
        "details": {
            "win_rate": round(win_prob, 3),
            "payoff_ratio": round(payoff, 3),
            "full_kelly": round(full_kelly, 4),
            "adjusted_kelly": round(adjusted, 4),
            "final_pct": round(final_pct, 4),
            "bet_size_usd": round(bet_size, 2),
            "fraction_used": f"{Config.KELLY_FRACTION}x Kelly",
            "data_source": source,
        },
    }

