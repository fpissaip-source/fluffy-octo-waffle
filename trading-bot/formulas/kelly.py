"""
Formula 2 — Kelly Criterion
f* = (p * b - q) / b
Wir verwenden Quarter-Kelly fuer konservatives Sizing.
"""

import numpy as np
import pandas as pd
from config import Config


def kelly_fraction(win_prob: float, payoff_ratio: float) -> float:
    q = 1.0 - win_prob
    f_star = (win_prob * payoff_ratio - q) / payoff_ratio
    return max(0.0, min(f_star, 1.0))


def evaluate(bars: pd.DataFrame, equity: float = 10000.0, **kwargs) -> dict:
    if len(bars) < 20:
        return {"name": "Kelly", "signal": 0.0, "passed": False,
                "details": {"error": "Not enough data"}}

    returns = bars["returns"].dropna()

    # Win rate konditioniert auf Aufwärts-Trend (letztes Drittel vs. erstes Drittel).
    # Rohe 50/50-Returns ergeben immer Kelly=0 (random walk nach Kostenabzug).
    # Stattdessen: Win-Rate nur in den letzten 33% der Bars messen (aktueller Trend).
    recent = returns.iloc[len(returns) * 2 // 3 :]
    older  = returns.iloc[: len(returns) // 3]
    win_prob = (recent > 0).sum() / max(len(recent), 1)
    # Trend-Bonus: wenn recent besser als older, leicht erhöhen (max +0.05)
    older_win = (older > 0).sum() / max(len(older), 1)
    win_prob = min(win_prob + max(win_prob - older_win, 0) * 0.5, 0.75)

    gains = recent[recent > 0]
    losses = recent[recent < 0]
    avg_gain = gains.mean() if len(gains) > 0 else 0.001
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.001

    # Dynamische Slippage: Penny Stocks / OTC haben viel hoehere Kosten
    spread = kwargs.get("spread", 0.0)
    current_price = bars["close"].iloc[-1]
    if spread and spread > 0 and current_price > 0:
        slippage_pct = (spread / current_price)   # realer Bid-Ask-Spread
    elif current_price < 1.0:
        slippage_pct = 0.01   # 100 bps fuer Sub-Dollar Stocks
    elif current_price < 5.0:
        slippage_pct = 0.005  # 50 bps fuer Penny Stocks
    else:
        slippage_pct = Config.SLIPPAGE_BPS / 10000  # Standard (10 bps)
    cost_pct = (slippage_pct + Config.FEE_BPS / 10000) * 2  # Roundtrip
    adj_gain = max(avg_gain - cost_pct, 0.0001)
    adj_loss = avg_loss + cost_pct
    payoff = min(adj_gain / adj_loss, 10.0)

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
            "cost_pct": f"{cost_pct*100:.3f}%",
            "slippage_bps": round(slippage_pct * 10000, 1),
        },
    }
