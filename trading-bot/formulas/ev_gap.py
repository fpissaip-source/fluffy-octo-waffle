"""
Formula 3 — EV Gap Detection
Vergleicht aktuellen Preis mit Fair Value (VWAP + Bollinger + EMA).
Sucht Gaps wo Expected Value positiv ist.
"""

import numpy as np
import pandas as pd
from config import Config


def estimate_fair_value(bars: pd.DataFrame) -> float:
    prices = bars["close"]
    volumes = bars["volume"]
    vwap = (prices * volumes).sum() / volumes.sum() if volumes.sum() > 0 else prices.mean()
    bb_mid = prices.rolling(20).mean().iloc[-1]
    ema21 = prices.ewm(span=21, adjust=False).mean().iloc[-1]
    return vwap * 0.4 + bb_mid * 0.3 + ema21 * 0.3


def evaluate(bars: pd.DataFrame, **kwargs) -> dict:
    if len(bars) < 25:
        return {"name": "EV-Gap", "signal": 0.0, "passed": False,
                "details": {"error": "Not enough data"}}

    current_price = bars["close"].iloc[-1]
    fair_value = estimate_fair_value(bars)
    win_prob = kwargs.get("win_prob", 0.55)

    gap_pct = (fair_value - current_price) / current_price
    upside = abs(gap_pct) if gap_pct > 0 else abs(gap_pct) * 0.3
    downside = abs(gap_pct) * 0.5

    # Dynamische Slippage: Penny Stocks / OTC haben viel hoehere Kosten
    spread = kwargs.get("spread", 0.0)
    if spread and spread > 0 and current_price > 0:
        slippage_pct = spread / current_price
    elif current_price < 1.0:
        slippage_pct = 0.01   # 100 bps
    elif current_price < 5.0:
        slippage_pct = 0.005  # 50 bps
    else:
        slippage_pct = Config.SLIPPAGE_BPS / 10000
    cost_pct = (slippage_pct + Config.FEE_BPS / 10000) * 2  # Roundtrip
    net_upside = max(upside - cost_pct, 0.0)
    net_downside = downside + cost_pct
    ev = win_prob * net_upside - (1 - win_prob) * net_downside

    direction = "LONG" if gap_pct > 0 else "SHORT" if gap_pct < -Config.MIN_EV_GAP else "NEUTRAL"

    return {
        "name": "EV-Gap",
        "signal": round(ev, 5),
        "passed": abs(ev) > Config.MIN_EV_GAP and gap_pct > 0,
        "details": {
            "current_price": round(current_price, 2),
            "fair_value": round(fair_value, 2),
            "gap_pct": f"{gap_pct:+.2%}",
            "ev": round(ev, 5),
            "direction": direction,
            "cost_pct": f"{cost_pct*100:.3f}%",
            "slippage_bps": round(slippage_pct * 10000, 1),
        },
    }
