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
    ev = win_prob * upside - (1 - win_prob) * downside

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
        },
    }

