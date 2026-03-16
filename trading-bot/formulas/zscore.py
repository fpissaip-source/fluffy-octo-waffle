import numpy as np
import pandas as pd

from config import Config


def evaluate(bars: pd.DataFrame, **kwargs) -> dict:
    if len(bars) < 30:
        return {"name": "Z-Score", "signal": 0.0, "passed": False,
                "details": {"error": "Need 30+ bars"}}

    closes = bars["close"]
    current_price = closes.iloc[-1]
    entry_threshold = kwargs.get("threshold", Config.ZSCORE_ENTRY_THRESHOLD)

    sma_20 = closes.rolling(20).mean().iloc[-1]
    std_20 = closes.rolling(20).std().iloc[-1]

    if std_20 < 1e-10:
        return {"name": "Z-Score", "signal": 0.0, "passed": False,
                "details": {"error": "Zero volatility"}}

    z_score = (current_price - sma_20) / std_20

    vwap = (closes * bars["volume"]).sum() / bars["volume"].sum() if bars["volume"].sum() > 0 else sma_20
    vwap_distance = (current_price - vwap) / vwap

    ema_50 = closes.ewm(span=50, adjust=False).mean().iloc[-1] if len(closes) >= 50 else sma_20
    trend_bullish = current_price > ema_50

    passed = z_score <= entry_threshold and trend_bullish

    return {
        "name": "Z-Score",
        "signal": round(z_score, 4),
        "passed": passed,
        "details": {
            "z_score": round(z_score, 4),
            "threshold": entry_threshold,
            "current_price": round(current_price, 2),
            "sma_20": round(sma_20, 2),
            "std_20": round(std_20, 4),
            "vwap_distance": f"{vwap_distance:+.2%}",
            "trend_bullish": trend_bullish,
        },
    }

