"""
Formula 1 — Momentum & Mean Reversion Detection

Signale:
  - RSI (14) fuer Overbought/Oversold
  - EMA Crossover (8/21) fuer Trendrichtung
  - Rate of Change (10-bar) fuer Staerke
  - Bollinger Band Position fuer Mean-Reversion

Output: signal (-1.0 bis +1.0), passed (bool)
"""

import numpy as np
import pandas as pd


def _rsi(closes: np.ndarray, period: int = 14) -> float:
    deltas = np.diff(closes[-(period + 1):])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains) if len(gains) > 0 else 0
    avg_loss = np.mean(losses) if len(losses) > 0 else 1e-10
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _ema(data: np.ndarray, period: int) -> np.ndarray:
    alpha = 2 / (period + 1)
    ema = np.zeros_like(data, dtype=float)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
    return ema


def _bollinger_position(closes: np.ndarray, period: int = 20, num_std: float = 2.0) -> float:
    if len(closes) < period:
        return 0.5
    window = closes[-period:]
    mean = np.mean(window)
    std = np.std(window)
    if std < 1e-10:
        return 0.5
    upper = mean + num_std * std
    lower = mean - num_std * std
    pos = (closes[-1] - lower) / (upper - lower)
    return float(np.clip(pos, 0.0, 1.0))


def evaluate(bars: pd.DataFrame, threshold: float = 0.6, **kwargs) -> dict:
    """
    Formula-Interface fuer engine.py.
    bars: pandas DataFrame mit 'close' Spalte.
    """
    name = "Momentum"

    if bars is None or bars.empty or len(bars) < 30:
        return {"name": name, "signal": 0.0, "passed": False,
                "details": {"error": "Not enough bars"}}

    closes = bars["close"].values.astype(float)

    rsi = _rsi(closes, 14)
    ema_fast = _ema(closes, 8)
    ema_slow = _ema(closes, 21)
    ema_diff = (ema_fast[-1] - ema_slow[-1]) / closes[-1]
    roc = (closes[-1] - closes[-11]) / closes[-11] if len(closes) > 11 else 0.0
    bb_pos = _bollinger_position(closes, 20, 2.0)

    score = 0.0
    signals = 0

    if rsi < 30:
        score += 0.3
        signals += 1
    elif rsi > 70:
        score -= 0.3
        signals += 1

    if ema_diff > 0.002:
        score += 0.25
        signals += 1
    elif ema_diff < -0.002:
        score -= 0.25
        signals += 1

    if roc > 0.02:
        score += 0.25
        signals += 1
    elif roc < -0.02:
        score -= 0.25
        signals += 1

    if bb_pos < 0.1:
        score += 0.2
        signals += 1
    elif bb_pos > 0.9:
        score -= 0.2
        signals += 1

    confidence = min(abs(score) / 0.7, 1.0)
    passed = confidence >= threshold and signals >= 2 and score > 0

    return {
        "name": name,
        "signal": round(float(score), 3),
        "passed": passed,
        "details": {
            "rsi": round(rsi, 1),
            "ema_diff": round(ema_diff, 4),
            "roc": round(roc, 4),
            "bb_pos": round(bb_pos, 3),
            "confidence": round(confidence, 3),
            "signals": signals,
        },
    }
