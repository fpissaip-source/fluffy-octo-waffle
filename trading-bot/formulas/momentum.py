"""
Formula 1 — Momentum & Mean Reversion Detection

Detects whether the current regime is trending or mean-reverting,
then generates a directional signal with confidence.

Uses:
  - RSI (14-period) for overbought/oversold
  - EMA crossover (8/21) for trend direction
  - Rate of Change for momentum strength
  - Bollinger Band position for mean-reversion signals
"""

import numpy as np
import pandas as pd

from config import Config


def evaluate(bars: pd.DataFrame, **kwargs) -> dict:
    if len(bars) < 30:
        return {"name": "Momentum", "signal": 0.0, "passed": False,
                "details": {"error": "Not enough bars (need 30+)"}}

    threshold = kwargs.get("threshold", Config.MIN_MOMENTUM_SCORE)
    closes = bars["close"].values

    # ── RSI (14) ──
    rsi = _rsi(closes, 14)

    # ── EMA crossover (8/21) ──
    ema_fast = _ema(closes, 8)
    ema_slow = _ema(closes, 21)
    ema_diff = (ema_fast[-1] - ema_slow[-1]) / closes[-1]

    # ── Rate of Change (10-bar) ──
    roc = (closes[-1] - closes[-11]) / closes[-11] if len(closes) > 11 else 0.0

    # ── Bollinger Band position ──
    bb_pos = _bollinger_position(closes, 20, 2.0)

    # ── Aggregate signal ──
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

    passed = score >= threshold and signals >= 2

    return {
        "name": "Momentum",
        "signal": round(score, 4),
        "passed": passed,
        "details": {
            "rsi": round(rsi, 1),
            "ema_diff": round(ema_diff, 5),
            "roc": round(roc, 4),
            "bb_pos": round(bb_pos, 3),
            "score": round(score, 4),
            "signals": signals,
            "threshold": threshold,
        },
    }


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
    return float((closes[-1] - lower) / (upper - lower))
