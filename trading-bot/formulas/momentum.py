"""
Formula 1 — Momentum & Mean Reversion Detection

Replaces LMSR (prediction-market specific) with a stock-market equivalent:
detects whether the current regime is trending or mean-reverting,
then generates a directional signal with confidence.

Uses:
  - RSI (14-period) for overbought/oversold
  - EMA crossover (8/21) for trend direction
  - Rate of Change for momentum strength
  - Bollinger Band position for mean-reversion signals
"""

import numpy as np
from models import BarData, FilterResult, FilterStatus, Direction


class MomentumFilter:
    """Detects momentum regime and outputs direction + confidence."""

    def __init__(self):
        self.name = "Momentum/MeanRev"

    def evaluate(self, bars: list[BarData]) -> tuple[Direction, float, FilterResult]:
        """
        Returns (direction, confidence, filter_result).
        Confidence is 0.0-1.0 representing how strong the signal is.
        """
        if len(bars) < 30:
            return Direction.NEUTRAL, 0.5, FilterResult(
                name=self.name,
                status=FilterStatus.SKIP,
                value=0.0,
                detail="Not enough bars (need 30+)"
            )

        closes = np.array([b.close for b in bars])

        # ── RSI (14) ──
        rsi = self._rsi(closes, 14)

        # ── EMA crossover (8/21) ──
        ema_fast = self._ema(closes, 8)
        ema_slow = self._ema(closes, 21)
        ema_diff = (ema_fast[-1] - ema_slow[-1]) / closes[-1]

        # ── Rate of Change (10-bar) ──
        roc = (closes[-1] - closes[-11]) / closes[-11] if len(closes) > 11 else 0

        # ── Bollinger Band position ──
        bb_pos = self._bollinger_position(closes, 20, 2.0)

        # ── Aggregate signal ──
        score = 0.0
        signals = 0

        # RSI signal
        if rsi < 30:
            score += 0.3  # Oversold → bullish
            signals += 1
        elif rsi > 70:
            score -= 0.3  # Overbought → bearish
            signals += 1

        # EMA crossover
        if ema_diff > 0.002:
            score += 0.25
            signals += 1
        elif ema_diff < -0.002:
            score -= 0.25
            signals += 1

        # Rate of change
        if roc > 0.02:
            score += 0.25
            signals += 1
        elif roc < -0.02:
            score -= 0.25
            signals += 1

        # Bollinger Band
        if bb_pos < 0.1:
            score += 0.2  # Near lower band → mean revert up
            signals += 1
        elif bb_pos > 0.9:
            score -= 0.2  # Near upper band → mean revert down
            signals += 1

        # Direction and confidence
        if score > 0.15:
            direction = Direction.LONG
        elif score < -0.15:
            direction = Direction.SHORT
        else:
            direction = Direction.NEUTRAL

        confidence = min(abs(score) / 0.7, 1.0)  # Normalize to 0-1
        threshold = 0.15

        passed = abs(score) >= threshold and signals >= 2

        detail = (
            f"RSI={rsi:.0f} EMA_diff={ema_diff:+.4f} "
            f"ROC={roc:+.3f} BB={bb_pos:.2f} → score={score:+.3f}"
        )

        return direction, confidence, FilterResult(
            name=self.name,
            status=FilterStatus.PASS if passed else FilterStatus.FAIL,
            value=score,
            detail=detail,
        )

    @staticmethod
    def _rsi(closes: np.ndarray, period: int = 14) -> float:
        deltas = np.diff(closes[-(period + 1):])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 1e-10
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _ema(data: np.ndarray, period: int) -> np.ndarray:
        alpha = 2 / (period + 1)
        ema = np.zeros_like(data, dtype=float)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
        return ema

    @staticmethod
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
        return (closes[-1] - lower) / (upper - lower)
