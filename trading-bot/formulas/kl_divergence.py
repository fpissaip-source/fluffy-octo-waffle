import numpy as np
import pandas as pd

from config import Config


def symmetric_kl(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, 1e-10, None)
    q = np.clip(q, 1e-10, None)
    p = p / p.sum()
    q = q / q.sum()
    kl_pq = float(np.sum(p * np.log(p / q)))
    kl_qp = float(np.sum(q * np.log(q / p)))
    return (kl_pq + kl_qp) / 2


def evaluate(bars: pd.DataFrame, **kwargs) -> dict:
    if len(bars) < 50:
        return {"name": "KL-Divergence", "signal": 0.0, "passed": False,
                "details": {"error": "Need 50+ bars"}}

    returns = bars["returns"].dropna()
    short_returns = returns.tail(10)
    long_returns = returns.tail(50)

    range_min = long_returns.min()
    range_max = long_returns.max()
    bins = np.linspace(range_min, range_max, 21)

    p_short, _ = np.histogram(short_returns, bins=bins, density=True)
    q_long, _ = np.histogram(long_returns, bins=bins, density=True)

    p_short = p_short + 1e-10
    q_long = q_long + 1e-10
    p_short = p_short / p_short.sum()
    q_long = q_long / q_long.sum()

    kl_div = symmetric_kl(p_short, q_long)
    threshold = kwargs.get("threshold", Config.KL_DIVERGENCE_THRESHOLD)

    short_mean = short_returns.mean()
    long_mean = long_returns.mean()
    drift = "SHORT_OUTPERFORMS" if short_mean > long_mean else "SHORT_UNDERPERFORMS"
    bullish_reversion = short_mean < long_mean and kl_div > threshold

    directional_signal = kl_div if bullish_reversion else -kl_div

    return {
        "name": "KL-Divergence",
        "signal": round(directional_signal, 4),
        "passed": bullish_reversion and kl_div > threshold,
        "details": {
            "kl_divergence": round(kl_div, 4),
            "threshold": threshold,
            "short_mean": f"{short_mean:+.4f}",
            "long_mean": f"{long_mean:+.4f}",
            "drift": drift,
            "bullish_reversion": bullish_reversion,
        },
    }

