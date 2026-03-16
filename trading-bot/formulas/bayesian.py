"""
Formula 5 — Bayesian Updates
P(H|E) = P(E|H) * P(H) / P(E)
Aktualisiert Trade-Wahrscheinlichkeit basierend auf Volume, Momentum, Volatilitaet.
"""

import numpy as np
import pandas as pd


class BayesianUpdater:
    def __init__(self, prior: float = 0.5):
        self.posterior = prior
        self.updates = []

    def update(self, likelihood_ratio: float, label: str = "") -> float:
        prior_odds = self.posterior / (1.0 - self.posterior + 1e-10)
        posterior_odds = prior_odds * likelihood_ratio
        self.posterior = posterior_odds / (1.0 + posterior_odds)
        self.posterior = max(0.05, min(0.95, self.posterior))
        self.updates.append({"label": label, "lr": round(likelihood_ratio, 3),
                             "posterior": round(self.posterior, 4)})
        return self.posterior


def evaluate(bars: pd.DataFrame, **kwargs) -> dict:
    if len(bars) < 30:
        return {"name": "Bayesian", "signal": 0.5, "passed": False,
                "details": {"error": "Not enough data"}}

    prior = kwargs.get("prior", 0.50)
    threshold = kwargs.get("threshold", 0.60)
    u = BayesianUpdater(prior)

    # Signal 1: Volume Spike
    vol = bars["volume"]
    avg_vol = vol.iloc[-21:-1].mean() if len(vol) > 21 else vol.mean()
    vol_ratio = vol.iloc[-1] / avg_vol if avg_vol > 0 else 1.0
    if vol_ratio > 2.0:
        recent_ret = bars["returns"].tail(3).mean()
        u.update(1.8 if recent_ret > 0 else 0.6, "volume_spike")
    else:
        u.update(1.0, "volume_normal")

    # Signal 2: Price Acceleration
    returns = bars["returns"]
    recent = returns.tail(5).mean()
    prior_ret = returns.iloc[-10:-5].mean() if len(returns) > 10 else 0
    if recent > prior_ret and recent > 0:
        u.update(1.5, "accelerating")
    elif recent < 0:
        u.update(0.7, "decelerating")
    else:
        u.update(1.0, "neutral")

    # Signal 3: Volatility Regime
    current_vol = returns.tail(20).std()
    hist_vol = returns.std()
    vol_r = current_vol / (hist_vol + 1e-10)
    if vol_r < 0.7:
        u.update(1.2, "low_vol")
    elif vol_r > 1.3:
        u.update(0.85, "high_vol")
    else:
        u.update(1.0, "normal_vol")

    # Signal 4: Trend Consistency
    pos_pct = (returns.tail(10) > 0).sum() / 10
    if pos_pct > 0.7:
        u.update(1.6, "strong_uptrend")
    elif pos_pct < 0.3:
        u.update(0.5, "strong_downtrend")
    else:
        u.update(1.0, "mixed")

    return {
        "name": "Bayesian",
        "signal": round(u.posterior, 4),
        "passed": u.posterior > threshold,
        "details": {
            "prior": prior,
            "posterior": round(u.posterior, 4),
            "updates": u.updates,
            "vol_ratio": round(vol_ratio, 2),
            "vol_regime": "low" if vol_r < 0.7 else "high" if vol_r > 1.3 else "normal",
        },
    }
