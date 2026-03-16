import pandas as pd
import numpy as np

from risk_manager import detect_regime, MarketRegime


def evaluate(bars: pd.DataFrame, **kwargs) -> dict:
    if len(bars) < 20:
        return {"name": "Regime", "signal": 0.0, "passed": False,
                "details": {"error": "Need 20+ bars"}}

    regime = detect_regime(bars)

    REGIME_SCORES = {
        MarketRegime.CALM: 1.0,
        MarketRegime.NORMAL: 0.7,
        MarketRegime.VOLATILE: 0.3,
        MarketRegime.CRISIS: 0.0,
    }

    REGIME_PASSABLE = {
        MarketRegime.CALM: True,
        MarketRegime.NORMAL: True,
        MarketRegime.VOLATILE: True,
        MarketRegime.CRISIS: False,
    }

    score = REGIME_SCORES.get(regime, 0.5)
    passed = REGIME_PASSABLE.get(regime, False)

    returns = bars["returns"].dropna()
    realized_vol = returns.tail(20).std() * np.sqrt(252)
    recent_high = bars["close"].tail(20).max()
    current = bars["close"].iloc[-1]
    drawdown = (current - recent_high) / recent_high

    return {
        "name": "Regime",
        "signal": round(score, 4),
        "passed": passed,
        "details": {
            "regime": regime.value,
            "realized_vol": f"{realized_vol:.2%}",
            "drawdown_20d": f"{drawdown:.2%}",
            "tradeable": passed,
        },
    }
