"""
Formula 6 — Stoikov Execution (Avellaneda-Stoikov)
Reservation Price: r = mid - q * gamma * sigma^2 * (T - t)
Bot kauft nur wenn Preis <= Reservation Price.
"""

import numpy as np
import pandas as pd
from config import Config


def evaluate(bars: pd.DataFrame, snapshot: dict = None, **kwargs) -> dict:
    if len(bars) < 20:
        return {"name": "Stoikov", "signal": 0.0, "passed": False,
                "details": {"error": "Not enough data"}}

    current_price = bars["close"].iloc[-1]

    # Mid price from bid/ask
    if snapshot and snapshot.get("bid") and snapshot.get("ask"):
        mid = (snapshot["bid"] + snapshot["ask"]) / 2.0
        spread = snapshot["ask"] - snapshot["bid"]
    else:
        mid = current_price
        spread = current_price * 0.002

    # Annualized volatility
    returns = bars["returns"].dropna()
    sigma = returns.std() * np.sqrt(252 * 78)  # 78 five-min bars/day
    sigma = max(sigma, 0.01)

    inventory_skew = kwargs.get("inventory_skew", 0.0)
    gamma = kwargs.get("gamma", Config.STOIKOV_SPREAD_MULT)
    time_remaining = kwargs.get("time_remaining", 0.5)

    # Reservation price
    reservation = mid - inventory_skew * gamma * (sigma ** 2) * time_remaining

    price_advantage = (reservation - current_price) / current_price
    passed = current_price <= reservation

    return {
        "name": "Stoikov",
        "signal": round(price_advantage, 5),
        "passed": passed,
        "details": {
            "current_price": round(current_price, 2),
            "mid_price": round(mid, 2),
            "reservation_price": round(reservation, 2),
            "price_advantage": f"{price_advantage:+.3%}",
            "spread": round(spread, 4),
            "volatility": round(sigma, 4),
            "inventory_skew": round(inventory_skew, 2),
        },
    }
