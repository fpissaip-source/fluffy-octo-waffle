"""
orderbook.py — Level-2 Orderbook Simulation

Simuliert Orderbook-Analyse mit verfügbaren Daten:
- Bid/Ask Spread (enger Spread = liquide, weiter = gefährlich)
- Buy/Sell Pressure via Trade-Flow
- Imbalance: mehr Käufer oder Verkäufer?
- Iceberg Detection: versteckte große Orders

Nutzt Alpaca Quote + Trade Data.
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional

logger = logging.getLogger("bot.orderbook")


def _analyze_spread(bid: float, ask: float, price: float) -> dict:
    if bid <= 0 or ask <= 0 or price <= 0:
        return {"spread_pct": 0.0, "spread_ok": True}

    spread = ask - bid
    spread_pct = spread / price

    # Für Penny Stocks: bis 2% Spread akzeptabel
    spread_ok = spread_pct < 0.02

    return {
        "spread": round(spread, 6),
        "spread_pct": round(spread_pct, 4),
        "spread_ok": spread_ok,
        "bid": round(bid, 4),
        "ask": round(ask, 4),
    }


def _compute_trade_flow(bars: pd.DataFrame) -> dict:
    """
    Schätzt Buy/Sell Druck aus OHLCV Bars.
    Wenn Close > Open = mehr Käufer (grüne Kerze).
    Volume-Gewichtet für bessere Genauigkeit.
    """
    if len(bars) < 5:
        return {"buy_pressure": 0.5, "imbalance": 0.0}

    recent = bars.tail(10)
    closes = recent["close"].values
    opens = recent["open"].values
    volumes = recent["volume"].values

    buy_vol = 0.0
    sell_vol = 0.0

    for i in range(len(recent)):
        c, o, v = closes[i], opens[i], volumes[i]
        if c > o:
            buy_vol += v * (c - o) / (c + o + 1e-10)
        else:
            sell_vol += v * (o - c) / (c + o + 1e-10)

    total = buy_vol + sell_vol
    buy_pressure = buy_vol / total if total > 0 else 0.5
    imbalance = (buy_vol - sell_vol) / total if total > 0 else 0.0

    return {
        "buy_pressure": round(buy_pressure, 3),
        "sell_pressure": round(1 - buy_pressure, 3),
        "imbalance": round(imbalance, 3),
    }


def _detect_accumulation(bars: pd.DataFrame) -> dict:
    """
    Erkennt stille Akkumulation (Institutionen kaufen leise).
    Merkmale: Hohes Volume aber enge Preisspanne (jemand kauft ohne den Preis zu bewegen).
    """
    if len(bars) < 10:
        return {"accumulation": False, "score": 0.0}

    recent = bars.tail(10)
    volumes = recent["volume"].values
    price_ranges = (recent["high"] - recent["low"]).values
    closes = recent["close"].values

    avg_vol = np.mean(volumes)
    avg_range = np.mean(price_ranges)

    # Hohes Volume + enge Range = Akkumulation
    vol_ratio = volumes[-1] / avg_vol if avg_vol > 0 else 1.0
    range_ratio = price_ranges[-1] / avg_range if avg_range > 0 else 1.0

    accum_score = 0.0
    if vol_ratio > 2.0 and range_ratio < 0.7:
        accum_score = 0.8  # Klassische Akkumulation
    elif vol_ratio > 1.5 and range_ratio < 0.9:
        accum_score = 0.4

    # Trend der letzten 5 Closes
    trend = (closes[-1] - closes[-5]) / closes[-5] if closes[-5] > 0 else 0

    return {
        "accumulation": accum_score > 0.5,
        "score": round(accum_score, 2),
        "vol_ratio": round(vol_ratio, 2),
        "range_ratio": round(range_ratio, 2),
        "trend_5bar": round(trend, 4),
    }


def evaluate(bars: pd.DataFrame, broker=None, symbol: str = "", **kwargs) -> dict:
    if len(bars) < 10:
        return {"name": "Orderbook", "signal": 0.0, "passed": True,
                "details": {"error": "Not enough bars"}}

    # Spread-Analyse via Snapshot
    spread_data = {"spread_pct": 0.0, "spread_ok": True}
    if broker and symbol:
        try:
            snap = broker.get_snapshot(symbol)
            if snap:
                spread_data = _analyze_spread(
                    snap.get("bid", 0),
                    snap.get("ask", 0),
                    bars["close"].iloc[-1],
                )
        except Exception:
            pass

    flow = _compute_trade_flow(bars)
    accum = _detect_accumulation(bars)

    score = 0.0

    # Spread: enger Spread = positiv
    if spread_data.get("spread_ok", True):
        score += 0.2
    else:
        score -= 0.3  # Zu weiter Spread = gefährlich

    # Buy Pressure
    buy_p = flow["buy_pressure"]
    if buy_p > 0.65:
        score += 0.3
    elif buy_p > 0.55:
        score += 0.1
    elif buy_p < 0.40:
        score -= 0.3

    # Akkumulation erkannt
    if accum["accumulation"]:
        score += 0.3

    # Imbalance
    if flow["imbalance"] > 0.3:
        score += 0.2
    elif flow["imbalance"] < -0.3:
        score -= 0.2

    score = max(-1.0, min(1.0, score))

    # Passed: kein extremer Spread + mehr Käufer als Verkäufer
    passed = spread_data.get("spread_ok", True) and flow["buy_pressure"] >= 0.45

    return {
        "name": "Orderbook",
        "signal": round(score, 4),
        "passed": passed,
        "details": {
            "spread_pct": f"{spread_data.get('spread_pct', 0):.2%}",
            "spread_ok": spread_data.get("spread_ok", True),
            "buy_pressure": f"{flow['buy_pressure']:.0%}",
            "sell_pressure": f"{flow['sell_pressure']:.0%}",
            "imbalance": flow["imbalance"],
            "accumulation": accum["accumulation"],
            "accum_score": accum["score"],
        },
    }
