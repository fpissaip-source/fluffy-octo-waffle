"""
catalyst.py — News-Katalysator & Pump/Dump Filter

Prüft ob eine Preisbewegung einen echten Grund hat:
- News vorhanden? (via Alpaca News API)
- Pump & Dump Muster? (zu schnell, zu hoch, kein Volume-Follow-through)
- Short Squeeze Setup? (hohes Short-Interest + Volume-Spike)
- VWAP Entry (Kurs über/unter VWAP?)
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger("bot.catalyst")


def _detect_pump_dump(bars: pd.DataFrame) -> dict:
    """
    Erkennt verdächtige Pump & Dump Muster.
    Merkmale:
    - Preis steigt >30% in <3 Bars, dann sofort Volumen-Einbruch
    - Volume in letzten Bars viel niedriger als beim Spike
    - Lange obere Dochten (Abverkauf von oben)
    """
    if len(bars) < 10:
        return {"is_pump_dump": False, "confidence": 0.0}

    closes = bars["close"].values
    volumes = bars["volume"].values
    highs = bars["high"].values

    # Letzter Preis vs. 10-Bar Tief
    recent_low = np.min(closes[-10:])
    current = closes[-1]
    spike_pct = (current - recent_low) / recent_low if recent_low > 0 else 0

    # Volume jetzt vs. beim Spike
    peak_vol_idx = np.argmax(volumes[-10:])
    peak_vol = volumes[-10:][peak_vol_idx]
    current_vol = volumes[-1]
    vol_decay = current_vol / peak_vol if peak_vol > 0 else 1.0

    # Obere Dochten (Abverkauf)
    body = abs(closes[-3:] - bars["open"].values[-3:])
    upper_wick = highs[-3:] - np.maximum(closes[-3:], bars["open"].values[-3:])
    wick_ratio = np.mean(upper_wick / (body + 1e-10))

    # Pump & Dump Score
    pd_score = 0.0
    if spike_pct > 0.50 and vol_decay < 0.3:  # >50% Spike, Volume bricht ein
        pd_score += 0.5
    if spike_pct > 0.30 and wick_ratio > 2.0:  # Lange Dochten nach Spike
        pd_score += 0.3
    if vol_decay < 0.2 and spike_pct > 0.20:
        pd_score += 0.2

    return {
        "is_pump_dump": pd_score >= 0.5,
        "confidence": round(pd_score, 2),
        "spike_pct": round(spike_pct, 3),
        "vol_decay": round(vol_decay, 3),
        "wick_ratio": round(wick_ratio, 2),
    }


def _detect_short_squeeze(bars: pd.DataFrame) -> dict:
    """
    Short-Squeeze Setup erkennen:
    - Hoher Volume-Spike (3x+)
    - Preis bricht über Widerstand
    - Shorts müssen eindecken -> weiterer Anstieg
    """
    if len(bars) < 20:
        return {"squeeze_potential": 0.0}

    closes = bars["close"].values
    volumes = bars["volume"].values

    # Volume-Ratio: aktuell vs. 20-Bar Durchschnitt
    avg_vol = np.mean(volumes[-20:])
    current_vol = volumes[-1]
    vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0

    # Breakout über 20-Bar High
    resistance = np.max(closes[-20:-1])
    breakout = closes[-1] > resistance

    # Konsistente Aufwärtsbewegung (letzte 3 Bars alle grün)
    green_bars = sum(1 for i in range(-3, 0) if closes[i] > bars["open"].values[i])

    squeeze_score = 0.0
    if vol_ratio > 3.0:
        squeeze_score += 0.4
    if breakout:
        squeeze_score += 0.3
    if green_bars >= 2:
        squeeze_score += 0.2
    if vol_ratio > 5.0:
        squeeze_score += 0.1

    return {
        "squeeze_potential": round(min(squeeze_score, 1.0), 2),
        "vol_ratio": round(vol_ratio, 2),
        "breakout": breakout,
        "green_bars": green_bars,
    }


def _compute_vwap(bars: pd.DataFrame) -> dict:
    """
    VWAP (Volume Weighted Average Price) berechnen.
    Kurs über VWAP = Bullen kontrollieren.
    Kurs unter VWAP = Bären kontrollieren.
    """
    if len(bars) < 5:
        return {"vwap": 0.0, "above_vwap": False, "vwap_distance": 0.0}

    typical_price = (bars["high"] + bars["low"] + bars["close"]) / 3
    vwap = (typical_price * bars["volume"]).sum() / bars["volume"].sum()
    current = bars["close"].iloc[-1]
    distance = (current - vwap) / vwap if vwap > 0 else 0

    return {
        "vwap": round(float(vwap), 4),
        "above_vwap": current > vwap,
        "vwap_distance": round(float(distance), 4),
    }


def _check_news(broker, symbol: str) -> dict:
    """
    Prüft ob es aktuelle News für das Symbol gibt.
    Alpaca News API ist kostenlos und bereits im SDK.
    """
    try:
        end = datetime.utcnow()
        start = end - timedelta(hours=24)
        news = broker.api.get_news(
            symbol,
            start=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            end=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            limit=5,
        )
        count = len(news) if news else 0
        headlines = [n.headline for n in (news or [])[:3]]
        return {
            "has_news": count > 0,
            "news_count": count,
            "headlines": headlines,
        }
    except Exception as e:
        logger.debug(f"News check failed {symbol}: {e}")
        return {"has_news": False, "news_count": 0, "headlines": []}


def evaluate(bars: pd.DataFrame, broker=None, symbol: str = "", **kwargs) -> dict:
    """
    Hauptfunktion: Katalysator-Analyse.

    Gibt Signal +1 wenn echter Katalysator vorliegt (News + kein P&D + VWAP ok)
    Gibt Signal -1 wenn Pump & Dump erkannt
    """
    if len(bars) < 20:
        return {
            "name": "Catalyst",
            "signal": 0.0,
            "passed": True,  # Im Zweifel nicht blockieren
            "details": {"error": "Not enough bars"},
        }

    pd_result = _detect_pump_dump(bars)
    squeeze = _detect_short_squeeze(bars)
    vwap = _compute_vwap(bars)

    # News nur wenn broker verfügbar
    news = {"has_news": False, "news_count": 0, "headlines": []}
    if broker and symbol:
        news = _check_news(broker, symbol)

    score = 0.0

    # Pump & Dump -> stark negativ
    if pd_result["is_pump_dump"]:
        score -= 0.8
    else:
        score += 0.2

    # Short Squeeze Potential -> positiv
    score += squeeze["squeeze_potential"] * 0.4

    # VWAP über VWAP -> positiv
    if vwap["above_vwap"]:
        score += 0.2
    else:
        score -= 0.1

    # News vorhanden -> leicht positiv
    if news["has_news"]:
        score += 0.2

    score = max(-1.0, min(1.0, score))

    # Passed: kein P&D + VWAP ok + mindestens etwas Momentum
    passed = (
        not pd_result["is_pump_dump"]
        and vwap["above_vwap"]
        and score >= 0.1
    )

    return {
        "name": "Catalyst",
        "signal": round(score, 4),
        "passed": passed,
        "details": {
            "pump_dump": pd_result["is_pump_dump"],
            "pd_confidence": pd_result["confidence"],
            "squeeze_potential": squeeze["squeeze_potential"],
            "vol_ratio": squeeze["vol_ratio"],
            "breakout": squeeze["breakout"],
            "vwap": vwap["vwap"],
            "above_vwap": vwap["above_vwap"],
            "vwap_distance": f"{vwap['vwap_distance']:+.2%}",
            "has_news": news["has_news"],
            "news_count": news["news_count"],
            "headlines": news["headlines"][:2],
        },
    }
