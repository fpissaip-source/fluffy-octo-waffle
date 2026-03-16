"""
ai_reasoning.py — GPT-4o-mini Trading Entscheidungs-Layer

Schicht 2: Der "Reasoning Layer"
Sendet alle gesammelten Daten an GPT-4o-mini und fragt:
"Soll ich diese Aktie kaufen? Wie hoch ist die Wahrscheinlichkeit für +35%?"

GPT bekommt:
- Technische Analyse (RSI, EMA, Bollinger, Z-Score, VWAP)
- Screener-Daten (Volume-Spike, % Change)
- News-Headlines (letzte 24h)
- Pump & Dump Risiko
- Short Squeeze Potenzial
- Markt-Regime
"""

import json
import logging
import os
from typing import Optional

import pandas as pd

logger = logging.getLogger("bot.ai")

MODEL = "gpt-4o-mini"


def _build_prompt(symbol: str, bars: pd.DataFrame, formula_results: dict,
                  news_headlines: list, regime: str) -> str:
    closes = bars["close"].values
    current_price = closes[-1]
    change_1d = (closes[-1] - closes[-20]) / closes[-20] * 100 if len(closes) >= 20 else 0

    mom = formula_results.get("Momentum", {}).get("details", {})
    cat = formula_results.get("Catalyst", {}).get("details", {})
    zsc = formula_results.get("Z-Score", {}).get("details", {})
    bay = formula_results.get("Bayesian", {}).get("details", {})

    headlines_text = "\n".join(f"  - {h}" for h in news_headlines[:5]) if news_headlines else "  - Keine aktuellen News"

    passed_filters = [name for name, r in formula_results.items() if r.get("passed")]
    failed_filters = [name for name, r in formula_results.items() if not r.get("passed")]

    return f"""Du bist ein erfahrener Quantitative Trader mit 20 Jahren Erfahrung bei einem Top-Hedgefonds.
Du analysierst gerade eine potenzielle Trade-Opportunity für eine Penny-Stock-Challenge (Ziel: 100x).

AKTIE: {symbol}
PREIS: ${current_price:.4f}
VERÄNDERUNG HEUTE: {change_1d:+.1f}%
MARKT-REGIME: {regime}

TECHNISCHE ANALYSE:
- RSI: {mom.get('rsi', 'N/A')}
- EMA-Diff: {mom.get('ema_diff', 'N/A')}
- Rate of Change: {mom.get('roc', 'N/A')}
- Bollinger Position: {mom.get('bb_position', 'N/A')} (0=unten, 1=oben)
- Z-Score Signal: {zsc.get('z_score', 'N/A')}
- Bayesian Posterior: {bay.get('posterior', 'N/A')}

MARKT-STRUKTUR:
- VWAP: {cat.get('vwap', 'N/A')} | Über VWAP: {cat.get('above_vwap', 'N/A')}
- Volume Ratio: {cat.get('vol_ratio', 'N/A')}x über Durchschnitt
- Short Squeeze Potenzial: {cat.get('squeeze_potential', 'N/A')}
- Pump & Dump Risiko: {cat.get('pump_dump', 'N/A')} (Konfidenz: {cat.get('pd_confidence', 0)})
- Breakout über Widerstand: {cat.get('breakout', 'N/A')}

FILTER-ERGEBNIS:
- Bestanden: {', '.join(passed_filters) if passed_filters else 'Keine'}
- Nicht bestanden: {', '.join(failed_filters) if failed_filters else 'Keine'}

AKTUELLE NEWS (letzte 24h):
{headlines_text}

STRATEGIE-KONTEXT:
- Challenge: Aus $100 → $10.000 machen (100x)
- Take Profit: +35% pro Trade
- Stop Loss: -10% pro Trade
- Max 2 Positionen gleichzeitig

AUFGABE:
Analysiere diese Daten wie ein Profi. Berücksichtige:
1. Ist das ein echter Ausbruch oder ein Fake-Move?
2. Gibt es einen Katalysator (News, Earnings, Short Squeeze)?
3. Wie hoch ist das Risiko von Pump & Dump?
4. Passt das zum Regime?

Antworte NUR als JSON (kein anderer Text):
{{"probability": <0-100>, "action": "<BUY|HOLD|AVOID>", "confidence": "<LOW|MEDIUM|HIGH>", "reason": "<1 Satz auf Deutsch>", "risk": "<LOW|MEDIUM|HIGH>"}}"""


def evaluate(bars: pd.DataFrame, broker=None, symbol: str = "",
             formula_results: dict = None, regime: str = "NORMAL", **kwargs) -> dict:
    """
    Sendet alle Daten an GPT-4o-mini für finale Handelsentscheidung.
    """
    if formula_results is None:
        formula_results = {}

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return {
            "name": "AI-Reasoning",
            "signal": 0.5,
            "passed": True,
            "details": {"error": "OPENAI_API_KEY not set"},
        }

    if len(bars) < 20:
        return {
            "name": "AI-Reasoning",
            "signal": 0.0,
            "passed": False,
            "details": {"error": "Not enough bars"},
        }

    # News aus Catalyst-Result holen
    news_headlines = formula_results.get("Catalyst", {}).get("details", {}).get("headlines", [])

    prompt = _build_prompt(symbol, bars, formula_results, news_headlines, regime)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Du bist ein präziser Trading-Analyst. Antworte immer nur als valides JSON."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=150,
            temperature=0.1,  # Niedrig = konsistenter, weniger kreativ
        )

        raw = response.choices[0].message.content.strip()
        # JSON extrahieren falls zusätzlicher Text
        if "{" in raw:
            raw = raw[raw.index("{"):raw.rindex("}") + 1]

        data = json.loads(raw)
        probability = float(data.get("probability", 50)) / 100
        action = data.get("action", "HOLD")
        confidence = data.get("confidence", "MEDIUM")
        reason = data.get("reason", "")
        risk = data.get("risk", "MEDIUM")

        # Signal: probability als Score (-1 bis +1)
        signal = (probability - 0.5) * 2  # 80% → +0.6, 20% → -0.6

        # Passed: GPT sagt BUY + hohe Wahrscheinlichkeit + kein HIGH risk
        passed = (
            action == "BUY"
            and probability >= 0.65
            and risk != "HIGH"
        )

        logger.info(f"AI [{symbol}]: {action} | Prob: {probability:.0%} | {reason}")

        return {
            "name": "AI-Reasoning",
            "signal": round(signal, 4),
            "passed": passed,
            "details": {
                "probability": f"{probability:.0%}",
                "action": action,
                "confidence": confidence,
                "risk": risk,
                "reason": reason,
                "model": MODEL,
            },
        }

    except Exception as e:
        logger.error(f"AI reasoning failed {symbol}: {e}")
        return {
            "name": "AI-Reasoning",
            "signal": 0.0,
            "passed": True,  # Nicht blockieren bei Fehler
            "details": {"error": str(e)},
        }
