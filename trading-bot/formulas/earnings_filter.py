"""
earnings_filter.py — Earnings-Kalender Filter

Vermeidet blindes Trading vor Earnings:
- Earnings in <2 Tagen = AVOID (zu hohes Risiko)
- Earnings in 2-5 Tagen = Warnung aber erlaubt
- Kein Earnings in Sicht = grünes Licht

Nutzt yfinance (kostenlos).
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

logger = logging.getLogger("bot.earnings")


def _get_next_earnings(symbol: str) -> Optional[datetime]:
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        cal = ticker.calendar
        if cal is None or cal.empty:
            return None
        if "Earnings Date" in cal.index:
            date_val = cal.loc["Earnings Date"].iloc[0] if hasattr(cal.loc["Earnings Date"], 'iloc') else cal.loc["Earnings Date"]
            if pd.notna(date_val):
                if hasattr(date_val, 'to_pydatetime'):
                    return date_val.to_pydatetime()
                return datetime.combine(date_val, datetime.min.time())
        return None
    except Exception as e:
        logger.debug(f"Earnings check failed {symbol}: {e}")
        return None


def evaluate(bars: pd.DataFrame, symbol: str = "", **kwargs) -> dict:
    if not symbol:
        return {"name": "Earnings", "signal": 0.0, "passed": True,
                "details": {"status": "no_symbol"}}

    next_earnings = _get_next_earnings(symbol)
    now = datetime.now()

    if next_earnings is None:
        return {
            "name": "Earnings",
            "signal": 0.2,
            "passed": True,
            "details": {"status": "no_earnings_found", "days_until": None},
        }

    # Stelle sicher dass next_earnings timezone-naive ist
    if hasattr(next_earnings, 'tzinfo') and next_earnings.tzinfo is not None:
        next_earnings = next_earnings.replace(tzinfo=None)

    days_until = (next_earnings - now).days

    if days_until < 0:
        # Earnings bereits vorbei
        return {
            "name": "Earnings",
            "signal": 0.3,
            "passed": True,
            "details": {"status": "earnings_passed", "days_until": days_until},
        }
    elif days_until <= 1:
        # Earnings morgen oder heute = BLOCKIEREN
        logger.warning(f"{symbol}: Earnings in {days_until} Tag(en) — Trade blockiert")
        return {
            "name": "Earnings",
            "signal": -0.8,
            "passed": False,
            "details": {
                "status": "earnings_imminent",
                "days_until": days_until,
                "earnings_date": next_earnings.strftime("%Y-%m-%d"),
                "warning": f"Earnings in {days_until} Tag(en)!",
            },
        }
    elif days_until <= 3:
        # Earnings in 2-3 Tagen = Warnung, aber erlaubt
        return {
            "name": "Earnings",
            "signal": -0.2,
            "passed": True,
            "details": {
                "status": "earnings_soon",
                "days_until": days_until,
                "earnings_date": next_earnings.strftime("%Y-%m-%d"),
                "warning": f"Earnings in {days_until} Tagen",
            },
        }
    else:
        # Earnings weit weg = kein Problem
        return {
            "name": "Earnings",
            "signal": 0.2,
            "passed": True,
            "details": {
                "status": "earnings_far",
                "days_until": days_until,
                "earnings_date": next_earnings.strftime("%Y-%m-%d"),
            },
        }
