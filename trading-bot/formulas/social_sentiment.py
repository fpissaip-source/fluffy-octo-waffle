"""
social_sentiment.py — StockTwits + Reddit Sentiment

Echtes soziales Momentum:
- StockTwits: kostenlose API, bullish/bearish Votes direkt von Tradern
- Reddit WSB: Erwähnungen und Stimmung aus r/wallstreetbets
"""

import logging
import requests
from typing import Optional

import pandas as pd

logger = logging.getLogger("bot.social")

STOCKTWITS_URL = "https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
REDDIT_URL = "https://www.reddit.com/r/wallstreetbets/search.json"


def _get_stocktwits(symbol: str) -> dict:
    try:
        url = STOCKTWITS_URL.format(symbol=symbol)
        r = requests.get(url, timeout=5, headers={"User-Agent": "TradingBot/1.0"})
        if r.status_code != 200:
            return {"bullish": 0, "bearish": 0, "total": 0, "score": 0.0}

        data = r.json()
        messages = data.get("messages", [])
        bullish = sum(1 for m in messages if m.get("entities", {}).get("sentiment", {}) and
                      m["entities"]["sentiment"].get("basic") == "Bullish")
        bearish = sum(1 for m in messages if m.get("entities", {}).get("sentiment", {}) and
                      m["entities"]["sentiment"].get("basic") == "Bearish")
        total = bullish + bearish

        score = (bullish - bearish) / total if total > 0 else 0.0

        return {
            "bullish": bullish,
            "bearish": bearish,
            "total": total,
            "score": round(score, 3),
            "message_count": len(messages),
        }
    except Exception as e:
        logger.debug(f"StockTwits failed {symbol}: {e}")
        return {"bullish": 0, "bearish": 0, "total": 0, "score": 0.0}


def _get_reddit_mentions(symbol: str) -> dict:
    try:
        params = {
            "q": symbol,
            "sort": "new",
            "limit": 25,
            "t": "day",
            "restrict_sr": "true",
        }
        headers = {"User-Agent": "TradingBot/1.0"}
        r = requests.get(REDDIT_URL, params=params, headers=headers, timeout=5)
        if r.status_code != 200:
            return {"mentions": 0, "score": 0.0}

        posts = r.json().get("data", {}).get("children", [])
        mentions = len(posts)
        upvotes = sum(p["data"].get("score", 0) for p in posts)
        avg_upvote = upvotes / mentions if mentions > 0 else 0

        # Score basiert auf Anzahl und Upvotes
        score = min(1.0, (mentions / 10) * 0.5 + min(1.0, avg_upvote / 100) * 0.5)

        return {
            "mentions": mentions,
            "avg_upvotes": round(avg_upvote, 1),
            "score": round(score, 3),
        }
    except Exception as e:
        logger.debug(f"Reddit failed {symbol}: {e}")
        return {"mentions": 0, "score": 0.0}


def evaluate(bars: pd.DataFrame, symbol: str = "", **kwargs) -> dict:
    if not symbol:
        return {"name": "Social", "signal": 0.0, "passed": True,
                "details": {"error": "No symbol"}}

    st = _get_stocktwits(symbol)
    reddit = _get_reddit_mentions(symbol)

    # Kombinierter Score: 70% StockTwits, 30% Reddit
    combined_score = st["score"] * 0.7 + reddit["score"] * 0.3

    # Passed: positives Sentiment oder keine Daten (nicht blockieren)
    passed = combined_score >= -0.2 or st["total"] == 0

    return {
        "name": "Social",
        "signal": round(combined_score, 4),
        "passed": passed,
        "details": {
            "stocktwits_bullish": st["bullish"],
            "stocktwits_bearish": st["bearish"],
            "stocktwits_score": st["score"],
            "reddit_mentions": reddit["mentions"],
            "reddit_score": reddit["score"],
            "combined_score": round(combined_score, 3),
        },
    }
