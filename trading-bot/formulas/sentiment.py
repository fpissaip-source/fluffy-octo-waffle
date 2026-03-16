import logging
import time
import re
from datetime import datetime, timedelta
from typing import Optional

from config import Config

logger = logging.getLogger("bot.sentiment")

BULLISH_KEYWORDS = {
    "beat earnings": 2, "beats estimates": 2, "record revenue": 2,
    "upgraded": 2, "all-time high": 2, "breakout": 2,
    "acquisition": 1.5, "buyback": 1.5, "share repurchase": 1.5,
    "dividend increase": 1.5, "fda approval": 2, "patent granted": 1.5,
    "partnership": 1, "expansion": 1, "growth": 1,
    "bullish": 1.5, "rally": 1.5, "surge": 1.5, "soar": 1.5,
    "outperform": 1.5, "buy rating": 2, "price target raised": 2,
    "strong demand": 1.5, "record sales": 2, "market share gain": 1.5,
    "positive": 1, "gains": 1, "higher": 1, "rises": 1,
    "optimistic": 1, "momentum": 1, "recovery": 1, "rebound": 1,
}

BEARISH_KEYWORDS = {
    "misses estimates": 2, "missed earnings": 2, "profit warning": 2,
    "downgraded": 2, "sell rating": 2, "price target cut": 2,
    "layoffs": 1.5, "restructuring": 1.5, "bankruptcy": 2,
    "investigation": 1.5, "lawsuit": 1.5, "fraud": 2,
    "recall": 1.5, "fda rejection": 2, "data breach": 1.5,
    "bearish": 1.5, "crash": 1.5, "plunge": 1.5, "tumble": 1.5,
    "decline": 1, "falls": 1, "drops": 1, "lower": 1, "loss": 1,
    "sell-off": 1.5, "correction": 1, "recession": 1.5,
    "war": 1.5, "conflict": 1, "attack": 1, "crisis": 1.5,
}

MACRO_BULLISH = {
    "rate cut": 2, "fed dovish": 2, "stimulus": 1.5,
    "ceasefire": 2, "peace deal": 2, "peace talks": 1.5,
    "inflation falls": 1.5, "strong jobs": 1.5, "gdp growth": 1.5,
}

MACRO_BEARISH = {
    "rate hike": 2, "fed hawkish": 2, "tightening": 1.5,
    "war escalation": 2, "missile strike": 2,
    "oil spike": 1.5, "inflation rises": 1.5, "stagflation": 2,
    "default": 2, "debt ceiling": 1.5, "government shutdown": 1.5,
}


class NewsFetcher:
    def __init__(self, broker):
        self.api = broker.api

    def get_news(self, symbol: str, limit: int = 10, hours_back: int = 24) -> list[dict]:
        try:
            end = datetime.now()
            start = end - timedelta(hours=hours_back)
            news = self.api.get_news(
                symbol=symbol,
                start=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                end=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                limit=limit,
            )
            return [{
                "headline": item.headline,
                "summary": getattr(item, "summary", "") or "",
                "source": getattr(item, "source", "unknown"),
                "created_at": str(item.created_at),
            } for item in news]
        except Exception as e:
            logger.warning(f"News fetch failed for {symbol}: {e}")
            return []

    def get_market_news(self, limit: int = 15) -> list[dict]:
        try:
            news = self.api.get_news(limit=limit)
            return [{
                "headline": item.headline,
                "summary": getattr(item, "summary", "") or "",
                "source": getattr(item, "source", "unknown"),
                "created_at": str(item.created_at),
            } for item in news]
        except Exception as e:
            logger.warning(f"Market news fetch failed: {e}")
            return []


class KeywordScorer:
    @staticmethod
    def score_text(text: str, bullish_dict: dict, bearish_dict: dict) -> dict:
        text_lower = text.lower()
        bull_score = 0.0
        bear_score = 0.0
        bull_hits = []
        bear_hits = []

        for keyword, weight in bullish_dict.items():
            count = text_lower.count(keyword)
            if count > 0:
                bull_score += weight * count
                bull_hits.append(keyword)

        for keyword, weight in bearish_dict.items():
            count = text_lower.count(keyword)
            if count > 0:
                bear_score += weight * count
                bear_hits.append(keyword)

        total = bull_score + bear_score
        normalized = (bull_score - bear_score) / total if total > 0 else 0.0

        return {
            "score": round(normalized, 3),
            "bull_score": round(bull_score, 2),
            "bear_score": round(bear_score, 2),
            "bull_hits": bull_hits,
            "bear_hits": bear_hits,
        }

    @staticmethod
    def score_articles(articles: list[dict], is_macro: bool = False) -> dict:
        if not articles:
            return {"score": 0.0, "article_count": 0, "details": []}

        bull_dict = MACRO_BULLISH if is_macro else BULLISH_KEYWORDS
        bear_dict = MACRO_BEARISH if is_macro else BEARISH_KEYWORDS

        scores = []
        details = []

        for article in articles:
            text = article["headline"] + " " + article.get("summary", "")
            result = KeywordScorer.score_text(text, bull_dict, bear_dict)
            scores.append(result["score"])
            if result["bull_hits"] or result["bear_hits"]:
                details.append({
                    "headline": article["headline"][:80],
                    "score": result["score"],
                    "signals": result["bull_hits"] + [f"-{b}" for b in result["bear_hits"]],
                })

        if scores:
            weights = [1.0 / (i + 1) for i in range(len(scores))]
            total_weight = sum(weights)
            weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
        else:
            weighted_score = 0.0

        return {
            "score": round(weighted_score, 3),
            "article_count": len(articles),
            "details": details[:5],
        }


class ClaudeAnalyzer:
    def __init__(self):
        self.api_key = Config.ANTHROPIC_API_KEY
        self.available = bool(self.api_key)
        self.last_call = 0
        self.min_interval = 60

    def analyze(self, headlines: list[str], symbol: str) -> Optional[dict]:
        if not self.available:
            return None
        now = time.time()
        if now - self.last_call < self.min_interval:
            return None
        try:
            import requests
            import json
            news_text = "\n".join(f"- {h}" for h in headlines[:10])
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "content-type": "application/json",
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 300,
                    "messages": [{"role": "user", "content": (
                        f"Analyze these headlines for {symbol} stock sentiment.\n\n"
                        f"{news_text}\n\n"
                        f"Respond ONLY with JSON:\n"
                        f'{{"score": <-1.0 to 1.0>, "confidence": <0 to 1>, '
                        f'"reason": "<one sentence>"}}'
                    )}],
                },
                timeout=15,
            )
            self.last_call = time.time()
            if response.status_code == 200:
                data = response.json()
                text = data["content"][0]["text"]
                match = re.search(r'\{[^}]+\}', text)
                if match:
                    return json.loads(match.group())
            return None
        except Exception as e:
            logger.warning(f"Claude analysis failed: {e}")
            return None


class SentimentEngine:
    def __init__(self, broker):
        self.news_fetcher = NewsFetcher(broker)
        self.scorer = KeywordScorer()
        self.claude = ClaudeAnalyzer()
        self._cache: dict = {}
        self._cache_ttl = 300

    def _get_cached(self, key: str) -> Optional[dict]:
        if key in self._cache:
            ts, val = self._cache[key]
            if time.time() - ts < self._cache_ttl:
                return val
        return None

    def _set_cache(self, key: str, value: dict):
        self._cache[key] = (time.time(), value)

    def analyze_symbol(self, symbol: str) -> dict:
        cached = self._get_cached(f"sentiment_{symbol}")
        if cached:
            return cached

        articles = self.news_fetcher.get_news(symbol, limit=10, hours_back=12)
        symbol_sentiment = self.scorer.score_articles(articles)

        macro_articles = self.news_fetcher.get_market_news(limit=10)
        macro_sentiment = self.scorer.score_articles(macro_articles, is_macro=True)

        claude_result = None
        headlines = [a["headline"] for a in articles]
        if headlines and abs(symbol_sentiment["score"]) > 0.3:
            claude_result = self.claude.analyze(headlines, symbol)

        combined = symbol_sentiment["score"] * 0.50 + macro_sentiment["score"] * 0.30
        if claude_result and "score" in claude_result:
            combined += claude_result["score"] * 0.20
        else:
            combined += symbol_sentiment["score"] * 0.20

        confidence = 0.3
        if symbol_sentiment["article_count"] > 3:
            confidence += 0.2
        if symbol_sentiment["article_count"] > 7:
            confidence += 0.1
        if macro_sentiment["article_count"] > 3:
            confidence += 0.1
        if claude_result:
            confidence += 0.2
        confidence = min(confidence, 1.0)

        result = {
            "score": round(max(-1.0, min(1.0, combined)), 3),
            "confidence": round(confidence, 2),
            "symbol_sentiment": symbol_sentiment,
            "macro_sentiment": macro_sentiment,
            "claude_analysis": claude_result,
            "article_count": symbol_sentiment["article_count"] + macro_sentiment["article_count"],
        }
        self._set_cache(f"sentiment_{symbol}", result)
        return result


import pandas as pd


def evaluate(bars: pd.DataFrame, **kwargs) -> dict:
    broker = kwargs.get("broker", None)
    symbol = kwargs.get("symbol", "")
    threshold = kwargs.get("threshold", -0.3)

    if not broker or not symbol:
        return {
            "name": "Sentiment",
            "signal": 0.0,
            "passed": True,
            "details": {"source": "no_broker", "note": "Neutral when broker unavailable"},
        }
    try:
        engine = SentimentEngine(broker)
        result = engine.analyze_symbol(symbol)
        score = result["score"]

        return {
            "name": "Sentiment",
            "signal": round(score, 3),
            "passed": score > threshold,
            "details": {
                "score": score,
                "confidence": result["confidence"],
                "articles": result["article_count"],
                "macro": result["macro_sentiment"]["score"],
                "symbol_news": result["symbol_sentiment"]["score"],
                "claude": result["claude_analysis"] is not None,
            },
        }
    except Exception as e:
        logger.warning(f"Sentiment analysis failed for {symbol}: {e}")
        return {
            "name": "Sentiment",
            "signal": 0.0,
            "passed": True,
            "details": {"error": str(e)},
        }

