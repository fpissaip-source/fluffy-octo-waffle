"""
sentiment.py — Multimodale Sentiment-Analyse

Drei Ebenen (alle Pflicht):
1. Alpaca News API — Echtzeit-Nachrichten pro Symbol
2. Keyword-Scoring — schnelle Sentiment-Bewertung
3. Google Gemini — Deep Analysis (IMMER aktiv, kein Fallback)

Output: Sentiment-Score von -1.0 (extrem bearish) bis +1.0 (extrem bullish)
"""

import logging
import time
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timedelta
from typing import Optional

from config import Config

logger = logging.getLogger("bot.sentiment")


# ═══════════════════════════════════════════════════════
#  KEYWORD DICTIONARIES
# ═══════════════════════════════════════════════════════

BULLISH_KEYWORDS = {
    # Starke Signale (Gewicht 2)
    "beat earnings": 2, "beats estimates": 2, "record revenue": 2,
    "upgraded": 2, "all-time high": 2, "breakout": 2,
    "acquisition": 1.5, "buyback": 1.5, "share repurchase": 1.5,
    "dividend increase": 1.5, "fda approval": 2, "patent granted": 1.5,
    "partnership": 1, "expansion": 1, "growth": 1,
    "bullish": 1.5, "rally": 1.5, "surge": 1.5, "soar": 1.5,
    "outperform": 1.5, "buy rating": 2, "price target raised": 2,
    "strong demand": 1.5, "record sales": 2, "market share gain": 1.5,
    # Moderate Signale (Gewicht 1)
    "positive": 1, "gains": 1, "higher": 1, "rises": 1,
    "optimistic": 1, "momentum": 1, "recovery": 1, "rebound": 1,
    "beat": 1, "exceeds": 1, "strong": 1, "profit": 1,
}

BEARISH_KEYWORDS = {
    # Starke Signale (Gewicht 2)
    "misses estimates": 2, "missed earnings": 2, "profit warning": 2,
    "downgraded": 2, "sell rating": 2, "price target cut": 2,
    "layoffs": 1.5, "restructuring": 1.5, "bankruptcy": 2,
    "investigation": 1.5, "lawsuit": 1.5, "fraud": 2,
    "recall": 1.5, "fda rejection": 2, "data breach": 1.5,
    "sanctions": 1.5, "tariff": 1.5, "trade war": 1.5,
    # Moderate Signale (Gewicht 1)
    "bearish": 1.5, "crash": 1.5, "plunge": 1.5, "tumble": 1.5,
    "decline": 1, "falls": 1, "drops": 1, "lower": 1, "loss": 1,
    "negative": 1, "concern": 1, "risk": 0.5, "uncertainty": 1,
    "sell-off": 1.5, "correction": 1, "recession": 1.5,
    "war": 1.5, "conflict": 1, "attack": 1, "crisis": 1.5,
}

# Makro-Keywords die den GESAMTEN Markt betreffen
MACRO_BULLISH = {
    "rate cut": 2, "fed dovish": 2, "stimulus": 1.5,
    "ceasefire": 2, "peace deal": 2, "peace talks": 1.5,
    "inflation falls": 1.5, "unemployment low": 1,
    "strong jobs": 1.5, "gdp growth": 1.5, "consumer confidence": 1,
}

MACRO_BEARISH = {
    "rate hike": 2, "fed hawkish": 2, "tightening": 1.5,
    "war escalation": 2, "missile strike": 2, "strait of hormuz": 2,
    "oil spike": 1.5, "oil surge": 1.5, "supply disruption": 1.5,
    "inflation rises": 1.5, "stagflation": 2,
    "default": 2, "debt ceiling": 1.5, "government shutdown": 1.5,
    "nuclear": 1.5, "invasion": 2, "bombing": 1.5,
}


# ═══════════════════════════════════════════════════════
#  NEWS FETCHER (via Alpaca)
# ═══════════════════════════════════════════════════════

class NewsFetcher:
    """Holt Nachrichten ueber die Alpaca News API."""

    def __init__(self, broker):
        self.api = broker.api

    def _fetch_with_timeout(self, fn, timeout: int = 5):
        """Führt eine API-Funktion mit Timeout aus. Gibt None zurück bei Timeout."""
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(fn)
            try:
                return future.result(timeout=timeout)
            except FuturesTimeoutError:
                logger.warning("News API timeout nach 5s — übersprungen")
                return None
            except Exception as e:
                logger.warning(f"News API Fehler: {e}")
                return None

    def get_news(self, symbol: str, limit: int = 10, hours_back: int = 24) -> list[dict]:
        """Alpaca News API deaktiviert — Gemini übernimmt Sentiment direkt."""
        return []

    def get_market_news(self, limit: int = 15) -> list[dict]:
        """Alpaca News API deaktiviert — Gemini übernimmt Sentiment direkt."""
        return []


# ═══════════════════════════════════════════════════════
#  KEYWORD SENTIMENT SCORER
# ═══════════════════════════════════════════════════════

class KeywordScorer:
    """Schnelle Sentiment-Bewertung basierend auf Keywords."""

    @staticmethod
    def score_text(text: str, bullish_dict: dict, bearish_dict: dict) -> dict:
        """Bewertet einen Text und gibt Score + Details zurueck."""
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
        if total == 0:
            normalized = 0.0
        else:
            # -1 (bearish) bis +1 (bullish)
            normalized = (bull_score - bear_score) / total

        return {
            "score": round(normalized, 3),
            "bull_score": round(bull_score, 2),
            "bear_score": round(bear_score, 2),
            "bull_hits": bull_hits,
            "bear_hits": bear_hits,
        }

    @staticmethod
    def score_articles(articles: list[dict], is_macro: bool = False) -> dict:
        """Bewertet mehrere Artikel und aggregiert."""
        if not articles:
            return {
                "score": 0.0,
                "article_count": 0,
                "details": [],
            }

        bull_dict = MACRO_BULLISH if is_macro else BULLISH_KEYWORDS
        bear_dict = MACRO_BEARISH if is_macro else BEARISH_KEYWORDS

        scores = []
        details = []

        for article in articles:
            text = article["headline"] + " " + article.get("summary", "")
            result = KeywordScorer.score_text(text, bull_dict, bear_dict)

            # Neuere Artikel haben mehr Gewicht
            scores.append(result["score"])
            if result["bull_hits"] or result["bear_hits"]:
                details.append({
                    "headline": article["headline"][:80],
                    "score": result["score"],
                    "signals": result["bull_hits"] + [f"-{b}" for b in result["bear_hits"]],
                })

        # Gewichteter Durchschnitt (neuere zuerst)
        if scores:
            weights = [1.0 / (i + 1) for i in range(len(scores))]
            total_weight = sum(weights)
            weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
        else:
            weighted_score = 0.0

        return {
            "score": round(weighted_score, 3),
            "article_count": len(articles),
            "details": details[:5],  # Top 5 relevanteste
        }


# ═══════════════════════════════════════════════════════
#  GEMINI DEEP ANALYSIS (optional)
# ═══════════════════════════════════════════════════════

class GeminiAnalyzer:
    """
    Nutzt Google Gemini fuer tiefere Sentiment-Analyse.
    Nur fuer kritische Momente (spart API-Kosten).

    Erfordert GEMINI_API_KEY in .env
    """

    def __init__(self):
        self.api_key = Config.GEMINI_API_KEY
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY fehlt — Sentiment Analysis nicht moeglich")
        self.last_call = 0
        self.min_interval = 60  # Max 1 Call pro Minute

    def analyze(self, articles: list[dict], symbol: str) -> Optional[dict]:
        """
        Fragt Gemini nach einer Einschaetzung.
        Bekommt vollständige Artikel-Daten (Headline + Datum) für Trend-Erkennung.
        Gibt Score (-1 bis +1), Confidence und Begründung zurück.
        """
        if not articles:
            return None

        # Rate limiting
        now = time.time()
        if now - self.last_call < self.min_interval:
            return None

        try:
            from google import genai
            from google.genai import types as genai_types
            import json

            client = genai.Client(api_key=self.api_key)

            # Top N Headlines mit Datum (für narrative Trends über mehrere Tage)
            top_articles = articles[:Config.SENTIMENT_GEMINI_HEADLINES]
            news_text = "\n".join(
                f"- [{a.get('created_at', '')[:10]}] {a['headline']}"
                for a in top_articles
            )

            prompt = (
                f"You are analyzing multi-day news flow for {symbol} stock.\n"
                f"News from the last {Config.NEWS_LOOKBACK_HOURS}h (newest first):\n\n"
                f"{news_text}\n\n"
                f"Identify the NARRATIVE TREND across all articles:\n"
                f"- Is sentiment improving, worsening, or stable over time?\n"
                f"- Are there fundamental catalysts (earnings, FDA, M&A, macro) that suggest a multi-day move?\n"
                f"- Score: +1.0 = strong bullish catalyst, 0.0 = neutral, -1.0 = strong bearish catalyst"
            )

            # Structured Output — kein JSON-Parsing nötig
            response_schema = genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "score":      genai_types.Schema(type=genai_types.Type.NUMBER),
                    "confidence": genai_types.Schema(type=genai_types.Type.NUMBER),
                    "trend":      genai_types.Schema(type=genai_types.Type.STRING,
                                                     enum=["improving", "stable", "worsening"]),
                    "reason":     genai_types.Schema(type=genai_types.Type.STRING),
                },
                required=["score", "confidence", "trend", "reason"],
            )

            response = client.models.generate_content(
                model=Config.REASONING_MODEL,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=response_schema,
                    max_output_tokens=300,
                    thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
                ),
            )

            self.last_call = time.time()
            result = json.loads(response.text or "{}")
            if result:
                logger.info(
                    f"Gemini sentiment {symbol}: score={result.get('score'):.2f} "
                    f"trend={result.get('trend')} | {result.get('reason', '')[:80]}"
                )
            return result if result else None

        except Exception as e:
            logger.warning(f"Gemini analysis failed: {e}")
            return None


# ═══════════════════════════════════════════════════════
#  HAUPTMODUL: SENTIMENT ENGINE
# ═══════════════════════════════════════════════════════

class SentimentEngine:
    """
    Kombiniert alle Sentiment-Quellen:
    1. Symbol-spezifische News (Alpaca)
    2. Makro-News (Markt allgemein)
    3. Optional: Google Gemini Deep Analysis

    Output wird als Signal in den Bayesian Updater gespeist.
    """

    def __init__(self, broker):
        self.news_fetcher = NewsFetcher(broker)
        self.scorer = KeywordScorer()
        self.gemini = GeminiAnalyzer()

        # Cache — für Swing Trading auf 15 Minuten erhöht
        # (News-Lage ändert sich nicht im Minutentakt)
        self._cache: dict = {}
        self._cache_ttl = 900  # 15 Minuten Cache

    def _is_cached(self, key: str) -> bool:
        if key in self._cache:
            ts, _ = self._cache[key]
            if time.time() - ts < self._cache_ttl:
                return True
        return False

    def _get_cached(self, key: str) -> Optional[dict]:
        if self._is_cached(key):
            return self._cache[key][1]
        return None

    def _set_cache(self, key: str, value: dict):
        self._cache[key] = (time.time(), value)

    def analyze_symbol(self, symbol: str) -> dict:
        """
        Vollstaendige Sentiment-Analyse fuer ein Symbol.

        Returns:
            score: -1.0 (bearish) bis +1.0 (bullish)
            confidence: 0 bis 1 (wie sicher)
            details: Aufschluesselung
        """
        # Cache Check
        cached = self._get_cached(f"sentiment_{symbol}")
        if cached:
            return cached

        # ── 1. Symbol-spezifische News (erweiterter Horizont für Swing-Kontext) ──
        articles = self.news_fetcher.get_news(
            symbol,
            limit=Config.NEWS_MAX_ARTICLES,
            hours_back=Config.NEWS_LOOKBACK_HOURS,
        )
        symbol_sentiment = self.scorer.score_articles(articles)

        # ── 2. Makro-News ──
        macro_articles = self.news_fetcher.get_market_news(limit=15)
        macro_sentiment = self.scorer.score_articles(macro_articles, is_macro=True)

        # ── 3. Gemini Trend-Analyse (Top 25 Headlines mit Datum) ──
        # Läuft immer wenn Artikel vorhanden (nicht nur bei extremem Score)
        gemini_result = None
        if articles:
            gemini_result = self.gemini.analyze(articles, symbol)

        # ── Kombination ──
        # Gewichtung: Symbol-News 50%, Makro 30%, Gemini 20%
        combined = symbol_sentiment["score"] * 0.50 + macro_sentiment["score"] * 0.30
        if gemini_result and "score" in gemini_result:
            combined += gemini_result["score"] * 0.20
        else:
            combined += symbol_sentiment["score"] * 0.20

        # Confidence basierend auf Datenqualitaet
        confidence = 0.3  # Baseline
        if symbol_sentiment["article_count"] > 3:
            confidence += 0.2
        if symbol_sentiment["article_count"] > 10:
            confidence += 0.1
        if macro_sentiment["article_count"] > 3:
            confidence += 0.1
        if gemini_result:
            confidence += 0.2
            # Klarer Trend = höhere Confidence
            if gemini_result.get("trend") in ("improving", "worsening"):
                confidence += 0.1
        confidence = min(confidence, 1.0)

        result = {
            "score": round(max(-1.0, min(1.0, combined)), 3),
            "confidence": round(confidence, 2),
            "symbol_sentiment": symbol_sentiment,
            "macro_sentiment": macro_sentiment,
            "gemini_analysis": gemini_result,
            "article_count": symbol_sentiment["article_count"] + macro_sentiment["article_count"],
        }

        self._set_cache(f"sentiment_{symbol}", result)
        return result

    def get_bayesian_likelihood(self, symbol: str) -> float:
        """
        Konvertiert Sentiment-Score in Bayesian Likelihood Ratio.

        Score +0.5 → LR 2.0 (starkes bullish Signal)
        Score  0.0 → LR 1.0 (neutral)
        Score -0.5 → LR 0.5 (starkes bearish Signal)
        """
        result = self.analyze_symbol(symbol)
        score = result["score"]
        confidence = result["confidence"]

        # Nur bei ausreichender Confidence den Score nutzen
        if confidence < 0.3:
            return 1.0  # Neutral

        # Score (-1 bis +1) → LR (0.3 bis 3.0)
        # Exponentiell skaliert
        lr = 2.0 ** (score * confidence * 2)
        return round(max(0.3, min(3.0, lr)), 3)


# ═══════════════════════════════════════════════════════
#  STANDALONE FORMULA INTERFACE
# ═══════════════════════════════════════════════════════

def evaluate(bars=None, broker=None, symbol: str = "", **kwargs) -> dict:
    """
    Formula-Interface fuer Integration in engine.py.

    Braucht broker-Instanz fuer News-API Zugang.
    """
    if not broker or not symbol:
        return {
            "name": "Sentiment",
            "signal": 0.0,
            "passed": True,  # Bei fehlendem Zugang nicht blockieren
            "details": {"error": "No broker/symbol provided"},
        }

    try:
        engine = SentimentEngine(broker)
        result = engine.analyze_symbol(symbol)

        score = result["score"]
        threshold = kwargs.get("threshold", -0.3)  # Blockiert nur bei stark bearish

        return {
            "name": "Sentiment",
            "signal": round(score, 3),
            "passed": score > threshold,  # Nur blockieren wenn STARK bearish
            "details": {
                "score": score,
                "confidence": result["confidence"],
                "articles": result["article_count"],
                "macro": result["macro_sentiment"]["score"],
                "symbol_news": result["symbol_sentiment"]["score"],
                "gemini": result["gemini_analysis"] is not None,
                "gemini_trend": result["gemini_analysis"].get("trend") if result["gemini_analysis"] else None,
                "top_signals": (
                    result["symbol_sentiment"]["details"][:3]
                    if result["symbol_sentiment"]["details"]
                    else []
                ),
            },
        }

    except Exception as e:
        logger.warning(f"Sentiment analysis failed for {symbol}: {e}")
        return {
            "name": "Sentiment",
            "signal": 0.0,
            "passed": True,  # Nicht blockieren bei Fehler
            "details": {"error": str(e)},
        }
