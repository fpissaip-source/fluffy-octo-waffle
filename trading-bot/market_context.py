"""
market_context.py — Echtzeit Markt-Kontext fuer den Reasoning Layer

Datenquellen:
  - VIX (Fear Index) via yfinance — kostenlos, kein Key noetig
  - LunarCrush (Crypto Sentiment) via API — kostenlos mit Key
  - Sector Performance via yfinance — kostenlos

Wird von ReasoningLayer genutzt um GPT-4o besseren Kontext zu geben.
"""

import logging
import time
from typing import Optional

import requests

from config import Config

logger = logging.getLogger("bot.context")


class VIXFetcher:
    """Holt den aktuellen VIX (Fear & Greed Index des Marktes)."""

    def __init__(self):
        self._cache: Optional[dict] = None
        self._cache_time: float = 0
        self._cache_ttl = 300  # 5 Minuten

    def get(self) -> dict:
        if self._cache and time.time() - self._cache_time < self._cache_ttl:
            return self._cache

        try:
            import yfinance as yf
            vix = yf.Ticker("^VIX")
            price = vix.fast_info.get("last_price") or vix.fast_info.get("previousClose")

            if price:
                if price < 15:
                    level = "SEHR NIEDRIG — Markt extrem entspannt, Selbstgefaelligkeit-Risiko"
                elif price < 20:
                    level = "NIEDRIG — Normaler ruhiger Markt"
                elif price < 30:
                    level = "ERHOHT — Erhoehte Unsicherheit"
                elif price < 40:
                    level = "HOCH — Starke Angst im Markt"
                else:
                    level = "EXTREM — Markt-Panik / Crash-Modus"

                result = {"vix": round(float(price), 2), "level": level}
                self._cache = result
                self._cache_time = time.time()
                logger.debug(f"VIX: {price:.1f} ({level})")
                return result

        except Exception as e:
            logger.warning(f"VIX fetch failed: {e}")

        return {"vix": None, "level": "Nicht verfuegbar"}


class LunarCrushFetcher:
    """
    Holt Crypto-Sentiment von LunarCrush.
    Kostenlos mit API Key (lunarcrush.com).
    Erfordert LUNARCRUSH_API_KEY in .env
    """

    SYMBOL_MAP = {
        "BTCUSD": "BTC",
        "ETHUSD": "ETH",
        "SOLUSD": "SOL",
        "AVAXUSD": "AVAX",
        "LINKUSD": "LINK",
    }

    def __init__(self):
        self.api_key = Config.LUNARCRUSH_API_KEY
        self.available = bool(self.api_key)
        self._cache: dict = {}
        self._cache_ttl = 600  # 10 Minuten

    def get(self, symbol: str) -> Optional[dict]:
        if not self.available:
            return None

        coin = self.SYMBOL_MAP.get(symbol.upper())
        if not coin:
            return None

        cache_key = f"lunar_{coin}"
        if cache_key in self._cache:
            ts, data = self._cache[cache_key]
            if time.time() - ts < self._cache_ttl:
                return data

        try:
            url = f"https://lunarcrush.com/api4/public/coins/{coin}/v1"
            resp = requests.get(
                url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json().get("data", {})
                result = {
                    "symbol": coin,
                    "galaxy_score": data.get("galaxy_score"),       # 0-100 Overall Score
                    "alt_rank": data.get("alt_rank"),                # Ranking vs andere Coins
                    "sentiment": data.get("sentiment"),              # 1-5 (bullish)
                    "social_volume": data.get("social_volume_24h"),  # Posts/Mentions
                    "social_score": data.get("social_score"),        # Social Engagement
                    "price_score": data.get("price_score"),          # Preis-Momentum
                }
                self._cache[cache_key] = (time.time(), result)
                logger.debug(f"LunarCrush {coin}: galaxy={result['galaxy_score']} sentiment={result['sentiment']}")
                return result

        except Exception as e:
            logger.warning(f"LunarCrush failed for {symbol}: {e}")

        return None


class SectorFetcher:
    """Holt Sektor-Performance via yfinance (SPY, QQQ, XLK etc.)."""

    SECTORS = {
        "SPY": "S&P 500",
        "QQQ": "Tech/Nasdaq",
        "XLK": "Tech Sektor",
        "XLF": "Financials",
    }

    def __init__(self):
        self._cache: Optional[dict] = None
        self._cache_time: float = 0
        self._cache_ttl = 600  # 10 Minuten

    def get(self) -> dict:
        if self._cache and time.time() - self._cache_time < self._cache_ttl:
            return self._cache

        result = {}
        try:
            import yfinance as yf
            for ticker, name in self.SECTORS.items():
                try:
                    t = yf.Ticker(ticker)
                    info = t.fast_info
                    price = info.get("last_price")
                    prev = info.get("previousClose")
                    if price and prev and prev > 0:
                        change_pct = (price - prev) / prev * 100
                        result[name] = round(change_pct, 2)
                except Exception:
                    pass

            self._cache = result
            self._cache_time = time.time()

        except Exception as e:
            logger.warning(f"Sector fetch failed: {e}")

        return result


class MarketContext:
    """
    Aggregiert alle Markt-Kontext-Daten fuer den Reasoning Layer.
    Einmal instanziiert, gecacht fuer Performance.
    """

    def __init__(self):
        self.vix = VIXFetcher()
        self.lunar = LunarCrushFetcher()
        self.sectors = SectorFetcher()

    def get_context(self, symbol: str) -> dict:
        """Gibt vollen Markt-Kontext fuer ein Symbol zurueck."""
        vix_data = self.vix.get()
        sector_data = self.sectors.get()
        lunar_data = self.lunar.get(symbol) if symbol.endswith("USD") else None

        return {
            "vix": vix_data,
            "sectors": sector_data,
            "lunar": lunar_data,
        }

    def format_for_prompt(self, symbol: str) -> str:
        """Formatiert Kontext als Text fuer GPT-4o Prompt."""
        ctx = self.get_context(symbol)
        lines = []

        # VIX
        vix = ctx["vix"]
        if vix["vix"]:
            lines.append(f"  - VIX (Fear Index): {vix['vix']} — {vix['level']}")
        else:
            lines.append("  - VIX: nicht verfuegbar")

        # Sektoren
        if ctx["sectors"]:
            sector_str = ", ".join(
                f"{name}: {chg:+.1f}%" for name, chg in ctx["sectors"].items()
            )
            lines.append(f"  - Sektoren heute: {sector_str}")

        # LunarCrush (nur fuer Crypto)
        if ctx["lunar"]:
            l = ctx["lunar"]
            lines.append(
                f"  - LunarCrush {l['symbol']}: "
                f"Galaxy={l.get('galaxy_score', 'n/a')} "
                f"Sentiment={l.get('sentiment', 'n/a')}/5 "
                f"SocialVol={l.get('social_volume', 'n/a')}"
            )

        return "\n".join(lines)
