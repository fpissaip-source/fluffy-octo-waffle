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


class FinnhubClient:
    """
    Finnhub Integration — zwei Signale:
    1. Earnings Calendar → Block wenn Earnings ≤ N Tage entfernt
    2. Insider Transactions → Kaufsignal wenn Insider kaufen
    """

    BASE_URL = "https://finnhub.io/api/v1"

    def __init__(self):
        self.api_key = Config.FINNHUB_API_KEY
        self.available = bool(self.api_key)
        self._cache: dict = {}
        self._cache_ttl = 3600  # 1 Stunde (Earnings ändern sich selten)

    def _get(self, endpoint: str, params: dict) -> Optional[dict]:
        if not self.available:
            return None
        cache_key = f"{endpoint}:{sorted(params.items())}"
        if cache_key in self._cache:
            ts, data = self._cache[cache_key]
            if time.time() - ts < self._cache_ttl:
                return data
        try:
            params["token"] = self.api_key
            resp = requests.get(
                f"{self.BASE_URL}/{endpoint}",
                params=params,
                timeout=5,
            )
            if resp.status_code == 200:
                data = resp.json()
                self._cache[cache_key] = (time.time(), data)
                return data
        except Exception as e:
            logger.warning(f"Finnhub {endpoint} failed: {e}")
        return None

    def days_to_earnings(self, symbol: str) -> Optional[int]:
        """
        Gibt zurück wie viele Tage bis zum nächsten Earnings-Termin.
        None = kein Termin bekannt.
        """
        from datetime import date, timedelta
        today = date.today()
        look_ahead = (today + timedelta(days=30)).isoformat()

        data = self._get("calendar/earnings", {
            "from": today.isoformat(),
            "to": look_ahead,
            "symbol": symbol,
        })
        if not data:
            return None

        earnings_list = data.get("earningsCalendar", [])
        if not earnings_list:
            return None

        # Nächster zukünftiger Termin
        for entry in sorted(earnings_list, key=lambda x: x.get("date", "")):
            try:
                earn_date = date.fromisoformat(entry["date"])
                days = (earn_date - today).days
                if days >= 0:
                    logger.info(f"[Finnhub] {symbol} Earnings in {days} Tagen ({earn_date})")
                    return days
            except Exception:
                continue
        return None

    def is_earnings_blackout(self, symbol: str) -> bool:
        """True wenn Earnings ≤ FINNHUB_EARNINGS_BLOCK_DAYS Tage entfernt."""
        if not self.available:
            return False
        days = self.days_to_earnings(symbol)
        if days is None:
            return False
        blocked = days <= Config.FINNHUB_EARNINGS_BLOCK_DAYS
        if blocked:
            logger.warning(
                f"[Finnhub] EARNINGS BLACKOUT {symbol}: "
                f"Earnings in {days} Tagen — kein Trade!"
            )
        return blocked

    def insider_signal(self, symbol: str) -> float:
        """
        Insider-Transaktions-Signal: +1 (starkes Kaufen) bis -1 (starkes Verkaufen).
        Basiert auf Netto-Käufe der letzten 90 Tage.
        """
        from datetime import date, timedelta
        data = self._get("stock/insider-transactions", {"symbol": symbol})
        if not data:
            return 0.0

        transactions = data.get("data", [])
        cutoff = (date.today() - timedelta(days=90)).isoformat()

        buy_value = 0.0
        sell_value = 0.0
        for tx in transactions:
            if tx.get("transactionDate", "") < cutoff:
                continue
            shares = abs(tx.get("share", 0) or 0)
            price = abs(tx.get("price", 0) or 0)
            value = shares * price
            tx_type = (tx.get("transactionCode") or "").upper()
            if tx_type in ("P",):       # Purchase
                buy_value += value
            elif tx_type in ("S",):     # Sale
                sell_value += value

        total = buy_value + sell_value
        if total == 0:
            return 0.0

        # Netto-Signal: +1 = alles Käufe, -1 = alles Verkäufe
        signal = (buy_value - sell_value) / total
        logger.info(
            f"[Finnhub] Insider {symbol}: "
            f"Käufe=${buy_value:,.0f} Verkäufe=${sell_value:,.0f} "
            f"Signal={signal:+.2f}"
        )
        return round(signal, 3)


class MarketContext:
    """
    Aggregiert alle Markt-Kontext-Daten fuer den Reasoning Layer.
    Einmal instanziiert, gecacht fuer Performance.
    """

    def __init__(self):
        self.vix = VIXFetcher()
        self.lunar = LunarCrushFetcher()
        self.sectors = SectorFetcher()
        self.finnhub = FinnhubClient()

    def get_context(self, symbol: str) -> dict:
        """Gibt vollen Markt-Kontext fuer ein Symbol zurueck."""
        vix_data = self.vix.get()
        sector_data = self.sectors.get()
        lunar_data = self.lunar.get(symbol) if symbol.endswith("USD") else None

        return {
            "vix": vix_data,
            "sectors": sector_data,
            "lunar": lunar_data,
            "finnhub_insider": self.finnhub.insider_signal(symbol) if not symbol.endswith("USD") else 0.0,
        }

    def is_earnings_blackout(self, symbol: str) -> bool:
        """Delegiert an FinnhubClient — wird von engine.py vor jeder Order geprüft."""
        return self.finnhub.is_earnings_blackout(symbol)

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

        # Finnhub Insider Signal (nur Aktien)
        insider = ctx.get("finnhub_insider", 0.0)
        if insider != 0.0:
            direction = "KAUFEN" if insider > 0 else "VERKAUFEN"
            lines.append(f"  - Insider Trades (90d): {direction} ({insider:+.2f})")

        return "\n".join(lines)
