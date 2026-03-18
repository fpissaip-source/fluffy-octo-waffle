"""broker.py — Alpaca API Wrapper. Handles account, market data, orders."""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timedelta
from typing import Optional

import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np

try:
    import yfinance as yf
    _YFINANCE_AVAILABLE = True
except ImportError:
    _YFINANCE_AVAILABLE = False

from config import Config

logger = logging.getLogger("bot.broker")

# Symbole ohne IEX-Daten — werden nicht nochmal versucht (bis Bot-Neustart)
_iex_blacklist: set[str] = set()

CRYPTO_SYMBOLS = {"BTCUSD", "ETHUSD", "SOLUSD", "AVAXUSD", "LINKUSD"}


def _alpaca_crypto(symbol: str) -> str:
    """BTCUSD → BTC/USD (Alpaca crypto API-Format)."""
    s = symbol.upper()
    if s.endswith("USD") and s in CRYPTO_SYMBOLS:
        return s[:-3] + "/USD"
    return s


class AlpacaBroker:
    def __init__(self):
        self.api = tradeapi.REST(
            key_id=Config.API_KEY,
            secret_key=Config.SECRET_KEY,
            base_url=Config.BASE_URL,
        )
        self._validate_connection()

    def _validate_connection(self):
        try:
            account = self.api.get_account()
            mode = "PAPER" if Config.is_paper() else "!! LIVE !!"
            logger.info(f"Connected [{mode}]  Equity: ${float(account.equity):,.2f}")
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            raise

    def get_equity(self) -> float:
        return float(self.api.get_account().equity)

    def get_buying_power(self) -> float:
        return float(self.api.get_account().buying_power)

    def get_cash(self) -> float:
        return float(self.api.get_account().cash)

    def get_positions(self) -> dict:
        positions = {}
        for p in self.api.list_positions():
            positions[p.symbol] = {
                "qty": float(p.qty),
                "avg_entry": float(p.avg_entry_price),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc),
                "side": p.side,
            }
        return positions

    def has_position(self, symbol: str) -> bool:
        return symbol in self.get_positions()

    def has_open_order(self, symbol: str) -> bool:
        """True wenn bereits eine offene Buy-Order für dieses Symbol existiert."""
        try:
            orders = self.api.list_orders(status="open", symbols=[symbol])
            return any(o.side == "buy" for o in orders)
        except Exception:
            return False

    def cancel_open_buy_orders(self, symbol: str) -> int:
        """Storniert alle offenen Buy-Orders für ein Symbol. Gibt Anzahl zurück."""
        try:
            orders = self.api.list_orders(status="open", symbols=[symbol])
            cancelled = 0
            for o in orders:
                if o.side == "buy":
                    self.api.cancel_order(o.id)
                    cancelled += 1
            return cancelled
        except Exception as e:
            logger.warning(f"cancel_open_buy_orders({symbol}): {e}")
            return 0

    def get_bars(self, symbol: str, timeframe: str = "5Min", limit: int = 100) -> pd.DataFrame:
        global _iex_blacklist

        # Sofort überspringen wenn kein IEX-Daten bekannt
        if symbol in _iex_blacklist:
            return pd.DataFrame()

        tf_map = {
            "1Min": tradeapi.TimeFrame.Minute,
            "5Min": tradeapi.TimeFrame(5, tradeapi.TimeFrameUnit.Minute),
            "15Min": tradeapi.TimeFrame(15, tradeapi.TimeFrameUnit.Minute),
            "1Hour": tradeapi.TimeFrame.Hour,
            "1Day": tradeapi.TimeFrame.Day,
        }
        tf = tf_map.get(timeframe, tradeapi.TimeFrame(5, tradeapi.TimeFrameUnit.Minute))
        end = datetime.now()
        _bars_per_day = {"1Min": 390, "5Min": 78, "15Min": 26, "1Hour": 7, "1Day": 1}
        _bpd = _bars_per_day.get(timeframe, 78)
        start = end - timedelta(days=max(14, (limit // _bpd + 3) * 2))

        is_crypto = symbol.upper() in CRYPTO_SYMBOLS

        kwargs = dict(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            limit=limit,
        )

        try:
            if is_crypto:
                def _fetch():
                    return self.api.get_crypto_bars(_alpaca_crypto(symbol), tf, **kwargs).df
            else:
                kwargs["feed"] = "iex"
                def _fetch():
                    return self.api.get_bars(symbol, tf, **kwargs).df

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_fetch)
                try:
                    bars = future.result(timeout=6)
                except FuturesTimeoutError:
                    logger.warning(f"Bars timeout für {symbol} — zur Blacklist hinzugefügt")
                    _iex_blacklist.add(symbol)
                    return pd.DataFrame()

        except Exception as e:
            logger.warning(f"Bars fetch fehlgeschlagen für {symbol}: {e}")
            return pd.DataFrame()

        if bars.empty:
            logger.warning(f"No bars for {symbol} via IEX — versuche yfinance Fallback")
            _iex_blacklist.add(symbol)
            return self._get_bars_yfinance(symbol, timeframe, limit)

        bars = bars.tail(limit).copy()
        bars["returns"] = bars["close"].pct_change()
        bars["log_returns"] = np.log(bars["close"] / bars["close"].shift(1))
        bars.dropna(inplace=True)
        return bars

    def _get_bars_yfinance(self, symbol: str, timeframe: str = "1Hour", limit: int = 100) -> pd.DataFrame:
        """Fallback: Bars via Yahoo Finance wenn IEX keine Daten liefert."""
        if not _YFINANCE_AVAILABLE:
            logger.warning(f"yfinance nicht installiert — kein Fallback für {symbol}")
            return pd.DataFrame()

        tf_map = {
            "1Min": "1m", "5Min": "5m", "15Min": "15m",
            "1Hour": "1h", "1Day": "1d",
        }
        yf_interval = tf_map.get(timeframe, "1h")

        # Zeitraum: genug History für limit Bars
        period_map = {
            "1m": "7d", "5m": "60d", "15m": "60d",
            "1h": "730d", "1d": "2y",
        }
        period = period_map.get(yf_interval, "60d")

        try:
            ticker = yf.Ticker(symbol)
            raw = ticker.history(period=period, interval=yf_interval, auto_adjust=True)
            if raw.empty:
                logger.warning(f"yfinance: Keine Daten für {symbol}")
                return pd.DataFrame()

            bars = raw.rename(columns={
                "Open": "open", "High": "high", "Low": "low",
                "Close": "close", "Volume": "volume",
            })[["open", "high", "low", "close", "volume"]].copy()

            bars = bars.tail(limit)
            bars["returns"] = bars["close"].pct_change()
            bars["log_returns"] = np.log(bars["close"] / bars["close"].shift(1))
            bars.dropna(inplace=True)

            logger.info(f"[yfinance] {symbol}: {len(bars)} Bars geladen (Fallback)")
            return bars

        except Exception as e:
            logger.warning(f"yfinance Fallback fehlgeschlagen für {symbol}: {e}")
            return pd.DataFrame()

    def _get_crypto_price(self, symbol: str) -> Optional[float]:
        """Crypto-Preis via letztem 1-Minuten-Bar (get_latest_crypto_trade nicht verfügbar)."""
        try:
            bars = self.api.get_crypto_bars(
                _alpaca_crypto(symbol),
                tradeapi.TimeFrame.Minute,
                start=(datetime.now() - timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                limit=1,
            ).df
            if not bars.empty:
                return float(bars["close"].iloc[-1])
        except Exception as e:
            logger.debug(f"Crypto bar fallback failed {symbol}: {e}")
        return None

    def get_latest_price(self, symbol: str) -> Optional[float]:
        try:
            if symbol.upper() in CRYPTO_SYMBOLS:
                return self._get_crypto_price(symbol)
            return float(self.api.get_latest_trade(symbol).price)
        except Exception as e:
            logger.warning(f"Price failed {symbol}: {e}")
            return None

    def get_snapshot(self, symbol: str) -> Optional[dict]:
        try:
            if symbol.upper() in CRYPTO_SYMBOLS:
                price = self._get_crypto_price(symbol)
                if price is None:
                    return None
                return {
                    "price": price,
                    "bid": price,
                    "ask": price,
                    "spread": 0.0,
                    "volume": 0,
                    "vwap": None,
                }
            snap = self.api.get_snapshot(symbol)
            return {
                "price": float(snap.latest_trade.price),
                "bid": float(snap.latest_quote.bid_price),
                "ask": float(snap.latest_quote.ask_price),
                "spread": float(snap.latest_quote.ask_price) - float(snap.latest_quote.bid_price),
                "volume": int(snap.daily_bar.volume) if snap.daily_bar else 0,
                "vwap": float(snap.daily_bar.vwap) if snap.daily_bar else None,
            }
        except Exception as e:
            logger.warning(f"Snapshot failed {symbol}: {e}")
            return None

    def market_buy(self, symbol: str, qty: int) -> Optional[str]:
        try:
            status = self.get_market_status()
            if status == "extended":
                # Extended hours: Limit-Order leicht über Marktpreis (0.1% Slippage)
                price = self.get_latest_price(symbol)
                if not price:
                    logger.warning(f"Buy skipped {symbol}: no price available")
                    return None
                limit_price = round(price * 1.001, 2)
                order = self.api.submit_order(
                    symbol=symbol, qty=qty, side="buy",
                    type="limit", time_in_force="day",
                    limit_price=limit_price,
                    extended_hours=True,
                )
                logger.info(f"BUY {qty}x {symbol} @ LIMIT ${limit_price:.2f} [EXT] -> Order {order.id}")
            else:
                order = self.api.submit_order(
                    symbol=symbol, qty=qty, side="buy", type="market", time_in_force="day",
                )
                logger.info(f"BUY {qty}x {symbol} @ MARKET -> Order {order.id}")
            return order.id
        except Exception as e:
            logger.error(f"Buy failed {symbol}: {e}")
            return None

    def market_sell(self, symbol: str, qty: int) -> Optional[str]:
        try:
            crypto = symbol.upper().endswith("USD") and not symbol.upper().endswith("BUSD")
            tif = "gtc" if crypto else "day"
            order = self.api.submit_order(
                symbol=symbol, qty=qty, side="sell", type="market", time_in_force=tif,
            )
            logger.info(f"SELL {qty}x {symbol} @ MARKET -> Order {order.id}")
            return order.id
        except Exception as e:
            logger.error(f"Sell failed {symbol}: {e}")
            return None

    def limit_buy(self, symbol: str, qty: int, limit_price: float) -> Optional[str]:
        try:
            order = self.api.submit_order(
                symbol=symbol, qty=qty, side="buy",
                type="limit", time_in_force="day",
                limit_price=round(limit_price, 2),
            )
            logger.info(f"BUY {qty}x {symbol} @ LIMIT ${limit_price:.2f} -> Order {order.id}")
            return order.id
        except Exception as e:
            logger.error(f"Limit buy failed {symbol}: {e}")
            return None

    def close_position(self, symbol: str) -> Optional[str]:
        try:
            order = self.api.close_position(symbol)
            logger.info(f"CLOSE {symbol} -> Order {order.id}")
            return order.id
        except Exception as e:
            if "insufficient qty" in str(e).lower() or "position does not exist" in str(e).lower():
                logger.info(f"CLOSE {symbol}: Position bereits geschlossen — ignoriert")
            else:
                logger.error(f"Close failed {symbol}: {e}")
            return None

    def get_snapshots_batch(self, symbols: list) -> dict:
        """Batch-Snapshot für bis zu 200 Symbole auf einmal (nur US Stocks, kein Crypto)."""
        try:
            result = self.api.get_snapshots(symbols, feed="iex")
            return result if result else {}
        except Exception as e:
            logger.warning(f"Batch snapshots failed: {e}")
            return {}

    def is_market_open(self) -> bool:
        return self.api.get_clock().is_open

    def get_market_status(self) -> str:
        """Returns 'open', 'extended', or 'closed'."""
        from datetime import timezone
        clock = self.api.get_clock()
        if clock.is_open:
            return "open"
        # ET-Zeit dynamisch berechnen (EDT=UTC-4 / EST=UTC-5 je nach Jahreszeit)
        now_utc = datetime.now(timezone.utc)
        try:
            import zoneinfo
            now_et = now_utc.astimezone(zoneinfo.ZoneInfo("America/New_York"))
        except Exception:
            # Fallback: EDT (UTC-4) wenn zoneinfo nicht verfügbar
            now_et = now_utc - timedelta(hours=4)
        et_hour = now_et.hour + now_et.minute / 60
        weekday = now_et.weekday()  # 0=Mo, 6=So
        if weekday < 5:  # Mo–Fr
            # Pre-market: 04:00–09:30 ET
            if 4.0 <= et_hour < 9.5:
                return "extended"
            # After-hours: 16:00–20:00 ET
            if 16.0 <= et_hour < 20.0:
                return "extended"
        return "closed"
