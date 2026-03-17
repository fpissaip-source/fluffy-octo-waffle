"""broker.py — Alpaca API Wrapper. Handles account, market data, orders."""

import logging
from datetime import datetime, timedelta
from typing import Optional

import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np

from config import Config

logger = logging.getLogger("bot.broker")

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

    def get_bars(self, symbol: str, timeframe: str = "5Min", limit: int = 100) -> pd.DataFrame:
        tf_map = {
            "1Min": tradeapi.TimeFrame.Minute,
            "5Min": tradeapi.TimeFrame(5, tradeapi.TimeFrameUnit.Minute),
            "15Min": tradeapi.TimeFrame(15, tradeapi.TimeFrameUnit.Minute),
            "1Hour": tradeapi.TimeFrame.Hour,
            "1Day": tradeapi.TimeFrame.Day,
        }
        tf = tf_map.get(timeframe, tradeapi.TimeFrame(5, tradeapi.TimeFrameUnit.Minute))
        end = datetime.now()
        # Korrekte Tagesberechnung je nach Timeframe (war: immer limit//78 = 5Min-Formel)
        _bars_per_day = {"1Min": 390, "5Min": 78, "15Min": 26, "1Hour": 7, "1Day": 1}
        _bpd = _bars_per_day.get(timeframe, 78)
        # x2 für Wochenenden/Feiertage, mindestens 14 Tage Puffer
        start = end - timedelta(days=max(14, (limit // _bpd + 3) * 2))

        is_crypto = symbol.upper() in CRYPTO_SYMBOLS

        kwargs = dict(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            limit=limit,
        )

        if is_crypto:
            bars = self.api.get_crypto_bars(_alpaca_crypto(symbol), tf, **kwargs).df
        else:
            kwargs["feed"] = "iex"
            bars = self.api.get_bars(symbol, tf, **kwargs).df

        if bars.empty:
            logger.warning(f"No bars for {symbol}")
            return pd.DataFrame()

        bars = bars.tail(limit).copy()
        bars["returns"] = bars["close"].pct_change()
        bars["log_returns"] = np.log(bars["close"] / bars["close"].shift(1))
        bars.dropna(inplace=True)
        return bars

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
        now = datetime.now(timezone.utc)
        next_open = clock.next_open.replace(tzinfo=timezone.utc)
        # Pre-market: 4:00–9:30 AM ET (5.5h vor Öffnung)
        pre_market_start = next_open - timedelta(hours=5.5)
        if pre_market_start <= now < next_open:
            return "extended"
        # After-hours: 16:00–20:00 ET, Mo–Fr
        # Bug-Fix: prev_close = next_close - 6.5h zeigte auf NÄCHSTEN Montag,
        # nicht auf den letzten Freitag → After-Hours wurde nie erkannt.
        # Stattdessen: direkt per ET-Uhrzeit prüfen (UTC-4 = EDT, gilt ~7 Monate)
        now_et = now - timedelta(hours=4)
        if now_et.weekday() < 5:  # Mo–Fr
            et_hour = now_et.hour + now_et.minute / 60
            if 16.0 <= et_hour < 20.0:
                return "extended"
        return "closed"
