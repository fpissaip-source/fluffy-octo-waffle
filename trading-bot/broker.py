"""broker.py — Alpaca API Wrapper. Handles account, market data, orders."""

import logging
from datetime import datetime, timedelta
from typing import Optional

import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np

from config import Config

logger = logging.getLogger("bot.broker")


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
        start = end - timedelta(days=max(7, limit // 78 + 3))

        crypto_symbols = {"BTCUSD", "ETHUSD", "SOLUSD", "AVAXUSD", "LINKUSD"}
        is_crypto = symbol.upper() in crypto_symbols

        kwargs = dict(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            limit=limit,
        )
        if not is_crypto:
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

    def get_latest_price(self, symbol: str) -> Optional[float]:
        try:
            return float(self.api.get_latest_trade(symbol).price)
        except Exception as e:
            logger.warning(f"Price failed {symbol}: {e}")
            return None

    def get_snapshot(self, symbol: str) -> Optional[dict]:
        try:
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
            crypto = symbol.upper().endswith("USD") and not symbol.upper().endswith("BUSD")
            tif = "gtc" if crypto else "day"
            params = dict(symbol=symbol, qty=qty, side="buy", type="market", time_in_force=tif)
            if not crypto:
                params["extended_hours"] = False  # market orders nicht in extended hours
            order = self.api.submit_order(**params)
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

    def is_market_open(self) -> bool:
        return self.api.get_clock().is_open
