import logging
import time
from datetime import datetime, timedelta
from typing import Optional

import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np

from config import Config

logger = logging.getLogger("bot.broker")

MAX_RETRIES = 3
RETRY_DELAY = 2


def _retry(func):
    def wrapper(*args, **kwargs):
        for attempt in range(MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAY * (2 ** attempt)
                    logger.warning(f"{func.__name__} failed (attempt {attempt + 1}): {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"{func.__name__} failed after {MAX_RETRIES} attempts: {e}")
                    raise
    return wrapper


class AlpacaBroker:
    def __init__(self):
        self.api = tradeapi.REST(
            key_id=Config.API_KEY,
            secret_key=Config.SECRET_KEY,
            base_url=Config.BASE_URL,
        )
        self._consecutive_errors = 0
        self._circuit_open = False
        self._circuit_reset_time = 0
        self._validate_connection()

    def _validate_connection(self):
        try:
            account = self.api.get_account()
            mode = "PAPER" if Config.is_paper() else "!! LIVE !!"
            logger.info(f"Connected [{mode}]  Equity: ${float(account.equity):,.2f}")
            self._consecutive_errors = 0
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            raise

    def _check_circuit(self) -> bool:
        if self._circuit_open:
            if time.time() > self._circuit_reset_time:
                self._circuit_open = False
                self._consecutive_errors = 0
                logger.info("Circuit breaker reset")
                return False
            return True
        return False

    def _record_error(self):
        self._consecutive_errors += 1
        if self._consecutive_errors >= 3:
            self._circuit_open = True
            self._circuit_reset_time = time.time() + 60
            logger.critical("Circuit breaker OPEN — pausing API calls for 60s")

    def _record_success(self):
        self._consecutive_errors = 0

    @_retry
    def get_equity(self) -> float:
        if self._check_circuit():
            raise RuntimeError("Circuit breaker open")
        try:
            result = float(self.api.get_account().equity)
            self._record_success()
            return result
        except Exception:
            self._record_error()
            raise

    @_retry
    def get_buying_power(self) -> float:
        return float(self.api.get_account().buying_power)

    @_retry
    def get_cash(self) -> float:
        return float(self.api.get_account().cash)

    @_retry
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

    @_retry
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

        bars = self.api.get_bars(
            symbol, tf,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            limit=limit, feed="iex",
        ).df

        if bars.empty:
            logger.warning(f"No bars for {symbol}")
            return pd.DataFrame()

        bars = bars.tail(limit).copy()
        bars["returns"] = bars["close"].pct_change()
        bars["log_returns"] = np.log(bars["close"] / bars["close"].shift(1))
        bars.dropna(inplace=True)
        return bars

    @_retry
    def get_latest_price(self, symbol: str) -> Optional[float]:
        try:
            price = float(self.api.get_latest_trade(symbol).price)
            self._record_success()
            return price
        except Exception as e:
            self._record_error()
            logger.warning(f"Price failed {symbol}: {e}")
            return None

    @_retry
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
            order = self.api.submit_order(
                symbol=symbol, qty=qty, side="buy",
                type="market", time_in_force="day",
            )
            logger.info(f"BUY {qty}x {symbol} @ MARKET -> Order {order.id}")
            return order.id
        except Exception as e:
            logger.error(f"Buy failed {symbol}: {e}")
            return None

    def market_sell(self, symbol: str, qty: int) -> Optional[str]:
        try:
            order = self.api.submit_order(
                symbol=symbol, qty=qty, side="sell",
                type="market", time_in_force="day",
            )
            logger.info(f"SELL {qty}x {symbol} @ MARKET -> Order {order.id}")
            return order.id
        except Exception as e:
            logger.error(f"Sell failed {symbol}: {e}")
            return None

    def close_position(self, symbol: str) -> Optional[str]:
        try:
            order = self.api.close_position(symbol)
            logger.info(f"CLOSE {symbol} -> Order {order.id}")
            return order.id
        except Exception as e:
            logger.error(f"Close failed {symbol}: {e}")
            return None

    @_retry
    def is_market_open(self) -> bool:
        return self.api.get_clock().is_open

