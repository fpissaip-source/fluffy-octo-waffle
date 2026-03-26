"""broker_binance.py — Binance Exchange Wrapper (Spot Trading, 24/7 Crypto).

Gleiche Schnittstelle wie AlpacaBroker — Engine bleibt unverändert.
Binance handelt 24/7: get_market_status() gibt immer 'open' zurück.
Positionen werden lokal gecacht (Binance Spot hat keine native Position-API).
"""

import json
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import Config

logger = logging.getLogger("bot.broker")

_POSITIONS_FILE = Path(__file__).parent / "data" / "positions_binance.json"


def _load_positions() -> dict:
    try:
        if _POSITIONS_FILE.exists():
            return json.loads(_POSITIONS_FILE.read_text())
    except Exception:
        pass
    return {}


def _save_positions(positions: dict):
    _POSITIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _POSITIONS_FILE.write_text(json.dumps(positions, indent=2, default=str))


class BinanceBroker:
    def __init__(self):
        from binance.client import Client
        testnet = getattr(Config, "BINANCE_TESTNET", True)
        self.client = Client(
            api_key=Config.BINANCE_API_KEY,
            api_secret=Config.BINANCE_SECRET_KEY,
            testnet=testnet,
        )
        # Local position tracking (Binance Spot hat keine native Positions-API)
        self._positions: dict = _load_positions()
        # Exchange filters per symbol (step size, tick size, min qty)
        self._exchange_info: dict = {}
        self._validate_connection()
        self._load_exchange_info()

    def _validate_connection(self):
        try:
            equity = self.get_equity()
            mode = "TESTNET" if getattr(Config, "BINANCE_TESTNET", True) else "!! LIVE !!"
            logger.info(f"Binance Connected [{mode}]  Equity: ~${equity:,.2f} USDT")
        except Exception as e:
            logger.error(f"Binance connection failed: {e}")
            raise

    def _load_exchange_info(self):
        """Cached LOT_SIZE und PRICE_FILTER für alle USDT-Paare."""
        try:
            exchange_info = self.client.get_exchange_info()
            for sym_info in exchange_info["symbols"]:
                if sym_info["symbol"].endswith("USDT") and sym_info["status"] == "TRADING":
                    filters = {f["filterType"]: f for f in sym_info["filters"]}
                    lot = filters.get("LOT_SIZE", {})
                    price_f = filters.get("PRICE_FILTER", {})
                    notional = filters.get("MIN_NOTIONAL", filters.get("NOTIONAL", {}))
                    self._exchange_info[sym_info["symbol"]] = {
                        "step_size": float(lot.get("stepSize", "0.001")),
                        "tick_size": float(price_f.get("tickSize", "0.01")),
                        "min_qty": float(lot.get("minQty", "0.001")),
                        "min_notional": float(notional.get("minNotional", "10")),
                    }
            logger.info(f"[BROKER] Exchange info geladen: {len(self._exchange_info)} USDT-Paare")
        except Exception as e:
            logger.warning(f"Exchange info load fehlgeschlagen: {e}")

    def _round_qty(self, symbol: str, qty: float) -> float:
        """Rundet qty auf Binance Step Size."""
        info = self._exchange_info.get(symbol, {})
        step = info.get("step_size", 0.001)
        if step <= 0:
            return round(qty, 8)
        # Floor to step size
        rounded = math.floor(qty / step) * step
        precision = max(0, int(round(-math.log10(step))))
        return round(rounded, precision)

    def _round_price(self, symbol: str, price: float) -> float:
        """Rundet Preis auf Binance Tick Size."""
        info = self._exchange_info.get(symbol, {})
        tick = info.get("tick_size", 0.01)
        if tick <= 0:
            return round(price, 8)
        rounded = round(price / tick) * tick
        precision = max(0, int(round(-math.log10(tick))))
        return round(rounded, precision)

    def _get_usdt_balance(self) -> float:
        try:
            balance = self.client.get_asset_balance(asset="USDT")
            if balance:
                return float(balance["free"]) + float(balance["locked"])
        except Exception as e:
            logger.warning(f"USDT balance failed: {e}")
        return 0.0

    def _get_asset_balance(self, asset: str) -> float:
        try:
            balance = self.client.get_asset_balance(asset=asset)
            if balance:
                return float(balance["free"]) + float(balance["locked"])
        except Exception:
            pass
        return 0.0

    def get_equity(self) -> float:
        """Gesamtwert in USDT (Cash + alle Position-Werte)."""
        try:
            account = self.client.get_account()
            # Batch prices für alle Coins auf einmal
            tickers = {t["symbol"]: float(t["price"])
                       for t in self.client.get_all_tickers()}

            total = 0.0
            for b in account["balances"]:
                qty = float(b["free"]) + float(b["locked"])
                if qty < 1e-8:
                    continue
                if b["asset"] == "USDT":
                    total += qty
                else:
                    sym = b["asset"] + "USDT"
                    price = tickers.get(sym, 0.0)
                    if price > 0:
                        total += qty * price
            return total
        except Exception as e:
            logger.warning(f"get_equity failed: {e}")
            return 0.0

    def get_buying_power(self) -> float:
        return self._get_usdt_balance()

    def get_cash(self) -> float:
        return self._get_usdt_balance()

    def get_positions(self) -> dict:
        """
        Gibt offene Positionen zurück (aus lokalem Cache, abgeglichen mit Binance Balance).
        Erkennt extern geschlossene Positionen (Stop-Loss / TP gefüllt).
        """
        result = {}
        to_remove = []

        for symbol, pos_data in self._positions.items():
            base_asset = symbol.replace("USDT", "")
            actual_qty = self._get_asset_balance(base_asset)
            min_qty = self._exchange_info.get(symbol, {}).get("min_qty", 0.0001)

            if actual_qty < min_qty:
                # Position wurde extern geschlossen (native Stop/TP gefüllt)
                to_remove.append(symbol)
                logger.info(f"[POSITIONS] {symbol}: Extern geschlossen (Balance={actual_qty:.8f})")
                continue

            avg_entry = pos_data.get("avg_entry", 0.0)
            current_price = self.get_latest_price(symbol)
            if not current_price:
                continue

            market_value = actual_qty * current_price
            cost_basis = actual_qty * avg_entry if avg_entry > 0 else market_value
            unrealized_pl = market_value - cost_basis
            unrealized_plpc = unrealized_pl / cost_basis if cost_basis > 0 else 0.0

            result[symbol] = {
                "qty": actual_qty,
                "avg_entry": avg_entry,
                "market_value": market_value,
                "unrealized_pl": unrealized_pl,
                "unrealized_plpc": unrealized_plpc,
                "side": "long",
            }

        if to_remove:
            for sym in to_remove:
                self._positions.pop(sym, None)
            _save_positions(self._positions)

        return result

    def has_position(self, symbol: str) -> bool:
        base_asset = symbol.replace("USDT", "")
        qty = self._get_asset_balance(base_asset)
        min_qty = self._exchange_info.get(symbol, {}).get("min_qty", 0.0001)
        return qty >= min_qty

    def has_open_order(self, symbol: str, side: str = "buy") -> bool:
        try:
            orders = self.client.get_open_orders(symbol=symbol)
            binance_side = "BUY" if side == "buy" else "SELL"
            return any(o["side"] == binance_side for o in orders)
        except Exception:
            return False

    def cancel_open_buy_orders(self, symbol: str) -> int:
        try:
            orders = self.client.get_open_orders(symbol=symbol)
            cancelled = 0
            for o in orders:
                if o["side"] == "BUY":
                    self.client.cancel_order(symbol=symbol, orderId=o["orderId"])
                    cancelled += 1
            return cancelled
        except Exception as e:
            logger.warning(f"cancel_open_buy_orders({symbol}): {e}")
            return 0

    def cancel_open_sell_orders(self, symbol: str) -> int:
        try:
            orders = self.client.get_open_orders(symbol=symbol)
            cancelled = 0
            for o in orders:
                if o["side"] == "SELL":
                    self.client.cancel_order(symbol=symbol, orderId=o["orderId"])
                    cancelled += 1
            return cancelled
        except Exception as e:
            logger.warning(f"cancel_open_sell_orders({symbol}): {e}")
            return 0

    def get_bars(self, symbol: str, timeframe: str = "15Min", limit: int = 100) -> pd.DataFrame:
        from binance.client import Client as _Client
        tf_map = {
            "1Min":  _Client.KLINE_INTERVAL_1MINUTE,
            "5Min":  _Client.KLINE_INTERVAL_5MINUTE,
            "15Min": _Client.KLINE_INTERVAL_15MINUTE,
            "1Hour": _Client.KLINE_INTERVAL_1HOUR,
            "1Day":  _Client.KLINE_INTERVAL_1DAY,
        }
        interval = tf_map.get(timeframe, _Client.KLINE_INTERVAL_15MINUTE)

        try:
            def _fetch():
                return self.client.get_klines(symbol=symbol, interval=interval, limit=limit + 5)

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_fetch)
                try:
                    klines = future.result(timeout=8)
                except FuturesTimeoutError:
                    logger.warning(f"Bars timeout für {symbol}")
                    return pd.DataFrame()

            if not klines:
                return pd.DataFrame()

            df = pd.DataFrame(klines, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades",
                "taker_buy_base", "taker_buy_quote", "ignore",
            ])
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
            df.set_index("open_time", inplace=True)
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)

            df = df[["open", "high", "low", "close", "volume"]].tail(limit).copy()
            df["returns"] = df["close"].pct_change()
            df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
            df.dropna(inplace=True)
            return df

        except Exception as e:
            logger.warning(f"Bars fetch fehlgeschlagen für {symbol}: {e}")
            return pd.DataFrame()

    def get_latest_price(self, symbol: str) -> Optional[float]:
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker["price"])
        except Exception as e:
            logger.warning(f"Price failed {symbol}: {e}")
            return None

    def get_snapshot(self, symbol: str) -> Optional[dict]:
        try:
            ticker = self.client.get_ticker(symbol=symbol)
            price = float(ticker["lastPrice"])
            bid = float(ticker["bidPrice"])
            ask = float(ticker["askPrice"])
            return {
                "price": price,
                "bid": bid,
                "ask": ask,
                "spread": ask - bid,
                "volume": float(ticker["volume"]),
                "vwap": float(ticker["weightedAvgPrice"]),
                "open": float(ticker["openPrice"]),
                "close": price,
                "pct_change": float(ticker["priceChangePercent"]) / 100.0,
            }
        except Exception as e:
            logger.warning(f"Snapshot failed {symbol}: {e}")
            return None

    def market_buy(self, symbol: str, qty: float) -> Optional[str]:
        if Config.DRY_RUN:
            logger.info(f"[DRY RUN] WÜRDE KAUFEN: {qty}x {symbol} @ MARKET")
            # Simuliere Kauf im Cache für DRY_RUN
            price = self.get_latest_price(symbol) or 0.0
            if price > 0:
                existing = self._positions.get(symbol, {"qty": 0.0, "avg_entry": 0.0})
                new_qty = existing["qty"] + qty
                new_avg = (
                    (existing["qty"] * existing["avg_entry"] + qty * price) / new_qty
                    if new_qty > 0 else price
                )
                self._positions[symbol] = {
                    "qty": new_qty, "avg_entry": new_avg,
                    "min_qty": self._exchange_info.get(symbol, {}).get("min_qty", 0.0001),
                }
                _save_positions(self._positions)
            return "dry-run-buy"

        try:
            rounded_qty = self._round_qty(symbol, qty)
            info = self._exchange_info.get(symbol, {})
            min_qty = info.get("min_qty", 0.0)
            if rounded_qty < min_qty:
                logger.warning(f"Buy skipped {symbol}: qty {rounded_qty} < min_qty {min_qty}")
                return None

            order = self.client.order_market_buy(symbol=symbol, quantity=rounded_qty)
            order_id = str(order["orderId"])

            # Durchschnittlichen Fill-Preis berechnen
            fills = order.get("fills", [])
            if fills:
                total_cost = sum(float(f["price"]) * float(f["qty"]) for f in fills)
                total_qty = sum(float(f["qty"]) for f in fills)
                fill_price = total_cost / total_qty if total_qty > 0 else 0.0
            else:
                fill_price = self.get_latest_price(symbol) or 0.0

            # Lokalen Position-Cache aktualisieren
            existing = self._positions.get(symbol, {"qty": 0.0, "avg_entry": 0.0})
            existing_qty = existing.get("qty", 0.0)
            existing_entry = existing.get("avg_entry", 0.0)
            new_qty = existing_qty + rounded_qty
            new_avg = (
                (existing_qty * existing_entry + rounded_qty * fill_price) / new_qty
                if new_qty > 0 and fill_price > 0 else existing_entry or fill_price
            )
            self._positions[symbol] = {
                "qty": new_qty,
                "avg_entry": new_avg,
                "min_qty": min_qty,
            }
            _save_positions(self._positions)

            logger.info(f"BUY {rounded_qty}x {symbol} @ ~${fill_price:.4f} → Order {order_id}")
            return order_id

        except Exception as e:
            logger.error(f"Buy failed {symbol}: {e}")
            return None

    def market_sell(self, symbol: str, qty: float) -> Optional[str]:
        if Config.DRY_RUN:
            logger.info(f"[DRY RUN] WÜRDE VERKAUFEN: {qty}x {symbol} @ MARKET")
            return "dry-run-sell"
        try:
            rounded_qty = self._round_qty(symbol, qty)
            order = self.client.order_market_sell(symbol=symbol, quantity=rounded_qty)
            order_id = str(order["orderId"])
            logger.info(f"SELL {rounded_qty}x {symbol} @ MARKET → Order {order_id}")
            return order_id
        except Exception as e:
            logger.error(f"Sell failed {symbol}: {e}")
            return None

    def limit_buy(self, symbol: str, qty: float, limit_price: float) -> Optional[str]:
        if Config.DRY_RUN:
            logger.info(f"[DRY RUN] WÜRDE KAUFEN: {qty}x {symbol} @ LIMIT ${limit_price:.6f}")
            return "dry-run-limit-buy"
        try:
            rounded_qty = self._round_qty(symbol, qty)
            rounded_price = self._round_price(symbol, limit_price)
            order = self.client.order_limit_buy(
                symbol=symbol,
                quantity=rounded_qty,
                price=str(rounded_price),
                timeInForce="GTC",
            )
            order_id = str(order["orderId"])
            logger.info(f"BUY {rounded_qty}x {symbol} @ LIMIT ${rounded_price} → Order {order_id}")
            return order_id
        except Exception as e:
            logger.error(f"Limit buy failed {symbol}: {e}")
            return None

    def close_position(self, symbol: str) -> Optional[str]:
        if Config.DRY_RUN:
            logger.info(f"[DRY RUN] WÜRDE POSITION SCHLIEßEN: {symbol}")
            self._positions.pop(symbol, None)
            _save_positions(self._positions)
            return "dry-run-close"
        try:
            base_asset = symbol.replace("USDT", "")
            qty = self._get_asset_balance(base_asset)
            if qty < self._exchange_info.get(symbol, {}).get("min_qty", 0.0001):
                logger.info(f"CLOSE {symbol}: Keine Balance vorhanden")
                self._positions.pop(symbol, None)
                _save_positions(self._positions)
                return None

            rounded_qty = self._round_qty(symbol, qty)
            order = self.client.order_market_sell(symbol=symbol, quantity=rounded_qty)
            order_id = str(order["orderId"])

            self._positions.pop(symbol, None)
            _save_positions(self._positions)

            logger.info(f"CLOSE {symbol} {rounded_qty}x @ MARKET → Order {order_id}")
            return order_id
        except Exception as e:
            logger.error(f"Close failed {symbol}: {e}")
            return None

    def place_native_stop(self, symbol: str, qty: float, stop_price: float) -> Optional[str]:
        """Platziert eine STOP_LOSS_LIMIT Order bei Binance (GTC).
        Limit 0.5% unter Stop-Preis um Fill sicherzustellen."""
        if Config.DRY_RUN:
            logger.info(f"[DRY RUN] WÜRDE STOP SETZEN: {symbol} {qty}x @ ${stop_price:.6f}")
            return "dry-run-stop"
        try:
            rounded_qty = self._round_qty(symbol, qty)
            rounded_stop = self._round_price(symbol, stop_price)
            limit_price = self._round_price(symbol, stop_price * 0.995)

            order = self.client.create_order(
                symbol=symbol,
                side="SELL",
                type="STOP_LOSS_LIMIT",
                timeInForce="GTC",
                quantity=rounded_qty,
                stopPrice=str(rounded_stop),
                price=str(limit_price),
            )
            order_id = str(order["orderId"])
            logger.info(f"NATIVE STOP {symbol} {rounded_qty}x @ ${rounded_stop} → Order {order_id}")
            return order_id
        except Exception as e:
            logger.error(f"Native stop failed {symbol}: {e}")
            return None

    def place_native_tp(self, symbol: str, qty: float, tp_price: float) -> Optional[str]:
        """Platziert eine GTC Limit-Sell-Order bei Binance (Take-Profit)."""
        if Config.DRY_RUN:
            logger.info(f"[DRY RUN] WÜRDE TP SETZEN: {symbol} {qty}x @ ${tp_price:.6f}")
            return "dry-run-tp"
        try:
            rounded_qty = self._round_qty(symbol, qty)
            rounded_price = self._round_price(symbol, tp_price)

            order = self.client.order_limit_sell(
                symbol=symbol,
                quantity=rounded_qty,
                price=str(rounded_price),
                timeInForce="GTC",
            )
            order_id = str(order["orderId"])
            logger.info(f"NATIVE TP {symbol} {rounded_qty}x @ ${rounded_price} → Order {order_id}")
            return order_id
        except Exception as e:
            logger.error(f"Native TP failed {symbol}: {e}")
            return None

    def cancel_order(self, order_id: str, symbol: str = None) -> bool:
        """Storniert eine Order. Symbol wird für Binance benötigt."""
        if not symbol:
            # Symbol aus lokalen Positionen oder Watchlist suchen
            for sym in list(Config.WATCHLIST):
                try:
                    orders = self.client.get_open_orders(symbol=sym)
                    for o in orders:
                        if str(o["orderId"]) == str(order_id):
                            symbol = sym
                            break
                    if symbol:
                        break
                except Exception:
                    pass

        if not symbol:
            logger.warning(f"cancel_order: Symbol für Order {order_id} nicht gefunden")
            return False

        try:
            self.client.cancel_order(symbol=symbol, orderId=int(order_id))
            return True
        except Exception as e:
            logger.warning(f"Cancel order {order_id} ({symbol}): {e}")
            return False

    def get_snapshots_batch(self, symbols: list) -> dict:
        """
        Batch 24h Ticker-Daten für mehrere Symbole.
        Rückgabe: {symbol: {price, open, close, volume, pct_change}}
        """
        try:
            tickers = self.client.get_ticker()  # Alle 24h Stats auf einmal
            sym_set = set(symbols)
            result = {}
            for t in tickers:
                sym = t["symbol"]
                if sym not in sym_set:
                    continue
                open_p = float(t["openPrice"])
                close_p = float(t["lastPrice"])
                vol = float(t["volume"])
                if open_p <= 0 or close_p <= 0:
                    continue
                result[sym] = {
                    "price": close_p,
                    "open": open_p,
                    "close": close_p,
                    "volume": vol,
                    "pct_change": float(t["priceChangePercent"]) / 100.0,
                }
            return result
        except Exception as e:
            logger.warning(f"Batch snapshots failed: {e}")
            return {}

    def is_market_open(self) -> bool:
        """Crypto handelt 24/7 — immer offen."""
        return True

    def get_market_status(self) -> str:
        """Crypto handelt 24/7 — immer 'open'."""
        return "open"
