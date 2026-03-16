import logging
from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("bot.risk")


class MarketRegime(Enum):
    CALM = "CALM"
    NORMAL = "NORMAL"
    VOLATILE = "VOLATILE"
    CRISIS = "CRISIS"


def detect_regime(bars: pd.DataFrame, vix_level: Optional[float] = None) -> MarketRegime:
    if len(bars) < 20:
        return MarketRegime.NORMAL

    returns = bars["returns"].dropna()
    realized_vol = returns.tail(20).std() * np.sqrt(252)
    recent_high = bars["close"].tail(20).max()
    current = bars["close"].iloc[-1]
    drawdown = (current - recent_high) / recent_high

    if vix_level is not None:
        if vix_level > 35:
            return MarketRegime.CRISIS
        elif vix_level > 25:
            return MarketRegime.VOLATILE
        elif vix_level > 15:
            return MarketRegime.NORMAL
        else:
            return MarketRegime.CALM

    if realized_vol > 0.50 or drawdown < -0.08:
        return MarketRegime.CRISIS
    elif realized_vol > 0.30 or drawdown < -0.05:
        return MarketRegime.VOLATILE
    elif realized_vol > 0.15:
        return MarketRegime.NORMAL
    else:
        return MarketRegime.CALM


def compute_atr(bars: pd.DataFrame, period: int = 14) -> float:
    if len(bars) < period + 1:
        return 0.0

    high = bars["high"]
    low = bars["low"]
    close = bars["close"].shift(1)

    tr1 = high - low
    tr2 = (high - close).abs()
    tr3 = (low - close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(period).mean().iloc[-1]
    return float(atr)


REGIME_PARAMS = {
    MarketRegime.CALM: {
        "stop_loss_atr_mult": 2.0,
        "take_profit_atr_mult": 4.0,
        "trailing_stop_atr_mult": 1.5,
        "max_position_pct": 0.12,
        "kelly_mult": 0.30,
        "max_open_positions": 5,
        "description": "Ruhiger Markt",
    },
    MarketRegime.NORMAL: {
        "stop_loss_atr_mult": 2.5,
        "take_profit_atr_mult": 5.0,
        "trailing_stop_atr_mult": 2.0,
        "max_position_pct": 0.10,
        "kelly_mult": 0.25,
        "max_open_positions": 4,
        "description": "Normaler Markt",
    },
    MarketRegime.VOLATILE: {
        "stop_loss_atr_mult": 3.0,
        "take_profit_atr_mult": 7.5,
        "trailing_stop_atr_mult": 2.5,
        "max_position_pct": 0.06,
        "kelly_mult": 0.15,
        "max_open_positions": 3,
        "description": "Volatile Phase",
    },
    MarketRegime.CRISIS: {
        "stop_loss_atr_mult": 4.0,
        "take_profit_atr_mult": 12.0,
        "trailing_stop_atr_mult": 3.5,
        "max_position_pct": 0.03,
        "kelly_mult": 0.10,
        "max_open_positions": 2,
        "description": "KRISEN-MODUS",
    },
}


class RiskManager:
    def __init__(self):
        self.regime = MarketRegime.NORMAL
        self.params = REGIME_PARAMS[self.regime]
        self.peak_equity = 0.0
        self.max_drawdown_limit = 0.15
        self.daily_loss_limit = 0.05
        self.start_of_day_equity = 0.0
        self.kill_switch_active = False
        self._last_check_date = None

    def update_regime(self, bars: pd.DataFrame, vix_level: Optional[float] = None):
        old = self.regime
        self.regime = detect_regime(bars, vix_level)
        self.params = REGIME_PARAMS[self.regime]
        if old != self.regime:
            logger.warning(f"REGIME CHANGE: {old.value} -> {self.regime.value} ({self.params['description']})")

    def compute_stops(self, entry_price: float, atr: float) -> dict:
        if atr <= 0:
            atr = entry_price * 0.01

        sl_distance = atr * self.params["stop_loss_atr_mult"]
        tp_distance = atr * self.params["take_profit_atr_mult"]
        trailing_distance = atr * self.params["trailing_stop_atr_mult"]

        return {
            "stop_loss": round(entry_price - sl_distance, 2),
            "take_profit": round(entry_price + tp_distance, 2),
            "trailing_stop": round(entry_price - trailing_distance, 2),
            "stop_loss_pct": f"{-sl_distance / entry_price:.1%}",
            "take_profit_pct": f"{+tp_distance / entry_price:.1%}",
            "risk_reward": round(tp_distance / sl_distance, 1),
            "atr": round(atr, 4),
            "regime": self.regime.value,
        }

    def update_trailing_stop(self, current_price: float, highest_since_entry: float, atr: float) -> float:
        trailing_dist = atr * self.params["trailing_stop_atr_mult"]
        return round(highest_since_entry - trailing_dist, 2)

    def max_position_size(self, equity: float, price: float) -> int:
        max_usd = equity * self.params["max_position_pct"]
        if price <= 0:
            return 0
        return max(1, int(max_usd / price))

    def adjust_kelly(self, kelly_fraction: float) -> float:
        return kelly_fraction * (self.params["kelly_mult"] / 0.25)

    def check_kill_switch(self, current_equity: float) -> bool:
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        today = datetime.now().date()
        if self._last_check_date != today:
            self.start_of_day_equity = current_equity
            self.kill_switch_active = False
            self._last_check_date = today
            logger.info(f"Daily reset: new day {today}")
        elif self.start_of_day_equity == 0:
            self.start_of_day_equity = current_equity
            self._last_check_date = today

        if self.peak_equity > 0:
            drawdown = (current_equity - self.peak_equity) / self.peak_equity
            if drawdown < -self.max_drawdown_limit:
                logger.critical(f"KILL SWITCH: Max Drawdown {drawdown:.1%}")
                self.kill_switch_active = True
                return True

        if self.start_of_day_equity > 0:
            daily_change = (current_equity - self.start_of_day_equity) / self.start_of_day_equity
            if daily_change < -self.daily_loss_limit:
                logger.critical(f"KILL SWITCH: Daily loss {daily_change:.1%}")
                self.kill_switch_active = True
                return True

        return False

    def can_open_position(self, current_positions: int) -> bool:
        if self.kill_switch_active:
            return False
        return current_positions < self.params["max_open_positions"]

    def reset_daily(self, equity: float):
        self.start_of_day_equity = equity
        self.kill_switch_active = False

    def should_exit(self, entry_price: float, current_price: float,
                    highest_price: float, atr: float,
                    bayesian_posterior: float = 0.5) -> dict:
        stops = self.compute_stops(entry_price, atr)
        trailing = self.update_trailing_stop(current_price, highest_price, atr)

        result = {
            "should_exit": False,
            "reason": "HOLD",
            "stops": stops,
            "trailing_stop": trailing,
        }

        if current_price <= stops["stop_loss"]:
            result["should_exit"] = True
            result["reason"] = f"STOP LOSS @ ${stops['stop_loss']}"
            return result

        if current_price > entry_price and current_price <= trailing:
            result["should_exit"] = True
            result["reason"] = f"TRAILING STOP @ ${trailing}"
            return result

        if current_price >= stops["take_profit"]:
            result["should_exit"] = True
            result["reason"] = f"TAKE PROFIT @ ${stops['take_profit']} ({stops['take_profit_pct']})"
            return result

        if bayesian_posterior < 0.30:
            result["should_exit"] = True
            result["reason"] = f"BAYESIAN EXIT (posterior={bayesian_posterior:.2f})"
            return result

        return result

    def status_summary(self) -> str:
        p = self.params
        return (
            f"Regime: {self.regime.value}\n"
            f"Max Position: {p['max_position_pct']:.0%}\n"
            f"Kelly Mult: {p['kelly_mult']}\n"
            f"SL: {p['stop_loss_atr_mult']}x ATR | TP: {p['take_profit_atr_mult']}x ATR\n"
            f"Max Positions: {p['max_open_positions']}\n"
            f"Kill Switch: {'ACTIVE' if self.kill_switch_active else 'OFF'}"
        )
