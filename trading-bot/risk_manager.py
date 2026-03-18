"""
risk_manager.py — Adaptives Risikomanagement

Ersetzt die fixen Stop-Loss (-3%) / Take-Profit (+5%) durch
dynamische Levels basierend auf:

1. Markt-Regime-Erkennung (Normal / Volatil / Crash)
2. ATR-basierte Stops (Average True Range)
3. Trailing Stop Loss
4. Max-Drawdown Kill-Switch
5. Krieg/Krise-Modus (aktuell: Iran-Konflikt)

Aus dem PDF "Der perfekte AI Trading Bot":
→ "Dynamische Positionsgroessen basierend auf Volatilitaet,
   automatisierte Stops und Black-Swan-Schutz"
"""

import logging
import time
from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from config import Config

logger = logging.getLogger("bot.risk")


# ═══════════════════════════════════════════════════════
#  MARKT-REGIME-ERKENNUNG
# ═══════════════════════════════════════════════════════

class MarketRegime(Enum):
    CALM = "CALM"           # VIX < 15, normale Volatilitaet
    NORMAL = "NORMAL"       # VIX 15-25
    VOLATILE = "VOLATILE"   # VIX 25-35, erhoehte Vorsicht
    CRISIS = "CRISIS"       # VIX > 35, Krieg/Crash/Black-Swan


def detect_regime(bars: pd.DataFrame, vix_level: Optional[float] = None) -> MarketRegime:
    """
    Erkennt das aktuelle Markt-Regime.

    Nutzt:
    - Realisierte Volatilitaet (aus Bars)
    - VIX-Level (wenn verfuegbar)
    - Drawdown vom lokalen Hoch
    """
    if len(bars) < 20:
        return MarketRegime.NORMAL

    returns = bars["returns"].dropna()

    # ── Realisierte Volatilitaet (annualisiert) ──
    realized_vol = returns.tail(20).std() * np.sqrt(252)

    # ── Drawdown vom 20-Bar-Hoch ──
    recent_high = bars["close"].tail(20).max()
    current = bars["close"].iloc[-1]
    drawdown = (current - recent_high) / recent_high

    # ── VIX-basierte Einschaetzung ──
    if vix_level is not None:
        if vix_level > 35:
            return MarketRegime.CRISIS
        elif vix_level > 25:
            return MarketRegime.VOLATILE
        elif vix_level > 15:
            return MarketRegime.NORMAL
        else:
            return MarketRegime.CALM

    # ── Fallback: Realisierte Volatilitaet ──
    if realized_vol > 0.50 or drawdown < -0.08:
        return MarketRegime.CRISIS
    elif realized_vol > 0.30 or drawdown < -0.05:
        return MarketRegime.VOLATILE
    elif realized_vol > 0.15:
        return MarketRegime.NORMAL
    else:
        return MarketRegime.CALM


# ═══════════════════════════════════════════════════════
#  ATR (AVERAGE TRUE RANGE)
# ═══════════════════════════════════════════════════════

def compute_atr(bars: pd.DataFrame, period: int = 14) -> float:
    """Berechnet ATR — misst die durchschnittliche Schwankungsbreite."""
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


# ═══════════════════════════════════════════════════════
#  REGIME-ADAPTIVE PARAMETER
# ═══════════════════════════════════════════════════════

# Jedes Regime hat andere Risk-Parameter
REGIME_PARAMS = {
    MarketRegime.CALM: {
        "stop_loss_atr_mult": 2.0,       # Enger Stop
        "take_profit_atr_mult": 4.0,     # 2:1 R/R
        "trailing_stop_atr_mult": 1.5,
        "max_position_pct": 0.12,        # Etwas groessere Positionen ok
        "kelly_mult": 0.30,              # 30% Kelly
        "max_open_positions": 5,
        "description": "Ruhiger Markt — normale Parameter",
    },
    MarketRegime.NORMAL: {
        "stop_loss_atr_mult": 2.5,
        "take_profit_atr_mult": 5.0,     # 2:1 R/R
        "trailing_stop_atr_mult": 2.0,
        "max_position_pct": 0.10,
        "kelly_mult": 0.25,              # Quarter Kelly
        "max_open_positions": 4,
        "description": "Normaler Markt — Standard-Parameter",
    },
    MarketRegime.VOLATILE: {
        "stop_loss_atr_mult": 3.0,       # Weiterer Stop (mehr Noise)
        "take_profit_atr_mult": 7.5,     # 2.5:1 R/R — groessere Moves moeglich
        "trailing_stop_atr_mult": 2.5,
        "max_position_pct": 0.06,        # Kleinere Positionen
        "kelly_mult": 0.15,              # 15% Kelly
        "max_open_positions": 3,
        "description": "Volatile Phase — kleinere Positionen, weitere Stops",
    },
    MarketRegime.CRISIS: {
        "stop_loss_atr_mult": 4.0,       # Sehr weit (extreme Swings)
        "take_profit_atr_mult": 12.0,    # GROSSE Moves — nicht zu frueh raus!
        "trailing_stop_atr_mult": 3.5,
        "max_position_pct": 0.03,        # Winzige Positionen
        "kelly_mult": 0.10,              # 10% Kelly
        "max_open_positions": 2,
        "description": "KRISEN-MODUS — minimale Positionen, maximale R/R",
    },
}


# ═══════════════════════════════════════════════════════
#  RISK MANAGER
# ═══════════════════════════════════════════════════════

class RiskManager:
    """Zentrales Risikomanagement mit adaptiven Parametern."""

    def __init__(self):
        self.regime = MarketRegime.NORMAL
        self.params = REGIME_PARAMS[self.regime]
        self.peak_equity = 0.0
        self.max_drawdown_limit = 0.15   # 15% Max-Drawdown → Kill Switch
        self.daily_loss_limit = 0.05     # 5% Tagesverlust → Stop
        self.start_of_day_equity = 0.0
        self.kill_switch_active = False
        self._regime_last_update: float = 0
        self._regime_cache_ttl: int = 300  # Regime max alle 5 Minuten updaten

    # ── Regime Update ──────────────────────────────────

    def update_regime(self, bars: pd.DataFrame, vix_level: Optional[float] = None, force: bool = False):
        """Aktualisiert das Markt-Regime — max alle 5 Minuten (global, nicht per Symbol)."""
        now = time.time()
        if not force and (now - self._regime_last_update) < self._regime_cache_ttl:
            return  # Noch frisch — kein Update
        self._regime_last_update = now

        old = self.regime
        self.regime = detect_regime(bars, vix_level)
        self.params = REGIME_PARAMS[self.regime]

        if old != self.regime:
            logger.warning(
                f"REGIME CHANGE: {old.value} → {self.regime.value} "
                f"({self.params['description']})"
            )

    # ── Stop-Loss / Take-Profit berechnen ──────────────

    def compute_stops(self, entry_price: float, atr: float) -> dict:
        """
        Berechnet ATR-basierte Stop-Loss und Take-Profit Levels.

        In einer Krise (Iran-Krieg etc.) sind die Moves GROESSER,
        also brauchen wir:
        - Weitere Stops (damit man nicht durch Noise ausgestoppt wird)
        - Hoehere Take-Profits (um die grossen Moves mitzunehmen)
        """
        if atr <= 0:
            atr = entry_price * 0.01  # Fallback: 1%

        sl_distance = atr * self.params["stop_loss_atr_mult"]
        tp_distance = atr * self.params["take_profit_atr_mult"]
        trailing_distance = atr * self.params["trailing_stop_atr_mult"]

        stop_loss = entry_price - sl_distance
        take_profit = entry_price + tp_distance
        trailing_stop = entry_price - trailing_distance

        return {
            "stop_loss": round(stop_loss, 2),
            "take_profit": round(take_profit, 2),
            "trailing_stop": round(trailing_stop, 2),
            "stop_loss_pct": f"{-sl_distance/entry_price:.1%}",
            "take_profit_pct": f"{+tp_distance/entry_price:.1%}",
            "risk_reward": round(tp_distance / sl_distance, 1),
            "atr": round(atr, 4),
            "regime": self.regime.value,
        }

    def update_trailing_stop(
        self, current_price: float, highest_since_entry: float, atr: float
    ) -> float:
        """Trailing Stop: Zieht mit wenn der Preis steigt."""
        trailing_dist = atr * self.params["trailing_stop_atr_mult"]
        return round(highest_since_entry - trailing_dist, 2)

    # ── Position Sizing ────────────────────────────────

    def max_position_size(self, equity: float, price: float) -> int:
        """Maximale Positionsgroesse basierend auf Regime."""
        max_usd = equity * self.params["max_position_pct"]
        if price <= 0:
            return 0
        return max(1, int(max_usd / price))

    def adjust_kelly(self, kelly_fraction: float) -> float:
        """Passt Kelly an das Regime an."""
        return kelly_fraction * (self.params["kelly_mult"] / 0.25)

    # ── Kill-Switch ────────────────────────────────────

    def check_kill_switch(self, current_equity: float) -> bool:
        """
        Black-Swan-Schutz:
        - Stoppt ALLE Trades wenn Drawdown > 15%
        - Stoppt ALLE Trades wenn Tagesverlust > 5%

        Aus dem PDF: "automatisierte Stops und Black-Swan-Schutz"
        """
        # Peak Equity tracken
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        # Tagesstart setzen
        if self.start_of_day_equity == 0:
            self.start_of_day_equity = current_equity

        # ── Max Drawdown Check ──
        if self.peak_equity > 0:
            drawdown = (current_equity - self.peak_equity) / self.peak_equity
            if drawdown < -self.max_drawdown_limit:
                logger.critical(
                    f"KILL SWITCH: Max Drawdown {drawdown:.1%} "
                    f"(Limit: {-self.max_drawdown_limit:.0%})"
                )
                self.kill_switch_active = True
                return True

        # ── Daily Loss Check ──
        if self.start_of_day_equity > 0:
            daily_change = (current_equity - self.start_of_day_equity) / self.start_of_day_equity
            if daily_change < -self.daily_loss_limit:
                logger.critical(
                    f"KILL SWITCH: Daily loss {daily_change:.1%} "
                    f"(Limit: {-self.daily_loss_limit:.0%})"
                )
                self.kill_switch_active = True
                return True

        return False

    def can_open_position(self, current_positions: int) -> bool:
        """Prueft ob wir noch neue Positionen oeffnen duerfen."""
        if self.kill_switch_active:
            return False
        return current_positions < self.params["max_open_positions"]

    def reset_daily(self, equity: float):
        """Taeglicher Reset (am Marktstart aufrufen)."""
        self.start_of_day_equity = equity
        self.kill_switch_active = False

    # ── Exit-Entscheidung ──────────────────────────────

    def should_exit(
        self,
        entry_price: float,
        current_price: float,
        highest_price: float,
        atr: float,
        bayesian_posterior: float = 0.5,
    ) -> dict:
        """
        Entscheidet ob eine Position geschlossen werden soll.

        Checks:
        1. Fixer Stop-Loss (ATR-basiert)
        2. Trailing Stop
        3. Take-Profit (ATR-basiert, ADAPTIV je Regime!)
        4. Bayesian Exit (Sentiment dreht)
        """
        stops = self.compute_stops(entry_price, atr)
        trailing = self.update_trailing_stop(current_price, highest_price, atr)

        result = {
            "should_exit": False,
            "reason": "HOLD",
            "stops": stops,
            "trailing_stop": trailing,
        }

        # 1. Stop-Loss
        if current_price <= stops["stop_loss"]:
            result["should_exit"] = True
            result["reason"] = f"STOP LOSS @ ${stops['stop_loss']}"
            return result

        # 2. Trailing Stop (nur wenn im Gewinn)
        if current_price > entry_price and current_price <= trailing:
            result["should_exit"] = True
            result["reason"] = f"TRAILING STOP @ ${trailing}"
            return result

        # 3. Take-Profit
        if current_price >= stops["take_profit"]:
            result["should_exit"] = True
            result["reason"] = f"TAKE PROFIT @ ${stops['take_profit']} ({stops['take_profit_pct']})"
            return result

        # 4. Bayesian Exit (Signal dreht stark bearish)
        if bayesian_posterior < 0.30:
            result["should_exit"] = True
            result["reason"] = f"BAYESIAN EXIT (posterior={bayesian_posterior:.2f})"
            return result

        return result

    # ── Summary ────────────────────────────────────────

    def status_summary(self) -> str:
        p = self.params
        return (
            f"Regime: {self.regime.value}\n"
            f"Max Position: {p['max_position_pct']:.0%}\n"
            f"Kelly Mult: {p['kelly_mult']}\n"
            f"SL: {p['stop_loss_atr_mult']}x ATR | "
            f"TP: {p['take_profit_atr_mult']}x ATR\n"
            f"Max Positions: {p['max_open_positions']}\n"
            f"Kill Switch: {'ACTIVE' if self.kill_switch_active else 'OFF'}"
        )
