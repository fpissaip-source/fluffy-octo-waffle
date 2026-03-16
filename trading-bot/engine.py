"""
engine.py — Orchestriert alle 7 Formeln.
Mindestens Config.MIN_FILTERS_REQUIRED (Standard: 5/7) muessen bestehen -> dann wird getradet.
"""

import logging
import time
from datetime import datetime
from typing import Optional

from broker import AlpacaBroker
from config import Config
from risk_manager import RiskManager, compute_atr
from adaptive import AdaptiveLearner
from formulas import momentum, kelly, ev_gap, kl_divergence, bayesian, stoikov
from formulas import sentiment as sentiment_formula

logger = logging.getLogger("bot.engine")


class TradeSignal:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.timestamp = datetime.now()
        self.results: dict = {}
        self.all_passed = False
        self.action: Optional[str] = None
        self.qty: int = 0
        self.reason: str = ""

    def add_result(self, result: dict):
        self.results[result["name"]] = result

    def evaluate(self):
        if len(self.results) < 7:
            self.all_passed = False
            self.action = "HOLD"
            self.reason = f"Only {len(self.results)}/7 formulas ran"
            return

        passed_list = [n for n, r in self.results.items() if r["passed"]]
        failed_list = [n for n, r in self.results.items() if not r["passed"]]
        passed_count = len(passed_list)

        self.all_passed = passed_count >= Config.MIN_FILTERS_REQUIRED

        if self.all_passed:
            self.action = "BUY"
            self.reason = f"{passed_count}/7 filters passed"
            if failed_list:
                self.reason += f" (skipped: {', '.join(failed_list)})"
        else:
            self.action = "HOLD"
            self.reason = f"Only {passed_count}/{Config.MIN_FILTERS_REQUIRED} required filters passed. Failed: {', '.join(failed_list)}"

    def summary(self) -> str:
        lines = [
            f"\n{'=' * 60}",
            f"  {self.symbol}  |  {self.timestamp.strftime('%H:%M:%S')}",
            f"{'=' * 60}",
        ]
        for name, r in self.results.items():
            status = "PASS" if r["passed"] else "FAIL"
            lines.append(f"  {name:<16} {status:<8} signal={r['signal']}")
        lines.append(f"{'-' * 60}")
        if self.all_passed:
            lines.append(f"  > ACTION: {self.action}  |  Qty: {self.qty}")
        else:
            lines.append(f"  > ACTION: {self.action}")
        lines.append(f"  > REASON: {self.reason}")
        lines.append(f"{'=' * 60}\n")
        return "\n".join(lines)


class Engine:
    def __init__(self):
        logger.info("Initializing engine...")
        self.broker = AlpacaBroker()
        self.risk = RiskManager()
        self.learner = AdaptiveLearner()
        self.trade_log: list[TradeSignal] = []
        self.position_highs: dict[str, float] = {}  # Track highest price per position

    def analyze_symbol(self, symbol: str) -> TradeSignal:
        signal = TradeSignal(symbol)

        bars = self.broker.get_bars(symbol, timeframe="5Min", limit=Config.LOOKBACK_BARS)
        if bars.empty or len(bars) < 50:
            logger.warning(f"{symbol}: Not enough data ({len(bars) if not bars.empty else 0} bars)")
            signal.reason = "Insufficient data"
            return signal

        snapshot = self.broker.get_snapshot(symbol)
        equity = self.broker.get_equity()
        positions = self.broker.get_positions()
        inventory_skew = 0.3 if symbol in positions else 0.0

        # ── F1: Momentum ──
        try:
            r1 = momentum.evaluate(bars, threshold=Config.MIN_MOMENTUM_SCORE)
            signal.add_result(r1)
        except Exception as e:
            signal.add_result({"name": "Momentum", "signal": 0, "passed": False, "details": {"error": str(e)}})

        # ── F2: Kelly ──
        try:
            r2 = kelly.evaluate(bars, equity=equity)
            signal.add_result(r2)
        except Exception as e:
            signal.add_result({"name": "Kelly", "signal": 0, "passed": False, "details": {"error": str(e)}})

        # ── F3: EV-Gap ──
        try:
            r3 = ev_gap.evaluate(bars, win_prob=0.55)
            signal.add_result(r3)
        except Exception as e:
            signal.add_result({"name": "EV-Gap", "signal": 0, "passed": False, "details": {"error": str(e)}})

        # ── F4: KL-Divergence ──
        try:
            r4 = kl_divergence.evaluate(bars, threshold=Config.KL_DIVERGENCE_THRESHOLD)
            signal.add_result(r4)
        except Exception as e:
            signal.add_result({"name": "KL-Divergence", "signal": 0, "passed": False, "details": {"error": str(e)}})

        # ── F5: Bayesian ──
        try:
            r5 = bayesian.evaluate(bars, prior=0.50, threshold=Config.MIN_BAYESIAN_POSTERIOR)
            signal.add_result(r5)
        except Exception as e:
            signal.add_result({"name": "Bayesian", "signal": 0.5, "passed": False, "details": {"error": str(e)}})

        # ── F6: Stoikov ──
        try:
            r6 = stoikov.evaluate(bars, snapshot=snapshot, inventory_skew=inventory_skew,
                                  time_remaining=0.5)
            signal.add_result(r6)
        except Exception as e:
            signal.add_result({"name": "Stoikov", "signal": 0, "passed": False, "details": {"error": str(e)}})

        # ── F7: Sentiment ──
        try:
            r7 = sentiment_formula.evaluate(bars, broker=self.broker, symbol=symbol)
            signal.add_result(r7)
        except Exception as e:
            signal.add_result({"name": "Sentiment", "signal": 0, "passed": True, "details": {"error": str(e)}})

        # ── Regime Update ──
        self.risk.update_regime(bars)

        # ── Decision ──
        signal.evaluate()

        # ── Adaptive Override Check ──
        # Only attempt override when close to the threshold (at least MIN-1 passed)
        if not signal.all_passed:
            passed_count = sum(1 for r in signal.results.values() if r.get("passed"))
            if passed_count >= Config.MIN_FILTERS_REQUIRED - 1:
                override = self.learner.should_override_entry(
                    self.risk.regime.value, signal.results
                )
                if override["override"]:
                    signal.all_passed = True
                    signal.action = "BUY"
                    signal.reason = override["reason"]
                    logger.info(f"{symbol}: ADAPTIVE OVERRIDE — {override['reason']}")

        # ── Weighted Score (fuer Logging) ──
        w_score = self.learner.weighted_score(self.risk.regime.value, signal.results)
        signal.reason += f" | score={w_score:.2f}"

        if signal.all_passed and "Kelly" in signal.results:
            bet_size = signal.results["Kelly"]["details"].get("bet_size_usd", 0)
            price = bars["close"].iloc[-1]
            if price > 0:
                signal.qty = max(1, int(bet_size / price))

        return signal

    def execute_signal(self, signal: TradeSignal) -> Optional[str]:
        if not signal.all_passed or signal.action != "BUY":
            return None
        if signal.qty <= 0:
            logger.warning(f"{signal.symbol}: All passed but qty=0")
            return None
        if self.broker.has_position(signal.symbol):
            logger.info(f"{signal.symbol}: Already have position, skipping")
            return None

        # Risk Manager checks
        positions = self.broker.get_positions()
        if not self.risk.can_open_position(len(positions)):
            logger.warning(
                f"{signal.symbol}: Max positions reached ({len(positions)}) "
                f"for regime {self.risk.regime.value}"
            )
            return None

        # Cap qty by risk manager
        equity = self.broker.get_equity()
        price = self.broker.get_latest_price(signal.symbol) or 1
        max_qty = self.risk.max_position_size(equity, price)
        signal.qty = min(signal.qty, max_qty)

        logger.info(f"{'=' * 40}")
        logger.info(f"EXECUTING: BUY {signal.qty}x {signal.symbol}")
        logger.info(f"Regime: {self.risk.regime.value} | {self.risk.params['description']}")
        logger.info(f"{'=' * 40}")

        order_id = self.broker.market_buy(signal.symbol, signal.qty)
        if order_id:
            signal.reason += f" -> Order {order_id}"
            self.trade_log.append(signal)
            self.position_highs[signal.symbol] = price

            # ── Adaptive Learning: Record Entry ──
            formula_scores = {
                name: r.get("signal", 0) for name, r in signal.results.items()
            }
            sentiment_score = signal.results.get("Sentiment", {}).get("signal", 0)
            self.learner.record_entry(
                symbol=signal.symbol,
                regime=self.risk.regime.value,
                formula_scores=formula_scores,
                sentiment_score=sentiment_score,
                entry_price=price,
                qty=signal.qty,
            )

        return order_id

    def check_exit_conditions(self):
        """Adaptive Exit-Checks basierend auf Markt-Regime."""
        positions = self.broker.get_positions()
        equity = self.broker.get_equity()

        # Kill-Switch Check
        if self.risk.check_kill_switch(equity):
            logger.critical("KILL SWITCH ACTIVE — closing ALL positions")
            for symbol in positions:
                self.broker.close_position(symbol)
            return

        for symbol, pos in positions.items():
            try:
                bars = self.broker.get_bars(symbol, timeframe="5Min", limit=50)
                if bars.empty:
                    continue

                # Regime updaten
                self.risk.update_regime(bars)

                # ATR berechnen
                atr = compute_atr(bars)
                entry_price = pos["avg_entry"]
                current_price = bars["close"].iloc[-1]

                # Highest price tracken (fuer Trailing Stop)
                if symbol not in self.position_highs:
                    self.position_highs[symbol] = entry_price
                self.position_highs[symbol] = max(
                    self.position_highs[symbol], current_price
                )

                # Bayesian posterior holen
                bay = bayesian.evaluate(bars, prior=0.50)
                posterior = bay.get("signal", 0.5)

                # Adaptive Exit-Entscheidung
                exit_decision = self.risk.should_exit(
                    entry_price=entry_price,
                    current_price=current_price,
                    highest_price=self.position_highs[symbol],
                    atr=atr,
                    bayesian_posterior=posterior,
                )

                if exit_decision["should_exit"]:
                    plpc = pos["unrealized_plpc"]
                    logger.info(
                        f"EXIT {symbol}: {exit_decision['reason']} "
                        f"| P/L: {plpc:+.1%} | Regime: {self.risk.regime.value}"
                    )
                    self.broker.close_position(symbol)
                    self.position_highs.pop(symbol, None)

                    # ── Adaptive Learning: Record Exit ──
                    self.learner.record_exit(
                        symbol, current_price, exit_decision["reason"]
                    )

            except Exception as e:
                logger.error(f"Exit check error {symbol}: {e}")

    def scan_once(self):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"  SCAN @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"  Watchlist: {', '.join(Config.WATCHLIST)}")
        logger.info(f"  Regime: {self.risk.regime.value}")
        logger.info(f"{'=' * 60}")

        self.check_exit_conditions()

        for symbol in Config.WATCHLIST:
            try:
                signal = self.analyze_symbol(symbol)
                print(signal.summary())
                if signal.all_passed:
                    self.execute_signal(signal)
            except Exception as e:
                logger.error(f"Error {symbol}: {e}")

        equity = self.broker.get_equity()
        positions = self.broker.get_positions()
        logger.info(f"Equity: ${equity:,.2f}  |  Positions: {len(positions)}  |  Trades: {len(self.trade_log)}")

    def run(self):
        logger.info("=" * 60)
        logger.info("  SIX FILTERS. ONE TRADE.")
        logger.info(f"  Mode: {'PAPER' if Config.is_paper() else '!! LIVE !!'}")
        logger.info(f"  Interval: {Config.SCAN_INTERVAL}s  |  Watchlist: {Config.WATCHLIST}")
        logger.info("=" * 60)

        while True:
            if not self.broker.is_market_open():
                logger.info("Market closed. Waiting 60s...")
                time.sleep(60)
                continue
            try:
                self.scan_once()
            except KeyboardInterrupt:
                logger.info("\nShutting down...")
                break
            except Exception as e:
                logger.error(f"Scan error: {e}")

            logger.info(f"Next scan in {Config.SCAN_INTERVAL}s...")
            time.sleep(Config.SCAN_INTERVAL)
