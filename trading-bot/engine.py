import logging
import time
from datetime import datetime
from typing import Optional

from broker import AlpacaBroker
from config import Config
from risk_manager import RiskManager, compute_atr
from adaptive import AdaptiveLearner
from formulas import momentum, kelly, ev_gap, kl_divergence, bayesian, zscore, regime
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
        self.weighted_score: float = 0.0

    def add_result(self, result: dict):
        self.results[result["name"]] = result

    def evaluate_weighted(self, required_filters: list[str], w_score: float, threshold: float):
        required_passed = all(
            self.results.get(f, {}).get("passed", False) for f in required_filters
            if f in self.results
        )

        self.weighted_score = w_score

        if required_passed and w_score >= threshold:
            self.all_passed = True
            self.action = "BUY"
            passed_count = sum(1 for r in self.results.values() if r.get("passed"))
            self.reason = f"Weighted score {w_score:.2f} >= {threshold} ({passed_count}/{len(self.results)} filters passed)"
        else:
            self.all_passed = False
            self.action = "HOLD"
            if not required_passed:
                failed_req = [f for f in required_filters if f in self.results and not self.results[f].get("passed")]
                self.reason = f"Required filters failed: {', '.join(failed_req)}"
            else:
                self.reason = f"Weighted score {w_score:.2f} < {threshold}"

    def summary(self) -> str:
        lines = [
            f"\n{'=' * 60}",
            f"  {self.symbol}  |  {self.timestamp.strftime('%H:%M:%S')}  |  Score: {self.weighted_score:.2f}",
            f"{'=' * 60}",
        ]
        for name, r in self.results.items():
            req = "*" if name in Config.REQUIRED_FILTERS else " "
            status = "PASS" if r["passed"] else "FAIL"
            lines.append(f" {req}{name:<16} {status:<8} signal={r['signal']}")
        lines.append(f"{'-' * 60}")
        lines.append(f"  > ACTION: {self.action}  |  Score: {self.weighted_score:.2f}")
        if self.all_passed:
            lines.append(f"  > Qty: {self.qty}")
        lines.append(f"  > {self.reason}")
        lines.append(f"{'=' * 60}\n")
        return "\n".join(lines)


class Engine:
    def __init__(self):
        logger.info("Initializing engine...")
        self.broker = AlpacaBroker()
        self.risk = RiskManager()
        self.learner = AdaptiveLearner()
        self.trade_log: list[TradeSignal] = []
        self.position_highs: dict[str, float] = {}

    def analyze_symbol(self, symbol: str) -> TradeSignal:
        signal = TradeSignal(symbol)

        bars = self.broker.get_bars(symbol, timeframe="5Min", limit=Config.LOOKBACK_BARS)
        if bars.empty or len(bars) < 50:
            logger.warning(f"{symbol}: Not enough data ({len(bars) if not bars.empty else 0} bars)")
            signal.reason = "Insufficient data"
            return signal

        snapshot = self.broker.get_snapshot(symbol)
        equity = self.broker.get_equity()
        trade_stats = self.learner.get_trade_history_stats()

        for name, func, extra_kwargs in [
            ("Momentum", momentum.evaluate, {"threshold": Config.MIN_MOMENTUM_SCORE}),
            ("Kelly", kelly.evaluate, {"equity": equity, "trade_history_stats": trade_stats}),
            ("EV-Gap", ev_gap.evaluate, {"win_prob": 0.55}),
            ("KL-Divergence", kl_divergence.evaluate, {"threshold": Config.KL_DIVERGENCE_THRESHOLD}),
            ("Bayesian", bayesian.evaluate, {"prior": 0.50, "threshold": Config.MIN_BAYESIAN_POSTERIOR}),
            ("Z-Score", zscore.evaluate, {"threshold": Config.ZSCORE_ENTRY_THRESHOLD}),
        ]:
            try:
                result = func(bars, **extra_kwargs)
                signal.add_result(result)
            except Exception as e:
                signal.add_result({"name": name, "signal": 0, "passed": False, "details": {"error": str(e)}})

        try:
            r7 = sentiment_formula.evaluate(bars, broker=self.broker, symbol=symbol, threshold=-0.3)
            signal.add_result(r7)
        except Exception as e:
            signal.add_result({"name": "Sentiment", "signal": 0, "passed": True, "details": {"error": str(e)}})

        try:
            r8 = regime.evaluate(bars)
            signal.add_result(r8)
        except Exception as e:
            signal.add_result({"name": "Regime", "signal": 0, "passed": False, "details": {"error": str(e)}})

        self.risk.update_regime(bars)

        w_score = self.learner.weighted_score(self.risk.regime.value, signal.results)
        signal.evaluate_weighted(Config.REQUIRED_FILTERS, w_score, Config.WEIGHTED_SCORE_THRESHOLD)

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
            return None
        if self.broker.has_position(signal.symbol):
            logger.info(f"{signal.symbol}: Already have position, skipping")
            return None

        positions = self.broker.get_positions()
        if not self.risk.can_open_position(len(positions)):
            logger.warning(f"{signal.symbol}: Max positions reached for regime {self.risk.regime.value}")
            return None

        equity = self.broker.get_equity()
        price = self.broker.get_latest_price(signal.symbol)
        if price is None:
            logger.error(f"{signal.symbol}: Could not get price, skipping trade")
            return None

        max_qty = self.risk.max_position_size(equity, price)
        signal.qty = min(signal.qty, max_qty)

        logger.info(f"EXECUTING: BUY {signal.qty}x {signal.symbol} | Regime: {self.risk.regime.value} | Score: {signal.weighted_score:.2f}")

        order_id = self.broker.market_buy(signal.symbol, signal.qty)
        if order_id:
            signal.reason += f" -> Order {order_id}"
            self.trade_log.append(signal)
            self.position_highs[signal.symbol] = price

            formula_scores = {name: r.get("signal", 0) for name, r in signal.results.items()}
            sentiment_score = signal.results.get("Sentiment", {}).get("signal", 0)
            self.learner.record_entry(
                symbol=signal.symbol,
                regime=self.risk.regime.value,
                formula_scores=formula_scores,
                sentiment_score=sentiment_score,
                entry_price=price,
                qty=signal.qty,
                weighted_score=signal.weighted_score,
            )

        return order_id

    def check_exit_conditions(self):
        positions = self.broker.get_positions()
        equity = self.broker.get_equity()

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

                self.risk.update_regime(bars)
                atr = compute_atr(bars)
                entry_price = pos["avg_entry"]
                current_price = bars["close"].iloc[-1]

                if symbol not in self.position_highs:
                    self.position_highs[symbol] = entry_price
                self.position_highs[symbol] = max(self.position_highs[symbol], current_price)

                bay = bayesian.evaluate(bars, prior=0.50)
                posterior = bay.get("signal", 0.5)

                exit_decision = self.risk.should_exit(
                    entry_price=entry_price,
                    current_price=current_price,
                    highest_price=self.position_highs[symbol],
                    atr=atr,
                    bayesian_posterior=posterior,
                )

                if exit_decision["should_exit"]:
                    plpc = pos["unrealized_plpc"]
                    logger.info(f"EXIT {symbol}: {exit_decision['reason']} | P/L: {plpc:+.1%}")
                    self.broker.close_position(symbol)
                    self.position_highs.pop(symbol, None)
                    self.learner.record_exit(symbol, current_price, exit_decision["reason"])

            except Exception as e:
                logger.error(f"Exit check error {symbol}: {e}")

    def scan_once(self):
        logger.info(f"SCAN @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Regime: {self.risk.regime.value}")
        self.check_exit_conditions()

        for symbol in Config.WATCHLIST:
            try:
                signal = self.analyze_symbol(symbol)
                self._last_signal = signal
                print(signal.summary())
                if signal.all_passed:
                    self.execute_signal(signal)
            except Exception as e:
                logger.error(f"Error {symbol}: {e}")

    def run(self):
        logger.info(f"SIX FILTERS. ONE TRADE. | Mode: {'PAPER' if Config.is_paper() else 'LIVE'}")
        while True:
            if not self.broker.is_market_open():
                logger.info("Market closed. Waiting 60s...")
                time.sleep(60)
                continue
            try:
                self.scan_once()
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break
            except Exception as e:
                logger.error(f"Scan error: {e}")
            time.sleep(Config.SCAN_INTERVAL)
