import asyncio
import logging
import threading
import time
from datetime import datetime
from typing import Optional

from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, ContextTypes

from config import Config
from broker import AlpacaBroker
from engine import Engine
from risk_manager import RiskManager, compute_atr

logger = logging.getLogger("bot.telegram")


class TradingTelegramBot:
    def __init__(self):
        self.token = Config.TELEGRAM_TOKEN
        self.chat_id = Config.TELEGRAM_CHAT_ID
        self.engine: Optional[Engine] = None
        self.is_running = False
        self.is_paused = False
        self.scan_thread: Optional[threading.Thread] = None
        self.app: Optional[Application] = None
        self._bot: Optional[Bot] = None
        self._lock = threading.Lock()

    async def send(self, text: str, parse_mode: str = "HTML"):
        try:
            if self._bot:
                await self._bot.send_message(chat_id=self.chat_id, text=text, parse_mode=parse_mode)
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")

    def send_sync(self, text: str):
        try:
            import requests
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            requests.post(url, json={"chat_id": self.chat_id, "text": text, "parse_mode": "HTML"}, timeout=10)
        except Exception as e:
            logger.error(f"Telegram sync send failed: {e}")

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        text = (
            "<b>TRADING BOT v2.0</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "<b>Monitoring:</b>\n"
            "/status  — Account\n"
            "/positions — Positionen\n"
            "/regime  — Markt-Regime\n\n"
            "<b>Trading:</b>\n"
            "/scan    — Einmal scannen\n"
            "/run     — Auto-Scan starten\n"
            "/pause   — Pausieren\n"
            "/resume  — Fortsetzen\n"
            "/stop    — Stoppen\n\n"
            "<b>Analyse:</b>\n"
            "/stats   — Performance\n"
            "/weights — Gewichte\n"
            "/trades  — Trades\n"
            "/backtest AAPL — Backtest\n\n"
            "<b>Watchlist:</b>\n"
            "/watchlist — Anzeigen\n"
            "/add TSLA — Hinzufuegen\n"
            "/remove TSLA — Entfernen\n"
        )
        await update.message.reply_text(text, parse_mode="HTML")

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            broker = AlpacaBroker()
            equity = broker.get_equity()
            cash = broker.get_cash()
            positions = broker.get_positions()
            market_open = broker.is_market_open()

            with self._lock:
                bot_status = "PAUSED" if self.is_paused else "RUNNING" if self.is_running else "IDLE"

            text = (
                f"<b>ACCOUNT STATUS</b>\n━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Mode: <code>{'PAPER' if Config.is_paper() else 'LIVE'}</code>\n"
                f"Market: {'Open' if market_open else 'Closed'}\n"
                f"Bot: <code>{bot_status}</code>\n━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Equity: <b>${equity:,.2f}</b>\n"
                f"Cash: ${cash:,.2f}\n"
                f"Positions: {len(positions)}\n"
            )
            await update.message.reply_text(text, parse_mode="HTML")
        except Exception as e:
            await update.message.reply_text(f"Error: {e}")

    async def cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            broker = AlpacaBroker()
            positions = broker.get_positions()
            if not positions:
                await update.message.reply_text("Keine offenen Positionen.")
                return

            text = "<b>POSITIONEN</b>\n━━━━━━━━━━━━━━━━━━━━━━\n"
            total_pl = 0.0
            for sym, pos in positions.items():
                pl = pos["unrealized_pl"]
                plpc = pos["unrealized_plpc"]
                total_pl += pl
                pre = "+" if pl >= 0 else ""
                text += f"\n<b>{sym}</b>  {pos['qty']:.0f}x @ ${pos['avg_entry']:.2f}\n  P/L: <code>{pre}${pl:.2f} ({pre}{plpc:.1%})</code>\n"

            pre = "+" if total_pl >= 0 else ""
            text += f"\n━━━━━━━━━━━━━━━━━━━━━━\nTotal: <b>{pre}${total_pl:.2f}</b>"
            await update.message.reply_text(text, parse_mode="HTML")
        except Exception as e:
            await update.message.reply_text(f"Error: {e}")

    async def cmd_scan(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("Scanning...")
        try:
            engine = Engine()
            for symbol in Config.WATCHLIST:
                signal = engine.analyze_symbol(symbol)
                lines = [f"<b>{signal.symbol}</b> | Score: {signal.weighted_score:.2f}"]
                for name, r in signal.results.items():
                    s = "+" if r["passed"] else "-"
                    lines.append(f"  {s} {name}: {r['signal']:+.4f}")
                lines.append(f"  > {signal.action}")
                await update.message.reply_text("\n".join(lines), parse_mode="HTML")
            await update.message.reply_text("Scan complete.")
        except Exception as e:
            await update.message.reply_text(f"Error: {e}")

    async def cmd_trades(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            from database import get_session, TradeRecord
            session = get_session()
            records = session.query(TradeRecord).order_by(TradeRecord.id.desc()).limit(10).all()
            session.close()

            if not records:
                await update.message.reply_text("Keine Trades.")
                return

            text = "<b>LETZTE TRADES</b>\n━━━━━━━━━━━━━━━━━━━━━━\n"
            for r in records:
                pre = "+" if (r.pnl or 0) >= 0 else ""
                pnl_str = f"{pre}${r.pnl:.2f}" if r.pnl is not None else "open"
                text += f"\n<b>{r.action} {r.qty}x {r.symbol}</b> @ ${r.entry_price:.2f}\n  P/L: {pnl_str}  {r.exit_reason or ''}\n"
            await update.message.reply_text(text, parse_mode="HTML")
        except Exception as e:
            await update.message.reply_text(f"Error: {e}")

    async def cmd_watchlist(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        text = f"<b>WATCHLIST</b> ({len(Config.WATCHLIST)})\n<code>{', '.join(Config.WATCHLIST)}</code>"
        await update.message.reply_text(text, parse_mode="HTML")

    async def cmd_add(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not context.args:
            await update.message.reply_text("Usage: /add TSLA")
            return
        symbol = context.args[0].upper()
        if symbol not in Config.WATCHLIST:
            Config.WATCHLIST.append(symbol)
        await update.message.reply_text(f"{symbol} hinzugefuegt. Watchlist: {', '.join(Config.WATCHLIST)}")

    async def cmd_remove(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not context.args:
            await update.message.reply_text("Usage: /remove TSLA")
            return
        symbol = context.args[0].upper()
        if symbol in Config.WATCHLIST:
            Config.WATCHLIST.remove(symbol)
        await update.message.reply_text(f"{symbol} entfernt. Watchlist: {', '.join(Config.WATCHLIST)}")

    async def cmd_pause(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        with self._lock:
            self.is_paused = True
        await update.message.reply_text("Bot pausiert. /resume zum Fortsetzen.")

    async def cmd_resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        with self._lock:
            self.is_paused = False
        await update.message.reply_text("Bot laeuft weiter.")

    async def cmd_run(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        with self._lock:
            if self.is_running:
                await update.message.reply_text("Bot laeuft bereits.")
                return
            self.is_running = True
            self.is_paused = False

        self.engine = Engine()
        self.scan_thread = threading.Thread(target=self._scan_loop, daemon=True)
        self.scan_thread.start()
        await update.message.reply_text(
            f"<b>Auto-Scan gestartet</b>\nIntervall: {Config.SCAN_INTERVAL}s\nWatchlist: {', '.join(Config.WATCHLIST)}",
            parse_mode="HTML",
        )

    async def cmd_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        with self._lock:
            self.is_running = False
            self.is_paused = False
        await update.message.reply_text("Auto-Scan gestoppt.")

    async def cmd_regime(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            broker = AlpacaBroker()
            bars = broker.get_bars("SPY", timeframe="5Min", limit=50)
            risk = RiskManager()
            risk.update_regime(bars)
            p = risk.params
            atr = compute_atr(bars)

            text = (
                f"<b>REGIME: {risk.regime.value}</b>\n━━━━━━━━━━━━━━━━━━━━━━\n"
                f"{p['description']}\n\n"
                f"SL: {p['stop_loss_atr_mult']}x ATR | TP: {p['take_profit_atr_mult']}x ATR\n"
                f"Max Position: {p['max_position_pct']:.0%}\n"
                f"Max Positionen: {p['max_open_positions']}\n"
                f"ATR: ${atr:.2f}\n"
            )
            await update.message.reply_text(text, parse_mode="HTML")
        except Exception as e:
            await update.message.reply_text(f"Error: {e}")

    async def cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            from adaptive import AdaptiveLearner
            learner = AdaptiveLearner()
            stats = learner.get_trade_history_stats()
            if stats.get("total_trades", 0) == 0:
                await update.message.reply_text("Noch keine Trades.")
                return
            text = (
                f"<b>PERFORMANCE</b>\n━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Trades: {stats['total_trades']}\n"
                f"Wins: {stats['wins']} ({stats['win_rate']:.0%})\n"
                f"Total P/L: <b>${stats['total_pnl']:+,.2f}</b>\n"
                f"Sharpe: {stats['sharpe']}\n"
            )
            await update.message.reply_text(text, parse_mode="HTML")
        except Exception as e:
            await update.message.reply_text(f"Error: {e}")

    async def cmd_weights(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            from adaptive import AdaptiveLearner
            learner = AdaptiveLearner()
            summary = learner.get_weights_summary()
            await update.message.reply_text(f"<b>GEWICHTE</b>\n<pre>{summary}</pre>", parse_mode="HTML")
        except Exception as e:
            await update.message.reply_text(f"Error: {e}")

    async def cmd_backtest(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        symbol = context.args[0].upper() if context.args else "AAPL"
        await update.message.reply_text(f"Backtest fuer {symbol}...")
        try:
            broker = AlpacaBroker()
            bars = broker.get_bars(symbol, timeframe="1Day", limit=200)
            if bars.empty:
                await update.message.reply_text(f"Keine Daten fuer {symbol}")
                return
            from backtester import BacktestEngine
            bt = BacktestEngine()
            results = bt.run(bars, symbol)
            text = (
                f"<b>BACKTEST {symbol}</b>\n━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Trades: {results['total_trades']}\n"
                f"Win Rate: {results.get('win_rate', 0):.0%}\n"
                f"Total P/L: ${results.get('total_pnl', 0):+,.2f}\n"
                f"Return: {results.get('total_return', 'N/A')}\n"
                f"Sharpe: {results.get('sharpe_ratio', 0)}\n"
                f"Max DD: {results.get('max_drawdown', 'N/A')}\n"
            )
            await update.message.reply_text(text, parse_mode="HTML")
        except Exception as e:
            await update.message.reply_text(f"Error: {e}")

    def _scan_loop(self):
        logger.info("Auto-scan thread started")
        self.send_sync("Auto-Scan gestartet.")

        while True:
            with self._lock:
                should_stop = not self.is_running
                is_paused = self.is_paused
            if should_stop:
                break
            if is_paused:
                time.sleep(5)
                continue

            try:
                broker = self.engine.broker
                if not broker.is_market_open():
                    time.sleep(60)
                    continue

                self.engine.check_exit_conditions()

                for symbol in Config.WATCHLIST:
                    with self._lock:
                        if not self.is_running:
                            break
                    try:
                        signal = self.engine.analyze_symbol(symbol)
                        if signal.all_passed:
                            self.send_sync(
                                f"<b>SIGNAL: BUY {signal.qty}x {signal.symbol}</b>\n"
                                f"Score: {signal.weighted_score:.2f}"
                            )
                            order_id = self.engine.execute_signal(signal)
                            if order_id:
                                self.send_sync(f"ORDER: BUY {signal.qty}x {signal.symbol} -> {order_id}")
                    except Exception as e:
                        logger.error(f"Scan error {symbol}: {e}")

            except Exception as e:
                logger.error(f"Scan loop error: {e}")

            time.sleep(Config.SCAN_INTERVAL)

        self.send_sync("Auto-Scan gestoppt.")

    def run(self):
        logger.info("Starting Telegram bot...")
        self.app = Application.builder().token(self.token).build()
        self._bot = self.app.bot

        handlers = [
            ("start", self.cmd_start), ("status", self.cmd_status),
            ("scan", self.cmd_scan), ("positions", self.cmd_positions),
            ("trades", self.cmd_trades), ("watchlist", self.cmd_watchlist),
            ("add", self.cmd_add), ("remove", self.cmd_remove),
            ("pause", self.cmd_pause), ("resume", self.cmd_resume),
            ("run", self.cmd_run), ("stop", self.cmd_stop),
            ("regime", self.cmd_regime), ("stats", self.cmd_stats),
            ("weights", self.cmd_weights), ("backtest", self.cmd_backtest),
        ]
        for cmd, handler in handlers:
            self.app.add_handler(CommandHandler(cmd, handler))

        logger.info("Telegram bot running.")
        self.app.run_polling(drop_pending_updates=True)

