"""
telegram_bot.py — Steuere den Trading-Bot vom iPhone via Telegram.

Befehle:
    /start        - Bot starten + Uebersicht
    /status       - Account Status (Equity, Positionen)
    /scan         - Einmal alle Symbole scannen
    /positions    - Offene Positionen anzeigen
    /trades       - Heutige Trades anzeigen
    /watchlist    - Aktuelle Watchlist
    /add TSLA     - Symbol zur Watchlist hinzufuegen
    /remove TSLA  - Symbol von Watchlist entfernen
    /pause        - Bot pausieren
    /resume       - Bot fortsetzen
    /stop         - Bot komplett stoppen
"""

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
from engine import Engine, TradeSignal
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

    # ── Nachrichten senden ──────────────────────────────

    async def send(self, text: str, parse_mode: str = "HTML"):
        """Nachricht an dein iPhone senden."""
        try:
            if self._bot:
                await self._bot.send_message(
                    chat_id=self.chat_id,
                    text=text,
                    parse_mode=parse_mode,
                )
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")

    def send_sync(self, text: str):
        """Synchrone Version fuer Threads."""
        try:
            import requests
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            requests.post(url, json={
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": "HTML",
            }, timeout=10)
        except Exception as e:
            logger.error(f"Telegram sync send failed: {e}")

    # ── Commands ────────────────────────────────────────

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        text = (
            "<b>SIX FILTERS. ONE TRADE.</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "Trading Engine v2.0 bereit.\n\n"
            "<b>Monitoring:</b>\n"
            "/status  — Account anzeigen\n"
            "/positions — Offene Positionen\n"
            "/regime  — Markt-Regime + Risk\n"
            "/sentiment AAPL — News-Analyse\n\n"
            "<b>Trading:</b>\n"
            "/scan    — Einmal scannen\n"
            "/run     — Auto-Scan starten\n"
            "/pause   — Bot pausieren\n"
            "/resume  — Bot fortsetzen\n"
            "/stop    — Auto-Scan stoppen\n\n"
            "<b>Analyse:</b>\n"
            "/stats   — Performance-Stats\n"
            "/weights — Gelernte Gewichte\n"
            "/trades  — Heutige Trades\n\n"
            "<b>Watchlist:</b>\n"
            "/watchlist — Symbole anzeigen\n"
            "/add TSLA — Hinzufuegen\n"
            "/remove TSLA — Entfernen\n"
        )
        await update.message.reply_text(text, parse_mode="HTML")

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            broker = AlpacaBroker()
            equity = broker.get_equity()
            cash = broker.get_cash()
            bp = broker.get_buying_power()
            positions = broker.get_positions()
            market_open = broker.is_market_open()

            status_icon = "🟢" if market_open else "🔴"
            mode = "PAPER" if Config.is_paper() else "LIVE"
            bot_status = "PAUSED" if self.is_paused else "RUNNING" if self.is_running else "IDLE"

            text = (
                f"<b>ACCOUNT STATUS</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Mode:     <code>{mode}</code>\n"
                f"Market:   {status_icon} {'Open' if market_open else 'Closed'}\n"
                f"Bot:      <code>{bot_status}</code>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Equity:       <b>${equity:,.2f}</b>\n"
                f"Cash:         ${cash:,.2f}\n"
                f"Buying Power: ${bp:,.2f}\n"
                f"Positions:    {len(positions)}\n"
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
                icon = "📈" if pl >= 0 else "📉"
                pre = "+" if pl >= 0 else ""

                text += (
                    f"\n{icon} <b>{sym}</b>\n"
                    f"   {pos['qty']:.0f} shares @ ${pos['avg_entry']:.2f}\n"
                    f"   P/L: <code>{pre}${pl:.2f} ({pre}{plpc:.1%})</code>\n"
                )

            pre = "+" if total_pl >= 0 else ""
            text += f"\n━━━━━━━━━━━━━━━━━━━━━━\nTotal P/L: <b>{pre}${total_pl:.2f}</b>"
            await update.message.reply_text(text, parse_mode="HTML")
        except Exception as e:
            await update.message.reply_text(f"Error: {e}")

    async def cmd_scan(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("Scanning...")

        try:
            engine = Engine()

            for symbol in Config.WATCHLIST:
                signal = engine.analyze_symbol(symbol)
                text = self._format_signal(signal)
                await update.message.reply_text(text, parse_mode="HTML")

                if signal.all_passed:
                    await update.message.reply_text(
                        f"🚨 <b>SIGNAL: BUY {signal.qty}x {signal.symbol}</b>\n"
                        f"Soll ich ausfuehren? (Bot fuehrt nur im Auto-Modus aus)",
                        parse_mode="HTML",
                    )

            await update.message.reply_text("Scan complete.")
        except Exception as e:
            await update.message.reply_text(f"Scan error: {e}")

    async def cmd_trades(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.engine or not self.engine.trade_log:
            await update.message.reply_text("Keine Trades heute.")
            return

        text = "<b>HEUTIGE TRADES</b>\n━━━━━━━━━━━━━━━━━━━━━━\n"
        for t in self.engine.trade_log[-10:]:
            text += (
                f"\n{t.timestamp.strftime('%H:%M')} "
                f"<b>{t.action} {t.qty}x {t.symbol}</b>\n"
                f"   {t.reason}\n"
            )
        await update.message.reply_text(text, parse_mode="HTML")

    async def cmd_watchlist(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        symbols = ", ".join(Config.WATCHLIST)
        text = (
            f"<b>WATCHLIST</b> ({len(Config.WATCHLIST)} Symbole)\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<code>{symbols}</code>\n\n"
            f"/add SYMBOL — Hinzufuegen\n"
            f"/remove SYMBOL — Entfernen"
        )
        await update.message.reply_text(text, parse_mode="HTML")

    async def cmd_add(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not context.args:
            await update.message.reply_text("Usage: /add TSLA")
            return
        symbol = context.args[0].upper()
        if symbol in Config.WATCHLIST:
            await update.message.reply_text(f"{symbol} ist schon in der Watchlist.")
            return
        Config.WATCHLIST.append(symbol)
        await update.message.reply_text(f"✅ {symbol} hinzugefuegt. Watchlist: {', '.join(Config.WATCHLIST)}")

    async def cmd_remove(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not context.args:
            await update.message.reply_text("Usage: /remove TSLA")
            return
        symbol = context.args[0].upper()
        if symbol not in Config.WATCHLIST:
            await update.message.reply_text(f"{symbol} nicht in Watchlist.")
            return
        Config.WATCHLIST.remove(symbol)
        await update.message.reply_text(f"🗑 {symbol} entfernt. Watchlist: {', '.join(Config.WATCHLIST)}")

    async def cmd_pause(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self.is_paused = True
        await update.message.reply_text("⏸ Bot pausiert. /resume zum Fortsetzen.")

    async def cmd_resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self.is_paused = False
        await update.message.reply_text("▶️ Bot laeuft weiter.")

    async def cmd_run(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if self.is_running:
            await update.message.reply_text("Bot laeuft bereits.")
            return

        self.is_running = True
        self.is_paused = False
        self.engine = Engine()
        self.scan_thread = threading.Thread(target=self._scan_loop, daemon=True)
        self.scan_thread.start()

        await update.message.reply_text(
            f"🟢 <b>Auto-Scan gestartet</b>\n"
            f"Intervall: {Config.SCAN_INTERVAL}s\n"
            f"Watchlist: {', '.join(Config.WATCHLIST)}\n\n"
            f"/pause — Pausieren\n"
            f"/stop — Stoppen",
            parse_mode="HTML",
        )

    async def cmd_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self.is_running = False
        self.is_paused = False
        await update.message.reply_text("🔴 Auto-Scan gestoppt.")

    async def cmd_regime(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Zeigt aktuelles Markt-Regime und Risk-Parameter."""
        try:
            broker = AlpacaBroker()
            # Regime auf Basis von SPY (S&P 500 ETF) erkennen
            bars = broker.get_bars("SPY", timeframe="5Min", limit=50)
            risk = RiskManager()
            risk.update_regime(bars)
            p = risk.params

            regime_icons = {
                "CALM": "😎", "NORMAL": "📊",
                "VOLATILE": "⚡", "CRISIS": "🚨",
            }
            icon = regime_icons.get(risk.regime.value, "📊")

            atr = compute_atr(bars)
            spy_price = bars["close"].iloc[-1] if not bars.empty else 0

            text = (
                f"{icon} <b>MARKT-REGIME: {risk.regime.value}</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"{p['description']}\n\n"
                f"<b>Risk-Parameter:</b>\n"
                f"  Stop-Loss:    {p['stop_loss_atr_mult']}x ATR\n"
                f"  Take-Profit:  {p['take_profit_atr_mult']}x ATR\n"
                f"  Trailing:     {p['trailing_stop_atr_mult']}x ATR\n"
                f"  Max Position: {p['max_position_pct']:.0%}\n"
                f"  Kelly Mult:   {p['kelly_mult']}\n"
                f"  Max Positionen: {p['max_open_positions']}\n\n"
                f"<b>SPY:</b> ${spy_price:.2f} | ATR: ${atr:.2f}\n"
                f"Kill-Switch: {'🔴 ACTIVE' if risk.kill_switch_active else '🟢 OFF'}"
            )
            await update.message.reply_text(text, parse_mode="HTML")
        except Exception as e:
            await update.message.reply_text(f"Error: {e}")

    async def cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Performance-Statistiken vom adaptiven Lernsystem."""
        try:
            from adaptive import AdaptiveLearner
            learner = AdaptiveLearner()
            stats = learner.get_stats()

            if stats.get("total_trades", 0) == 0:
                await update.message.reply_text("Noch keine abgeschlossenen Trades.")
                return

            text = (
                f"📊 <b>PERFORMANCE STATS</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Trades:    <b>{stats['total_trades']}</b>\n"
                f"Wins:      {stats['wins']} ({stats['win_rate']:.0%})\n"
                f"Losses:    {stats['losses']}\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Avg Win:   <code>{stats['avg_win']:+.2%}</code>\n"
                f"Avg Loss:  <code>{stats['avg_loss']:+.2%}</code>\n"
                f"Best:      <code>{stats['best_trade']:+.2%}</code>\n"
                f"Worst:     <code>{stats['worst_trade']:+.2%}</code>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Total P/L: <b>${stats['total_pnl']:+,.2f}</b>\n"
                f"Sharpe:    {stats['sharpe']}\n"
            )

            # Per-Regime
            if stats.get("per_regime"):
                text += "\n<b>Per Regime:</b>\n"
                for regime, rs in stats["per_regime"].items():
                    text += (
                        f"  {regime}: {rs['trades']} trades, "
                        f"WR {rs['win_rate']:.0%}, "
                        f"avg {rs['avg_pnl']:+.2%}\n"
                    )

            await update.message.reply_text(text, parse_mode="HTML")
        except Exception as e:
            await update.message.reply_text(f"Error: {e}")

    async def cmd_weights(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Zeigt gelernte Formel-Gewichte pro Regime."""
        try:
            from adaptive import AdaptiveLearner
            learner = AdaptiveLearner()
            summary = learner.get_weights_summary()

            text = f"🧠 <b>GELERNTE GEWICHTE</b>\n<pre>{summary}</pre>"
            await update.message.reply_text(text, parse_mode="HTML")
        except Exception as e:
            await update.message.reply_text(f"Error: {e}")

    async def cmd_sentiment(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Sentiment-Analyse fuer ein Symbol oder die ganze Watchlist."""
        try:
            from formulas.sentiment import SentimentEngine
            broker = AlpacaBroker()
            se = SentimentEngine(broker)

            symbols = [context.args[0].upper()] if context.args else Config.WATCHLIST[:5]

            for symbol in symbols:
                result = se.analyze_symbol(symbol)
                score = result["score"]

                if score > 0.3:
                    icon = "🟢"
                elif score > 0:
                    icon = "🔵"
                elif score > -0.3:
                    icon = "🟡"
                else:
                    icon = "🔴"

                text = (
                    f"{icon} <b>{symbol}</b> Sentiment: <code>{score:+.3f}</code>\n"
                    f"  Confidence: {result['confidence']:.0%}\n"
                    f"  News: {result['article_count']} Artikel\n"
                    f"  Symbol: {result['symbol_sentiment']['score']:+.3f} | "
                    f"Makro: {result['macro_sentiment']['score']:+.3f}\n"
                )

                # Top Signals
                details = result["symbol_sentiment"].get("details", [])
                if details:
                    text += "  Signale:\n"
                    for d in details[:3]:
                        text += f"    {d['score']:+.2f} {d['headline'][:50]}...\n"

                await update.message.reply_text(text, parse_mode="HTML")

        except Exception as e:
            await update.message.reply_text(f"Error: {e}")

    # ── Auto-Scan Loop (runs in thread) ─────────────────

    def _scan_loop(self):
        logger.info("Auto-scan thread started")
        self.send_sync("🤖 Auto-Scan Thread gestartet.")

        while self.is_running:
            if self.is_paused:
                time.sleep(5)
                continue

            try:
                broker = self.engine.broker
                if not broker.is_market_open():
                    time.sleep(60)
                    continue

                # Exit checks
                self.engine.check_exit_conditions()

                # Scan
                for symbol in Config.WATCHLIST:
                    if not self.is_running:
                        break
                    try:
                        signal = self.engine.analyze_symbol(symbol)

                        if signal.all_passed:
                            # Alert senden
                            self.send_sync(
                                f"🚨 <b>TRADE SIGNAL</b>\n\n"
                                f"<b>BUY {signal.qty}x {signal.symbol}</b>\n"
                                f"Filter bestanden!\n\n"
                                f"{self._format_signal_text(signal)}"
                            )
                            # Trade ausfuehren
                            order_id = self.engine.execute_signal(signal)
                            if order_id:
                                self.send_sync(
                                    f"✅ <b>ORDER EXECUTED</b>\n"
                                    f"BUY {signal.qty}x {signal.symbol}\n"
                                    f"Order ID: <code>{order_id}</code>"
                                )

                    except Exception as e:
                        logger.error(f"Scan error {symbol}: {e}")

                # Exit-Alerts
                positions = broker.get_positions()
                for sym, pos in positions.items():
                    plpc = pos["unrealized_plpc"]
                    if plpc < -0.025:
                        self.send_sync(f"⚠️ <b>{sym}</b> bei {plpc:+.1%} — nahe Stop Loss (-3%)")

            except Exception as e:
                logger.error(f"Scan loop error: {e}")
                self.send_sync(f"⚠️ Scan error: {e}")

            time.sleep(Config.SCAN_INTERVAL)

        self.send_sync("🔴 Auto-Scan gestoppt.")
        logger.info("Auto-scan thread ended")

    # ── Helpers ──────────────────────────────────────────

    def _format_signal(self, signal: TradeSignal) -> str:
        icon = "🟢" if signal.all_passed else "⚪"
        lines = [
            f"{icon} <b>{signal.symbol}</b>  {signal.timestamp.strftime('%H:%M:%S')}",
            "━━━━━━━━━━━━━━━━━━━━━━",
        ]
        for name, r in signal.results.items():
            s = "✅" if r["passed"] else "❌"
            lines.append(f"{s} {name:<14} {r['signal']:+.4f}")

        lines.append("━━━━━━━━━━━━━━━━━━━━━━")
        lines.append(f"Action: <b>{signal.action}</b>")
        if signal.all_passed:
            lines.append(f"Qty: <b>{signal.qty}</b>")
        return "\n".join(lines)

    def _format_signal_text(self, signal: TradeSignal) -> str:
        lines = []
        for name, r in signal.results.items():
            s = "✅" if r["passed"] else "❌"
            lines.append(f"{s} {name}: {r['signal']:+.4f}")
        return "\n".join(lines)

    # ── Start ───────────────────────────────────────────

    def run(self):
        """Startet den Telegram Bot."""
        logger.info("Starting Telegram bot...")

        self.app = Application.builder().token(self.token).build()
        self._bot = self.app.bot

        # Commands registrieren
        self.app.add_handler(CommandHandler("start", self.cmd_start))
        self.app.add_handler(CommandHandler("status", self.cmd_status))
        self.app.add_handler(CommandHandler("scan", self.cmd_scan))
        self.app.add_handler(CommandHandler("positions", self.cmd_positions))
        self.app.add_handler(CommandHandler("trades", self.cmd_trades))
        self.app.add_handler(CommandHandler("watchlist", self.cmd_watchlist))
        self.app.add_handler(CommandHandler("add", self.cmd_add))
        self.app.add_handler(CommandHandler("remove", self.cmd_remove))
        self.app.add_handler(CommandHandler("pause", self.cmd_pause))
        self.app.add_handler(CommandHandler("resume", self.cmd_resume))
        self.app.add_handler(CommandHandler("run", self.cmd_run))
        self.app.add_handler(CommandHandler("stop", self.cmd_stop))
        self.app.add_handler(CommandHandler("regime", self.cmd_regime))
        self.app.add_handler(CommandHandler("stats", self.cmd_stats))
        self.app.add_handler(CommandHandler("weights", self.cmd_weights))
        self.app.add_handler(CommandHandler("sentiment", self.cmd_sentiment))

        logger.info("Telegram bot running. Send /start to begin.")
        self.app.run_polling(drop_pending_updates=True)
