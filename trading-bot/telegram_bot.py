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
    /screener     - Top-Mover Screener (penny/micro/sub)
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
            "<b>Screener:</b>\n"
            "/screener       — Top-Mover (alle)\n"
            "/screener penny — Penny Stocks &lt;$5\n"
            "/screener micro — unter $1\n"
            "/screener sub   — Sub-Penny &lt;$0.01\n\n"
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
            market_status = broker.get_market_status()

            status_icon = {"open": "🟢", "extended": "🟡", "closed": "🔴"}.get(market_status, "🔴")
            status_label = {"open": "Open", "extended": "Extended Hours", "closed": "Closed"}.get(market_status, "Closed")
            mode = "PAPER" if Config.is_paper() else "LIVE"
            bot_status = "PAUSED" if self.is_paused else "RUNNING" if self.is_running else "IDLE"

            text = (
                f"<b>ACCOUNT STATUS</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Mode:     <code>{mode}</code>\n"
                f"Market:   {status_icon} {status_label}\n"
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

    async def cmd_screener(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Screener: Top-Mover nach Kategorie (penny/micro/sub)."""
        await update.message.reply_text("🔍 Screener läuft...")

        mode = context.args[0].lower() if context.args else "all"

        mode_config = {
            "penny": {"label": "Penny Stocks (<$5)", "max_price": 5.0},
            "micro": {"label": "Micro Stocks (<$1)", "max_price": 1.0},
            "sub":   {"label": "Sub-Penny (<$0.01)", "max_price": 0.01},
            "all":   {"label": "Top-Mover (alle)", "max_price": None},
        }
        cfg = mode_config.get(mode, mode_config["all"])

        price_filter = (
            f"Nur Aktien mit Kurs unter ${cfg['max_price']} USD."
            if cfg["max_price"]
            else "Alle Preisbereiche erlaubt."
        )

        prompt = f"""Du bist ein Day-Trader-Screener. {price_filter}
Welche 10 US-Aktien haben HEUTE das höchste Momentum und relatives Volumen (mind. 3x Durchschnitt)?
Fokus auf: starke News-Katalysatoren, Short Squeeze, FDA, Earnings heute.
Antworte NUR mit JSON:
{{"results": [{{"symbol": "SYM", "price_est": 1.23, "catalyst": "kurzer Grund", "volume_mult": 4.5}}]}}"""

        try:
            import json
            from google import genai
            from google.genai import types as genai_types

            client = genai.Client(api_key=Config.GEMINI_API_KEY)
            response = client.models.generate_content(
                model=Config.REASONING_MODEL,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.3,
                    max_output_tokens=500,
                ),
            )
            text = response.text or ""
            result = None
            depth, start = 0, None
            for i, c in enumerate(text):
                if c == '{':
                    if depth == 0:
                        start = i
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0 and start is not None:
                        try:
                            result = json.loads(text[start:i + 1])
                        except json.JSONDecodeError:
                            pass
                        break

            if not result or not result.get("results"):
                await update.message.reply_text("Keine Ergebnisse vom Screener.")
                return

            lines = [f"📊 <b>SCREENER — {cfg['label']}</b>", "━━━━━━━━━━━━━━━━━━━━━━"]
            for r in result["results"][:10]:
                sym = r.get("symbol", "?").upper()
                price = r.get("price_est", 0)
                catalyst = r.get("catalyst", "—")[:40]
                vol = r.get("volume_mult", 0)
                lines.append(
                    f"\n<b>{sym}</b>  ~${price:.2f}  |  {vol:.1f}x Vol\n"
                    f"  📌 {catalyst}"
                )
            lines.append(f"\n<i>Quelle: Gemini Echtzeit-Analyse</i>")
            lines.append(f"/screener penny  /screener micro  /screener sub")

            await update.message.reply_text("\n".join(lines), parse_mode="HTML")

        except Exception as e:
            await update.message.reply_text(f"Screener Error: {e}")

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
        self.send_sync("🤖 Auto-Scan gestartet. SpikeSensor aktiv (300+ Symbole).")

        while self.is_running:
            if self.is_paused:
                time.sleep(5)
                continue

            try:
                broker = self.engine.broker
                market_status = broker.get_market_status()
                status_labels = {"open": "REGULÄR", "extended": "VOR-/NACHBÖRSE", "closed": "NUR CRYPTO"}
                logger.info(f"Scan [{status_labels.get(market_status, market_status)}]")

                # Spike-Alert BEFORE scan so Telegram gets notified immediately
                if market_status in ("open", "extended"):
                    spike_symbols = self.engine.spike_sensor.scan()
                    if spike_symbols:
                        self.send_sync(
                            f"⚡ <b>SPIKE DETECTED</b> — {len(spike_symbols)} Symbole\n"
                            + "\n".join(f"  • <b>{s}</b>" for s in spike_symbols[:10])
                            + "\n<i>Vollanalyse läuft...</i>"
                        )
                        # Spike-Symbole sofort in die Engine-Watchlist einbauen
                        for s in spike_symbols:
                            if s not in self.engine.watchlist.dynamic_symbols:
                                self.engine.watchlist.dynamic_symbols.append(s)

                # Haupt-Scan (Watchlist + Spike-Symbole + Execute)
                # scan_once übernimmt: exit_checks, watchlist, spike-merge, execute
                spike_results = self.engine.scan_once(market_status)

                # Trade-Alerts aus Engine trade_log (neue seit letztem Scan)
                # (Engine sendet keine Telegram-Messages direkt, das macht der Loop hier)
                for signal in self.engine.trade_log[-5:]:
                    if signal.all_passed and signal.qty > 0:
                        # Nur neue (letzten 5 Sekunden) alertieren
                        age = (datetime.now() - signal.timestamp).total_seconds()
                        if age < Config.SCAN_INTERVAL + 10:
                            self.send_sync(
                                f"🚨 <b>TRADE SIGNAL</b>\n\n"
                                f"<b>BUY {signal.qty}x {signal.symbol}</b>\n"
                                f"Kaskade: {signal.cascade_label}\n\n"
                                f"{self._format_signal_text(signal)}"
                            )

                # Stop-Loss Nähe Warnung
                positions = broker.get_positions()
                for sym, pos in positions.items():
                    plpc = pos["unrealized_plpc"]
                    if plpc < -0.025:
                        self.send_sync(f"⚠️ <b>{sym}</b> bei {plpc:+.1%} — nahe Stop Loss (-3%)")

            except Exception as e:
                logger.error(f"Scan loop error: {e}")
                self.send_sync(f"⚠️ Scan error: {e}")

            # Intervall je nach Marktphase
            ms = self.engine.broker.get_market_status()
            interval = Config.SCAN_INTERVAL if ms == "open" else (60 if ms == "extended" else 120)
            time.sleep(interval)

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
        self.app.add_handler(CommandHandler("screener", self.cmd_screener))

        logger.info("Telegram bot running. Send /start to begin.")
        self.app.run_polling(drop_pending_updates=True)
