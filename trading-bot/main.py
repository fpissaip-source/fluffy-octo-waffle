#!/usr/bin/env python3
import sys
import os
import logging
import argparse
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config


def setup_logging():
    level = getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO)
    fmt = "%(asctime)s | %(name)-12s | %(levelname)-7s | %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S")
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("websocket").setLevel(logging.WARNING)


def init_database():
    from database import init_db
    init_db()


def start_api_server():
    import uvicorn
    from api import app
    port = Config.BOT_API_PORT
    logging.getLogger("bot.main").info(f"Starting FastAPI on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


def cmd_status():
    from broker import AlpacaBroker
    broker = AlpacaBroker()
    equity = broker.get_equity()
    cash = broker.get_cash()
    positions = broker.get_positions()

    print(f"\n{'=' * 50}")
    print(f"  ACCOUNT STATUS  ({'PAPER' if Config.is_paper() else 'LIVE'})")
    print(f"{'=' * 50}")
    print(f"  Equity:       ${equity:,.2f}")
    print(f"  Cash:         ${cash:,.2f}")
    print(f"  Positions:    {len(positions)}")
    if positions:
        for sym, pos in positions.items():
            pl = pos['unrealized_pl']
            pre = "+" if pl >= 0 else ""
            print(f"  {sym:<8} {pos['qty']:>6} @ ${pos['avg_entry']:.2f}  {pre}${pl:.2f}")
    print(f"{'=' * 50}\n")


def cmd_backtest():
    from broker import AlpacaBroker
    from backtester import BacktestEngine

    broker = AlpacaBroker()
    bt = BacktestEngine()

    print(f"\n{'=' * 50}")
    print(f"  BACKTEST")
    print(f"{'=' * 50}\n")

    for symbol in Config.WATCHLIST:
        try:
            bars = broker.get_bars(symbol, timeframe="1Day", limit=200)
            if bars.empty:
                print(f"  {symbol}: No data")
                continue
            results = bt.run(bars, symbol)
            print(f"  {symbol}: {results['total_trades']} trades | "
                  f"WR {results.get('win_rate', 0):.0%} | "
                  f"P/L ${results.get('total_pnl', 0):+,.2f} | "
                  f"Sharpe {results.get('sharpe_ratio', 0)}")
        except Exception as e:
            print(f"  {symbol}: Error - {e}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Trading Bot v2.0")
    parser.add_argument("--status", action="store_true", help="Account status")
    parser.add_argument("--scan-once", action="store_true", help="Single scan")
    parser.add_argument("--backtest", action="store_true", help="Run backtest")
    parser.add_argument("--telegram", action="store_true", help="Telegram mode")
    parser.add_argument("--api-only", action="store_true", help="API server only")
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger("bot.main")

    if not Config.validate():
        print("\n  API Keys nicht konfiguriert!")
        print("  Setze ALPACA_API_KEY und ALPACA_SECRET_KEY als Umgebungsvariablen.\n")
        sys.exit(1)

    if Config.DATABASE_URL:
        init_database()

    print()
    print("  +==========================================+")
    print("  |   TRADING BOT v2.0                      |")
    print("  |   Weighted Scoring + Adaptive Learning   |")
    print("  +==========================================+")
    print()

    if args.api_only:
        start_api_server()
    elif args.telegram:
        if not Config.TELEGRAM_TOKEN:
            print("  TELEGRAM_TOKEN nicht gesetzt!\n")
            sys.exit(1)
        api_thread = threading.Thread(target=start_api_server, daemon=True)
        api_thread.start()
        from telegram_bot import TradingTelegramBot
        bot = TradingTelegramBot()
        bot.run()
    elif args.status:
        cmd_status()
    elif args.scan_once:
        from engine import Engine
        Engine().scan_once()
    elif args.backtest:
        cmd_backtest()
    else:
        api_thread = threading.Thread(target=start_api_server, daemon=True)
        api_thread.start()
        from engine import Engine
        Engine().run()


if __name__ == "__main__":
    main()
