#!/usr/bin/env python3
"""
main.py — SIX FILTERS. ONE TRADE.

Usage:
    python main.py              # Bot starten (Endlosschleife)
    python main.py --scan-once  # Einmaliger Scan
    python main.py --status     # Account Status
    python main.py --backtest   # Quick Analyse
"""

import os
import sys
import logging
import argparse

from config import Config
from engine import Engine
from broker import AlpacaBroker


def setup_logging():
    level = getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO)
    fmt = "%(asctime)s | %(name)-12s | %(levelname)-7s | %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S")
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("websocket").setLevel(logging.WARNING)


def cmd_status():
    broker = AlpacaBroker()
    equity = broker.get_equity()
    cash = broker.get_cash()
    bp = broker.get_buying_power()
    positions = broker.get_positions()

    print(f"\n{'=' * 50}")
    print(f"  ACCOUNT STATUS  ({'PAPER' if Config.is_paper() else 'LIVE'})")
    print(f"{'=' * 50}")
    print(f"  Equity:        ${equity:,.2f}")
    print(f"  Cash:          ${cash:,.2f}")
    print(f"  Buying Power:  ${bp:,.2f}")
    print(f"  Positions:     {len(positions)}")
    print(f"{'-' * 50}")
    if positions:
        for sym, pos in positions.items():
            pl = pos['unrealized_pl']
            plpc = pos['unrealized_plpc']
            pre = "+" if pl >= 0 else ""
            print(f"  {sym:<8} {pos['qty']:>6} shares @ ${pos['avg_entry']:.2f}"
                  f"  {pre}${pl:.2f} ({pre}{plpc:.1%})")
    else:
        print("  No open positions.")
    print(f"{'=' * 50}\n")


def cmd_backtest():
    print(f"\n{'=' * 50}")
    print(f"  QUICK ANALYSIS")
    print(f"{'=' * 50}\n")
    engine = Engine()
    for symbol in Config.WATCHLIST:
        try:
            signal = engine.analyze_symbol(symbol)
            print(signal.summary())
            for name, r in signal.results.items():
                d = r.get("details", {})
                detail_str = "  ".join(f"{k}={v}" for k, v in d.items()
                                       if k not in ("error", "updates"))
                if detail_str:
                    print(f"    {name}: {detail_str}")
            print()
        except Exception as e:
            print(f"  {symbol}: Error - {e}\n")


def main():
    parser = argparse.ArgumentParser(description="Six Filters. One Trade.")
    parser.add_argument("--scan-once", action="store_true", help="Single scan")
    parser.add_argument("--status", action="store_true", help="Account status")
    parser.add_argument("--backtest", action="store_true", help="Quick analysis")
    parser.add_argument("--telegram", action="store_true", help="Start with Telegram control")
    args = parser.parse_args()

    setup_logging()

    if not Config.validate():
        print("\n  API Keys nicht konfiguriert!")
        print("  1. cp .env.example .env")
        print("  2. Trage deine Alpaca Keys in .env ein")
        print("  3. Starte erneut\n")
        sys.exit(1)

    print()
    print("  +==========================================+")
    print("  |   SIX FILTERS. ONE TRADE.               |")
    print("  |   Quantitative Trading Engine v1.0       |")
    print("  +==========================================+")
    print()

    if args.status:
        cmd_status()
    elif args.scan_once:
        e = Engine()
        Engine().scan_once(market_open=e.broker.is_market_open())
    elif args.backtest:
        cmd_backtest()
    else:
        # Always start the API dashboard for long-running modes
        from api import start_api_server
        engine = Engine()
        start_api_server(broker=engine.broker, port=int(os.getenv("BOT_API_PORT", "5001")))

        telegram_configured = (
            Config.TELEGRAM_TOKEN
            and Config.TELEGRAM_TOKEN != "your_telegram_bot_token_here"
            and Config.TELEGRAM_CHAT_ID
        )

        if telegram_configured:
            import threading
            t = threading.Thread(target=engine.run, daemon=True, name="engine")
            t.start()
            from telegram_bot import TradingTelegramBot
            bot = TradingTelegramBot(engine=engine)
            bot.run()
        elif args.telegram:
            print("  Telegram nicht konfiguriert!")
            print("  Trage TELEGRAM_TOKEN und TELEGRAM_CHAT_ID in .env ein.")
            print("  Siehe README fuer Anleitung.\n")
            sys.exit(1)
        else:
            engine.run()


if __name__ == "__main__":
    main()
