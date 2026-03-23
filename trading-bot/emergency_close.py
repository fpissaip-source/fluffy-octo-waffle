#!/usr/bin/env python3
"""
emergency_close.py — Alle Positionen schließen + Bot beenden.

Löst 403-Fehler durch Stornierung aller offenen Orders ZUERST,
dann schließt alle Positionen, dann killt den Bot-Prozess.

Usage:
    python emergency_close.py
"""
import os
import sys
import time
import signal
import logging

# .env laden
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("emergency")

_LOCK_FILE = "/tmp/trading_bot.pid"


def kill_bot():
    """Bot-Prozess via PID-Datei beenden."""
    if not os.path.exists(_LOCK_FILE):
        log.info("Keine PID-Datei gefunden — Bot läuft möglicherweise nicht.")
        return
    try:
        with open(_LOCK_FILE) as f:
            pid = int(f.read().strip())
        os.kill(pid, signal.SIGTERM)
        log.info(f"SIGTERM an Bot-Prozess PID {pid} gesendet.")
        time.sleep(2)
        # Prüfen ob Prozess noch läuft
        try:
            os.kill(pid, 0)
            log.warning(f"Prozess {pid} läuft noch — sende SIGKILL.")
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            log.info(f"Bot-Prozess {pid} erfolgreich beendet.")
        # Lock-Datei entfernen
        try:
            os.remove(_LOCK_FILE)
        except FileNotFoundError:
            pass
    except (ValueError, FileNotFoundError):
        log.warning("PID-Datei ungültig oder bereits gelöscht.")
    except PermissionError:
        log.error("Keine Berechtigung, Bot-Prozess zu beenden.")


def main():
    import alpaca_trade_api as tradeapi

    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

    if not api_key or not secret_key:
        log.error("ALPACA_API_KEY / ALPACA_SECRET_KEY nicht gesetzt!")
        log.error("Setze die Variablen in .env oder als Umgebungsvariablen.")
        sys.exit(1)

    api = tradeapi.REST(key_id=api_key, secret_key=secret_key, base_url=base_url)

    # Schritt 1: Bot-Prozess killen
    log.info("=" * 50)
    log.info("SCHRITT 1: Bot-Prozess beenden")
    log.info("=" * 50)
    kill_bot()

    # Schritt 2: Alle offenen Orders stornieren (verhindert 403!)
    log.info("=" * 50)
    log.info("SCHRITT 2: Alle offenen Orders stornieren")
    log.info("=" * 50)
    try:
        open_orders = api.list_orders(status="open")
        if not open_orders:
            log.info("Keine offenen Orders.")
        else:
            log.info(f"{len(open_orders)} offene Orders gefunden — storniere alle...")
            for order in open_orders:
                try:
                    api.cancel_order(order.id)
                    log.info(f"  Storniert: {order.side.upper()} {order.qty}x {order.symbol} (ID: {order.id})")
                except Exception as e:
                    log.warning(f"  Fehler beim Stornieren {order.id}: {e}")
        # Kurz warten bis Stornierungen verarbeitet
        time.sleep(2)
    except Exception as e:
        log.error(f"Fehler beim Laden der Orders: {e}")
        sys.exit(1)

    # Schritt 3: Alle Positionen schließen
    log.info("=" * 50)
    log.info("SCHRITT 3: Alle Positionen schließen")
    log.info("=" * 50)
    try:
        positions = api.list_positions()
        if not positions:
            log.info("Keine offenen Positionen.")
        else:
            log.info(f"{len(positions)} Positionen gefunden — schließe alle...")
            for pos in positions:
                symbol = pos.symbol
                qty = float(pos.qty)
                entry = float(pos.avg_entry_price)
                pl = float(pos.unrealized_pl)
                log.info(f"  Schließe: {symbol} — {qty} Shares @ ${entry:.2f} | P&L: ${pl:+.2f}")
                try:
                    # close_position API: schließt gesamte Position mit Market-Order
                    order = api.close_position(symbol)
                    log.info(f"  -> Order ID: {order.id}")
                except Exception as e:
                    err = str(e)
                    if "403" in err or "forbidden" in err.lower():
                        log.error(f"  403 bei {symbol}: Möglicherweise noch offene Orders vorhanden.")
                        log.error(f"  Warte 3s und versuche nochmal...")
                        time.sleep(3)
                        try:
                            # Nochmal alle Orders stornieren
                            remaining = api.list_orders(status="open", symbols=[symbol])
                            for o in remaining:
                                api.cancel_order(o.id)
                                log.info(f"  Storniert verbleibende Order {o.id}")
                            time.sleep(1)
                            order = api.close_position(symbol)
                            log.info(f"  -> OK, Order ID: {order.id}")
                        except Exception as e2:
                            log.error(f"  Immer noch Fehler bei {symbol}: {e2}")
                            log.error(f"  Bitte manuell im Alpaca-Dashboard schließen!")
                    else:
                        log.error(f"  Fehler bei {symbol}: {e}")

        time.sleep(2)

        # Abschließende Prüfung
        remaining_positions = api.list_positions()
        if remaining_positions:
            log.warning(f"Noch {len(remaining_positions)} Positionen offen:")
            for pos in remaining_positions:
                log.warning(f"  {pos.symbol}: {pos.qty} Shares — bitte manuell im Alpaca-Dashboard schließen")
        else:
            log.info("Alle Positionen erfolgreich geschlossen!")

    except Exception as e:
        log.error(f"Fehler beim Schließen der Positionen: {e}")
        sys.exit(1)

    log.info("=" * 50)
    log.info("FERTIG. Bot gestoppt + alle Positionen geschlossen.")
    log.info("=" * 50)


if __name__ == "__main__":
    main()
