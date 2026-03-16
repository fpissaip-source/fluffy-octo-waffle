"""
dashboard.py — Live Web-Dashboard fuer den Trading Bot

Start: python dashboard.py
URL:   http://SERVER_IP:5001

Zeigt:
  - Equity & Portfolio
  - Offene Positionen
  - Trade-Historie
  - Bot-Status (laeuft / gestoppt)
"""

import json
import os
import logging
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread

from broker import AlpacaBroker
from config import Config
from adaptive import AdaptiveLearner

logger = logging.getLogger("bot.dashboard")


def get_dashboard_data() -> dict:
    try:
        broker = AlpacaBroker()
        equity = broker.get_equity()
        cash = broker.get_cash()
        positions = broker.get_positions()
        learner = AdaptiveLearner()
        stats = learner.get_stats()

        return {
            "equity": equity,
            "cash": cash,
            "positions": positions,
            "stats": stats,
            "watchlist": Config.WATCHLIST,
            "mode": "PAPER" if Config.is_paper() else "LIVE",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    except Exception as e:
        return {"error": str(e)}


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta http-equiv="refresh" content="30">
<title>Trading Bot Dashboard</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0a0a0a; color: #e0e0e0; font-family: 'Courier New', monospace; padding: 20px; }}
  h1 {{ color: #00ff88; font-size: 1.5em; margin-bottom: 20px; }}
  h2 {{ color: #00aaff; font-size: 1em; margin: 15px 0 8px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px; }}
  .card {{ background: #111; border: 1px solid #222; border-radius: 8px; padding: 15px; }}
  .card .label {{ color: #888; font-size: 0.75em; margin-bottom: 5px; }}
  .card .value {{ color: #00ff88; font-size: 1.4em; font-weight: bold; }}
  .card .value.red {{ color: #ff4444; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.85em; }}
  th {{ background: #1a1a1a; color: #888; padding: 8px; text-align: left; border-bottom: 1px solid #333; }}
  td {{ padding: 8px; border-bottom: 1px solid #1a1a1a; }}
  .pass {{ color: #00ff88; }} .fail {{ color: #ff4444; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.75em; }}
  .badge.paper {{ background: #1a3a1a; color: #00ff88; }}
  .badge.live {{ background: #3a1a1a; color: #ff4444; }}
  .ts {{ color: #555; font-size: 0.75em; margin-top: 20px; }}
</style>
</head>
<body>
<h1>⚡ TRADING BOT DASHBOARD <span class="badge {mode_class}">{mode}</span></h1>

<div class="grid">
  <div class="card">
    <div class="label">EQUITY</div>
    <div class="value">${equity:,.2f}</div>
  </div>
  <div class="card">
    <div class="label">CASH</div>
    <div class="value">${cash:,.2f}</div>
  </div>
  <div class="card">
    <div class="label">OFFENE POSITIONEN</div>
    <div class="value">{pos_count}</div>
  </div>
  <div class="card">
    <div class="label">TOTAL TRADES</div>
    <div class="value">{total_trades}</div>
  </div>
  <div class="card">
    <div class="label">WIN RATE</div>
    <div class="value {wr_class}">{win_rate}</div>
  </div>
  <div class="card">
    <div class="label">WATCHLIST</div>
    <div class="value" style="font-size:0.8em">{watchlist}</div>
  </div>
</div>

{positions_html}

<p class="ts">Letzte Aktualisierung: {timestamp} &nbsp;|&nbsp; Auto-Refresh alle 30s</p>
</body>
</html>"""


def build_html(data: dict) -> str:
    if "error" in data:
        return f"<h1>Fehler: {data['error']}</h1>"

    stats = data.get("stats", {})
    positions = data.get("positions", {})

    win_rate_val = stats.get("win_rate", 0)
    win_rate_str = f"{win_rate_val:.0%}" if win_rate_val else "N/A"
    wr_class = "pass" if win_rate_val >= 0.5 else "fail"

    # Positionen Tabelle
    if positions:
        rows = ""
        for sym, pos in positions.items():
            pl = pos["unrealized_pl"]
            plpc = pos["unrealized_plpc"]
            color = "pass" if pl >= 0 else "fail"
            rows += f"""<tr>
              <td>{sym}</td>
              <td>{pos['qty']:.0f}</td>
              <td>${pos['avg_entry']:.2f}</td>
              <td class="{color}">${pl:+.2f} ({plpc:+.1%})</td>
              <td>${pos['market_value']:,.2f}</td>
            </tr>"""
        positions_html = f"""
        <h2>OFFENE POSITIONEN</h2>
        <table>
          <tr><th>Symbol</th><th>Qty</th><th>Einstieg</th><th>P&L</th><th>Wert</th></tr>
          {rows}
        </table>"""
    else:
        positions_html = "<h2>OFFENE POSITIONEN</h2><p style='color:#555;padding:10px'>Keine offenen Positionen.</p>"

    return HTML_TEMPLATE.format(
        mode=data["mode"],
        mode_class="paper" if data["mode"] == "PAPER" else "live",
        equity=data["equity"],
        cash=data["cash"],
        pos_count=len(positions),
        total_trades=stats.get("total_trades", 0),
        win_rate=win_rate_str,
        wr_class=wr_class,
        watchlist=", ".join(data["watchlist"][:6]) + ("..." if len(data["watchlist"]) > 6 else ""),
        positions_html=positions_html,
        timestamp=data["timestamp"],
    )


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api":
            data = get_dashboard_data()
            body = json.dumps(data, default=str).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        else:
            data = get_dashboard_data()
            html = build_html(data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html)

    def log_message(self, format, *args):
        pass  # Kein HTTP-Log Spam


def run(port: int = 5001):
    server = HTTPServer(("0.0.0.0", port), DashboardHandler)
    logger.info(f"Dashboard laeuft auf http://0.0.0.0:{port}")
    server.serve_forever()


if __name__ == "__main__":
    import sys
    from config import Config
    logging.basicConfig(level=logging.INFO)
    port = int(os.getenv("BOT_API_PORT", "5001"))
    print(f"\n  Dashboard: http://0.0.0.0:{port}\n")
    run(port)
