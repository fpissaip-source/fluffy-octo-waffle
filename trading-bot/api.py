"""
api.py — Flask REST API + Web Dashboard

Startet automatisch als Background-Thread wenn der Bot laeuft.
Kein separater Prozess noetig.

Endpoints:
  GET /          -> HTML Dashboard
  GET /api/status -> JSON mit allen Bot-Daten
"""

import json
import logging
import threading
from datetime import datetime
from typing import Optional

logger = logging.getLogger("bot.api")

# Geteilter State zwischen Bot und API
_bot_state = {
    "equity": 0.0,
    "cash": 0.0,
    "positions": {},
    "trades": [],
    "regime": "UNKNOWN",
    "watchlist": [],
    "last_scan": None,
    "mode": "PAPER",
    "running": True,
}


def update_state(**kwargs):
    """Wird vom Bot aufgerufen um den State zu aktualisieren."""
    _bot_state.update(kwargs)
    _bot_state["last_scan"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_state() -> dict:
    return dict(_bot_state)


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Trading Bot</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:#0d0d0d;color:#e0e0e0;font-family:'Courier New',monospace;padding:16px;max-width:900px;margin:0 auto}
  h1{color:#00ff88;font-size:1.3em;margin-bottom:16px;display:flex;align-items:center;gap:10px}
  .dot{width:10px;height:10px;border-radius:50%;background:#00ff88;animation:pulse 2s infinite}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:0.3}}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-bottom:20px}
  .card{background:#111;border:1px solid #1e1e1e;border-radius:8px;padding:14px}
  .card .lbl{color:#666;font-size:0.7em;margin-bottom:4px;text-transform:uppercase}
  .card .val{color:#00ff88;font-size:1.3em;font-weight:bold}
  .card .val.red{color:#ff4444}
  .card .val.yellow{color:#ffaa00}
  h2{color:#00aaff;font-size:0.9em;margin:16px 0 8px;text-transform:uppercase;letter-spacing:1px}
  table{width:100%;border-collapse:collapse;font-size:0.82em}
  th{background:#161616;color:#555;padding:8px 10px;text-align:left;border-bottom:1px solid #222}
  td{padding:8px 10px;border-bottom:1px solid #161616}
  .green{color:#00ff88}.red{color:#ff4444}.gray{color:#555}
  .badge{padding:2px 8px;border-radius:4px;font-size:0.7em;font-weight:bold}
  .badge.paper{background:#0a2a0a;color:#00ff88;border:1px solid #00ff8833}
  .badge.live{background:#2a0a0a;color:#ff4444;border:1px solid #ff444433}
  .ts{color:#333;font-size:0.7em;margin-top:16px;text-align:center}
  .no-data{color:#333;padding:12px;text-align:center;font-style:italic}
  #error{display:none;background:#2a0a0a;border:1px solid #ff4444;padding:10px;border-radius:6px;margin-bottom:16px;color:#ff4444;font-size:0.8em}
</style>
</head>
<body>
<h1><div class="dot"></div> TRADING BOT <span id="badge" class="badge paper">PAPER</span></h1>
<div id="error"></div>

<div class="grid" id="cards">
  <div class="card"><div class="lbl">Equity</div><div class="val" id="equity">--</div></div>
  <div class="card"><div class="lbl">Cash</div><div class="val" id="cash">--</div></div>
  <div class="card"><div class="lbl">Positionen</div><div class="val" id="pos_count">--</div></div>
  <div class="card"><div class="lbl">Trades heute</div><div class="val" id="trades">--</div></div>
  <div class="card"><div class="lbl">Regime</div><div class="val yellow" id="regime">--</div></div>
  <div class="card"><div class="lbl">Letzter Scan</div><div class="val gray" id="last_scan" style="font-size:0.8em">--</div></div>
</div>

<h2>Offene Positionen</h2>
<div id="positions_table"><p class="no-data">Keine Positionen</p></div>

<h2>Watchlist</h2>
<div id="watchlist" style="color:#555;font-size:0.8em;padding:8px 0">--</div>

<p class="ts" id="ts">Verbinde...</p>

<script>
async function refresh() {
  try {
    const r = await fetch('/api/status');
    const d = await r.json();
    document.getElementById('error').style.display = 'none';

    document.getElementById('equity').textContent = '$' + d.equity.toLocaleString('de-DE', {minimumFractionDigits:2, maximumFractionDigits:2});
    document.getElementById('cash').textContent = '$' + d.cash.toLocaleString('de-DE', {minimumFractionDigits:2, maximumFractionDigits:2});
    document.getElementById('pos_count').textContent = Object.keys(d.positions).length;
    document.getElementById('trades').textContent = d.trades.length;
    document.getElementById('regime').textContent = d.regime;
    document.getElementById('last_scan').textContent = d.last_scan || '--';
    document.getElementById('watchlist').textContent = (d.watchlist || []).join(', ');

    const badge = document.getElementById('badge');
    badge.textContent = d.mode;
    badge.className = 'badge ' + d.mode.toLowerCase();

    // Positionen Tabelle
    const pos = d.positions;
    const keys = Object.keys(pos);
    if (keys.length === 0) {
      document.getElementById('positions_table').innerHTML = '<p class="no-data">Keine offenen Positionen</p>';
    } else {
      let rows = keys.map(sym => {
        const p = pos[sym];
        const pl = p.unrealized_pl;
        const plpc = (p.unrealized_plpc * 100).toFixed(1);
        const cls = pl >= 0 ? 'green' : 'red';
        const sign = pl >= 0 ? '+' : '';
        return `<tr>
          <td><b>${sym}</b></td>
          <td>${p.qty}</td>
          <td>$${p.avg_entry.toFixed(2)}</td>
          <td class="${cls}">${sign}$${pl.toFixed(2)} (${sign}${plpc}%)</td>
          <td>$${p.market_value.toLocaleString('de-DE', {maximumFractionDigits:0})}</td>
        </tr>`;
      }).join('');
      document.getElementById('positions_table').innerHTML =
        `<table><tr><th>Symbol</th><th>Qty</th><th>Einstieg</th><th>P&L</th><th>Wert</th></tr>${rows}</table>`;
    }

    document.getElementById('ts').textContent = 'Aktualisiert: ' + new Date().toLocaleTimeString('de-DE');
  } catch(e) {
    document.getElementById('error').style.display = 'block';
    document.getElementById('error').textContent = 'Verbindungsfehler: ' + e.message;
  }
}
refresh();
setInterval(refresh, 10000);
</script>
</body>
</html>"""


def create_app(broker=None):
    """Erstellt die Flask App. broker wird fuer Live-Daten genutzt."""
    try:
        from flask import Flask, jsonify, Response
    except ImportError:
        logger.error("Flask nicht installiert. Fuehre aus: pip install flask")
        return None

    app = Flask(__name__)
    app.logger.setLevel(logging.ERROR)
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    @app.route("/")
    def dashboard():
        return Response(DASHBOARD_HTML, mimetype="text/html")

    @app.route("/api/status")
    def status():
        state = get_state()
        # Live-Daten vom Broker holen falls verfuegbar
        if broker:
            try:
                state["equity"] = broker.get_equity()
                state["cash"] = broker.get_cash()
                state["positions"] = broker.get_positions()
            except Exception:
                pass
        return jsonify(state)

    return app


def start_api_server(broker=None, port: int = 5001):
    """Startet den API-Server als Background-Thread."""
    app = create_app(broker)
    if not app:
        return None

    def run():
        try:
            app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
        except Exception as e:
            logger.error(f"API Server Fehler: {e}")

    thread = threading.Thread(target=run, daemon=True, name="api-server")
    thread.start()
    logger.info(f"Dashboard: http://0.0.0.0:{port}")
    return thread
