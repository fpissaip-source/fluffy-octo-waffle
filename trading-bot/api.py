"""
api.py — Web Dashboard + REST API (kein Flask, kein externes Framework)

Startet automatisch als Background-Thread wenn der Bot laeuft.

Endpoints:
  GET /           -> HTML Dashboard (auto-refresh 10s)
  GET /api/status -> JSON mit allen Bot-Daten
"""

import json
import logging
import threading
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler

logger = logging.getLogger("bot.api")

_broker = None
_state = {
    "equity": 0.0,
    "cash": 0.0,
    "positions": {},
    "trades": [],
    "regime": "UNKNOWN",
    "watchlist": [],
    "last_scan": None,
    "mode": "PAPER",
}


def update_state(**kwargs):
    _state.update(kwargs)
    _state["last_scan"] = datetime.now().strftime("%H:%M:%S")


def _get_live_state() -> dict:
    state = dict(_state)
    if _broker:
        try:
            state["equity"] = _broker.get_equity()
            state["cash"] = _broker.get_cash()
            state["positions"] = _broker.get_positions()
        except Exception:
            pass
    return state


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
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:20px}
.card{background:#111;border:1px solid #1e1e1e;border-radius:8px;padding:14px}
.card .lbl{color:#666;font-size:0.7em;margin-bottom:4px;text-transform:uppercase}
.card .val{color:#00ff88;font-size:1.3em;font-weight:bold}
.yellow{color:#ffaa00!important}.gray{color:#555!important}.red{color:#ff4444!important}
h2{color:#00aaff;font-size:0.85em;margin:16px 0 8px;text-transform:uppercase;letter-spacing:1px}
table{width:100%;border-collapse:collapse;font-size:0.82em}
th{background:#161616;color:#555;padding:8px;text-align:left;border-bottom:1px solid #222}
td{padding:8px;border-bottom:1px solid #161616}
.green{color:#00ff88}.badge{padding:2px 8px;border-radius:4px;font-size:0.7em;font-weight:bold;background:#0a2a0a;color:#00ff88;border:1px solid #00ff8833}
.no-data{color:#333;padding:12px;font-style:italic}
.ts{color:#333;font-size:0.7em;margin-top:16px;text-align:center}
</style>
</head>
<body>
<h1><div class="dot"></div> TRADING BOT <span class="badge" id="mode">PAPER</span></h1>
<div class="grid">
  <div class="card"><div class="lbl">Equity</div><div class="val" id="equity">--</div></div>
  <div class="card"><div class="lbl">Cash</div><div class="val" id="cash">--</div></div>
  <div class="card"><div class="lbl">Positionen</div><div class="val" id="pos_count">--</div></div>
  <div class="card"><div class="lbl">Trades</div><div class="val" id="trades">--</div></div>
  <div class="card"><div class="lbl">Regime</div><div class="val yellow" id="regime">--</div></div>
  <div class="card"><div class="lbl">Letzter Scan</div><div class="val gray" id="scan">--</div></div>
</div>
<h2>Offene Positionen</h2>
<div id="pos_div"><p class="no-data">Lade...</p></div>
<h2>Watchlist</h2>
<p id="watchlist" style="color:#555;font-size:0.8em;padding:6px 0">--</p>
<p class="ts" id="ts">Verbinde...</p>
<script>
function fmt(n){return n.toLocaleString('de-DE',{minimumFractionDigits:2,maximumFractionDigits:2})}
async function refresh(){
  try{
    const d=await(await fetch('/api/status')).json();
    document.getElementById('equity').textContent='$'+fmt(d.equity);
    document.getElementById('cash').textContent='$'+fmt(d.cash);
    document.getElementById('pos_count').textContent=Object.keys(d.positions||{}).length;
    document.getElementById('trades').textContent=(d.trades||[]).length;
    document.getElementById('regime').textContent=d.regime||'--';
    document.getElementById('scan').textContent=d.last_scan||'--';
    document.getElementById('mode').textContent=d.mode||'PAPER';
    document.getElementById('watchlist').textContent=(d.watchlist||[]).join(', ');
    const pos=d.positions||{};
    const keys=Object.keys(pos);
    if(!keys.length){document.getElementById('pos_div').innerHTML='<p class="no-data">Keine offenen Positionen</p>';return;}
    document.getElementById('pos_div').innerHTML='<table><tr><th>Symbol</th><th>Qty</th><th>Einstieg</th><th>P&L</th><th>Wert</th></tr>'+
      keys.map(s=>{const p=pos[s];const pl=p.unrealized_pl;const c=pl>=0?'green':'red';
        return`<tr><td><b>${s}</b></td><td>${p.qty}</td><td>$${p.avg_entry.toFixed(2)}</td><td class="${c}">${pl>=0?'+':''}$${pl.toFixed(2)} (${(p.unrealized_plpc*100).toFixed(1)}%)</td><td>$${fmt(p.market_value)}</td></tr>`;
      }).join('')+'</table>';
    document.getElementById('ts').textContent='Aktualisiert: '+new Date().toLocaleTimeString('de-DE');
  }catch(e){document.getElementById('ts').textContent='Fehler: '+e.message;}
}
refresh();setInterval(refresh,10000);
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/status":
            body = json.dumps(_get_live_state(), default=str).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
        else:
            body = DASHBOARD_HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, *args):
        pass


def start_api_server(broker=None, port: int = 5001):
    global _broker
    _broker = broker

    def run():
        try:
            server = HTTPServer(("0.0.0.0", port), Handler)
            logger.info(f"Dashboard: http://0.0.0.0:{port}")
            server.serve_forever()
        except Exception as e:
            logger.error(f"Dashboard Fehler: {e}")

    t = threading.Thread(target=run, daemon=True, name="api-server")
    t.start()
    return t
