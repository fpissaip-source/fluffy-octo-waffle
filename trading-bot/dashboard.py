"""
dashboard.py — Live Web-Dashboard fuer den Trading Bot

Start: python dashboard.py
URL:   http://localhost:5001

Zeigt:
  - API Health Check (Alpaca, Gemini)
  - Equity, Cash, Buying Power, Invested, Day P&L
  - Offene Positionen mit P&L
  - Trade-Historie (letzte 20 Trades)
  - Performance-Stats (Win Rate, Sharpe, etc.)
  - Markt-Regime + Watchlist
"""

import json
import logging
import os
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from threading import Thread

logger = logging.getLogger("bot.dashboard")


def _check_gemini(api_key: str) -> dict:
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        m = genai.GenerativeModel("gemini-2.5-flash")
        m.generate_content("ping", generation_config={"max_output_tokens": 5})
        return {"ok": True, "msg": "Verbunden"}
    except Exception as e:
        return {"ok": False, "msg": str(e)[:80]}


def get_dashboard_data() -> dict:
    from config import Config
    data: dict = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "PAPER" if Config.is_paper() else "LIVE",
        "api_alpaca": {"ok": False, "msg": ""},
        "api_gemini": {"ok": False, "msg": ""},
        "equity": 0.0,
        "cash": 0.0,
        "buying_power": 0.0,
        "invested": 0.0,
        "day_pl": 0.0,
        "day_pl_pct": 0.0,
        "positions": {},
        "stats": {},
        "trade_history": [],
        "watchlist": Config.WATCHLIST,
        "regime": "UNKNOWN",
        "blacklist": [],
    }

    # ── Alpaca API Check ──
    try:
        from broker import AlpacaBroker
        broker = AlpacaBroker()
        account = broker.api.get_account()

        data["api_alpaca"] = {"ok": True, "msg": f"Account {account.id[:8]}..."}
        data["equity"] = float(account.equity)
        data["cash"] = float(account.cash)
        data["buying_power"] = float(account.buying_power)
        data["day_pl"] = float(account.equity) - float(account.last_equity)
        data["day_pl_pct"] = data["day_pl"] / float(account.last_equity) if float(account.last_equity) > 0 else 0

        positions = broker.get_positions()
        data["positions"] = positions
        data["invested"] = sum(p["market_value"] for p in positions.values())

    except Exception as e:
        data["api_alpaca"] = {"ok": False, "msg": str(e)[:120]}

    # ── Gemini API Check ──
    if Config.GEMINI_API_KEY:
        data["api_gemini"] = _check_gemini(Config.GEMINI_API_KEY)
    else:
        data["api_gemini"] = {"ok": False, "msg": "GEMINI_API_KEY fehlt in .env"}

    # ── Trade History + Stats ──
    try:
        from adaptive import AdaptiveLearner
        learner = AdaptiveLearner()
        data["stats"] = learner.get_stats()

        history = []
        for t in reversed(learner.trade_history[-20:]):
            history.append({
                "symbol": t.symbol,
                "side": "BUY",
                "entry": round(t.entry_price, 4),
                "exit": round(t.exit_price, 4) if t.exit_price else None,
                "pnl": round(t.pnl, 2) if t.pnl is not None else None,
                "pnl_pct": round(t.pnl_pct * 100, 2) if t.pnl_pct is not None else None,
                "regime": t.regime,
                "cascade": t.cascade_level if hasattr(t, "cascade_level") else "—",
                "ts": t.entry_time if hasattr(t, "entry_time") else "—",
            })
        data["trade_history"] = history

        # Blacklist
        bl = learner.blacklist if hasattr(learner, "blacklist") else {}
        data["blacklist"] = list(bl.keys()) if isinstance(bl, dict) else list(bl)

    except Exception:
        pass

    return data


# ═══════════════════════════════════════════════════════════════════
#  HTML DASHBOARD
# ═══════════════════════════════════════════════════════════════════

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Trading Bot Dashboard</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#080808;color:#d0d0d0;font-family:'Courier New',monospace;padding:16px;max-width:1100px;margin:0 auto}
h1{color:#00ff88;font-size:1.4em;margin-bottom:18px;display:flex;align-items:center;gap:10px}
h2{color:#00aaff;font-size:0.78em;margin:20px 0 8px;text-transform:uppercase;letter-spacing:2px}
.dot{width:9px;height:9px;border-radius:50%;background:#00ff88;animation:pulse 2s infinite;flex-shrink:0}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.25}}
/* Status bar */
.statusbar{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:18px}
.api-badge{display:flex;align-items:center;gap:6px;background:#111;border:1px solid #1e1e1e;border-radius:6px;padding:7px 12px;font-size:.75em}
.api-badge .icon{font-size:1em}
.api-badge .name{color:#666}
.api-badge .status{font-weight:bold}
.ok{color:#00ff88}.fail{color:#ff4444}.warn{color:#ffaa00}
/* Cards */
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin-bottom:6px}
.card{background:#111;border:1px solid #1a1a1a;border-radius:8px;padding:14px}
.card .lbl{color:#555;font-size:.68em;margin-bottom:5px;text-transform:uppercase;letter-spacing:.5px}
.card .val{color:#00ff88;font-size:1.35em;font-weight:bold;word-break:break-all}
.card .val.red{color:#ff4444}
.card .val.yellow{color:#ffaa00}
.card .val.white{color:#e0e0e0}
/* Tables */
table{width:100%;border-collapse:collapse;font-size:.8em;margin-bottom:4px}
th{background:#111;color:#444;padding:8px 10px;text-align:left;border-bottom:1px solid #1e1e1e;white-space:nowrap}
td{padding:7px 10px;border-bottom:1px solid #111;vertical-align:middle}
tr:hover td{background:#0f0f0f}
.green{color:#00ff88}.red{color:#ff4444}
.badge{display:inline-block;padding:1px 7px;border-radius:4px;font-size:.68em;font-weight:bold}
.badge-paper{background:#0a2a0a;color:#00ff88;border:1px solid #00ff8830}
.badge-live{background:#2a0a0a;color:#ff4444;border:1px solid #ff444430}
.badge-regime{background:#0a1a2a;color:#00aaff;border:1px solid #00aaff30}
.no-data{color:#333;padding:14px 10px;font-style:italic;font-size:.82em}
/* Stats row */
.stats-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:8px;margin-bottom:6px}
.stat{background:#0d0d0d;border:1px solid #1a1a1a;border-radius:6px;padding:10px 12px}
.stat .slbl{color:#444;font-size:.65em;text-transform:uppercase;letter-spacing:.5px;margin-bottom:3px}
.stat .sval{color:#00aaff;font-size:1.1em;font-weight:bold}
/* Footer */
.footer{color:#282828;font-size:.65em;text-align:center;margin-top:22px;padding-top:12px;border-top:1px solid #111}
.bl{color:#333;font-size:.72em;margin-top:6px}
/* Scrollable tables on mobile */
.tscroll{overflow-x:auto}
</style>
</head>
<body>

<h1>
  <div class="dot" id="dot"></div>
  TRADING BOT
  <span class="badge badge-paper" id="mode-badge">PAPER</span>
  <span class="badge badge-regime" id="regime-badge" style="display:none">—</span>
</h1>

<!-- API Status -->
<div class="statusbar" id="api-status">
  <div class="api-badge"><span class="icon">🔌</span><span class="name">Alpaca</span><span class="status" id="alpaca-status">...</span></div>
  <div class="api-badge"><span class="icon">🤖</span><span class="name">Gemini</span><span class="status" id="gemini-status">...</span></div>
  <div class="api-badge"><span class="icon">🕐</span><span class="name">Stand</span><span class="status white" id="ts-badge">—</span></div>
</div>

<!-- Portfolio Cards -->
<h2>Portfolio</h2>
<div class="grid">
  <div class="card"><div class="lbl">Equity</div><div class="val" id="c-equity">—</div></div>
  <div class="card"><div class="lbl">Cash</div><div class="val white" id="c-cash">—</div></div>
  <div class="card"><div class="lbl">Buying Power</div><div class="val white" id="c-bp">—</div></div>
  <div class="card"><div class="lbl">Investiert</div><div class="val yellow" id="c-invested">—</div></div>
  <div class="card"><div class="lbl">Day P&amp;L</div><div class="val" id="c-daypl">—</div></div>
  <div class="card"><div class="lbl">Positionen</div><div class="val white" id="c-poscount">—</div></div>
</div>

<!-- Positions -->
<h2>Offene Positionen</h2>
<div class="tscroll">
<table id="pos-table">
  <thead><tr><th>Symbol</th><th>Qty</th><th>Einstieg</th><th>Akt. Wert</th><th>P&amp;L $</th><th>P&amp;L %</th></tr></thead>
  <tbody id="pos-body"><tr><td colspan="6" class="no-data">Lade...</td></tr></tbody>
</table>
</div>

<!-- Performance Stats -->
<h2>Performance</h2>
<div class="stats-grid" id="stats-grid">
  <div class="stat"><div class="slbl">Trades gesamt</div><div class="sval" id="s-total">—</div></div>
  <div class="stat"><div class="slbl">Win Rate</div><div class="sval" id="s-wr">—</div></div>
  <div class="stat"><div class="slbl">Ø Gewinn</div><div class="sval green" id="s-avgwin">—</div></div>
  <div class="stat"><div class="slbl">Ø Verlust</div><div class="sval red" id="s-avgloss">—</div></div>
  <div class="stat"><div class="slbl">Bester Trade</div><div class="sval green" id="s-best">—</div></div>
  <div class="stat"><div class="slbl">Schlechtester</div><div class="sval red" id="s-worst">—</div></div>
  <div class="stat"><div class="slbl">Total P&amp;L</div><div class="sval" id="s-totalpnl">—</div></div>
  <div class="stat"><div class="slbl">Sharpe Ratio</div><div class="sval" id="s-sharpe">—</div></div>
</div>

<!-- Trade History -->
<h2>Trade-Historie (letzte 20)</h2>
<div class="tscroll">
<table id="hist-table">
  <thead><tr><th>Symbol</th><th>Einstieg</th><th>Ausstieg</th><th>P&amp;L $</th><th>P&amp;L %</th><th>Cascade</th><th>Regime</th></tr></thead>
  <tbody id="hist-body"><tr><td colspan="7" class="no-data">Lade...</td></tr></tbody>
</table>
</div>

<!-- Watchlist + Blacklist -->
<h2>Watchlist</h2>
<p id="watchlist" style="color:#555;font-size:.8em;line-height:1.8em">—</p>
<p class="bl" id="blacklist"></p>

<div class="footer" id="footer">Auto-Refresh alle 15s</div>

<script>
const fmt = n => (n == null ? '—' : n.toLocaleString('de-DE', {minimumFractionDigits:2, maximumFractionDigits:2}));
const fmtPct = n => (n == null ? '—' : (n >= 0 ? '+' : '') + n.toFixed(2) + '%');
const cls = n => n >= 0 ? 'green' : 'red';

async function refresh() {
  let d;
  try {
    d = await (await fetch('/api/data')).json();
  } catch(e) {
    document.getElementById('footer').textContent = 'Fehler beim Laden: ' + e.message;
    return;
  }

  // Mode badge
  const mb = document.getElementById('mode-badge');
  mb.textContent = d.mode || 'PAPER';
  mb.className = 'badge ' + (d.mode === 'LIVE' ? 'badge-live' : 'badge-paper');

  // Regime
  if (d.regime && d.regime !== 'UNKNOWN') {
    const rb = document.getElementById('regime-badge');
    rb.textContent = d.regime;
    rb.style.display = 'inline-block';
  }

  // API Badges
  const alpacaEl = document.getElementById('alpaca-status');
  if (d.api_alpaca.ok) {
    alpacaEl.textContent = '✓ ' + d.api_alpaca.msg;
    alpacaEl.className = 'status ok';
  } else {
    alpacaEl.textContent = '✗ ' + d.api_alpaca.msg;
    alpacaEl.className = 'status fail';
  }
  const geminiEl = document.getElementById('gemini-status');
  if (d.api_gemini.ok) {
    geminiEl.textContent = '✓ ' + d.api_gemini.msg;
    geminiEl.className = 'status ok';
  } else {
    geminiEl.textContent = '✗ ' + d.api_gemini.msg;
    geminiEl.className = 'status fail';
  }
  document.getElementById('ts-badge').textContent = d.timestamp || '—';

  // Portfolio Cards
  const daypl = d.day_pl || 0;
  const dayplPct = (d.day_pl_pct || 0) * 100;
  document.getElementById('c-equity').textContent = '$' + fmt(d.equity);
  document.getElementById('c-cash').textContent = '$' + fmt(d.cash);
  document.getElementById('c-bp').textContent = '$' + fmt(d.buying_power);
  document.getElementById('c-invested').textContent = '$' + fmt(d.invested);
  const dplEl = document.getElementById('c-daypl');
  dplEl.textContent = (daypl >= 0 ? '+' : '') + '$' + fmt(daypl) + ' (' + fmtPct(dayplPct) + ')';
  dplEl.className = 'val ' + cls(daypl);
  document.getElementById('c-poscount').textContent = Object.keys(d.positions || {}).length;

  // Positions
  const pos = d.positions || {};
  const posKeys = Object.keys(pos);
  const posBody = document.getElementById('pos-body');
  if (!posKeys.length) {
    posBody.innerHTML = '<tr><td colspan="6" class="no-data">Keine offenen Positionen</td></tr>';
  } else {
    posBody.innerHTML = posKeys.map(sym => {
      const p = pos[sym];
      const pl = p.unrealized_pl;
      const plpct = p.unrealized_plpc * 100;
      const c = cls(pl);
      return `<tr>
        <td><b>${sym}</b></td>
        <td>${p.qty}</td>
        <td>$${p.avg_entry.toFixed(4)}</td>
        <td>$${fmt(p.market_value)}</td>
        <td class="${c}">${pl >= 0 ? '+' : ''}$${fmt(pl)}</td>
        <td class="${c}">${fmtPct(plpct)}</td>
      </tr>`;
    }).join('');
  }

  // Stats
  const s = d.stats || {};
  if (s.total_trades) {
    document.getElementById('s-total').textContent = s.total_trades;
    document.getElementById('s-wr').textContent = s.win_rate != null ? (s.win_rate * 100).toFixed(1) + '%' : '—';
    document.getElementById('s-avgwin').textContent = s.avg_win != null ? fmtPct(s.avg_win * 100) : '—';
    document.getElementById('s-avgloss').textContent = s.avg_loss != null ? fmtPct(s.avg_loss * 100) : '—';
    document.getElementById('s-best').textContent = s.best_trade != null ? fmtPct(s.best_trade * 100) : '—';
    document.getElementById('s-worst').textContent = s.worst_trade != null ? fmtPct(s.worst_trade * 100) : '—';
    const tpnl = document.getElementById('s-totalpnl');
    tpnl.textContent = s.total_pnl != null ? (s.total_pnl >= 0 ? '+' : '') + '$' + fmt(s.total_pnl) : '—';
    tpnl.className = 'sval ' + cls(s.total_pnl || 0);
    document.getElementById('s-sharpe').textContent = s.sharpe != null ? s.sharpe.toFixed(2) : '—';
  } else {
    document.getElementById('s-total').textContent = '0';
  }

  // Trade History
  const hist = d.trade_history || [];
  const histBody = document.getElementById('hist-body');
  if (!hist.length) {
    histBody.innerHTML = '<tr><td colspan="7" class="no-data">Noch keine abgeschlossenen Trades</td></tr>';
  } else {
    histBody.innerHTML = hist.map(t => {
      const c = t.pnl != null ? cls(t.pnl) : '';
      const plStr = t.pnl != null ? (t.pnl >= 0 ? '+' : '') + '$' + fmt(t.pnl) : '—';
      const plpctStr = t.pnl_pct != null ? fmtPct(t.pnl_pct) : '—';
      return `<tr>
        <td><b>${t.symbol}</b></td>
        <td>$${t.entry != null ? t.entry.toFixed(4) : '—'}</td>
        <td>$${t.exit != null ? t.exit.toFixed(4) : '<span style="color:#555">offen</span>'}</td>
        <td class="${c}">${plStr}</td>
        <td class="${c}">${plpctStr}</td>
        <td style="color:#666">${t.cascade}/7</td>
        <td style="color:#555">${t.regime || '—'}</td>
      </tr>`;
    }).join('');
  }

  // Watchlist
  document.getElementById('watchlist').textContent = (d.watchlist || []).join('  ·  ') || '—';

  // Blacklist
  const bl = d.blacklist || [];
  if (bl.length) {
    document.getElementById('blacklist').textContent = '⛔ Blacklist: ' + bl.join(', ');
  }

  document.getElementById('footer').textContent = 'Auto-Refresh alle 15s  |  Letztes Update: ' + d.timestamp;
}

refresh();
setInterval(refresh, 15000);
</script>
</body>
</html>"""


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/data":
            try:
                data = get_dashboard_data()
                body = json.dumps(data, default=str).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)
            except Exception as e:
                body = json.dumps({"error": str(e)}).encode()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
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


def run(port: int = 5001):
    server = HTTPServer(("0.0.0.0", port), DashboardHandler)
    print(f"\n  Dashboard: http://localhost:{port}\n")
    logger.info(f"Dashboard laeuft auf http://0.0.0.0:{port}")
    server.serve_forever()


def start_background(port: int = 5001):
    t = Thread(target=run, args=(port,), daemon=True, name="dashboard")
    t.start()
    return t


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    port = int(os.getenv("BOT_API_PORT", "5001"))
    run(port)
