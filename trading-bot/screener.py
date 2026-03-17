"""
screener.py — Echtzeit Markt-Spike-Scanner.

WebSocket-Modus (Standard): Abonniert Alpaca 1-Minuten-Bars per WSS.
  → Spikes werden in Millisekunden erkannt statt in 60-Sekunden-Polling-Fenstern.
  → Drastisch weniger API-Calls, kein Ratenlimit-Risiko.

REST-Fallback: Falls WebSocket-Verbindung nicht aufgebaut werden kann,
  fällt der Scanner automatisch auf Batch-Snapshots zurück.
"""

import logging
import threading
import time
from datetime import date
from typing import TYPE_CHECKING

from config import Config

if TYPE_CHECKING:
    from broker import AlpacaBroker

logger = logging.getLogger("bot.screener")


class SpikeSensor:
    """
    Echtzeit Spike-Erkennung via Alpaca WebSocket (1-Minuten-Bars).
    Universum: ~300 bekannte volatile Small-Cap / Penny / Biotech Stocks.
    Trigger: >3% intraday-Bewegung ODER Volumen >3x Durchschnitt.

    Interface bleibt identisch zum REST-Modus:
      sensor.scan() → list[str] der erkannten Spike-Symbole
    """

    # Breites Universum: volatile Small/Mid Caps, Biotech, Meme, EV, Crypto-Proxies
    # Bereinigt: delisted/bankrotte Stocks entfernt (BBBY, RIDE, GOEV, HYLN, HEXO,
    # TTOO, BBIG, MULN, ENDP, MILE/METROMILE, NAVB, SOLO, XELA, WISH, VIEW, AYRO,
    # SUNW/NOVA/FSR (bankruptcy), EXPR, KOSS, PHUN, NXGL, GFAI, DPSI, LIQT)
    UNIVERSE = [
        # ── Meme / Reddit ──────────────────────────────────────
        "AMC", "GME", "BB", "NOK", "SPCE",
        # ── Biotech / Pharma ───────────────────────────────────
        "OCGN", "NVAX", "INO", "SRNE", "VXRT", "IOVA", "SAVA",
        "CRSP", "EDIT", "NTLA", "BEAM", "VERV", "AMRN",
        "MNKD", "ADMA", "CPRX", "PRGO", "IRWD",
        "DARE", "CLRB", "INPX", "CLPS", "KOPN", "VUZI",
        "AGEN", "SURF", "PRME", "SUPN", "NKTR", "ACAD",
        # ── EV / Wasserstoff / Clean Energy ───────────────────
        "NKLA", "WKHS", "PLUG", "FCEL", "BLNK", "CHPT", "QS", "LCID", "RIVN",
        "SPWR", "RUN", "ARRY", "STEM",
        # ── Crypto-Proxies ─────────────────────────────────────
        "MARA", "RIOT", "BTBT", "SOS", "MOGO",
        "HUT", "CIFR", "BTDR", "CORZ",
        # ── Cannabis ───────────────────────────────────────────
        "SNDL", "TLRY", "ACB", "CGC", "CRON",
        "OGI", "GRWG", "IIPR",
        # ── Fintech / Meme Finance ─────────────────────────────
        "SOFI", "UPST", "AFRM", "HOOD", "COIN", "OPEN", "DKNG",
        "PENN", "RBLX",
        # ── Tech Small Caps ────────────────────────────────────
        "MVIS", "IDEX", "HIMS", "NNDM", "GNUS",
        "CTRM", "SHIP", "FREE", "IMPP",
        "IMVT", "ALDX", "RETA",
        # ── Energy / Oil Micro ─────────────────────────────────
        "BORR", "TELL", "AMTX", "REI",
        # ── Volatile Mid-Caps mit Momentum ────────────────────
        "RKT", "IRBT", "ANGI", "LMND", "ROOT",
        # ── Short-Squeeze Kandidaten ───────────────────────────
        "PRTY", "VSTO", "SCVL", "HRTX", "ACET",
        "NUVB", "AEYE", "SKIN", "BKSY", "SPIR", "OSAT",
        # ── Biotech mit nahen Katalysatoren ───────────────────
        "FATE", "RGEN", "ARWR", "PTGX", "IMCR", "TGTX",
        "KRTX", "RXDX", "ARVN", "RCKT", "ALLO",
        "MGTX", "RAPT", "NRIX", "PRAX", "FOLD",
        # ── Micro Cap (handelbar via Alpaca) ───────────────────
        "TNXP", "JAGX", "LGVN", "TPVG",
    ]

    def __init__(self, broker: "AlpacaBroker"):
        self.broker = broker
        self.min_pct = getattr(Config, "SPIKE_MIN_PCT", 0.03)
        self.min_vol_mult = 3.0
        self._avg_volumes: dict[str, float] = {}
        self._daily_opens: dict[str, float] = {}    # Tages-Open pro Symbol
        self._last_open_date: date = date.min        # Für tägliches Reset

        # Spike-Queue: Thread-sicher, wird von scan() abgeholt
        self._spike_queue: list[str] = []
        self._lock = threading.Lock()

        # WebSocket-Status
        self._ws_active = False
        self._ws_thread: threading.Thread | None = None
        self._use_websocket = True   # Auf False setzen wenn WS nicht verfügbar

        logger.info(f"[SPIKE] SpikeSensor init — Universum: {len(self.UNIVERSE)} Symbole")
        self._start_websocket()

    # ── WebSocket Modus ─────────────────────────────────

    def _start_websocket(self):
        """Startet WebSocket-Listener im Hintergrund-Thread."""
        t = threading.Thread(
            target=self._run_websocket_loop,
            daemon=True,
            name="spike-sensor-ws",
        )
        t.start()
        self._ws_thread = t

    def _run_websocket_loop(self):
        """
        Endlos-Loop mit automatischem Reconnect.
        Fallback auf REST wenn alpaca.data.live nicht verfügbar.
        """
        backoff = 5
        while True:
            try:
                self._connect_websocket()
                backoff = 5  # Reset nach erfolgreicher Verbindung
            except ImportError:
                logger.warning(
                    "[SPIKE] alpaca.data.live nicht verfügbar — "
                    "WebSocket deaktiviert, nutze REST-Fallback"
                )
                self._use_websocket = False
                return
            except Exception as e:
                logger.warning(f"[SPIKE] WebSocket Fehler: {e} — Reconnect in {backoff}s")
                self._ws_active = False
                time.sleep(backoff)
                backoff = min(backoff * 2, 120)

    def _connect_websocket(self):
        """Verbindet mit Alpaca WebSocket und abonniert 1-Min-Bars."""
        from alpaca.data.live import StockDataStream

        wss = StockDataStream(Config.API_KEY, Config.SECRET_KEY)

        def on_bar(bar):
            try:
                symbol = bar.symbol
                close = float(bar.close)
                open_bar = float(bar.open)
                volume = int(bar.volume)

                # Tages-Open: jeden neuen Handelstag resetten
                today = date.today()
                if today != self._last_open_date:
                    self._daily_opens.clear()
                    self._last_open_date = today
                    logger.debug("[SPIKE] Tages-Open zurückgesetzt (neuer Handelstag)")

                # Ersten Tages-Open pro Symbol speichern
                if symbol not in self._daily_opens:
                    self._daily_opens[symbol] = open_bar

                daily_open = self._daily_opens[symbol]
                pct_move = (close - daily_open) / daily_open if daily_open > 0 else 0

                # Rolling-Average Volumen (EMA)
                avg_vol = self._avg_volumes.get(symbol, volume)
                vol_mult = volume / avg_vol if avg_vol > 0 else 1.0
                self._avg_volumes[symbol] = avg_vol * 0.9 + volume * 0.1

                is_spike = abs(pct_move) >= self.min_pct or vol_mult >= self.min_vol_mult

                if is_spike:
                    direction = "+" if pct_move >= 0 else ""
                    logger.info(
                        f"[SPIKE] {symbol}: {direction}{pct_move:.1%} intraday "
                        f"| Vol: {vol_mult:.1f}x | Preis: ${close:.3f}"
                    )
                    with self._lock:
                        if symbol not in self._spike_queue:
                            self._spike_queue.append(symbol)

            except Exception as e:
                logger.debug(f"[SPIKE] Bar-Handler Fehler ({bar.symbol if hasattr(bar, 'symbol') else '?'}): {e}")

        # Symbole in Batches abonnieren (Alpaca WS-Limit: max 1024)
        self._ws_active = True
        logger.info(f"[SPIKE] WebSocket verbunden — abonniere {len(self.UNIVERSE)} Symbole")
        wss.subscribe_bars(on_bar, *self.UNIVERSE)
        wss.run()   # Blockiert bis Verbindung abbricht

    # ── REST-Fallback ────────────────────────────────────

    def _scan_rest(self) -> list[str]:
        """
        Batch-Snapshot aller Symbole — wird nur genutzt wenn WebSocket nicht verfügbar.
        """
        spikes: list[str] = []
        batch_size = 200
        for i in range(0, len(self.UNIVERSE), batch_size):
            batch = self.UNIVERSE[i:i + batch_size]
            snaps = self.broker.get_snapshots_batch(batch)
            for symbol, snap in snaps.items():
                try:
                    daily = snap.daily_bar
                    if not daily:
                        continue
                    open_p = float(daily.open)
                    close_p = float(daily.close)
                    volume = int(daily.volume)
                    if open_p <= 0 or close_p <= 0:
                        continue
                    pct_move = (close_p - open_p) / open_p
                    avg_vol = self._avg_volumes.get(symbol, volume)
                    vol_mult = volume / avg_vol if avg_vol > 0 else 1.0
                    self._avg_volumes[symbol] = avg_vol * 0.9 + volume * 0.1
                    if abs(pct_move) >= self.min_pct or vol_mult >= self.min_vol_mult:
                        direction = "+" if pct_move >= 0 else ""
                        logger.info(
                            f"[SPIKE][REST] {symbol}: {direction}{pct_move:.1%} "
                            f"| Vol: {vol_mult:.1f}x | ${close_p:.3f}"
                        )
                        spikes.append(symbol)
                except Exception as e:
                    logger.debug(f"[SPIKE] {symbol} snapshot parse error: {e}")
        return spikes

    # ── Öffentliches Interface ───────────────────────────

    def should_scan(self) -> bool:
        """Im WebSocket-Modus immer True (Spikes kommen per Push)."""
        return True

    def scan(self) -> list[str]:
        """
        Gibt erkannte Spikes zurück und leert die Queue.

        WebSocket-Modus: Sammelt akkumulierte Push-Spikes seit letztem Aufruf.
        REST-Fallback: Führt synchrones Batch-Snapshot durch.
        """
        if not self._use_websocket:
            # REST-Fallback (WebSocket nicht verfügbar)
            spikes = self._scan_rest()
            if spikes:
                logger.info(f"[SPIKE][REST] {len(spikes)} Spikes: {', '.join(spikes)}")
            return spikes

        if not self._ws_active:
            logger.debug("[SPIKE] WebSocket noch nicht verbunden — überspringe Scan")
            return []

        # WebSocket-Modus: Akkumulierte Spikes aus Queue holen
        with self._lock:
            spikes = list(self._spike_queue)
            self._spike_queue.clear()

        if spikes:
            logger.info(f"[SPIKE] {len(spikes)} Spikes (WebSocket): {', '.join(spikes)}")
        else:
            logger.debug(f"[SPIKE] Keine neuen Spikes seit letztem Scan")

        return spikes
