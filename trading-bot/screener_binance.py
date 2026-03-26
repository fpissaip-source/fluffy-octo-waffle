"""
screener_binance.py — Binance Echtzeit Crypto Spike-Scanner.

WebSocket-Modus: Abonniert alle Mini-Ticker via Binance !miniTicker@arr Stream.
  → Spikes werden in Echtzeit erkannt (24/7, kein Markt-Schluss).
  → Drastisch weniger API-Calls als REST-Polling.

Gleiche Schnittstelle wie SpikeSensor — Engine bleibt kompatibel.
"""

import logging
import threading
import time
from typing import TYPE_CHECKING

from config import Config

if TYPE_CHECKING:
    from broker_binance import BinanceBroker

logger = logging.getLogger("bot.screener")


class BinanceSpikeSensor:
    """
    Echtzeit Spike-Erkennung via Binance WebSocket (All Market Mini Tickers).
    Universum: ~80 volatile Crypto-Paare auf Binance.
    Trigger: >2% 24h-Bewegung (niedriger als Aktien wegen Crypto-Volatilität).

    Interface identisch zu SpikeSensor:
      sensor.scan() → list[str] der erkannten Spike-Symbole
    """

    UNIVERSE = [
        # ── Major ─────────────────────────────────────────────
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
        "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
        # ── Layer 2 / Scaling ──────────────────────────────────
        "MATICUSDT", "ARBUSDT", "OPUSDT", "STRKUSDT", "MANAUSDT",
        # ── DeFi ──────────────────────────────────────────────
        "UNIUSDT", "AAVEUSDT", "MKRUSDT", "CRVUSDT", "COMPUSDT",
        "LDOUSDT", "DYDXUSDT", "GMXUSDT", "SNXUSDT", "1INCHUSDT",
        # ── AI / Infrastruktur ─────────────────────────────────
        "FETUSDT", "AGIXUSDT", "RNDRUSDT", "WLDUSDT", "OCEANUSDT",
        "TAONETUSDT", "ARUSDT",
        # ── Neue Layer 1 ───────────────────────────────────────
        "INJUSDT", "SUIUSDT", "APTUSDT", "SEIUSDT", "TIAUSDT",
        "PYTHUSDT", "JUPUSDT", "WIFUSDT", "NEARUSDT", "ICPUSDT",
        # ── Mid Cap Volatile ───────────────────────────────────
        "ATOMUSDT", "FTMUSDT", "ALGOUSDT", "EGLDUSDT", "HBARUSDT",
        "VETUSDT", "ZILUSDT", "ONEUSDT", "IOSTUSDT", "IOTXUSDT",
        # ── Gaming / NFT / Metaverse ───────────────────────────
        "SANDUSDT", "AXSUSDT", "GALAUSDT", "ENJUSDT", "IMXUSDT",
        "FLOWUSDT", "CHZUSDT", "HIGHUSDT",
        # ── Meme / Hochvolatil ─────────────────────────────────
        "DOGEUSDT", "SHIBUSDT", "PEPEUSDT", "BONKUSDT", "FLOKIUSDT",
        "MEMEUSDT", "WIFUSDT", "BOMEUSDT", "TURBOUSDT",
        # ── Crypto-Infrastruktur ───────────────────────────────
        "LPTUSDT", "STXUSDT", "RUNEUSDT", "THORUSDT", "CELOUSDT",
        "KASUSDT", "BEAMUSDT", "ALTUSDT",
    ]

    def __init__(self, broker: "BinanceBroker"):
        self.broker = broker
        self.min_pct = getattr(Config, "SPIKE_MIN_PCT", 0.02)
        self._spike_queue: list[str] = []
        self._lock = threading.Lock()
        self._ws_active = False
        self._twm = None

        logger.info(f"[SPIKE] BinanceSpikeSensor init — Universum: {len(self.UNIVERSE)} Symbole")
        self._start_websocket()

    # ── WebSocket ────────────────────────────────────────

    def _start_websocket(self):
        t = threading.Thread(
            target=self._run_websocket_loop,
            daemon=True,
            name="spike-sensor-ws",
        )
        t.start()

    def _run_websocket_loop(self):
        """Endlos-Loop mit automatischem Reconnect."""
        backoff = 5
        while True:
            try:
                self._connect_websocket()
                backoff = 5
            except ImportError:
                logger.warning("[SPIKE] python-binance nicht installiert — WebSocket deaktiviert")
                return
            except Exception as e:
                logger.warning(f"[SPIKE] WebSocket Fehler: {e} — Reconnect in {backoff}s")
                self._ws_active = False
                if self._twm:
                    try:
                        self._twm.stop()
                    except Exception:
                        pass
                    self._twm = None
                time.sleep(backoff)
                backoff = min(backoff * 2, 120)

    def _connect_websocket(self):
        """Verbindet mit Binance WebSocket — All Market Mini Tickers."""
        from binance import ThreadedWebsocketManager

        universe_set = set(self.UNIVERSE)

        twm = ThreadedWebsocketManager(
            api_key=Config.BINANCE_API_KEY,
            api_secret=Config.BINANCE_SECRET_KEY,
            testnet=getattr(Config, "BINANCE_TESTNET", True),
        )
        twm.start()
        self._twm = twm

        def on_message(msg):
            try:
                # !miniTicker@arr liefert eine Liste
                data = msg if isinstance(msg, list) else [msg]
                for ticker in data:
                    sym = ticker.get("s", "")
                    if sym not in universe_set:
                        continue
                    close = float(ticker.get("c", 0))
                    open_p = float(ticker.get("o", 0))
                    if close <= 0 or open_p <= 0:
                        continue
                    pct_move = (close - open_p) / open_p
                    if abs(pct_move) >= self.min_pct:
                        direction = "+" if pct_move >= 0 else ""
                        logger.info(
                            f"[SPIKE] {sym}: {direction}{pct_move:.1%} (24h)"
                            f" | ${close:.4f}"
                        )
                        with self._lock:
                            if sym not in self._spike_queue:
                                self._spike_queue.append(sym)
            except Exception as e:
                logger.debug(f"[SPIKE] Ticker-Handler Fehler: {e}")

        self._ws_active = True
        logger.info("[SPIKE] Binance WebSocket verbunden — !miniTicker@arr Stream aktiv")
        twm.start_all_market_mini_tickers_socket(callback=on_message)
        twm.join()  # Blockiert bis Verbindung abbricht

    # ── Öffentliches Interface ───────────────────────────

    def should_scan(self) -> bool:
        """Immer True — Crypto handelt 24/7."""
        return True

    def scan(self) -> list[str]:
        """Gibt erkannte Spikes zurück und leert die Queue."""
        if not self._ws_active:
            logger.debug("[SPIKE] WebSocket noch nicht verbunden — überspringe Scan")
            return []

        with self._lock:
            spikes = list(self._spike_queue)
            self._spike_queue.clear()

        if spikes:
            logger.info(f"[SPIKE] {len(spikes)} Crypto-Spikes: {', '.join(spikes)}")
        return spikes
