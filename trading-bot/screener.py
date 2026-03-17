"""
screener.py — Echtzeit Markt-Spike-Scanner.

Scannt ~300 Small-Cap / Penny Stocks per Alpaca Batch-Snapshot.
Erkennt Preissprünge (>3% intraday ODER >5x Volumen) in Echtzeit.
Fügt erkannte Symbole sofort zur aktiven Analyse-Queue hinzu.
"""

import logging
import time
from typing import TYPE_CHECKING

from config import Config

if TYPE_CHECKING:
    from broker import AlpacaBroker

logger = logging.getLogger("bot.screener")


class SpikeSensor:
    """
    Scannt den breiten Markt alle 60s auf Preissprünge via Batch-Snapshots.
    Universum: ~300 bekannte volatile Small-Cap / Penny / Biotech Stocks.
    Trigger: >3% intraday-Bewegung ODER Volumen >3x Durchschnitt.
    """

    # Breites Universum: ~300 volatile Small/Micro Caps, Biotech, Meme, EV, Crypto-Proxies
    UNIVERSE = [
        # ── Meme / Reddit ──────────────────────────────────────
        "AMC", "GME", "BBBY", "KOSS", "EXPR", "BB", "NOK", "SPCE",
        # ── Biotech / Pharma ───────────────────────────────────
        "OCGN", "NVAX", "INO", "SRNE", "ATOS", "VXRT", "IOVA", "SAVA",
        "CRSP", "EDIT", "NTLA", "BEAM", "VERV", "HOOK", "FREQ", "AMRN",
        "MNKD", "ADMA", "OTIC", "NAVB", "CPRX", "PRGO", "IRWD", "ENDP",
        "DARE", "CLRB", "TTOO", "INPX", "CLPS", "KOPN", "VUZI",
        "AGEN", "SURF", "PHUN", "PRME", "SUPN", "NKTR", "ACAD",
        # ── EV / Wasserstoff / Clean Energy ───────────────────
        "NKLA", "GOEV", "RIDE", "WKHS", "AYRO", "HYLN", "FSR",
        "PLUG", "FCEL", "BLNK", "CHPT", "QS", "LCID", "RIVN",
        "SPWR", "SUNW", "NOVA", "RUN", "ARRY", "STEM",
        # ── Crypto-Proxies ─────────────────────────────────────
        "MARA", "RIOT", "BTBT", "SOS", "NCTY", "MOGO", "EBON",
        "CAN", "HUT", "CIFR", "BTDR", "CORZ",
        # ── Cannabis ───────────────────────────────────────────
        "SNDL", "APHA", "TLRY", "ACB", "CGC", "CRON", "HEXO",
        "OGI", "GRWG", "IIPR",
        # ── Fintech / Meme Finance ─────────────────────────────
        "SOFI", "UPST", "AFRM", "HOOD", "COIN", "OPEN", "DKNG",
        "PENN", "SKLZ", "RBLX", "CLOV", "WISH",
        # ── Tech Small Caps ────────────────────────────────────
        "MVIS", "IDEX", "HIMS", "NNDM", "PLBY", "GNUS", "PRED",
        "SOLO", "XELA", "CTRM", "SHIP", "FREE", "IMPP", "ZOM",
        "PHGE", "CIDM", "CELZ", "IMVT", "ALDX", "RETA", "NVAX",
        # ── Energy / Oil Micro ─────────────────────────────────
        "BORR", "TELL", "NEXT", "HLTH", "AMTX", "REI", "IMPP",
        # ── SPACs / Recent IPOs ────────────────────────────────
        "PSFE", "SKLZ", "VIEW", "VELO", "GRNV",
        # ── Volatile Mid-Caps mit Momentum ────────────────────
        "RKT", "OPEN", "OFFERPAD", "IRBT", "VZIO", "ANGI",
        "LMND", "ROOT", "HIPPO", "MILE", "METROMILE",
        # ── Short-Squeeze Kandidaten ───────────────────────────
        "PRTY", "VSTO", "SCVL", "PAYA", "OLBG", "HRTX", "ACET",
        "NUVB", "AEYE", "SKIN", "BKSY", "SPIR", "OSAT",
        # ── Biotech mit nahen Katalysatoren ───────────────────
        "FATE", "RGEN", "ARWR", "PTGX", "YMAB", "IMCR", "TGTX",
        "KRTX", "RXDX", "ARVN", "KDNY", "RCKT", "ALLO",
        "CABA", "MGTX", "RAPT", "NRIX", "PRAX", "FOLD",
        # ── OTC / Micro Cap (handelbar via Alpaca) ─────────────
        "TNXP", "JAGX", "GFAI", "MULN", "BBIG", "PHUN",
        "NXGL", "LIQT", "LGVN", "TPVG", "DPSI",
    ]

    def __init__(self, broker: "AlpacaBroker"):
        self.broker = broker
        self.min_pct = getattr(Config, "SPIKE_MIN_PCT", 0.03)   # 3% intraday
        self.min_vol_mult = 3.0                                   # 3x avg volume
        self.last_scan = 0
        self.scan_interval = getattr(Config, "SPIKE_SCAN_INTERVAL", 60)
        self._avg_volumes: dict[str, float] = {}                  # rolling avg
        logger.info(f"[SPIKE] SpikeSensor init — Universum: {len(self.UNIVERSE)} Symbole")

    def should_scan(self) -> bool:
        return time.time() - self.last_scan >= self.scan_interval

    def scan(self) -> list[str]:
        """
        Batch-Snapshot aller Symbole → filtert Spikes.
        Gibt Liste der Symbole zurück die sofort analysiert werden sollen.
        """
        if not self.should_scan():
            return []

        self.last_scan = time.time()
        spikes: list[str] = []

        # Batches à 200 (Alpaca-Limit pro Request)
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

                    # Intraday % Bewegung (von Open)
                    pct_move = (close_p - open_p) / open_p

                    # Volumen-Multiplikator (gegen Rolling Average)
                    avg_vol = self._avg_volumes.get(symbol, volume)
                    vol_mult = volume / avg_vol if avg_vol > 0 else 1.0
                    # Rolling Average updaten (EMA)
                    self._avg_volumes[symbol] = avg_vol * 0.9 + volume * 0.1

                    # Spike erkannt?
                    is_spike = abs(pct_move) >= self.min_pct or vol_mult >= self.min_vol_mult

                    if is_spike:
                        direction = "+" if pct_move >= 0 else ""
                        logger.info(
                            f"[SPIKE] 🚀 {symbol}: {direction}{pct_move:.1%} intraday "
                            f"| Vol: {vol_mult:.1f}x | Preis: ${close_p:.3f}"
                        )
                        spikes.append(symbol)

                except Exception as e:
                    logger.debug(f"[SPIKE] {symbol} snapshot parse error: {e}")

        if spikes:
            logger.info(f"[SPIKE] {len(spikes)} Spikes gefunden: {', '.join(spikes)}")
        else:
            logger.debug(f"[SPIKE] Kein Spike im Universum ({len(self.UNIVERSE)} Symbole)")

        return spikes
