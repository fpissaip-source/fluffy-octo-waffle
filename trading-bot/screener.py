"""
screener.py — Dynamischer Markt-Screener

Scannt den gesamten US-Aktienmarkt nach:
- Volume-Spikes (aktuelles Vol >> Durchschnitt)
- Starken Preisbewegungen (% Change)
- Penny Stocks & Sub-Penny Stocks (hohes Chance/Risiko-Potenzial)

Liefert eine dynamische Watchlist mit den heissesten Titeln.
"""

import logging
import time
from typing import Optional

logger = logging.getLogger("bot.screener")

# Batch-Groesse fuer Snapshot-API-Calls
BATCH_SIZE = 200


class DynamicScreener:
    """
    Scannt den Gesamtmarkt und gibt die Top-Mover zurueck.

    Kriterien (alle konfigurierbar):
    - Mindest-Volumen (vermeidet illiquide Titel ohne Bewegung)
    - Mindest-Preisbewegung in % (heutige Veraenderung)
    - Preis-Bereich: alle (inkl. Penny/Sub-Penny) oder gefiltert
    - Volume-Spike-Ratio (aktuell vs. Durchschnitt)
    """

    def __init__(self, broker):
        self.broker = broker
        self._symbol_cache: list[str] = []
        self._cache_ts: float = 0
        self._cache_ttl: int = 3600  # Symbolliste 1h cachen

    def _get_all_symbols(self) -> list[str]:
        """Holt alle aktiven, handelbaren US-Aktien von Alpaca."""
        now = time.time()
        if self._symbol_cache and (now - self._cache_ts) < self._cache_ttl:
            return self._symbol_cache

        try:
            assets = self.broker.api.list_assets(
                status="active",
                asset_class="us_equity",
            )
            symbols = [
                a.symbol for a in assets
                if a.tradable
                and a.exchange in ("NYSE", "NASDAQ", "ARCA", "BATS", "OTC")
                and "." not in a.symbol  # Keine Warrants / Sonderklassen
                and len(a.symbol) <= 5    # Keine langen OTC-Kuerzel
            ]
            self._symbol_cache = symbols
            self._cache_ts = now
            logger.info(f"Screener: {len(symbols)} handelbare Symbole geladen")
            return symbols
        except Exception as e:
            logger.error(f"Symbol-Liste laden fehlgeschlagen: {e}")
            return self._symbol_cache or []

    def _get_snapshots_batched(self, symbols: list[str]) -> dict:
        """Holt Snapshots in Batches um API-Limits zu umgehen."""
        all_snaps = {}
        for i in range(0, len(symbols), BATCH_SIZE):
            batch = symbols[i: i + BATCH_SIZE]
            try:
                snaps = self.broker.api.get_snapshots(batch)
                all_snaps.update(snaps)
            except Exception as e:
                logger.warning(f"Snapshot-Batch {i}-{i+BATCH_SIZE} fehlgeschlagen: {e}")
            time.sleep(0.1)  # Rate-Limit schonen
        return all_snaps

    def screen(
        self,
        max_results: int = 50,
        min_volume: int = 50_000,          # Mindest-Tagesvolumen
        min_price_change_pct: float = 2.0,  # Mindest-Bewegung in %
        min_price: float = 0.001,          # Auch Sub-Penny erlaubt
        max_price: Optional[float] = None, # None = kein Limit
        min_volume_ratio: float = 1.5,     # Vol muss X-fach ueber Avg sein
    ) -> list[dict]:
        """
        Scannt den Markt und gibt sortierte Top-Mover zurueck.

        Returns: Liste von dicts mit symbol, price, change_pct, volume, volume_ratio
        """
        symbols = self._get_all_symbols()
        if not symbols:
            return []

        logger.info(f"Screener: Scanne {len(symbols)} Symbole...")
        snapshots = self._get_snapshots_batched(symbols)

        candidates = []

        for symbol, snap in snapshots.items():
            try:
                # Preis aus Daily Bar
                daily = snap.daily_bar
                prev = snap.prev_daily_bar
                if not daily or not prev:
                    continue

                price = float(daily.close)
                prev_close = float(prev.close)
                volume = int(daily.volume)

                if price < min_price:
                    continue
                if max_price and price > max_price:
                    continue
                if volume < min_volume:
                    continue

                # Preisveraenderung heute
                if prev_close <= 0:
                    continue
                change_pct = ((price - prev_close) / prev_close) * 100

                if abs(change_pct) < min_price_change_pct:
                    continue

                # Volume-Ratio (nur wenn prev_volume vorhanden)
                prev_volume = int(prev.volume) if prev.volume else 0
                vol_ratio = (volume / prev_volume) if prev_volume > 0 else 1.0

                if vol_ratio < min_volume_ratio:
                    continue

                candidates.append({
                    "symbol": symbol,
                    "price": round(price, 4),
                    "change_pct": round(change_pct, 2),
                    "volume": volume,
                    "volume_ratio": round(vol_ratio, 2),
                    "prev_close": round(prev_close, 4),
                    "category": _categorize(price),
                })

            except Exception:
                continue

        # Sortieren: Volume-Ratio * abs(change) als Score
        candidates.sort(
            key=lambda x: x["volume_ratio"] * abs(x["change_pct"]),
            reverse=True,
        )

        result = candidates[:max_results]
        logger.info(
            f"Screener: {len(candidates)} Kandidaten gefunden, "
            f"Top {len(result)} ausgewaehlt"
        )
        return result

    def get_watchlist(
        self,
        max_results: int = 50,
        min_volume: int = 50_000,
        min_price_change_pct: float = 2.0,
        min_price: float = 0.001,
        max_price: Optional[float] = None,
        min_volume_ratio: float = 1.5,
    ) -> list[str]:
        """Gibt nur die Symbol-Liste zurueck (fuer Engine-Integration)."""
        hits = self.screen(
            max_results=max_results,
            min_volume=min_volume,
            min_price_change_pct=min_price_change_pct,
            min_price=min_price,
            max_price=max_price,
            min_volume_ratio=min_volume_ratio,
        )
        return [h["symbol"] for h in hits]

    def format_summary(self, hits: list[dict], limit: int = 15) -> str:
        """Formatiert Top-Mover fuer Telegram-Ausgabe."""
        if not hits:
            return "Keine Mover gefunden."

        lines = [f"<b>TOP MOVER ({len(hits)} gefunden)</b>\n━━━━━━━━━━━━━━━━━━━━━━"]
        for h in hits[:limit]:
            arrow = "+" if h["change_pct"] > 0 else ""
            lines.append(
                f"<b>{h['symbol']}</b> ${h['price']:.4f}  "
                f"<code>{arrow}{h['change_pct']:.1f}%</code>  "
                f"Vol: {h['volume']:,}x  ({h['category']})"
            )
        return "\n".join(lines)


def _categorize(price: float) -> str:
    if price < 0.001:
        return "sub-penny"
    elif price < 0.01:
        return "micro-penny"
    elif price < 1.0:
        return "penny"
    elif price < 5.0:
        return "low-cap"
    else:
        return "normal"
