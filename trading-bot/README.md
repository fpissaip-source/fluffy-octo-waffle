# SIX FILTERS. ONE TRADE.

**Quantitative Trading Engine für Alpaca (US-Aktien)**

Ein modularer Trading-Bot der 6 quantitative Formeln parallel ausführt.
Nur wenn **alle 6 Filter** bestehen, wird ein Trade ausgeführt.

---

## Architektur

```
main.py              ← Einstiegspunkt (CLI)
├── config.py        ← Konfiguration aus .env
├── broker.py        ← Alpaca API Wrapper (Daten + Orders)
├── engine.py        ← Orchestriert alle 6 Formeln
│
└── formulas/
    ├── momentum.py      ← F1: Multi-Timeframe Momentum (RSI, MACD, MA)
    ├── kelly.py         ← F2: Kelly Criterion (Position Sizing)
    ├── ev_gap.py        ← F3: Expected Value Gap Detection
    ├── kl_divergence.py ← F4: KL-Divergenz (Cross-Timeframe)
    ├── bayesian.py      ← F5: Bayesian Updates (Signal-Fusion)
    └── stoikov.py       ← F6: Avellaneda-Stoikov Execution
```

---

## Die 6 Formeln

### F1 — Momentum Score (ersetzt LMSR)
LMSR ist spezifisch für Prediction Markets. Für Aktien nutzen wir
Multi-Timeframe Momentum: RSI + MACD Crossover + Moving Average Position.
Erkennt Trends bevor sie voll eingepreist sind.

### F2 — Kelly Criterion
Berechnet die mathematisch optimale Positionsgröße.
Nutzt Quarter-Kelly (25%) für konservatives Sizing.
Nie zu groß, nie zu klein.

### F3 — EV Gap Detection
Vergleicht den aktuellen Preis mit dem geschätzten Fair Value
(VWAP + Bollinger + EMA). Sucht Gaps > 2% wo der Expected Value positiv ist.

### F4 — KL-Divergenz
Misst die statistische Distanz zwischen kurzfristigen (10 Bars)
und langfristigen (50 Bars) Return-Verteilungen.
Hohe Divergenz = Zeitrahmen aus dem Gleichgewicht = Opportunity.

### F5 — Bayesian Updates
Startet mit Prior 50% und aktualisiert basierend auf:
Volume-Spikes, Preis-Beschleunigung, Volatilitäts-Regime, Trend-Konsistenz.
Posterior muss > 60% sein.

### F6 — Stoikov Execution
Berechnet den Reservation Price (Avellaneda-Stoikov Modell).
Der Bot kauft nur wenn der aktuelle Preis ≤ Reservation Price.
Kein Chasing, kein FOMO.

---

## Setup

### 1. Alpaca Account erstellen

1. Gehe zu [https://app.alpaca.markets/](https://app.alpaca.markets/)
2. Erstelle einen **Paper Trading** Account (kostenlos)
3. Gehe zu "API Keys" und generiere ein Key-Paar

### 2. Installation

```bash
cd trading-bot

# (Optional) Virtual Environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# oder: venv\Scripts\activate  # Windows

# Dependencies installieren
pip install -r requirements.txt
```

### 3. Konfiguration

```bash
# .env Datei erstellen
cp .env.example .env

# Keys eintragen
nano .env  # oder mit einem Editor deiner Wahl
```

Trage deine Alpaca API Keys ein:
```
ALPACA_API_KEY=PK...
ALPACA_SECRET_KEY=...
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

### 4. Bot starten

```bash
# Account Status prüfen
python main.py --status

# Einmaliger Scan (gut zum Testen)
python main.py --scan-once

# Quick-Analyse aller Watchlist-Symbole
python main.py --backtest

# Bot starten (Endlosschleife)
python main.py
```

---

## Konfiguration (.env)

| Variable           | Default     | Beschreibung                          |
|-------------------|-------------|---------------------------------------|
| `WATCHLIST`       | AAPL,TSLA.. | Komma-getrennte Symbole               |
| `MAX_POSITION_PCT`| 0.10        | Max 10% des Portfolios pro Trade      |
| `KELLY_FRACTION`  | 0.25        | Quarter-Kelly (konservativ)           |
| `MIN_EV_GAP`     | 0.02        | Mindest-EV-Gap (2%)                   |
| `SCAN_INTERVAL`  | 30          | Sekunden zwischen Scans               |
| `LOG_LEVEL`      | INFO        | DEBUG für mehr Details                |

---

## Wichtige Hinweise

**⚠️ STARTE IMMER MIT PAPER TRADING.**

- Dieser Bot ist ein Werkzeug, keine Gelddruckmaschine
- Die Formeln erkennen Muster, aber der Markt ist nicht vorhersagbar
- Vergangene Performance sagt nichts über zukünftige Ergebnisse
- Quarter-Kelly und der 10%-Position-Cap sind bewusst konservativ
- Teste ausgiebig mit Paper Trading bevor du echtes Geld einsetzt
- Der Bot handelt nur während US-Marktzeiten (9:30-16:00 ET)

---

## Erweitern

### Eigene Formel hinzufügen

Erstelle eine neue Datei in `formulas/` mit dieser Struktur:

```python
def evaluate(bars, **kwargs):
    return {
        "name": "MeineFormel",
        "signal": 0.75,       # -1 bis +1
        "passed": True,        # Filter bestanden?
        "details": {...},      # Zusätzliche Infos
    }
```

Dann in `engine.py` importieren und im `analyze_symbol()` aufrufen.

### Alerts hinzufügen (Telegram/Discord)

Ergänze in `engine.py` nach `execute_signal()` einen
Webhook-Call zu Telegram oder Discord.

---

## Lizenz

Privat. Nutzung auf eigenes Risiko.
