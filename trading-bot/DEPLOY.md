# DEPLOYMENT GUIDE
## Bot auf Cloud-Server + iPhone-Steuerung via Telegram

Dieses Guide bringt dich von Null zum laufenden Bot.
Geschaetzte Zeit: 30-45 Minuten.

---

## SCHRITT 1: Telegram Bot erstellen (5 Min)

Alles auf deinem iPhone in der Telegram App:

### 1.1 Bot erstellen
1. Oeffne Telegram
2. Suche nach **@BotFather**
3. Sende `/newbot`
4. Name eingeben: z.B. `Mein Trading Bot`
5. Username eingeben: z.B. `mein_trading_123_bot` (muss auf `_bot` enden)
6. **BotFather gibt dir einen Token** — kopiere ihn! Sieht so aus:
   ```
   7123456789:AAHxxx-xxxxxxxxxxxxxxxxxxxxx
   ```

### 1.2 Deine Chat-ID finden
1. Sende eine beliebige Nachricht an deinen neuen Bot
2. Oeffne diese URL im Browser (Token einsetzen):
   ```
   https://api.telegram.org/bot<DEIN_TOKEN>/getUpdates
   ```
3. Suche nach `"chat":{"id":` — die Zahl dahinter ist deine **Chat-ID**
   ```
   z.B. 987654321
   ```

**Speichere beides:**
- Token: `7123456789:AAHxxx...`
- Chat-ID: `987654321`

---

## SCHRITT 2: Alpaca Account (5 Min)

1. Gehe zu [https://app.alpaca.markets/signup](https://app.alpaca.markets/signup)
2. Account erstellen (kostenlos)
3. Gehe zu **Paper Trading** → **API Keys**
4. Klicke **Generate New Key**
5. **Kopiere Key + Secret** (Secret wird nur einmal gezeigt!)

---

## SCHRITT 3: Cloud-Server einrichten (15 Min)

### Option A: Oracle Cloud — KOSTENLOS (empfohlen)

1. Gehe zu [https://cloud.oracle.com/](https://cloud.oracle.com/)
2. Account erstellen (Kreditkarte noetig, wird NICHT belastet)
3. Erstelle eine **VM Instance**:
   - Shape: `VM.Standard.E2.1.Micro` (Always Free)
   - Image: **Ubuntu 22.04**
   - SSH Key: Generiere einen und lade den Private Key herunter
4. Notiere die **Public IP** deiner VM

### Option B: Hetzner — 4€/Monat

1. Gehe zu [https://www.hetzner.com/cloud](https://www.hetzner.com/cloud)
2. Server erstellen: **CX11** (2GB RAM), Ubuntu 22.04
3. SSH Key hinterlegen
4. Notiere die IP

### Option C: Railway — Einfachste Option

1. Gehe zu [https://railway.app/](https://railway.app/)
2. GitHub Account verbinden
3. Neues Projekt → Deploy from GitHub Repo
4. Environment Variables dort eintragen
5. Fertig — kein SSH noetig

---

## SCHRITT 4: Server einrichten (SSH)

Von deinem Computer (Terminal/PowerShell):

```bash
# Mit Server verbinden
ssh ubuntu@DEINE_SERVER_IP

# System updaten
sudo apt update && sudo apt upgrade -y

# Python + Tools installieren
sudo apt install -y python3 python3-pip python3-venv git

# Projekt-Ordner erstellen
mkdir -p ~/trading-bot
cd ~/trading-bot
```

### Dateien hochladen

**Option A: ZIP hochladen**
```bash
# Von deinem Computer:
scp trading-bot.zip ubuntu@DEINE_SERVER_IP:~/

# Auf dem Server:
cd ~
unzip trading-bot.zip
cd trading-bot
```

**Option B: Git (wenn du ein Repo hast)**
```bash
git clone https://github.com/DEIN_USER/trading-bot.git
cd trading-bot
```

### Dependencies installieren

```bash
# Virtual Environment
python3 -m venv venv
source venv/bin/activate

# Packages installieren
pip install -r requirements.txt
```

### .env Datei erstellen

```bash
cp .env.example .env
nano .env
```

Trage ein:
```
ALPACA_API_KEY=PK...dein_key...
ALPACA_SECRET_KEY=...dein_secret...
ALPACA_BASE_URL=https://paper-api.alpaca.markets

WATCHLIST=AAPL,TSLA,NVDA,AMD,META

MAX_POSITION_PCT=0.10
KELLY_FRACTION=0.25
MIN_EV_GAP=0.02
SCAN_INTERVAL=30
LOG_LEVEL=INFO

TELEGRAM_TOKEN=7123456789:AAHxxx...
TELEGRAM_CHAT_ID=987654321
```

Speichern: `Ctrl+O` → `Enter` → `Ctrl+X`

### Testen

```bash
# Verbindung testen
python main.py --status

# Quick-Analyse
python main.py --backtest

# Telegram-Bot testen (kurz laufen lassen, dann Ctrl+C)
python main.py --telegram
```

Gehe jetzt in Telegram und sende `/start` an deinen Bot.
Wenn er antwortet → alles funktioniert!

---

## SCHRITT 5: Bot dauerhaft laufen lassen

### Mit systemd (empfohlen — startet auch nach Reboot)

```bash
# Service-Datei erstellen
sudo nano /etc/systemd/system/trading-bot.service
```

Inhalt:
```ini
[Unit]
Description=Six Filters Trading Bot
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/trading-bot
Environment=PATH=/home/ubuntu/trading-bot/venv/bin:$PATH
ExecStart=/home/ubuntu/trading-bot/venv/bin/python main.py --telegram
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Speichern, dann:
```bash
# Service aktivieren + starten
sudo systemctl daemon-reload
sudo systemctl enable trading-bot
sudo systemctl start trading-bot

# Status pruefen
sudo systemctl status trading-bot

# Logs anschauen
sudo journalctl -u trading-bot -f
```

### Nuetzliche Befehle

```bash
# Bot stoppen
sudo systemctl stop trading-bot

# Bot neustarten
sudo systemctl restart trading-bot

# Logs der letzten Stunde
sudo journalctl -u trading-bot --since "1 hour ago"
```

---

## SCHRITT 6: Vom iPhone steuern

Oeffne Telegram auf deinem iPhone. Dein Bot versteht:

| Befehl | Was passiert |
|--------|-------------|
| `/start` | Uebersicht aller Befehle |
| `/status` | Equity, Cash, Positionen |
| `/scan` | Einmal alle Symbole analysieren |
| `/positions` | Offene Positionen mit P/L |
| `/trades` | Heutige Trades anzeigen |
| `/watchlist` | Aktuelle Watchlist |
| `/add TSLA` | TSLA zur Watchlist |
| `/remove TSLA` | TSLA entfernen |
| `/run` | Auto-Scan starten |
| `/pause` | Bot pausieren |
| `/resume` | Bot fortsetzen |
| `/stop` | Auto-Scan stoppen |

### Was der Bot automatisch tut:

- Scannt alle 30s die Watchlist
- Sendet dir eine Nachricht wenn alle 6 Filter bestehen
- Fuehrt den Trade automatisch aus
- Warnt dich wenn eine Position nahe am Stop Loss ist
- Stop Loss (-3%) und Take Profit (+5%) laufen automatisch

---

## TROUBLESHOOTING

**Bot antwortet nicht in Telegram?**
- Prüfe Token und Chat-ID in .env
- Prüfe ob Service läuft: `sudo systemctl status trading-bot`

**"Market closed" Meldung?**
- US-Markt ist nur 15:30-22:00 Uhr (DE Zeit) offen
- Der Bot wartet automatisch

**"No bars" Warnung?**
- Ausserhalb der Handelszeiten gibt es keine Daten
- Warte bis der Markt oeffnet

**Server-Verbindung verloren?**
- Der systemd Service startet den Bot automatisch neu

**Bot updaten?**
```bash
# Dateien hochladen, dann:
sudo systemctl restart trading-bot
```

---

## SICHERHEIT

- Nutze IMMER zuerst Paper Trading
- Setze `MAX_POSITION_PCT` nicht hoeher als 0.10 (10%)
- Beobachte den Bot mindestens 2 Wochen im Paper-Modus
- Der Bot tradet NUR waehrend US-Marktzeiten
- Halte deine .env Datei geheim (nie committen!)
