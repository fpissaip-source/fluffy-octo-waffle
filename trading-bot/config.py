"""config.py — Zentrale Konfiguration."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")


class Config:
    API_KEY: str = os.getenv("ALPACA_API_KEY", "")
    SECRET_KEY: str = os.getenv("ALPACA_SECRET_KEY", "")
    BASE_URL: str = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

    WATCHLIST: list[str] = [
        s.strip() for s in os.getenv(
            "WATCHLIST", "NVDA,META,BTCUSD,ETHUSD,DVLT"
        ).split(",")
    ]

    MAX_POSITION_PCT: float = float(os.getenv("MAX_POSITION_PCT", "0.10"))
    KELLY_FRACTION: float = float(os.getenv("KELLY_FRACTION", "0.25"))
    MIN_EV_GAP: float = float(os.getenv("MIN_EV_GAP", "0.02"))
    MIN_MOMENTUM_SCORE: float = float(os.getenv("MIN_MOMENTUM_SCORE", "0.50"))
    KL_DIVERGENCE_THRESHOLD: float = 0.15
    MIN_BAYESIAN_POSTERIOR: float = 0.60
    STOIKOV_SPREAD_MULT: float = 1.5

    SCAN_INTERVAL: int = int(os.getenv("SCAN_INTERVAL", "30"))
    LOOKBACK_BARS: int = 100
    SHORT_WINDOW: int = 10
    LONG_WINDOW: int = 30

    # ── Chart-Timeframe (Daytrading: "5Min" / Swing: "1Hour") ──
    # Bestimmt: Kerzenbreite für alle Formeln, ATR-Weite der Stops,
    # Gleitende Durchschnitte, EV-Gap, Momentum — alles skaliert mit.
    TRADING_TIMEFRAME: str = os.getenv("TRADING_TIMEFRAME", "1Hour")

    # ── News / Sentiment Horizont ──
    NEWS_LOOKBACK_HOURS: int = int(os.getenv("NEWS_LOOKBACK_HOURS", "72"))   # 3 Tage für Swing-Kontext
    NEWS_MAX_ARTICLES:   int = int(os.getenv("NEWS_MAX_ARTICLES",   "30"))   # Mehr Artikel = breiteres Bild
    SENTIMENT_GEMINI_HEADLINES: int = 25                                      # Gemini bekommt Top 25 Headlines

    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # ── Telegram ──
    TELEGRAM_TOKEN: str = os.getenv("TELEGRAM_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

    # ── Google Gemini API (REQUIRED — Reasoning Layer vor jeder Order) ──
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    # ── LunarCrush (optional — Crypto Sentiment) ──
    LUNARCRUSH_API_KEY: str = os.getenv("LUNARCRUSH_API_KEY", "")

    # ── Finnhub (Earnings Calendar + Insider Trades) ──
    FINNHUB_API_KEY: str = os.getenv("FINNHUB_API_KEY", "")
    FINNHUB_EARNINGS_BLOCK_DAYS: int = 3   # Kein Trade wenn Earnings ≤3 Tage entfernt

    # ── Reasoning Layer Einstellungen ──
    REASONING_MODEL: str = "gemini-2.5-flash"
    REASONING_MIN_CONFIDENCE: float = 0.55  # 55% — weniger restriktiv (war 65%)
    REASONING_TIMEOUT: int = 20

    # ── Slippage & Fee Model ──
    SLIPPAGE_BPS: float = float(os.getenv("SLIPPAGE_BPS", "10"))   # 10 bps = 0.10% Slippage
    FEE_BPS: float = float(os.getenv("FEE_BPS", "1"))              # 1 bps regulatorische Gebuehren

    # ── Spike-Sensor (Echtzeit Markt-Scanner) ──
    SPIKE_SCAN_INTERVAL: int = 60     # Alle 60s breiten Markt scannen
    SPIKE_MIN_PCT: float = 0.03       # 3% intraday-Bewegung = Spike

    @classmethod
    def is_paper(cls) -> bool:
        return "paper" in cls.BASE_URL

    @classmethod
    def validate(cls) -> bool:
        if not cls.API_KEY or cls.API_KEY == "your_paper_api_key_here":
            return False
        if not cls.SECRET_KEY or cls.SECRET_KEY == "your_paper_secret_key_here":
            return False
        if not cls.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY fehlt in .env — Reasoning Layer nicht verfuegbar. Bot gestoppt.")
        return True
