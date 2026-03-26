"""config.py — Zentrale Konfiguration (Binance Crypto)."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")


class Config:
    # ── Binance API (Hauptbörse) ──
    BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY", "")
    BINANCE_SECRET_KEY: str = os.getenv("BINANCE_SECRET_KEY", "")
    # True = Binance Testnet (Paper Trading), False = Live
    BINANCE_TESTNET: bool = os.getenv("BINANCE_TESTNET", "true").lower() in ("1", "true", "yes")

    # ── Alpaca (Legacy — wird nicht mehr verwendet) ──
    API_KEY: str = os.getenv("ALPACA_API_KEY", "")
    SECRET_KEY: str = os.getenv("ALPACA_SECRET_KEY", "")
    BASE_URL: str = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

    # ── Crypto Watchlist (Binance USDT-Paare) ──
    WATCHLIST: list[str] = [
        s.strip() for s in os.getenv(
            "WATCHLIST",
            "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,AVAXUSDT,LINKUSDT,INJUSDT,SUIUSDT"
        ).split(",")
    ]

    MAX_POSITION_PCT: float = float(os.getenv("MAX_POSITION_PCT", "0.10"))
    KELLY_FRACTION: float = float(os.getenv("KELLY_FRACTION", "0.25"))
    MIN_EV_GAP: float = float(os.getenv("MIN_EV_GAP", "0.01"))
    MIN_MOMENTUM_SCORE: float = float(os.getenv("MIN_MOMENTUM_SCORE", "0.50"))
    KL_DIVERGENCE_THRESHOLD: float = 0.15
    MIN_BAYESIAN_POSTERIOR: float = 0.60
    STOIKOV_SPREAD_MULT: float = 1.5

    SCAN_INTERVAL: int = int(os.getenv("SCAN_INTERVAL", "20"))   # Kürzer als Aktien (Crypto bewegt sich schneller)
    LOOKBACK_BARS: int = 200
    SHORT_WINDOW: int = 10
    LONG_WINDOW: int = 30

    # ── Chart-Timeframe ──
    TRADING_TIMEFRAME: str = os.getenv("TRADING_TIMEFRAME", "15Min")

    # ── News / Sentiment Horizont ──
    NEWS_LOOKBACK_HOURS: int = int(os.getenv("NEWS_LOOKBACK_HOURS", "48"))
    NEWS_MAX_ARTICLES:   int = int(os.getenv("NEWS_MAX_ARTICLES",   "30"))
    SENTIMENT_GEMINI_HEADLINES: int = 25

    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # ── Dry Run ──
    DRY_RUN: bool = os.getenv("DRY_RUN", "true").lower() in ("1", "true", "yes")  # Standard: True für Crypto

    # ── Telegram ──
    TELEGRAM_TOKEN: str = os.getenv("TELEGRAM_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

    # ── Google Gemini API (REQUIRED) ──
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    # ── LunarCrush (optional — Crypto Sentiment) ──
    LUNARCRUSH_API_KEY: str = os.getenv("LUNARCRUSH_API_KEY", "")

    # ── Finnhub (optional — für Crypto nicht relevant, aber kompatibel) ──
    FINNHUB_API_KEY: str = os.getenv("FINNHUB_API_KEY", "")
    FINNHUB_EARNINGS_BLOCK_DAYS: int = 0   # Für Crypto deaktiviert

    # ── Reasoning Layer ──
    REASONING_MODEL: str = "gemini-2.5-flash"
    REASONING_MIN_CONFIDENCE: float = 0.55
    REASONING_TIMEOUT: int = 20

    # ── Slippage & Fee Model (Binance Maker/Taker) ──
    SLIPPAGE_BPS: float = float(os.getenv("SLIPPAGE_BPS", "5"))   # 5 bps = 0.05% (Binance Taker)
    FEE_BPS: float = float(os.getenv("FEE_BPS", "10"))            # 10 bps = 0.10% Binance Taker Fee

    # ── Spike-Sensor ──
    SPIKE_SCAN_INTERVAL: int = 60
    SPIKE_MIN_PCT: float = 0.02   # 2% für Crypto (war 3% für Aktien)

    @classmethod
    def is_paper(cls) -> bool:
        return cls.BINANCE_TESTNET

    @classmethod
    def validate(cls) -> bool:
        if not cls.BINANCE_API_KEY or cls.BINANCE_API_KEY == "your_binance_api_key_here":
            return False
        if not cls.BINANCE_SECRET_KEY or cls.BINANCE_SECRET_KEY == "your_binance_secret_key_here":
            return False
        if not cls.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY fehlt in .env — Reasoning Layer nicht verfuegbar. Bot gestoppt.")
        return True
