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
            "WATCHLIST", "NVDA,META,BTCUSD,ETHUSD"
        ).split(",")
    ]

    MAX_POSITION_PCT: float = float(os.getenv("MAX_POSITION_PCT", "0.10"))
    KELLY_FRACTION: float = float(os.getenv("KELLY_FRACTION", "0.25"))
    MIN_EV_GAP: float = float(os.getenv("MIN_EV_GAP", "0.02"))
    MIN_MOMENTUM_SCORE: float = 0.6
    KL_DIVERGENCE_THRESHOLD: float = 0.15
    MIN_BAYESIAN_POSTERIOR: float = 0.60
    STOIKOV_SPREAD_MULT: float = 1.5

    SCAN_INTERVAL: int = int(os.getenv("SCAN_INTERVAL", "30"))
    LOOKBACK_BARS: int = 100
    SHORT_WINDOW: int = 10
    LONG_WINDOW: int = 30

    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # ── Telegram ──
    TELEGRAM_TOKEN: str = os.getenv("TELEGRAM_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

    # ── Google Gemini API (REQUIRED — Reasoning Layer vor jeder Order) ──
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    # ── LunarCrush (optional — Crypto Sentiment) ──
    LUNARCRUSH_API_KEY: str = os.getenv("LUNARCRUSH_API_KEY", "")

    # ── Reasoning Layer Einstellungen ──
    REASONING_MODEL: str = "gemini-2.5-flash-preview-04-17"
    REASONING_MIN_CONFIDENCE: float = 0.65  # Gemini muss mind. 65% sicher sein
    REASONING_TIMEOUT: int = 20

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
