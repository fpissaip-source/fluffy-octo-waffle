import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean,
    DateTime, JSON, Text, func
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from config import Config

logger = logging.getLogger("bot.database")

Base = declarative_base()


class TradeRecord(Base):
    __tablename__ = "bot_trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False, index=True)
    regime = Column(String(20), nullable=False)
    action = Column(String(10), nullable=False)
    qty = Column(Integer, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    pnl = Column(Float, nullable=True)
    pnl_pct = Column(Float, nullable=True)
    formula_scores = Column(JSON, nullable=True)
    sentiment_score = Column(Float, default=0.0)
    entry_time = Column(DateTime, default=func.now(), nullable=False)
    exit_time = Column(DateTime, nullable=True)
    exit_reason = Column(String(200), nullable=True)
    weighted_score = Column(Float, nullable=True)


class PositionSnapshot(Base):
    __tablename__ = "bot_positions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False, index=True)
    qty = Column(Float, nullable=False)
    avg_entry = Column(Float, nullable=False)
    market_value = Column(Float, nullable=False)
    unrealized_pl = Column(Float, nullable=False)
    unrealized_plpc = Column(Float, nullable=False)
    side = Column(String(10), nullable=False)
    timestamp = Column(DateTime, default=func.now(), nullable=False)


class BotStatus(Base):
    __tablename__ = "bot_status"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=func.now(), nullable=False)
    equity = Column(Float, nullable=False)
    cash = Column(Float, nullable=True)
    regime = Column(String(20), nullable=True)
    is_running = Column(Boolean, default=False)
    is_paused = Column(Boolean, default=False)
    open_positions = Column(Integer, default=0)
    last_scan = Column(DateTime, nullable=True)
    kill_switch_active = Column(Boolean, default=False)


class FormulaWeight(Base):
    __tablename__ = "bot_formula_weights"

    id = Column(Integer, primary_key=True, autoincrement=True)
    regime = Column(String(20), nullable=False, index=True)
    formula_name = Column(String(50), nullable=False)
    weight = Column(Float, nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class BacktestResult(Base):
    __tablename__ = "bot_backtest_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False, index=True)
    timeframe = Column(String(20), nullable=False)
    start_date = Column(String(20), nullable=False)
    end_date = Column(String(20), nullable=False)
    total_trades = Column(Integer, default=0)
    wins = Column(Integer, default=0)
    losses = Column(Integer, default=0)
    win_rate = Column(Float, default=0.0)
    total_pnl = Column(Float, default=0.0)
    sharpe_ratio = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)
    pnl_curve = Column(JSON, nullable=True)
    trades_detail = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=func.now())


_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    if _engine is None:
        if not Config.DATABASE_URL:
            raise RuntimeError("DATABASE_URL not set")
        _engine = create_engine(Config.DATABASE_URL, pool_pre_ping=True, pool_size=5)
    return _engine


def get_session() -> Session:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine())
    return _SessionLocal()


def init_db():
    engine = get_engine()
    Base.metadata.create_all(engine)
    logger.info("Database tables created/verified")
