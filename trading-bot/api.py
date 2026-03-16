import logging
import threading
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import Config

logger = logging.getLogger("bot.api")

app = FastAPI(title="Trading Bot API", version="2.0")

import os
_allowed_origins = os.getenv("CORS_ORIGINS", "").split(",") if os.getenv("CORS_ORIGINS") else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_engine = None
_bot_thread: Optional[threading.Thread] = None
_bot_lock = threading.Lock()
_bot_running = False
_bot_paused = False


def get_engine():
    global _engine
    if _engine is None:
        from engine import Engine
        _engine = Engine()
    return _engine


def _bot_scan_loop():
    global _bot_running, _bot_paused
    import time
    engine = get_engine()
    logger.info("Bot scan loop started")
    while True:
        with _bot_lock:
            should_stop = not _bot_running
            is_paused = _bot_paused
        if should_stop:
            break
        if is_paused:
            time.sleep(5)
            continue
        try:
            if not engine.broker.is_market_open():
                time.sleep(60)
                continue
            engine.scan_once()
        except Exception as e:
            logger.error(f"Scan loop error: {e}")
        time.sleep(Config.SCAN_INTERVAL)
    logger.info("Bot scan loop stopped")


class ControlAction(BaseModel):
    action: str


class WatchlistAction(BaseModel):
    action: str
    symbol: str


@app.get("/bot-api/status")
def get_status():
    try:
        engine = get_engine()
        equity = engine.broker.get_equity()
        cash = engine.broker.get_cash()
        bp = engine.broker.get_buying_power()
        positions = engine.broker.get_positions()
        market_open = engine.broker.is_market_open()

        from database import get_session, BotStatus
        session = get_session()
        session.add(BotStatus(
            equity=equity,
            cash=cash,
            regime=engine.risk.regime.value,
            is_running=_bot_running,
            is_paused=_bot_paused,
            open_positions=len(positions),
            last_scan=datetime.now(),
            kill_switch_active=engine.risk.kill_switch_active,
        ))
        session.commit()
        session.close()

        total_unrealized_pl = sum(p.get("unrealized_pl", 0) for p in positions.values())
        today_pl = equity - float(engine.broker.api.get_account().last_equity) if hasattr(engine.broker.api.get_account(), 'last_equity') else 0.0

        return {
            "equity": equity,
            "cash": cash,
            "buying_power": bp,
            "positions_count": len(positions),
            "market_open": market_open,
            "mode": "PAPER" if Config.is_paper() else "LIVE",
            "regime": engine.risk.regime.value,
            "is_running": _bot_running,
            "is_paused": _bot_paused,
            "kill_switch": engine.risk.kill_switch_active,
            "watchlist": Config.WATCHLIST,
            "unrealized_pl": round(total_unrealized_pl, 2),
            "today_pl": round(today_pl, 2),
        }
    except Exception as e:
        logger.error(f"Status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/bot-api/positions")
def get_positions():
    try:
        engine = get_engine()
        positions = engine.broker.get_positions()

        from database import get_session, PositionSnapshot
        session = get_session()
        for sym, pos in positions.items():
            session.add(PositionSnapshot(
                symbol=sym,
                qty=pos["qty"],
                avg_entry=pos["avg_entry"],
                market_value=pos["market_value"],
                unrealized_pl=pos["unrealized_pl"],
                unrealized_plpc=pos["unrealized_plpc"],
                side=pos["side"],
            ))
        session.commit()
        session.close()

        result = []
        for sym, pos in positions.items():
            current_price = pos["market_value"] / pos["qty"] if pos["qty"] else 0
            result.append({
                "symbol": sym,
                "qty": pos["qty"],
                "avg_entry": pos["avg_entry"],
                "current_price": round(current_price, 2),
                "market_value": pos["market_value"],
                "unrealized_pl": pos["unrealized_pl"],
                "unrealized_plpc": pos["unrealized_plpc"],
                "side": pos["side"],
            })
        return {"positions": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/bot-api/trades")
def get_trades(limit: int = Query(default=50, le=500)):
    try:
        from database import get_session, TradeRecord
        session = get_session()
        records = session.query(TradeRecord).order_by(
            TradeRecord.id.desc()
        ).limit(limit).all()
        session.close()

        return {"trades": [{
            "id": r.id,
            "symbol": r.symbol,
            "regime": r.regime,
            "action": r.action,
            "qty": r.qty,
            "entry_price": r.entry_price,
            "exit_price": r.exit_price,
            "pnl": r.pnl,
            "pnl_pct": r.pnl_pct,
            "entry_time": r.entry_time.isoformat() if r.entry_time else None,
            "exit_time": r.exit_time.isoformat() if r.exit_time else None,
            "exit_reason": r.exit_reason,
            "weighted_score": r.weighted_score,
        } for r in records]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/bot-api/weights")
def get_weights():
    try:
        engine = get_engine()
        return {
            "weights": engine.learner.weights,
            "regime": engine.risk.regime.value,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/bot-api/stats")
def get_stats():
    try:
        engine = get_engine()
        stats = engine.learner.get_trade_history_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/bot-api/scan/{symbol}")
def scan_symbol(symbol: str):
    try:
        engine = get_engine()
        signal = engine.analyze_symbol(symbol.upper())
        return {
            "symbol": signal.symbol,
            "timestamp": signal.timestamp.isoformat(),
            "action": signal.action,
            "all_passed": signal.all_passed,
            "weighted_score": signal.weighted_score,
            "qty": signal.qty,
            "reason": signal.reason,
            "results": {
                name: {
                    "signal": r.get("signal", 0),
                    "passed": r.get("passed", False),
                    "details": r.get("details", {}),
                } for name, r in signal.results.items()
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/bot-api/last-scores")
def get_last_scores():
    try:
        engine = get_engine()
        if not hasattr(engine, '_last_signal') or engine._last_signal is None:
            if Config.WATCHLIST:
                signal = engine.analyze_symbol(Config.WATCHLIST[0])
                engine._last_signal = signal
            else:
                return {"scores": [], "symbol": "", "timestamp": ""}

        signal = engine._last_signal
        scores = []
        for name, r in signal.results.items():
            scores.append({
                "name": name,
                "signal": r.get("signal", 0),
                "passed": r.get("passed", False),
            })
        return {
            "scores": scores,
            "symbol": signal.symbol,
            "timestamp": signal.timestamp.isoformat() if signal.timestamp else "",
            "weighted_score": signal.weighted_score,
            "action": signal.action,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/bot-api/backtest")
def run_backtest(
    symbol: str = Query(default="AAPL"),
    timeframe: str = Query(default="1Day"),
    limit: int = Query(default=200, le=1000),
    start_date: Optional[str] = Query(default=None, description="Start date YYYY-MM-DD"),
    end_date: Optional[str] = Query(default=None, description="End date YYYY-MM-DD"),
):
    try:
        engine = get_engine()
        bars = engine.broker.get_bars(symbol.upper(), timeframe=timeframe, limit=limit)
        if bars.empty:
            raise HTTPException(status_code=404, detail=f"No data for {symbol}")

        if start_date:
            bars = bars[bars.index >= start_date]
        if end_date:
            bars = bars[bars.index <= end_date]
        if bars.empty:
            raise HTTPException(status_code=404, detail=f"No data for {symbol} in specified date range")

        from backtester import BacktestEngine
        bt = BacktestEngine()
        results = bt.run(bars, symbol.upper())

        from database import get_session, BacktestResult
        session = get_session()
        max_dd_val = 0.0
        if isinstance(results.get("max_drawdown"), str):
            try:
                max_dd_val = float(results["max_drawdown"].rstrip("%")) / 100
            except (ValueError, AttributeError):
                pass
        session.add(BacktestResult(
            symbol=symbol.upper(),
            timeframe=timeframe,
            start_date=str(bars.index[0]),
            end_date=str(bars.index[-1]),
            total_trades=results.get("total_trades", 0),
            wins=results.get("wins", 0),
            losses=results.get("losses", 0),
            win_rate=results.get("win_rate", 0),
            total_pnl=results.get("total_pnl", 0),
            sharpe_ratio=results.get("sharpe_ratio", 0),
            max_drawdown=max_dd_val,
            pnl_curve=results.get("pnl_curve"),
            trades_detail=results.get("trades"),
        ))
        session.commit()
        session.close()

        return results
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/bot-api/control")
def control_bot(action: ControlAction):
    global _bot_running, _bot_paused, _bot_thread
    with _bot_lock:
        if action.action == "start":
            if _bot_running:
                return {"status": "already_running"}
            _bot_running = True
            _bot_paused = False
            _bot_thread = threading.Thread(target=_bot_scan_loop, daemon=True)
            _bot_thread.start()
            return {"status": "started"}
        elif action.action == "stop":
            _bot_running = False
            _bot_paused = False
            return {"status": "stopped"}
        elif action.action == "pause":
            if not _bot_running:
                return {"status": "not_running"}
            _bot_paused = True
            return {"status": "paused"}
        elif action.action == "resume":
            if not _bot_running:
                return {"status": "not_running"}
            _bot_paused = False
            return {"status": "resumed"}
        else:
            raise HTTPException(status_code=400, detail=f"Unknown action: {action.action}")


@app.post("/bot-api/watchlist")
def manage_watchlist(action: WatchlistAction):
    symbol = action.symbol.upper()
    if action.action == "add":
        if symbol not in Config.WATCHLIST:
            Config.WATCHLIST.append(symbol)
        return {"watchlist": Config.WATCHLIST}
    elif action.action == "remove":
        if symbol in Config.WATCHLIST:
            Config.WATCHLIST.remove(symbol)
        return {"watchlist": Config.WATCHLIST}
    else:
        raise HTTPException(status_code=400, detail=f"Unknown action: {action.action}")


@app.get("/bot-api/equity-history")
def get_equity_history(limit: int = Query(default=100, le=500)):
    try:
        from database import get_session, BotStatus
        session = get_session()
        records = session.query(BotStatus).order_by(
            BotStatus.id.desc()
        ).limit(limit).all()
        session.close()

        return {"history": [{
            "timestamp": r.timestamp.isoformat() if r.timestamp else None,
            "equity": r.equity,
            "regime": r.regime,
        } for r in reversed(records)]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
