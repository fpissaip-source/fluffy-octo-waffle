const BOT_API = "/bot-api";

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BOT_API}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(err || `API error ${res.status}`);
  }
  return res.json();
}

export interface BotStatus {
  equity: number;
  cash: number;
  buying_power: number;
  positions_count: number;
  market_open: boolean;
  mode: string;
  regime: string;
  is_running: boolean;
  is_paused: boolean;
  kill_switch: boolean;
  watchlist: string[];
  unrealized_pl: number;
  today_pl: number;
}

export interface Position {
  symbol: string;
  qty: number;
  avg_entry: number;
  current_price: number;
  market_value: number;
  unrealized_pl: number;
  unrealized_plpc: number;
  side: string;
}

export interface Trade {
  id: number;
  symbol: string;
  regime: string;
  action: string;
  qty: number;
  entry_price: number;
  exit_price: number | null;
  pnl: number | null;
  pnl_pct: number | null;
  entry_time: string | null;
  exit_time: string | null;
  exit_reason: string | null;
  weighted_score: number | null;
}

export interface ScanResult {
  symbol: string;
  timestamp: string;
  action: string;
  all_passed: boolean;
  weighted_score: number;
  qty: number;
  reason: string;
  results: Record<string, {
    signal: number;
    passed: boolean;
    details: Record<string, unknown>;
  }>;
}

export interface TradeStats {
  total_trades: number;
  wins?: number;
  losses?: number;
  win_rate?: number;
  avg_win?: number;
  avg_loss?: number;
  payoff_ratio?: number;
  best_trade?: number;
  worst_trade?: number;
  total_pnl?: number;
  sharpe?: number;
}

export interface WeightsData {
  weights: Record<string, Record<string, number>>;
  regime: string;
}

export interface EquityPoint {
  timestamp: string | null;
  equity: number;
  regime: string | null;
}

export interface BacktestResult {
  symbol: string;
  total_trades: number;
  wins?: number;
  losses?: number;
  win_rate?: number;
  total_pnl?: number;
  total_return?: string;
  sharpe_ratio?: number;
  max_drawdown?: string;
  best_trade?: string;
  worst_trade?: string;
  avg_win?: string;
  avg_loss?: string;
  final_equity?: number;
  initial_equity?: number;
  trades?: Array<{
    symbol: string;
    entry_price: number;
    exit_price: number;
    qty: number;
    pnl: number;
    pnl_pct: number;
    reason: string;
    entry_time: string;
    exit_time: string;
  }>;
  pnl_curve?: Array<{ timestamp: string; equity: number }>;
  message?: string;
}

export interface FormulaScore {
  name: string;
  signal: number;
  passed: boolean;
}

export interface LastScores {
  scores: FormulaScore[];
  symbol: string;
  timestamp: string;
  weighted_score: number;
  action: string;
}

export const api = {
  getStatus: () => apiFetch<BotStatus>("/status"),
  getPositions: () => apiFetch<{ positions: Position[] }>("/positions"),
  getTrades: (limit = 50) => apiFetch<{ trades: Trade[] }>(`/trades?limit=${limit}`),
  getStats: () => apiFetch<TradeStats>("/stats"),
  getWeights: () => apiFetch<WeightsData>("/weights"),
  getEquityHistory: (limit = 100) => apiFetch<{ history: EquityPoint[] }>(`/equity-history?limit=${limit}`),
  scanSymbol: (symbol: string) => apiFetch<ScanResult>(`/scan/${symbol}`),
  getLastScores: () => apiFetch<LastScores>("/last-scores"),
  runBacktest: (symbol: string, timeframe = "1Day", limit = 200, startDate?: string, endDate?: string) => {
    let url = `/backtest?symbol=${symbol}&timeframe=${timeframe}&limit=${limit}`;
    if (startDate) url += `&start_date=${startDate}`;
    if (endDate) url += `&end_date=${endDate}`;
    return apiFetch<BacktestResult>(url);
  },
  controlBot: (action: string) =>
    apiFetch<{ status: string }>("/control", {
      method: "POST",
      body: JSON.stringify({ action }),
    }),
  manageWatchlist: (action: string, symbol: string) =>
    apiFetch<{ watchlist: string[] }>("/watchlist", {
      method: "POST",
      body: JSON.stringify({ action, symbol }),
    }),
};

