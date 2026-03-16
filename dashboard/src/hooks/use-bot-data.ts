import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";

export function useStatus() {
  return useQuery({
    queryKey: ["bot-status"],
    queryFn: api.getStatus,
    refetchInterval: 10000,
    retry: false,
  });
}

export function usePositions() {
  return useQuery({
    queryKey: ["positions"],
    queryFn: api.getPositions,
    refetchInterval: 15000,
    retry: 1,
  });
}

export function useTrades(limit = 50) {
  return useQuery({
    queryKey: ["trades", limit],
    queryFn: () => api.getTrades(limit),
    refetchInterval: 30000,
    retry: 1,
  });
}

export function useStats() {
  return useQuery({
    queryKey: ["stats"],
    queryFn: api.getStats,
    refetchInterval: 30000,
    retry: 1,
  });
}

export function useWeights() {
  return useQuery({
    queryKey: ["weights"],
    queryFn: api.getWeights,
    refetchInterval: 60000,
    retry: 1,
  });
}

export function useEquityHistory(limit = 100) {
  return useQuery({
    queryKey: ["equity-history", limit],
    queryFn: () => api.getEquityHistory(limit),
    refetchInterval: 30000,
    retry: 1,
  });
}

export function useLastScores() {
  return useQuery({
    queryKey: ["last-scores"],
    queryFn: api.getLastScores,
    refetchInterval: 30000,
    retry: 1,
  });
}

export function useScanSymbol() {
  return useMutation({
    mutationFn: (symbol: string) => api.scanSymbol(symbol),
  });
}

export function useBacktest() {
  return useMutation({
    mutationFn: (params: { symbol: string; timeframe?: string; limit?: number; startDate?: string; endDate?: string }) =>
      api.runBacktest(params.symbol, params.timeframe, params.limit, params.startDate, params.endDate),
  });
}

export function useControlBot() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (action: string) => api.controlBot(action),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["bot-status"] }),
  });
}

export function useWatchlist() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (params: { action: string; symbol: string }) =>
      api.manageWatchlist(params.action, params.symbol),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["bot-status"] }),
  });
}

