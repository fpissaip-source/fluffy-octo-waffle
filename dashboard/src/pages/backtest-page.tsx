import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from "recharts";
import { useBacktest } from "@/hooks/use-bot-data";
import type { BacktestResult } from "@/lib/api";
import { formatCurrency } from "@/lib/format";

export default function BacktestPage() {
  const [symbol, setSymbol] = useState("AAPL");
  const [timeframe, setTimeframe] = useState("1Day");
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const backtest = useBacktest();
  const [result, setResult] = useState<BacktestResult | null>(null);

  const handleBacktest = () => {
    if (!symbol.trim()) return;
    backtest.mutate({
      symbol: symbol.trim().toUpperCase(),
      timeframe,
      startDate: startDate || undefined,
      endDate: endDate || undefined,
    }, {
      onSuccess: (data) => setResult(data),
    });
  };

  return (
    <div className="max-w-[1400px] mx-auto space-y-4">
      <Card>
        <CardHeader><CardTitle className="text-base">Backtest-Konfiguration</CardTitle></CardHeader>
        <CardContent>
          <div className="flex gap-2 flex-wrap items-end">
            <div>
              <label className="text-xs text-muted-foreground block mb-1">Symbol</label>
              <Input value={symbol} onChange={(e) => setSymbol(e.target.value)} className="w-32" />
            </div>
            <div>
              <label className="text-xs text-muted-foreground block mb-1">Zeitrahmen</label>
              <select className="bg-secondary text-secondary-foreground border border-border rounded-md px-3 py-1.5 text-sm h-9" value={timeframe} onChange={(e) => setTimeframe(e.target.value)}>
                <option value="5Min">5 Min</option>
                <option value="15Min">15 Min</option>
                <option value="1Hour">1 Stunde</option>
                <option value="1Day">1 Tag</option>
              </select>
            </div>
            <div>
              <label className="text-xs text-muted-foreground block mb-1">Start-Datum</label>
              <Input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} className="w-40" />
            </div>
            <div>
              <label className="text-xs text-muted-foreground block mb-1">End-Datum</label>
              <Input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} className="w-40" />
            </div>
            <Button onClick={handleBacktest} disabled={backtest.isPending} className="h-9">
              {backtest.isPending ? "Berechne..." : "Backtest starten"}
            </Button>
          </div>
          {backtest.isError && <p className="text-destructive text-sm mt-2">Fehler: {(backtest.error as Error).message}</p>}
        </CardContent>
      </Card>

      {result && (
        <>
          {result.message ? (
            <Card>
              <CardContent className="p-6 text-center text-muted-foreground">{result.message}</CardContent>
            </Card>
          ) : (
            <>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <Card>
                  <CardContent className="p-4">
                    <p className="text-xs text-muted-foreground">Trades</p>
                    <p className="text-2xl font-bold">{result.total_trades}</p>
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className="p-4">
                    <p className="text-xs text-muted-foreground">Win Rate</p>
                    <p className="text-2xl font-bold text-green-400">{((result.win_rate || 0) * 100).toFixed(0)}%</p>
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className="p-4">
                    <p className="text-xs text-muted-foreground">Gesamt P/L</p>
                    <p className={`text-2xl font-bold ${(result.total_pnl || 0) >= 0 ? "text-green-400" : "text-red-400"}`}>{formatCurrency(result.total_pnl)}</p>
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className="p-4">
                    <p className="text-xs text-muted-foreground">Return</p>
                    <p className="text-2xl font-bold">{result.total_return || "-"}</p>
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className="p-4">
                    <p className="text-xs text-muted-foreground">Sharpe Ratio</p>
                    <p className="text-2xl font-bold">{result.sharpe_ratio?.toFixed(2) || "-"}</p>
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className="p-4">
                    <p className="text-xs text-muted-foreground">Max Drawdown</p>
                    <p className="text-2xl font-bold text-red-400">{result.max_drawdown || "-"}</p>
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className="p-4">
                    <p className="text-xs text-muted-foreground">Bester Trade</p>
                    <p className="text-2xl font-bold text-green-400">{result.best_trade || "-"}</p>
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className="p-4">
                    <p className="text-xs text-muted-foreground">Schlechtester Trade</p>
                    <p className="text-2xl font-bold text-red-400">{result.worst_trade || "-"}</p>
                  </CardContent>
                </Card>
              </div>

              {result.pnl_curve && result.pnl_curve.length > 0 && (
                <Card>
                  <CardHeader><CardTitle className="text-base">Equity-Kurve ({result.symbol})</CardTitle></CardHeader>
                  <CardContent>
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart data={result.pnl_curve}>
                        <CartesianGrid strokeDasharray="3 3" stroke="hsl(217 33% 18%)" />
                        <XAxis dataKey="timestamp" tick={false} />
                        <YAxis tick={{ fontSize: 11 }} stroke="hsl(215 20% 55%)" tickFormatter={(v) => `$${v.toLocaleString()}`} domain={["auto", "auto"]} />
                        <Tooltip contentStyle={{ background: "hsl(222 47% 11%)", border: "1px solid hsl(217 33% 18%)", borderRadius: 8 }} formatter={(v: number) => [`$${v.toFixed(2)}`, "Equity"]} />
                        <Line type="monotone" dataKey="equity" stroke="hsl(217 91% 60%)" strokeWidth={2} dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              )}

              {result.trades && result.trades.length > 0 && (
                <Card>
                  <CardHeader><CardTitle className="text-base">Simulierte Trades ({result.trades.length})</CardTitle></CardHeader>
                  <CardContent className="p-0">
                    <div className="overflow-x-auto">
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>#</TableHead>
                            <TableHead>Symbol</TableHead>
                            <TableHead className="text-right">Qty</TableHead>
                            <TableHead className="text-right">Entry</TableHead>
                            <TableHead className="text-right">Exit</TableHead>
                            <TableHead className="text-right">P/L</TableHead>
                            <TableHead className="text-right">P/L %</TableHead>
                            <TableHead>Grund</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {result.trades.map((t, i) => (
                            <TableRow key={i}>
                              <TableCell className="text-muted-foreground text-xs">{i + 1}</TableCell>
                              <TableCell className="font-medium">{t.symbol}</TableCell>
                              <TableCell className="text-right">{t.qty}</TableCell>
                              <TableCell className="text-right">{formatCurrency(t.entry_price)}</TableCell>
                              <TableCell className="text-right">{formatCurrency(t.exit_price)}</TableCell>
                              <TableCell className={`text-right font-medium ${t.pnl >= 0 ? "text-green-400" : "text-red-400"}`}>{formatCurrency(t.pnl)}</TableCell>
                              <TableCell className={`text-right ${t.pnl_pct >= 0 ? "text-green-400" : "text-red-400"}`}>{(t.pnl_pct * 100).toFixed(2)}%</TableCell>
                              <TableCell className="text-xs text-muted-foreground">{t.reason}</TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </div>
                  </CardContent>
                </Card>
              )}

              <Card>
                <CardContent className="p-4 flex justify-between text-sm text-muted-foreground">
                  <span>Startkapital: {formatCurrency(result.initial_equity)}</span>
                  <span>Endkapital: <span className="font-medium text-foreground">{formatCurrency(result.final_equity)}</span></span>
                  <span>Avg Win: <span className="text-green-400">{result.avg_win || "-"}</span></span>
                  <span>Avg Loss: <span className="text-red-400">{result.avg_loss || "-"}</span></span>
                </CardContent>
              </Card>
            </>
          )}
        </>
      )}
    </div>
  );
}

