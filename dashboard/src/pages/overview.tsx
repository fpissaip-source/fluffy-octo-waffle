import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import {
  XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, BarChart, Bar, Cell, RadarChart,
  PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar,
  AreaChart, Area
} from "recharts";
import {
  useStatus, usePositions, useTrades, useStats,
  useWeights, useEquityHistory, useScanSymbol,
  useControlBot, useWatchlist, useLastScores
} from "@/hooks/use-bot-data";
import type { ScanResult } from "@/lib/api";
import { formatCurrency, formatPct, REGIME_COLORS } from "@/lib/format";

function StatusCards() {
  const { data: status, isLoading, error } = useStatus();
  const control = useControlBot();

  if (error) {
    return (
      <Card className="col-span-full border-destructive/50">
        <CardContent className="p-6 text-center">
          <p className="text-destructive font-medium text-lg mb-2">Bot API nicht erreichbar</p>
          <p className="text-muted-foreground text-sm">Setze ALPACA_API_KEY und ALPACA_SECRET_KEY als Umgebungsvariablen und starte den Bot-Workflow neu.</p>
        </CardContent>
      </Card>
    );
  }
  if (isLoading || !status) return <Card className="col-span-full"><CardContent className="p-6 text-center text-muted-foreground">Laden...</CardContent></Card>;

  const regime = status.regime || "NORMAL";

  return (
    <>
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-muted-foreground">Equity</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-3xl font-bold tracking-tight">{formatCurrency(status.equity)}</p>
          <p className="text-xs text-muted-foreground mt-1">Buying Power: {formatCurrency(status.buying_power)}</p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-muted-foreground">Offener P/L</CardTitle>
        </CardHeader>
        <CardContent>
          <p className={`text-3xl font-bold tracking-tight ${(status.unrealized_pl ?? 0) >= 0 ? "text-green-400" : "text-red-400"}`}>{formatCurrency(status.unrealized_pl)}</p>
          <p className="text-xs text-muted-foreground mt-1">Positionen: {status.positions_count}</p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-muted-foreground">Heutiger P/L</CardTitle>
        </CardHeader>
        <CardContent>
          <p className={`text-3xl font-bold tracking-tight ${(status.today_pl ?? 0) >= 0 ? "text-green-400" : "text-red-400"}`}>{formatCurrency(status.today_pl)}</p>
          <p className="text-xs text-muted-foreground mt-1">
            Regime: <span className={REGIME_COLORS[regime] || "text-blue-400"}>{regime}</span>
            {" | "}
            <span className={status.market_open ? "text-green-400" : "text-red-400"}>{status.market_open ? "Open" : "Closed"}</span>
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-muted-foreground">Bot Steuerung</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-2 mb-2">
            <div className={`w-2.5 h-2.5 rounded-full ${status.is_running ? (status.is_paused ? "bg-yellow-400" : "bg-green-400 animate-pulse") : "bg-gray-400"}`} />
            <span className="text-sm font-medium">{status.is_running ? (status.is_paused ? "Pausiert" : "Aktiv") : "Gestoppt"}</span>
            <Badge variant="outline" className="text-xs ml-auto">{status.mode}</Badge>
          </div>
          <div className="flex gap-1.5">
            {!status.is_running ? (
              <Button size="sm" variant="default" onClick={() => control.mutate("start")} disabled={control.isPending}>Start</Button>
            ) : (
              <>
                {status.is_paused ? (
                  <Button size="sm" variant="secondary" onClick={() => control.mutate("resume")} disabled={control.isPending}>Resume</Button>
                ) : (
                  <Button size="sm" variant="secondary" onClick={() => control.mutate("pause")} disabled={control.isPending}>Pause</Button>
                )}
                <Button size="sm" variant="destructive" onClick={() => control.mutate("stop")} disabled={control.isPending}>Stop</Button>
              </>
            )}
          </div>
          {status.kill_switch && <Badge variant="destructive" className="mt-2">KILL SWITCH</Badge>}
        </CardContent>
      </Card>
    </>
  );
}

function EquityChart() {
  const { data } = useEquityHistory(100);
  if (!data?.history?.length) return null;

  const chartData = data.history.map((p) => ({
    time: p.timestamp ? new Date(p.timestamp).toLocaleTimeString("de-DE", { hour: "2-digit", minute: "2-digit" }) : "",
    equity: p.equity,
  }));

  return (
    <Card className="col-span-full">
      <CardHeader><CardTitle className="text-base">Equity-Verlauf</CardTitle></CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={280}>
          <AreaChart data={chartData}>
            <defs>
              <linearGradient id="eqGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="hsl(142 76% 36%)" stopOpacity={0.3} />
                <stop offset="95%" stopColor="hsl(142 76% 36%)" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(217 33% 18%)" />
            <XAxis dataKey="time" tick={{ fontSize: 11 }} stroke="hsl(215 20% 55%)" />
            <YAxis tick={{ fontSize: 11 }} stroke="hsl(215 20% 55%)" tickFormatter={(v) => `$${v.toLocaleString()}`} domain={["auto", "auto"]} />
            <Tooltip contentStyle={{ background: "hsl(222 47% 11%)", border: "1px solid hsl(217 33% 18%)", borderRadius: 8 }} labelStyle={{ color: "hsl(215 20% 55%)" }} formatter={(v: number) => [`$${v.toLocaleString("en-US", { minimumFractionDigits: 2 })}`, "Equity"]} />
            <Area type="monotone" dataKey="equity" stroke="hsl(142 76% 36%)" fill="url(#eqGrad)" strokeWidth={2} />
          </AreaChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}

function PositionsTable() {
  const { data } = usePositions();
  const positions = data?.positions || [];
  if (!positions.length) return <Card><CardContent className="p-6 text-center text-muted-foreground text-sm">Keine offenen Positionen</CardContent></Card>;

  return (
    <Card>
      <CardHeader><CardTitle className="text-base">Offene Positionen</CardTitle></CardHeader>
      <CardContent className="p-0">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Symbol</TableHead>
              <TableHead className="text-right">Qty</TableHead>
              <TableHead className="text-right">Entry</TableHead>
              <TableHead className="text-right">Kurs</TableHead>
              <TableHead className="text-right">Wert</TableHead>
              <TableHead className="text-right">P/L</TableHead>
              <TableHead className="text-right">P/L %</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {positions.map((p) => (
              <TableRow key={p.symbol}>
                <TableCell className="font-medium">{p.symbol}</TableCell>
                <TableCell className="text-right">{p.qty}</TableCell>
                <TableCell className="text-right">{formatCurrency(p.avg_entry)}</TableCell>
                <TableCell className="text-right">{formatCurrency(p.current_price)}</TableCell>
                <TableCell className="text-right">{formatCurrency(p.market_value)}</TableCell>
                <TableCell className={`text-right font-medium ${p.unrealized_pl >= 0 ? "text-green-400" : "text-red-400"}`}>{formatCurrency(p.unrealized_pl)}</TableCell>
                <TableCell className={`text-right font-medium ${p.unrealized_plpc >= 0 ? "text-green-400" : "text-red-400"}`}>{formatPct(p.unrealized_plpc)}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}

function RecentTrades() {
  const { data } = useTrades(5);
  const trades = data?.trades || [];

  return (
    <Card>
      <CardHeader><CardTitle className="text-base">Letzte Trades</CardTitle></CardHeader>
      <CardContent className="p-0">
        {!trades.length ? (
          <p className="p-6 text-center text-muted-foreground text-sm">Keine Trades</p>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Symbol</TableHead>
                <TableHead>Aktion</TableHead>
                <TableHead className="text-right">P/L</TableHead>
                <TableHead>Grund</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {trades.map((t) => (
                <TableRow key={t.id}>
                  <TableCell className="font-medium">{t.symbol}</TableCell>
                  <TableCell><Badge variant={t.action === "BUY" ? "default" : "secondary"}>{t.action}</Badge></TableCell>
                  <TableCell className={`text-right font-medium ${(t.pnl ?? 0) >= 0 ? "text-green-400" : "text-red-400"}`}>{t.pnl != null ? formatCurrency(t.pnl) : "-"}</TableCell>
                  <TableCell className="text-xs text-muted-foreground truncate max-w-24">{t.exit_reason || "-"}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  );
}

function FormulaScoresChart() {
  const { data } = useLastScores();
  if (!data?.scores?.length) return null;

  const FORMULA_COLORS: Record<string, string> = {
    Momentum: "hsl(142 76% 36%)", Kelly: "hsl(217 91% 60%)", "EV-Gap": "hsl(47 96% 53%)",
    "KL-Divergence": "hsl(262 83% 58%)", Bayesian: "hsl(0 84% 60%)", "Z-Score": "hsl(180 70% 50%)",
    Sentiment: "hsl(30 90% 55%)", Regime: "hsl(200 80% 50%)",
  };

  return (
    <Card className="col-span-full">
      <CardHeader>
        <CardTitle className="text-base">
          Formel-Scores
          {data.symbol && <Badge variant="outline" className="ml-2">{data.symbol}</Badge>}
          {data.action && <Badge variant={data.action === "BUY" ? "default" : "secondary"} className="ml-1">{data.action}</Badge>}
          {data.weighted_score != null && <span className="text-sm text-muted-foreground ml-2">Score: {data.weighted_score.toFixed(2)}</span>}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          {data.scores.map((s) => (
            <div key={s.name} className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${s.passed ? "bg-green-400" : "bg-red-400"}`} />
              <span className="text-sm w-28" style={{ color: FORMULA_COLORS[s.name] || "inherit" }}>{s.name}</span>
              <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
                <div className="h-full rounded-full transition-all" style={{ width: `${Math.min(100, Math.abs(s.signal) * 100)}%`, background: FORMULA_COLORS[s.name] || "hsl(142 76% 36%)" }} />
              </div>
              <span className="text-xs text-muted-foreground w-14 text-right">{s.signal >= 0 ? "+" : ""}{s.signal.toFixed(3)}</span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

function ScanPanel() {
  const [symbol, setSymbol] = useState("");
  const scan = useScanSymbol();
  const [result, setResult] = useState<ScanResult | null>(null);

  const handleScan = () => {
    if (!symbol.trim()) return;
    scan.mutate(symbol.trim().toUpperCase(), {
      onSuccess: (data) => setResult(data),
    });
  };

  const FORMULA_COLORS: Record<string, string> = {
    Momentum: "hsl(142 76% 36%)", Kelly: "hsl(217 91% 60%)", "EV-Gap": "hsl(47 96% 53%)",
    "KL-Divergence": "hsl(262 83% 58%)", Bayesian: "hsl(0 84% 60%)", "Z-Score": "hsl(180 70% 50%)",
    Sentiment: "hsl(30 90% 55%)", Regime: "hsl(200 80% 50%)",
  };

  return (
    <Card>
      <CardHeader><CardTitle className="text-base">Symbol Scanner</CardTitle></CardHeader>
      <CardContent>
        <div className="flex gap-2 mb-4">
          <Input placeholder="Symbol (z.B. AAPL)" value={symbol} onChange={(e) => setSymbol(e.target.value)} onKeyDown={(e) => e.key === "Enter" && handleScan()} className="flex-1" />
          <Button onClick={handleScan} disabled={scan.isPending}>{scan.isPending ? "..." : "Scan"}</Button>
        </div>
        {scan.isError && <p className="text-destructive text-sm mb-2">Fehler: {(scan.error as Error).message}</p>}
        {result && (
          <div>
            <div className="flex items-center gap-2 mb-3">
              <span className="font-bold text-lg">{result.symbol}</span>
              <Badge variant={result.all_passed ? "default" : "secondary"}>{result.action}</Badge>
              <span className="text-sm text-muted-foreground">Score: {result.weighted_score.toFixed(2)}</span>
            </div>
            <div className="space-y-2">
              {Object.entries(result.results).map(([name, r]) => (
                <div key={name} className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${r.passed ? "bg-green-400" : "bg-red-400"}`} />
                  <span className="text-sm w-28" style={{ color: FORMULA_COLORS[name] || "inherit" }}>{name}</span>
                  <div className="flex-1 h-1.5 bg-muted rounded-full overflow-hidden">
                    <div className="h-full rounded-full transition-all" style={{ width: `${Math.min(100, Math.abs(r.signal) * 100)}%`, background: FORMULA_COLORS[name] || "hsl(142 76% 36%)" }} />
                  </div>
                  <span className="text-xs text-muted-foreground w-14 text-right">{r.signal >= 0 ? "+" : ""}{r.signal.toFixed(3)}</span>
                </div>
              ))}
            </div>
            <p className="text-xs text-muted-foreground mt-3">{result.reason}</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function StatsPanel() {
  const { data: stats } = useStats();
  if (!stats || !stats.total_trades) return null;

  const winLossData = [
    { name: "Wins", value: stats.wins || 0, fill: "hsl(142 76% 36%)" },
    { name: "Losses", value: stats.losses || 0, fill: "hsl(0 84% 60%)" },
  ];

  return (
    <Card>
      <CardHeader><CardTitle className="text-base">Performance</CardTitle></CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-4 mb-4">
          <div><p className="text-xs text-muted-foreground">Trades</p><p className="text-xl font-bold">{stats.total_trades}</p></div>
          <div><p className="text-xs text-muted-foreground">Win Rate</p><p className="text-xl font-bold text-green-400">{((stats.win_rate || 0) * 100).toFixed(0)}%</p></div>
          <div><p className="text-xs text-muted-foreground">Total P/L</p><p className={`text-xl font-bold ${(stats.total_pnl ?? 0) >= 0 ? "text-green-400" : "text-red-400"}`}>{formatCurrency(stats.total_pnl)}</p></div>
          <div><p className="text-xs text-muted-foreground">Sharpe</p><p className="text-xl font-bold">{stats.sharpe?.toFixed(2) || "-"}</p></div>
        </div>
        <ResponsiveContainer width="100%" height={120}>
          <BarChart data={winLossData} layout="vertical">
            <XAxis type="number" hide />
            <YAxis type="category" dataKey="name" tick={{ fontSize: 12 }} stroke="hsl(215 20% 55%)" width={50} />
            <Bar dataKey="value" radius={[0, 4, 4, 0]}>
              {winLossData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.fill} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}

function WeightsPanel() {
  const { data } = useWeights();
  if (!data) return null;

  const regime = data.regime || "NORMAL";
  const weights = data.weights[regime] || {};

  const radarData = Object.entries(weights).map(([name, weight]) => ({
    formula: name.replace("-", "\n"),
    weight: weight,
  }));

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Formel-Gewichte <Badge variant="outline" className={REGIME_COLORS[regime]}>{regime}</Badge></CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={250}>
          <RadarChart cx="50%" cy="50%" outerRadius="70%" data={radarData}>
            <PolarGrid stroke="hsl(217 33% 18%)" />
            <PolarAngleAxis dataKey="formula" tick={{ fontSize: 10, fill: "hsl(215 20% 55%)" }} />
            <PolarRadiusAxis tick={{ fontSize: 9 }} stroke="hsl(217 33% 18%)" domain={[0, 2]} />
            <Radar name="Weight" dataKey="weight" stroke="hsl(142 76% 36%)" fill="hsl(142 76% 36%)" fillOpacity={0.2} strokeWidth={2} />
          </RadarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}

function WatchlistPanel() {
  const { data: status } = useStatus();
  const watchlist = useWatchlist();
  const [newSymbol, setNewSymbol] = useState("");

  const handleAdd = () => {
    if (!newSymbol.trim()) return;
    watchlist.mutate({ action: "add", symbol: newSymbol.trim().toUpperCase() });
    setNewSymbol("");
  };

  const handleRemove = (symbol: string) => {
    watchlist.mutate({ action: "remove", symbol });
  };

  return (
    <Card>
      <CardHeader><CardTitle className="text-base">Watchlist</CardTitle></CardHeader>
      <CardContent>
        <div className="flex gap-2 mb-3">
          <Input placeholder="Symbol" value={newSymbol} onChange={(e) => setNewSymbol(e.target.value)} onKeyDown={(e) => e.key === "Enter" && handleAdd()} className="flex-1" />
          <Button size="sm" onClick={handleAdd} disabled={watchlist.isPending}>+</Button>
        </div>
        <div className="flex flex-wrap gap-1.5">
          {(status?.watchlist || []).map((s) => (
            <Badge key={s} variant="secondary" className="cursor-pointer hover:bg-destructive/20" onClick={() => handleRemove(s)}>
              {s} <span className="ml-1 opacity-60">x</span>
            </Badge>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

export default function OverviewPage() {
  return (
    <div className="max-w-[1400px] mx-auto space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatusCards />
      </div>

      <EquityChart />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <PositionsTable />
        <RecentTrades />
      </div>

      <FormulaScoresChart />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <ScanPanel />
        <WatchlistPanel />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <StatsPanel />
        <WeightsPanel />
      </div>
    </div>
  );
}

