import { useState, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { useTrades } from "@/hooks/use-bot-data";
import { formatCurrency, formatPct, REGIME_COLORS } from "@/lib/format";

type SortKey = "id" | "symbol" | "pnl" | "entry_price" | "exit_price" | "qty";
type SortDir = "asc" | "desc";

export default function TradesPage() {
  const { data, isLoading } = useTrades(200);
  const trades = data?.trades || [];

  const [filterSymbol, setFilterSymbol] = useState("");
  const [filterDateFrom, setFilterDateFrom] = useState("");
  const [filterDateTo, setFilterDateTo] = useState("");
  const [sortKey, setSortKey] = useState<SortKey>("id");
  const [sortDir, setSortDir] = useState<SortDir>("desc");

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir(sortDir === "asc" ? "desc" : "asc");
    } else {
      setSortKey(key);
      setSortDir("desc");
    }
  };

  const sortIndicator = (key: SortKey) => {
    if (sortKey !== key) return "";
    return sortDir === "asc" ? " ↑" : " ↓";
  };

  const filtered = useMemo(() => {
    let result = [...trades];

    if (filterSymbol.trim()) {
      const sym = filterSymbol.trim().toUpperCase();
      result = result.filter((t) => t.symbol.includes(sym));
    }

    if (filterDateFrom) {
      result = result.filter((t) => {
        const d = t.entry_time || t.exit_time || "";
        return d >= filterDateFrom;
      });
    }

    if (filterDateTo) {
      result = result.filter((t) => {
        const d = t.entry_time || t.exit_time || "";
        return d <= filterDateTo + "T23:59:59";
      });
    }

    result.sort((a, b) => {
      let aVal: number | string = 0;
      let bVal: number | string = 0;
      switch (sortKey) {
        case "id": aVal = a.id; bVal = b.id; break;
        case "symbol": aVal = a.symbol; bVal = b.symbol; break;
        case "pnl": aVal = a.pnl ?? 0; bVal = b.pnl ?? 0; break;
        case "entry_price": aVal = a.entry_price; bVal = b.entry_price; break;
        case "exit_price": aVal = a.exit_price ?? 0; bVal = b.exit_price ?? 0; break;
        case "qty": aVal = a.qty; bVal = b.qty; break;
      }
      if (typeof aVal === "string" && typeof bVal === "string") {
        return sortDir === "asc" ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
      }
      return sortDir === "asc" ? (aVal as number) - (bVal as number) : (bVal as number) - (aVal as number);
    });

    return result;
  }, [trades, filterSymbol, filterDateFrom, filterDateTo, sortKey, sortDir]);

  const exportCSV = () => {
    const headers = ["ID", "Symbol", "Aktion", "Qty", "Entry", "Exit", "P/L", "P/L %", "Regime", "Grund", "Entry Time", "Exit Time"];
    const rows = filtered.map((t) => [
      t.id, t.symbol, t.action, t.qty, t.entry_price, t.exit_price ?? "",
      t.pnl ?? "", t.pnl_pct ?? "", t.regime, t.exit_reason ?? "",
      t.entry_time ?? "", t.exit_time ?? "",
    ]);
    const csv = [headers.join(","), ...rows.map((r) => r.join(","))].join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `trades_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const totalPnl = filtered.reduce((sum, t) => sum + (t.pnl ?? 0), 0);
  const winCount = filtered.filter((t) => (t.pnl ?? 0) > 0).length;
  const winRate = filtered.length > 0 ? winCount / filtered.length : 0;

  return (
    <div className="max-w-[1400px] mx-auto space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <CardContent className="p-4">
            <p className="text-xs text-muted-foreground">Gefilterte Trades</p>
            <p className="text-2xl font-bold">{filtered.length}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <p className="text-xs text-muted-foreground">Gesamt P/L</p>
            <p className={`text-2xl font-bold ${totalPnl >= 0 ? "text-green-400" : "text-red-400"}`}>{formatCurrency(totalPnl)}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <p className="text-xs text-muted-foreground">Win Rate</p>
            <p className="text-2xl font-bold text-green-400">{(winRate * 100).toFixed(0)}%</p>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <div className="flex items-center justify-between flex-wrap gap-2">
            <CardTitle className="text-base">Trade-Historie</CardTitle>
            <Button size="sm" variant="outline" onClick={exportCSV}>CSV Export</Button>
          </div>
        </CardHeader>
        <CardContent>
          <div className="flex gap-2 mb-4 flex-wrap">
            <Input placeholder="Symbol filtern..." value={filterSymbol} onChange={(e) => setFilterSymbol(e.target.value)} className="w-40" />
            <Input type="date" value={filterDateFrom} onChange={(e) => setFilterDateFrom(e.target.value)} className="w-40" />
            <Input type="date" value={filterDateTo} onChange={(e) => setFilterDateTo(e.target.value)} className="w-40" />
            {(filterSymbol || filterDateFrom || filterDateTo) && (
              <Button size="sm" variant="ghost" onClick={() => { setFilterSymbol(""); setFilterDateFrom(""); setFilterDateTo(""); }}>
                Zurucksetzen
              </Button>
            )}
          </div>

          {isLoading ? (
            <p className="text-center text-muted-foreground py-8">Laden...</p>
          ) : !filtered.length ? (
            <p className="text-center text-muted-foreground py-8">Keine Trades gefunden</p>
          ) : (
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="cursor-pointer select-none" onClick={() => handleSort("id")}>#{sortIndicator("id")}</TableHead>
                    <TableHead className="cursor-pointer select-none" onClick={() => handleSort("symbol")}>Symbol{sortIndicator("symbol")}</TableHead>
                    <TableHead>Aktion</TableHead>
                    <TableHead className="text-right cursor-pointer select-none" onClick={() => handleSort("qty")}>Qty{sortIndicator("qty")}</TableHead>
                    <TableHead className="text-right cursor-pointer select-none" onClick={() => handleSort("entry_price")}>Entry{sortIndicator("entry_price")}</TableHead>
                    <TableHead className="text-right cursor-pointer select-none" onClick={() => handleSort("exit_price")}>Exit{sortIndicator("exit_price")}</TableHead>
                    <TableHead className="text-right cursor-pointer select-none" onClick={() => handleSort("pnl")}>P/L{sortIndicator("pnl")}</TableHead>
                    <TableHead className="text-right">P/L %</TableHead>
                    <TableHead>Regime</TableHead>
                    <TableHead>Grund</TableHead>
                    <TableHead>Zeit</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {filtered.map((t) => (
                    <TableRow key={t.id}>
                      <TableCell className="text-muted-foreground text-xs">{t.id}</TableCell>
                      <TableCell className="font-medium">{t.symbol}</TableCell>
                      <TableCell><Badge variant={t.action === "BUY" ? "default" : "secondary"}>{t.action}</Badge></TableCell>
                      <TableCell className="text-right">{t.qty}</TableCell>
                      <TableCell className="text-right">{formatCurrency(t.entry_price)}</TableCell>
                      <TableCell className="text-right">{t.exit_price ? formatCurrency(t.exit_price) : <span className="text-muted-foreground">offen</span>}</TableCell>
                      <TableCell className={`text-right font-medium ${(t.pnl ?? 0) >= 0 ? "text-green-400" : "text-red-400"}`}>{t.pnl != null ? formatCurrency(t.pnl) : "-"}</TableCell>
                      <TableCell className={`text-right ${(t.pnl_pct ?? 0) >= 0 ? "text-green-400" : "text-red-400"}`}>{t.pnl_pct != null ? formatPct(t.pnl_pct) : "-"}</TableCell>
                      <TableCell><Badge variant="outline" className={REGIME_COLORS[t.regime]}>{t.regime}</Badge></TableCell>
                      <TableCell className="text-xs text-muted-foreground max-w-32 truncate">{t.exit_reason || "-"}</TableCell>
                      <TableCell className="text-xs text-muted-foreground whitespace-nowrap">{t.entry_time ? new Date(t.entry_time).toLocaleDateString("de-DE") : "-"}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}


