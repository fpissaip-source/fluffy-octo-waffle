import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useState } from "react";
import { useStatus, useWeights, useControlBot, useWatchlist } from "@/hooks/use-bot-data";
import {
  ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar,
} from "recharts";
import { REGIME_COLORS } from "@/lib/format";

function BotControlPanel() {
  const { data: status } = useStatus();
  const control = useControlBot();

  if (!status) return null;

  return (
    <Card>
      <CardHeader><CardTitle className="text-base">Bot-Steuerung</CardTitle></CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center gap-3">
          <div className={`w-3 h-3 rounded-full ${status.is_running ? (status.is_paused ? "bg-yellow-400" : "bg-green-400 animate-pulse") : "bg-gray-400"}`} />
          <span className="font-medium">{status.is_running ? (status.is_paused ? "Pausiert" : "Aktiv") : "Gestoppt"}</span>
          <Badge variant="outline" className="ml-auto">{status.mode}</Badge>
        </div>

        <div className="flex gap-2">
          {!status.is_running ? (
            <Button onClick={() => control.mutate("start")} disabled={control.isPending} className="flex-1">Start</Button>
          ) : (
            <>
              {status.is_paused ? (
                <Button variant="secondary" onClick={() => control.mutate("resume")} disabled={control.isPending} className="flex-1">Resume</Button>
              ) : (
                <Button variant="secondary" onClick={() => control.mutate("pause")} disabled={control.isPending} className="flex-1">Pause</Button>
              )}
              <Button variant="destructive" onClick={() => control.mutate("stop")} disabled={control.isPending} className="flex-1">Stop</Button>
            </>
          )}
        </div>

        {status.kill_switch && <Badge variant="destructive" className="w-full justify-center">KILL SWITCH AKTIV</Badge>}

        <div className="text-sm text-muted-foreground space-y-1 pt-2 border-t border-border">
          <p>Market: <span className={status.market_open ? "text-green-400" : "text-red-400"}>{status.market_open ? "Open" : "Closed"}</span></p>
          <p>Positionen: {status.positions_count}</p>
          <p>Regime: <span className={REGIME_COLORS[status.regime] || "text-blue-400"}>{status.regime}</span></p>
        </div>
      </CardContent>
    </Card>
  );
}

function WatchlistSettings() {
  const { data: status } = useStatus();
  const watchlist = useWatchlist();
  const [newSymbol, setNewSymbol] = useState("");

  const handleAdd = () => {
    if (!newSymbol.trim()) return;
    watchlist.mutate({ action: "add", symbol: newSymbol.trim().toUpperCase() });
    setNewSymbol("");
  };

  return (
    <Card>
      <CardHeader><CardTitle className="text-base">Watchlist</CardTitle></CardHeader>
      <CardContent className="space-y-3">
        <div className="flex gap-2">
          <Input placeholder="Symbol hinzufugen..." value={newSymbol} onChange={(e) => setNewSymbol(e.target.value)} onKeyDown={(e) => e.key === "Enter" && handleAdd()} className="flex-1" />
          <Button onClick={handleAdd} disabled={watchlist.isPending}>Hinzufugen</Button>
        </div>
        <div className="flex flex-wrap gap-2">
          {(status?.watchlist || []).map((s) => (
            <Badge key={s} variant="secondary" className="cursor-pointer hover:bg-destructive/20 px-3 py-1" onClick={() => watchlist.mutate({ action: "remove", symbol: s })}>
              {s} <span className="ml-1.5 text-destructive opacity-70">x</span>
            </Badge>
          ))}
        </div>
        {(!status?.watchlist || status.watchlist.length === 0) && (
          <p className="text-sm text-muted-foreground">Keine Symbole in der Watchlist</p>
        )}
      </CardContent>
    </Card>
  );
}

function WeightsDisplay() {
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
        <ResponsiveContainer width="100%" height={300}>
          <RadarChart cx="50%" cy="50%" outerRadius="70%" data={radarData}>
            <PolarGrid stroke="hsl(217 33% 18%)" />
            <PolarAngleAxis dataKey="formula" tick={{ fontSize: 10, fill: "hsl(215 20% 55%)" }} />
            <PolarRadiusAxis tick={{ fontSize: 9 }} stroke="hsl(217 33% 18%)" domain={[0, 2]} />
            <Radar name="Weight" dataKey="weight" stroke="hsl(142 76% 36%)" fill="hsl(142 76% 36%)" fillOpacity={0.2} strokeWidth={2} />
          </RadarChart>
        </ResponsiveContainer>

        <div className="mt-4 space-y-1">
          {Object.entries(weights).map(([name, w]) => (
            <div key={name} className="flex justify-between text-sm">
              <span className="text-muted-foreground">{name}</span>
              <span className="font-medium">{(w as number).toFixed(2)}</span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

export default function SettingsPage() {
  return (
    <div className="max-w-[1000px] mx-auto space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <BotControlPanel />
        <WatchlistSettings />
      </div>
      <WeightsDisplay />
    </div>
  );
}

