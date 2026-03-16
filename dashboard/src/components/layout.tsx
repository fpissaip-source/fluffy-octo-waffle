import { useState } from "react";
import { useStatus } from "@/hooks/use-bot-data";
import { cn } from "@/lib/utils";
import OverviewPage from "@/pages/overview";
import TradesPage from "@/pages/trades-page";
import BacktestPage from "@/pages/backtest-page";
import SettingsPage from "@/pages/settings-page";

const NAV_ITEMS = [
  { key: "dashboard", label: "Dashboard", icon: "📊" },
  { key: "trades", label: "Trades", icon: "📈" },
  { key: "backtest", label: "Backtest", icon: "🧪" },
  { key: "settings", label: "Einstellungen", icon: "⚙️" },
];

function BotStatusDot() {
  const { data: status } = useStatus();
  const isRunning = status?.is_running ?? false;
  const isPaused = status?.is_paused ?? false;

  let color = "bg-gray-500";
  let label = "Offline";
  if (isRunning && !isPaused) {
    color = "bg-green-500 animate-pulse";
    label = "Aktiv";
  } else if (isRunning && isPaused) {
    color = "bg-yellow-500";
    label = "Pausiert";
  } else if (status) {
    color = "bg-red-500";
    label = "Gestoppt";
  }

  return (
    <div className="flex items-center gap-2">
      <div className={cn("w-2.5 h-2.5 rounded-full", color)} />
      <span className="text-xs font-medium text-muted-foreground">{label}</span>
    </div>
  );
}

export default function AppLayout() {
  const [activePage, setActivePage] = useState("dashboard");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  return (
    <div className="min-h-screen bg-background text-foreground dark flex">
      <aside className={cn(
        "border-r border-border flex flex-col transition-all duration-200 shrink-0",
        sidebarCollapsed ? "w-16" : "w-56"
      )}>
        <div className="p-4 border-b border-border flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center shrink-0">
            <span className="text-primary-foreground font-bold text-sm">TB</span>
          </div>
          {!sidebarCollapsed && (
            <div className="min-w-0">
              <h1 className="text-sm font-semibold tracking-tight truncate">Trading Bot</h1>
              <BotStatusDot />
            </div>
          )}
        </div>

        <nav className="flex-1 p-2 space-y-1">
          {NAV_ITEMS.map((item) => (
            <button
              key={item.key}
              onClick={() => setActivePage(item.key)}
              className={cn(
                "w-full flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-colors",
                activePage === item.key
                  ? "bg-primary/10 text-primary"
                  : "text-muted-foreground hover:bg-muted hover:text-foreground"
              )}
            >
              <span className="text-base">{item.icon}</span>
              {!sidebarCollapsed && <span>{item.label}</span>}
            </button>
          ))}
        </nav>

        <div className="p-2 border-t border-border">
          <button
            onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
            className="w-full flex items-center justify-center px-3 py-2 rounded-md text-sm text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
          >
            {sidebarCollapsed ? "→" : "← Einklappen"}
          </button>
        </div>
      </aside>

      <div className="flex-1 flex flex-col min-w-0">
        <header className="border-b border-border px-6 py-3 flex items-center justify-between shrink-0">
          <h2 className="text-lg font-semibold tracking-tight">
            {NAV_ITEMS.find((i) => i.key === activePage)?.label || "Dashboard"}
          </h2>
          <div className="flex items-center gap-4">
            <BotStatusDot />
            <span className="text-xs text-muted-foreground">v2.0 | Weighted Scoring</span>
          </div>
        </header>

        <main className="flex-1 overflow-auto p-4 md:p-6">
          {activePage === "dashboard" && <OverviewPage />}
          {activePage === "trades" && <TradesPage />}
          {activePage === "backtest" && <BacktestPage />}
          {activePage === "settings" && <SettingsPage />}
        </main>
      </div>
    </div>
  );
}

