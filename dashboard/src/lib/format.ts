export function formatCurrency(val: number | null | undefined): string {
  if (val == null) return "-";
  return val >= 0
    ? `$${val.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
    : `-$${Math.abs(val).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

export function formatPct(val: number | null | undefined): string {
  if (val == null) return "-";
  return `${val >= 0 ? "+" : ""}${(val * 100).toFixed(2)}%`;
}

export const REGIME_COLORS: Record<string, string> = {
  CALM: "text-green-400",
  NORMAL: "text-blue-400",
  VOLATILE: "text-yellow-400",
  CRISIS: "text-red-400",
};

