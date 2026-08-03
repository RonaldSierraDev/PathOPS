import { useQuery } from "@tanstack/react-query";
import { fetchHealth, fetchRecentPredictions } from "../lib/api";
import { STATUS, Tile, TILE_NUMBER, StatusDot } from "./Tile";

export function OverviewTiles() {
  const health = useQuery({ queryKey: ["health"], queryFn: fetchHealth });
  const recent = useQuery({
    queryKey: ["recent"],
    queryFn: () => fetchRecentPredictions(200),
  });

  const rows = recent.data ?? [];
  const latencies = rows.filter((r) => r.latency_ms != null).map((r) => r.latency_ms!);
  const avgLatency =
    latencies.length > 0 ? latencies.reduce((a, b) => a + b, 0) / latencies.length : null;
  const tumorShare =
    rows.length > 0 ? rows.filter((r) => r.predicted_label === "tumor").length / rows.length : null;

  return (
    <div className="grid grid-cols-4 gap-4">
      <Tile label="API status">
        {health.isPending ? (
          <StatusDot color={STATUS.neutral} text="Checking" />
        ) : health.isError ? (
          <StatusDot color={STATUS.danger} text="Unreachable" />
        ) : (
          <StatusDot color={STATUS.ok} text="Healthy" />
        )}
      </Tile>
      <Tile label="Predictions · last 200">
        <div className={TILE_NUMBER}>{recent.isSuccess ? rows.length : "—"}</div>
      </Tile>
      <Tile label="Avg latency">
        <div className={TILE_NUMBER}>
          {avgLatency != null ? `${avgLatency.toFixed(0)}` : "—"}
          {avgLatency != null && <span className="ml-1 text-[13px] text-ink-mute">ms</span>}
        </div>
      </Tile>
      <Tile label="Tumor share">
        <div className={TILE_NUMBER}>
          {tumorShare != null ? `${(tumorShare * 100).toFixed(1)}%` : "—"}
        </div>
      </Tile>
    </div>
  );
}
