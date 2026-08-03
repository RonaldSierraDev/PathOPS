import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchRecentPredictions } from "../lib/api";
import { LineChart, type LinePoint } from "./LineChart";

export function LatencyChart() {
  const recent = useQuery({
    queryKey: ["recent"],
    queryFn: () => fetchRecentPredictions(200),
  });

  // API returns newest-first; the chart reads left-to-right in time.
  const points = useMemo<LinePoint[]>(
    () =>
      (recent.data ?? [])
        .filter((r) => r.latency_ms != null)
        .reverse()
        .map((r) => ({
          value: r.latency_ms!,
          label: `${r.latency_ms!.toFixed(1)} ms · ${new Date(r.created_at).toLocaleTimeString()}`,
        })),
    [recent.data],
  );

  return (
    <div className="card px-6 py-5">
      <div className="mb-3">
        <div className="eyebrow">Inference latency</div>
        <div className="text-[13px] text-ink-mute">per prediction, oldest → newest</div>
      </div>
      {points.length < 2 ? (
        <div className="py-8 text-[14px] text-ink-mute">
          {recent.isError
            ? `Latency data unavailable: ${recent.error.message}`
            : "Not enough predictions to chart yet."}
        </div>
      ) : (
        <LineChart points={points} formatTick={(v) => String(v)} />
      )}
    </div>
  );
}
