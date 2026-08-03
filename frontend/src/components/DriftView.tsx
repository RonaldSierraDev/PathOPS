import { useQuery } from "@tanstack/react-query";
import { fetchDriftStatus } from "../lib/api";
import { LineChart, type LinePoint } from "./LineChart";
import { STATUS, Tile, TILE_NUMBER, StatusDot } from "./Tile";

const MONITORED_FEATURES = "mean_r, mean_g, mean_b, std_r, std_g, std_b, brightness";

const formatPercent = (share: number) => `${(share * 100).toFixed(0)}%`;

export function DriftView() {
  const drift = useQuery({ queryKey: ["drift"], queryFn: fetchDriftStatus });

  if (drift.isError) {
    return (
      <div className="card px-6 py-5">
        <div className="mb-1 eyebrow">Data drift</div>
        <div className="py-6 text-[14px] text-ink-mute">
          Drift monitoring unavailable: {drift.error.message}
        </div>
      </div>
    );
  }

  const data = drift.data;
  const breached = data?.current_share != null && data.current_share >= data.threshold;
  const points: LinePoint[] =
    data?.history.map((point) => ({
      value: point.share,
      label: `${formatPercent(point.share)} · ${new Date(point.timestamp).toLocaleString()}`,
    })) ?? [];

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-3 gap-4">
        <Tile label="Drift status">
          {data?.current_share == null ? (
            <StatusDot color={STATUS.neutral} text="No checks yet" />
          ) : breached ? (
            <StatusDot color={STATUS.warn} text="Above alarm threshold" />
          ) : (
            <StatusDot color={STATUS.ok} text="Within threshold" />
          )}
        </Tile>
        <Tile label="Drifted features">
          <div className={TILE_NUMBER}>
            {data?.current_share != null ? formatPercent(data.current_share) : "—"}
            {data != null && (
              <span className="ml-2 text-[13px] font-normal text-ink-mute">
                of {formatPercent(data.threshold)} threshold
              </span>
            )}
          </div>
        </Tile>
        <Tile label="Last checked">
          <div className="text-[15px] text-ink-soft">
            {data?.checked_at ? new Date(data.checked_at).toLocaleString() : "—"}
          </div>
        </Tile>
      </div>

      <div className="card px-6 py-5">
        <div className="mb-3">
          <div className="eyebrow">Drift share over time</div>
          <div className="text-[13px] text-ink-mute">
            Fraction of monitored features whose live distribution has diverged from the training
            baseline. Checks run on a schedule; the dashed line is where the CloudWatch alarm fires.
          </div>
        </div>
        {points.length < 2 ? (
          <div className="py-8 text-[14px] text-ink-mute">
            {drift.isPending
              ? "Loading drift history…"
              : "Not enough completed checks to chart yet."}
          </div>
        ) : (
          <LineChart
            points={points}
            yMax={1}
            formatTick={formatPercent}
            threshold={{ value: data!.threshold, label: "alarm threshold" }}
            showPoints
            pointColor={(v) => (v >= data!.threshold ? STATUS.warn : "#1C2127")}
          />
        )}
      </div>

      <div className="flex items-start gap-6">
        <div className="card min-w-0 flex-1 px-6 py-5">
          <div className="mb-1 eyebrow">Reports</div>
          <p className="mb-4 text-[13px] text-ink-mute">
            Full Evidently report per check. Links are temporary and expire an hour after this page
            loaded them.
          </p>
          {data && data.reports.length === 0 ? (
            <div className="py-4 text-[14px] text-ink-mute">
              No reports yet. The monitor writes one per completed check.
            </div>
          ) : (
            <table className="w-full border-collapse">
              <thead>
                <tr>
                  <th className="border-b border-hairline pb-2 text-left font-mono text-[11px] font-medium uppercase tracking-eyebrow text-ink-mute">
                    generated
                  </th>
                  <th className="border-b border-hairline pb-2 text-left font-mono text-[11px] font-medium uppercase tracking-eyebrow text-ink-mute">
                    report
                  </th>
                </tr>
              </thead>
              <tbody>
                {(data?.reports ?? []).map((report) => (
                  <tr key={report.key} className="h-9 border-b border-hairline last:border-b-0">
                    <td className="pr-4 font-mono text-[12px] text-ink-soft">
                      {new Date(report.created_at).toLocaleString()}
                    </td>
                    <td>
                      <a
                        href={report.url}
                        target="_blank"
                        rel="noreferrer noopener"
                        className="text-[13px] text-accent hover:underline"
                      >
                        Open report
                      </a>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        <aside className="w-80 shrink-0">
          <div className="card px-6 py-5">
            <div className="mb-1 eyebrow">What is monitored</div>
            <p className="text-[13px] leading-relaxed text-ink-soft">
              Each patch is reduced to seven numeric features — per-channel means and standard
              deviations plus overall brightness — and compared against the training distribution.
              Stain and exposure shifts, and obviously out-of-domain inputs, both show up clearly
              in these.
            </p>
            <p className="mt-3 font-mono text-[11px] leading-relaxed text-ink-mute">
              {MONITORED_FEATURES}
            </p>
          </div>
        </aside>
      </div>
    </div>
  );
}
