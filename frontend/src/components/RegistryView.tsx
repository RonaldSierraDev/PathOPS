import { useQuery } from "@tanstack/react-query";
import { fetchModels } from "../lib/api";

const TH = "border-b border-hairline pb-2 text-left font-mono text-[11px] font-medium uppercase tracking-eyebrow text-ink-mute";

export function RegistryView() {
  const models = useQuery({ queryKey: ["models"], queryFn: fetchModels });

  return (
    <div className="card px-6 py-5">
      <div className="mb-1 eyebrow">Model versions</div>
      <p className="mb-4 text-[13px] text-ink-mute">
        Every version recorded in the registry, with the live prediction volume it has served.
        The serving row is what this API instance loaded at startup.
      </p>
      {models.isError ? (
        <div className="py-6 text-[14px] text-ink-mute">
          Registry unavailable: {models.error.message}
        </div>
      ) : models.isSuccess && models.data.length === 0 ? (
        <div className="py-6 text-[14px] text-ink-mute">
          No model versions recorded yet. The API records one at startup once a database is
          configured.
        </div>
      ) : (
        <table className="w-full border-collapse">
          <thead>
            <tr>
              <th className={TH}>model</th>
              <th className={TH}>version</th>
              <th className={TH}>alias</th>
              <th className={TH}>mlflow run</th>
              <th className={TH}>registered</th>
              <th className={`${TH} text-right`}>predictions</th>
              <th className={TH}>status</th>
            </tr>
          </thead>
          <tbody>
            {(models.data ?? []).map((m) => (
              <tr key={`${m.model_name}-${m.version}`} className="h-9 border-b border-hairline last:border-b-0">
                <td className="pr-4 text-[13px] font-medium">{m.model_name}</td>
                <td className="pr-4 font-mono text-[12px]">v{m.version}</td>
                <td className="pr-4 text-[13px] text-ink-soft">{m.alias ?? "—"}</td>
                <td className="pr-4 font-mono text-[12px] text-ink-mute" title={m.mlflow_run_id ?? undefined}>
                  {m.mlflow_run_id ? `${m.mlflow_run_id.slice(0, 10)}…` : "—"}
                </td>
                <td className="pr-4 font-mono text-[12px] text-ink-soft">
                  {new Date(m.created_at).toLocaleDateString()}
                </td>
                <td className="pr-4 text-right font-mono text-[12px]">
                  {m.prediction_count.toLocaleString()}
                </td>
                <td>
                  {m.serving ? (
                    <span className="flex items-center gap-1.5">
                      <span className="h-1.5 w-1.5 rounded-full bg-ok" />
                      <span className="text-[12px] text-ink-soft">Serving</span>
                    </span>
                  ) : (
                    <span className="text-[12px] text-ink-mute">Registered</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
