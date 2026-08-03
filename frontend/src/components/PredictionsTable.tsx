import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  createColumnHelper,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable,
  type SortingState,
} from "@tanstack/react-table";
import { fetchRecentPredictions, type RecentPrediction } from "../lib/api";

const col = createColumnHelper<RecentPrediction>();

export function PredictionsTable() {
  const recent = useQuery({
    queryKey: ["recent"],
    queryFn: () => fetchRecentPredictions(200),
  });
  const [sorting, setSorting] = useState<SortingState>([{ id: "created_at", desc: true }]);

  const columns = useMemo(
    () => [
      col.accessor("created_at", {
        header: "time",
        cell: (c) => (
          <span className="font-mono text-[12px] text-ink-soft">
            {new Date(c.getValue()).toLocaleTimeString()}
          </span>
        ),
      }),
      col.accessor("input_hash", {
        header: "input",
        enableSorting: false,
        cell: (c) => (
          <span className="font-mono text-[12px] text-ink-mute" title={c.getValue()}>
            {c.getValue().slice(0, 12)}…
          </span>
        ),
      }),
      col.accessor("predicted_label", {
        header: "label",
        cell: (c) => {
          const tumor = c.getValue() === "tumor";
          return (
            <span className="flex items-center gap-2">
              <span
                className="h-1.5 w-1.5 rounded-full"
                style={{ background: tumor ? "#C87619" : "#238551" }}
              />
              <span className="text-[13px]">{c.getValue()}</span>
            </span>
          );
        },
      }),
      col.accessor("confidence", {
        header: "conf",
        cell: (c) => (
          <span className="block text-right font-mono text-[12px]">
            {(c.getValue() * 100).toFixed(1)}%
          </span>
        ),
      }),
      col.accessor("latency_ms", {
        header: "latency",
        cell: (c) => {
          const v = c.getValue();
          return (
            <span className="block text-right font-mono text-[12px]">
              {v != null ? `${v.toFixed(1)} ms` : "—"}
            </span>
          );
        },
      }),
    ],
    [],
  );

  const table = useReactTable({
    data: recent.data ?? [],
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  });

  return (
    <div className="card px-6 py-5">
      <div className="mb-3 eyebrow">Recent predictions</div>
      {recent.isError ? (
        <div className="py-6 text-[14px] text-ink-mute">
          Predictions unavailable: {recent.error.message}
        </div>
      ) : recent.isSuccess && recent.data.length === 0 ? (
        <div className="py-6 text-[14px] text-ink-mute">
          No predictions yet. Run one from the panel on the right.
        </div>
      ) : (
        <div className="max-h-96 overflow-y-auto">
          <table className="w-full border-collapse">
            <thead>
              {table.getHeaderGroups().map((hg) => (
                <tr key={hg.id}>
                  {hg.headers.map((h) => (
                    <th
                      key={h.id}
                      onClick={h.column.getToggleSortingHandler()}
                      className={`sticky top-0 border-b border-hairline bg-paper pb-2 text-left font-mono text-[11px] font-medium uppercase tracking-eyebrow text-ink-mute ${
                        h.column.getCanSort() ? "cursor-pointer select-none" : ""
                      } ${["conf", "latency"].includes(h.column.columnDef.header as string) ? "text-right" : ""}`}
                    >
                      {flexRender(h.column.columnDef.header, h.getContext())}
                      {{ asc: " ↑", desc: " ↓" }[h.column.getIsSorted() as string] ?? ""}
                    </th>
                  ))}
                </tr>
              ))}
            </thead>
            <tbody>
              {table.getRowModel().rows.map((row) => (
                <tr key={row.id} className="h-8 border-b border-hairline last:border-b-0">
                  {row.getVisibleCells().map((cell) => (
                    <td key={cell.id} className="pr-4">
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
