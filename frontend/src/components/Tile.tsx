export function Tile({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="card px-6 py-5">
      <div className="eyebrow mb-2">{label}</div>
      {children}
    </div>
  );
}

export function StatusDot({ color, text }: { color: string; text: string }) {
  return (
    <div className="flex items-center gap-2">
      <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: color }} />
      <span className="text-[15px] text-ink-soft">{text}</span>
    </div>
  );
}

/** Shared with the SVG charts, which need literal colors rather than Tailwind classes. */
export const STATUS = {
  neutral: "#ABB3BF",
  ok: "#238551",
  warn: "#C87619",
  danger: "#CD4246",
} as const;

export const TILE_NUMBER = "font-mono text-[24px] font-medium leading-none";
