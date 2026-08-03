import { Icon, Tooltip, type IconName } from "@blueprintjs/core";

export type View = "console" | "pipeline" | "registry" | "drift";

const ITEMS: { icon: IconName; label: string; view: View }[] = [
  { icon: "dashboard", label: "Console", view: "console" },
  { icon: "flows", label: "Pipeline", view: "pipeline" },
  { icon: "database", label: "Registry", view: "registry" },
  { icon: "timeline-line-chart", label: "Drift", view: "drift" },
];

export function Rail({ view, onNavigate }: { view: View; onNavigate: (v: View) => void }) {
  return (
    <nav className="sticky top-0 flex h-screen w-14 shrink-0 flex-col items-center gap-1 bg-rail py-4">
      <div className="mb-4 font-mono text-[13px] font-medium text-paper">P/</div>
      {ITEMS.map((item) => {
        const active = item.view === view;
        return (
          <Tooltip key={item.label} content={item.label} placement="right" compact>
            <button
              className={`relative flex h-10 w-10 items-center justify-center rounded transition-colors duration-150 ${
                active
                  ? "bg-[#1C2127] before:absolute before:left-[-8px] before:top-2 before:h-6 before:w-[2px] before:bg-accent"
                  : "opacity-60 hover:opacity-100"
              }`}
              onClick={() => onNavigate(item.view)}
              aria-label={item.label}
            >
              <Icon icon={item.icon} color={active ? "#FFFFFF" : "#8F99A8"} />
            </button>
          </Tooltip>
        );
      })}
    </nav>
  );
}
