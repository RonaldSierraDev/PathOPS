import { useMemo, useRef, useState } from "react";

const W = 720;
const H = 140;
const PAD = { top: 12, right: 8, bottom: 22, left: 44 };

export interface LinePoint {
  value: number;
  /** Tooltip text shown when this point is the nearest to the cursor. */
  label: string;
}

interface Props {
  points: LinePoint[];
  formatTick: (value: number) => string;
  /** Fixed axis maximum; defaults to the data's max rounded up. */
  yMax?: number;
  threshold?: { value: number; label: string };
  /** Draw a dot per point -- worth it for sparse series, noise for dense ones. */
  showPoints?: boolean;
  pointColor?: (value: number) => string;
}

/** Schematic single-series line chart: thin ink line, hairline gridlines, crosshair on hover. */
export function LineChart({
  points,
  formatTick,
  yMax: fixedYMax,
  threshold,
  showPoints = false,
  pointColor,
}: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);

  const { path, coords, yTicks, thresholdY } = useMemo(() => {
    const values = points.map((p) => p.value);
    const dataMax = Math.max(...values, threshold?.value ?? 0);
    const yMax = fixedYMax ?? (Math.ceil(dataMax / 50) * 50 || 50);
    const x = (i: number) => PAD.left + (i / (points.length - 1)) * (W - PAD.left - PAD.right);
    const y = (v: number) => PAD.top + (1 - v / yMax) * (H - PAD.top - PAD.bottom);
    const coords = points.map((p, i) => ({ x: x(i), y: y(p.value) }));
    return {
      path: coords.map((c, i) => `${i === 0 ? "M" : "L"}${c.x.toFixed(1)},${c.y.toFixed(1)}`).join(""),
      coords,
      yTicks: [0, yMax / 2, yMax].map((v) => ({ v, y: y(v) })),
      thresholdY: threshold ? y(threshold.value) : null,
    };
  }, [points, fixedYMax, threshold]);

  function onMove(e: React.MouseEvent<SVGSVGElement>) {
    if (coords.length === 0 || !svgRef.current) return;
    const rect = svgRef.current.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * W;
    let best = 0;
    for (let i = 1; i < coords.length; i++) {
      if (Math.abs(coords[i].x - px) < Math.abs(coords[best].x - px)) best = i;
    }
    setHoverIdx(best);
  }

  const hover = hoverIdx != null && coords[hoverIdx] ? { c: coords[hoverIdx], p: points[hoverIdx] } : null;
  // Flip the tooltip to the left of the cursor near the right edge so it never clips.
  const flip = (hover?.c.x ?? 0) > W - 150;

  return (
    <svg
      ref={svgRef}
      viewBox={`0 0 ${W} ${H}`}
      className="block w-full"
      onMouseMove={onMove}
      onMouseLeave={() => setHoverIdx(null)}
    >
      {yTicks.map((t) => (
        <g key={t.v}>
          <line x1={PAD.left} x2={W - PAD.right} y1={t.y} y2={t.y} stroke="#DCE0E5" strokeWidth={1} />
          <text x={PAD.left - 8} y={t.y + 3.5} textAnchor="end" className="fill-ink-mute font-mono" fontSize={10}>
            {formatTick(t.v)}
          </text>
        </g>
      ))}

      {threshold && thresholdY != null && (
        <g>
          <line
            x1={PAD.left}
            x2={W - PAD.right}
            y1={thresholdY}
            y2={thresholdY}
            stroke="#C87619"
            strokeWidth={1}
            strokeDasharray="4 4"
          />
          <text x={W - PAD.right} y={thresholdY - 5} textAnchor="end" className="fill-warn font-mono" fontSize={10}>
            {threshold.label}
          </text>
        </g>
      )}

      <path d={path} fill="none" stroke="#1C2127" strokeWidth={1.5} strokeLinejoin="round" />

      {showPoints &&
        coords.map((c, i) => (
          <circle
            key={i}
            cx={c.x}
            cy={c.y}
            r={3}
            fill={pointColor ? pointColor(points[i].value) : "#1C2127"}
            stroke="#FFFFFF"
            strokeWidth={1}
          />
        ))}

      {hover && (
        <g>
          <line x1={hover.c.x} x2={hover.c.x} y1={PAD.top} y2={H - PAD.bottom} stroke="#ABB3BF" strokeWidth={1} />
          <circle cx={hover.c.x} cy={hover.c.y} r={3.5} fill="#FFFFFF" stroke="#2D72D2" strokeWidth={1.5} />
          <text
            x={flip ? hover.c.x - 8 : hover.c.x + 8}
            y={PAD.top + 10}
            textAnchor={flip ? "end" : "start"}
            className="fill-ink font-mono"
            fontSize={11}
          >
            {hover.p.label}
          </text>
        </g>
      )}
    </svg>
  );
}
