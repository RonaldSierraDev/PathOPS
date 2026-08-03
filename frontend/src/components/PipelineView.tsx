import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Background,
  BackgroundVariant,
  Handle,
  Position,
  ReactFlow,
  type Edge,
  type Node,
  type NodeProps,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { fetchFeedbackExport, fetchHealth, fetchRecentPredictions } from "../lib/api";

type StageStatus = { color: string; text: string };

const EXTERNAL: StageStatus = { color: "#ABB3BF", text: "Runs in CI / AWS" };

type StageData = {
  eyebrow: string;
  name: string;
  detail: string;
  status: StageStatus;
};

type StageNode = Node<StageData, "stage">;

function StageNodeView({ data }: NodeProps<StageNode>) {
  return (
    <div className="w-52 rounded-card border border-hairline bg-paper px-4 py-3">
      <Handle type="target" position={Position.Left} className="!h-1 !w-1 !min-h-0 !min-w-0 !border-0 !bg-ink-mute" />
      <div className="eyebrow mb-1">{data.eyebrow}</div>
      <div className="text-[14px] font-bold leading-tight">{data.name}</div>
      <div className="mt-1 text-[12px] leading-snug text-ink-soft">{data.detail}</div>
      <div className="mt-2 flex items-center gap-1.5">
        <span className="h-1.5 w-1.5 rounded-full" style={{ background: data.status.color }} />
        <span className="font-mono text-[10px] text-ink-mute">{data.status.text}</span>
      </div>
      <Handle type="source" position={Position.Right} className="!h-1 !w-1 !min-h-0 !min-w-0 !border-0 !bg-ink-mute" />
    </div>
  );
}

const nodeTypes = { stage: StageNodeView };

const EDGE_STYLE = { stroke: "#ABB3BF", strokeWidth: 1.25 };

export function PipelineView() {
  const health = useQuery({ queryKey: ["health"], queryFn: fetchHealth });
  const recent = useQuery({ queryKey: ["recent"], queryFn: () => fetchRecentPredictions(200) });
  const feedback = useQuery({ queryKey: ["feedback-export"], queryFn: fetchFeedbackExport });

  const { nodes, edges } = useMemo(() => {
    const serving: StageStatus = health.isPending
      ? { color: "#ABB3BF", text: "Checking" }
      : health.isError
        ? { color: "#CD4246", text: "Unreachable" }
        : { color: "#238551", text: "Healthy" };
    const predLog: StageStatus = recent.isSuccess
      ? { color: "#238551", text: `${recent.data.length} rows in window` }
      : recent.isError
        ? { color: "#C87619", text: recent.error.message }
        : { color: "#ABB3BF", text: "Checking" };
    const fb: StageStatus = feedback.isSuccess
      ? {
          color: feedback.data.length > 0 ? "#238551" : "#ABB3BF",
          text: `${feedback.data.length} corrections`,
        }
      : feedback.isError
        ? { color: "#C87619", text: feedback.error.message }
        : { color: "#ABB3BF", text: "Checking" };

    const mk = (
      id: string,
      x: number,
      y: number,
      data: StageData,
    ): StageNode => ({ id, type: "stage", position: { x, y }, data });

    const nodes: StageNode[] = [
      mk("dataset", 0, 140, {
        eyebrow: "Source",
        name: "PCam dataset",
        detail: "327k patches · S3 artifacts",
        status: EXTERNAL,
      }),
      mk("train", 260, 140, {
        eyebrow: "Compute",
        name: "Training",
        detail: "ResNet18 fine-tune · retrain.yml",
        status: EXTERNAL,
      }),
      mk("gate", 520, 140, {
        eyebrow: "Gate",
        name: "Eval + promotion",
        detail: "AUC / sensitivity floor · manual promote",
        status: EXTERNAL,
      }),
      mk("registry", 780, 140, {
        eyebrow: "Registry",
        name: "MLflow + S3",
        detail: "Versioned checkpoints",
        status: EXTERNAL,
      }),
      mk("serve", 1040, 140, {
        eyebrow: "Serving",
        name: "FastAPI on ECS",
        detail: "Fargate Spot · /predict",
        status: serving,
      }),
      mk("predlog", 1040, 320, {
        eyebrow: "Storage",
        name: "Prediction log",
        detail: "RDS Postgres · hash, label, latency",
        status: predLog,
      }),
      mk("drift", 780, 320, {
        eyebrow: "Monitoring",
        name: "Drift monitor",
        detail: "Lambda + Evidently · CloudWatch/SNS",
        status: EXTERNAL,
      }),
      mk("feedback", 520, 320, {
        eyebrow: "Loop",
        name: "Feedback",
        detail: "Corrected labels → retraining",
        status: fb,
      }),
    ];

    const mkEdge = (source: string, target: string, label?: string): Edge => ({
      id: `${source}-${target}`,
      source,
      target,
      label,
      type: "smoothstep",
      style: EDGE_STYLE,
      labelStyle: { fontFamily: "'JetBrains Mono', monospace", fontSize: 10, fill: "#404854" },
      labelBgStyle: { fill: "#F6F7F9" },
    });

    const edges: Edge[] = [
      mkEdge("dataset", "train"),
      mkEdge("train", "gate"),
      mkEdge("gate", "registry"),
      mkEdge("registry", "serve"),
      mkEdge("serve", "predlog"),
      mkEdge("predlog", "drift"),
      mkEdge("predlog", "feedback"),
      { ...mkEdge("feedback", "train", "corrections"), style: { ...EDGE_STYLE, stroke: "#2D72D2" } },
    ];

    return { nodes, edges };
  }, [health, recent, feedback]);

  return (
    <div className="card flex h-[calc(100vh-220px)] min-h-[420px] flex-col px-6 py-5">
      <div className="mb-3">
        <div className="eyebrow">Model lifecycle</div>
        <div className="text-[13px] text-ink-mute">
          Live stages read from the API; gray stages run in GitHub Actions / AWS. The accent edge
          is the feedback loop — retraining learns from corrections, not the static dataset.
        </div>
      </div>
      <div className="min-h-0 flex-1">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          fitView
          fitViewOptions={{ padding: 0.15 }}
          nodesDraggable={false}
          nodesConnectable={false}
          elementsSelectable={false}
          zoomOnScroll={false}
          panOnScroll
          preventScrolling={false}
        >
          <Background variant={BackgroundVariant.Dots} gap={24} size={1} color="#DCE0E5" />
        </ReactFlow>
      </div>
    </div>
  );
}
