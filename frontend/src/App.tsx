import { useState } from "react";
import { Rail, type View } from "./components/Rail";
import { OverviewTiles } from "./components/OverviewTiles";
import { LatencyChart } from "./components/LatencyChart";
import { PredictionsTable } from "./components/PredictionsTable";
import { PredictPanel } from "./components/PredictPanel";
import { PipelineView } from "./components/PipelineView";
import { RegistryView } from "./components/RegistryView";
import { DriftView } from "./components/DriftView";

function ConsoleView() {
  return (
    <>
      <OverviewTiles />
      <div className="mt-12 flex items-start gap-6">
        <section className="min-w-0 flex-1">
          <LatencyChart />
          <div className="mt-6">
            <PredictionsTable />
          </div>
        </section>
        <aside className="w-80 shrink-0">
          <PredictPanel />
        </aside>
      </div>
    </>
  );
}

const VIEWS: Record<View, { title: string; render: () => React.ReactNode }> = {
  console: { title: "Operations console", render: () => <ConsoleView /> },
  pipeline: { title: "Pipeline", render: () => <PipelineView /> },
  registry: { title: "Model registry", render: () => <RegistryView /> },
  drift: { title: "Data drift", render: () => <DriftView /> },
};

export default function App() {
  const [view, setView] = useState<View>("console");

  return (
    <div className="flex min-h-screen">
      <Rail view={view} onNavigate={setView} />
      <main className="mx-auto w-full max-w-6xl px-12 py-10">
        <header className="mb-12">
          <div className="eyebrow mb-1">PathML Pipeline</div>
          <h1 className="text-[30px] font-bold leading-tight tracking-tight">
            {VIEWS[view].title}
          </h1>
        </header>
        {VIEWS[view].render()}
      </main>
    </div>
  );
}
