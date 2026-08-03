import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import "@blueprintjs/core/lib/css/blueprint.css";
import "./index.css";
import App from "./App.tsx";

const queryClient = new QueryClient({
  defaultOptions: {
    // networkMode "always": navigator.onLine is unreliable in embedded browsers
    // and the API may be on localhost/VPC where "offline" still works.
    // retry 0: the 30s poll is the retry; failures should surface immediately.
    queries: { refetchInterval: 30_000, retry: 0, networkMode: "always" },
    mutations: { networkMode: "always" },
  },
});

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </StrictMode>,
);
