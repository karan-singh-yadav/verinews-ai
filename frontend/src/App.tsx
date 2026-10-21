import { lazy, Suspense } from "react";
import { NavLink, Route, Routes } from "react-router-dom";
import type { Health } from "./api/types";
import { useApi } from "./components/hooks";
import CheckPage from "./pages/CheckPage";
import HistoryPage from "./pages/HistoryPage";

// Chart pages are loaded on demand so the first screen stays small.
const AnalyticsPage = lazy(() => import("./pages/AnalyticsPage"));
const DriftPage = lazy(() => import("./pages/DriftPage"));
const ExperimentsPage = lazy(() => import("./pages/ExperimentsPage"));
const ModelPage = lazy(() => import("./pages/ModelPage"));

const NAV = [
  { to: "/", label: "Check news", end: true },
  { to: "/history", label: "History" },
  { to: "/analytics", label: "Analytics" },
  { to: "/model", label: "Model" },
  { to: "/experiments", label: "Experiments" },
  { to: "/drift", label: "Drift and quality" },
];

function Status() {
  const { data, error } = useApi<Health>("/api/health");
  if (error) return <p className="text-xs text-contradicted">API unreachable</p>;
  if (!data) return <p className="text-xs text-muted">Connecting…</p>;
  const providers = Object.entries(data.providers_configured);
  return (
    <div className="space-y-1 text-xs text-muted">
      <p>
        <span className={data.model_loaded ? "text-supported" : "text-mixed"}>●</span>{" "}
        {data.model_loaded ? `Model ${data.model_version}` : "No model trained yet"}
      </p>
      {providers.map(([name, ok]) => (
        <p key={name}><span className={ok ? "text-supported" : "text-mixed"}>●</span> {name}: {ok ? "ready" : "not set up"}</p>
      ))}
    </div>
  );
}

export default function App() {
  return (
    <div className="min-h-screen lg:grid lg:grid-cols-[15rem_1fr]">
      <aside className="border-b border-line bg-panel/60 px-4 py-4 lg:sticky lg:top-0 lg:h-screen lg:border-b-0 lg:border-r lg:py-8">
        <p className="font-serif text-2xl font-medium">VeriNews</p>
        <nav aria-label="Main" className="mt-3 flex gap-1 overflow-x-auto lg:mt-8 lg:flex-col">
          {NAV.map((n) => (
            <NavLink key={n.to} to={n.to} end={n.end}
              className={({ isActive }) =>
                `whitespace-nowrap rounded-md px-3 py-2 text-sm ${isActive ? "bg-raised font-medium text-text" : "text-muted hover:text-text"}`}
            >{n.label}</NavLink>
          ))}
        </nav>
        <div className="mt-4 hidden border-t border-line pt-4 lg:block"><Status /></div>
      </aside>
      <main className="mx-auto w-full max-w-6xl px-5 py-8 lg:px-10 lg:py-12">
        <Suspense fallback={<p className="text-sm text-muted">Loading…</p>}>
        <Routes>
          <Route path="/" element={<CheckPage />} />
          <Route path="/history" element={<HistoryPage />} />
          <Route path="/analytics" element={<AnalyticsPage />} />
          <Route path="/model" element={<ModelPage />} />
          <Route path="/experiments" element={<ExperimentsPage />} />
          <Route path="/drift" element={<DriftPage />} />
          <Route path="*" element={<p className="text-muted">Page not found.</p>} />
        </Routes>
        </Suspense>
      </main>
    </div>
  );
}
