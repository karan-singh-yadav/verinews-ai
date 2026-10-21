import { Bar, BarChart, CartesianGrid, Cell, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { Analytics, ServiceMetrics } from "../api/types";
import { AXIS, GRID, MODEL, MODEL_DIM, TOOLTIP } from "../components/charts";
import { useApi } from "../components/hooks";
import { Empty, ErrorNote, PageHeader, Section, Spinner, Stat } from "../components/ui";
import { pct } from "../lib/format";

const VERDICT_COLOR: Record<string, string> = {
  Supported: "#5ccba6", Contradicted: "#f2766c", "Mixed Evidence": "#e8b94c", Inconclusive: "#a7b3d0",
};

export default function AnalyticsPage() {
  const a = useApi<Analytics>("/api/analytics");
  const m = useApi<ServiceMetrics>("/api/metrics");

  return (
    <div>
      <PageHeader title="Analytics">
        What this server has actually handled. These are counts of logged requests, not accuracy.
      </PageHeader>
      {a.loading && <Spinner label="Loading…" />}
      {a.error && <ErrorNote error={a.error} onRetry={a.reload} />}

      {a.data && a.data.total === 0 && (
        <Empty title="No requests logged yet">Run a few checks and the charts will fill in. No sample data is shown.</Empty>
      )}

      {a.data && a.data.total > 0 && (
        <>
          <dl className="grid max-w-3xl grid-cols-2 gap-6 sm:grid-cols-4">
            <Stat label="Requests logged" value={a.data.total.toLocaleString()} />
            <Stat label="Unfamiliar text" value={pct(a.data.rates.out_of_distribution, 1)} />
            <Stat label="Low confidence" value={pct(a.data.rates.low_confidence, 1)} />
            <Stat label="Searched online" value={pct(a.data.rates.verification_triggered, 1)} />
          </dl>

          <Section title="Requests per day" aside={<span className="text-sm text-muted">by what the model leaned toward</span>}>
            <div className="h-64" role="img" aria-label="Stacked bar chart of daily requests split by model lean">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={a.data.by_day.map((d) => ({ ...d, real: d.total - d.fake }))}>
                  <CartesianGrid stroke={GRID} vertical={false} />
                  <XAxis dataKey="day" {...AXIS} /><YAxis allowDecimals={false} {...AXIS} />
                  <Tooltip {...TOOLTIP} /><Legend />
                  <Bar dataKey="fake" name="Leaned fake" stackId="a" fill={MODEL} />
                  <Bar dataKey="real" name="Leaned real" stackId="a" fill={MODEL_DIM} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Section>

          <Section title="Online verdicts">
            {Object.keys(a.data.verdicts).length === 0 ? (
              <p className="text-sm text-muted">No online verification has run yet.</p>
            ) : (
              <div className="h-56" role="img" aria-label="Bar chart of online verdict counts">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart layout="vertical" data={Object.entries(a.data.verdicts).map(([name, count]) => ({ name, count }))}>
                    <CartesianGrid stroke={GRID} horizontal={false} />
                    <XAxis type="number" allowDecimals={false} {...AXIS} />
                    <YAxis type="category" dataKey="name" width={120} {...AXIS} />
                    <Tooltip {...TOOLTIP} />
                    <Bar dataKey="count" name="Requests">
                      {Object.keys(a.data.verdicts).map((k) => <Cell key={k} fill={VERDICT_COLOR[k] ?? MODEL} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
          </Section>
        </>
      )}

      <Section title="Service health">
        {m.error && <ErrorNote error={m.error} onRetry={m.reload} />}
        {m.data && (
          <>
            <dl className="grid max-w-3xl grid-cols-2 gap-6 sm:grid-cols-4">
              <Stat label="Requests handled" value={m.data.total_requests.toLocaleString()} />
              <Stat label="Uptime" value={`${Math.round(m.data.uptime_seconds / 60)} min`} />
              <Stat label="Verifications finished" value={m.data.verification.completed ?? 0} />
              <Stat label="Verifications incomplete" value={m.data.verification.incomplete ?? 0} />
            </dl>
            {Object.keys(m.data.verification_provider_failures).length > 0 && (
              <p className="mt-4 text-sm text-alert">
                Provider failures: {Object.entries(m.data.verification_provider_failures).map(([k, v]) => `${k} (${v})`).join(", ")}
              </p>
            )}
            <div className="mt-6 overflow-x-auto">
              <table className="w-full min-w-[32rem] text-left text-sm">
                <caption className="sr-only">Requests, errors and latency by route</caption>
                <thead className="text-muted"><tr>
                  <th className="py-2 pr-4 font-normal">Route</th><th className="pr-4 font-normal">Requests</th>
                  <th className="pr-4 font-normal">Client errors</th><th className="pr-4 font-normal">Server errors</th>
                  <th className="pr-4 font-normal">Median ms</th><th className="font-normal">95th pct ms</th></tr></thead>
                <tbody className="divide-y divide-line">
                  {Object.entries(m.data.routes).map(([route, r]) => (
                    <tr key={route}><td className="py-2 pr-4">{route}</td><td>{r.count}</td><td>{r.errors_4xx}</td>
                      <td>{r.errors_5xx}</td><td>{r.latency_ms_p50 ?? "n/a"}</td><td>{r.latency_ms_p95 ?? "n/a"}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </Section>
    </div>
  );
}
