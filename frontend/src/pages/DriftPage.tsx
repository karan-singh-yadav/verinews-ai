import { Bar, BarChart, CartesianGrid, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { DriftReport, Performance } from "../api/types";
import { ALERT, AXIS, GRID, MODEL, TOOLTIP } from "../components/charts";
import { ConfusionMatrix } from "./ModelPage";
import { useApi } from "../components/hooks";
import { ErrorNote, PageHeader, Section, Spinner, Stat } from "../components/ui";
import { num, pct } from "../lib/format";

const STATUS_TEXT: Record<DriftReport["status"], string> = {
  ok: "No significant drift", warning: "Possible drift", drift: "Drift detected",
  insufficient_data: "Not enough data yet", no_reference: "No reference profile",
};

export default function DriftPage() {
  const d = useApi<DriftReport>("/api/monitoring/drift");
  const p = useApi<Performance>("/api/monitoring/performance");
  const flagged = d.data && (d.data.status === "drift" || d.data.status === "warning");

  return (
    <div>
      <PageHeader title="Drift and quality">
        Whether live traffic looks different from the data the model was trained on, and how the model does
        on cases a human has reviewed. These are two separate questions.
      </PageHeader>

      <h2 className="font-serif text-2xl font-medium">Data drift</h2>
      {d.loading && <div className="mt-3"><Spinner label="Loading…" /></div>}
      {d.error && <div className="mt-3"><ErrorNote error={d.error} onRetry={d.reload} /></div>}
      {d.data && (
        <div className="mt-3">
          <p className={`font-serif text-3xl ${flagged ? "text-alert" : ""}`}>{STATUS_TEXT[d.data.status]}</p>
          <p className="mt-1 max-w-2xl text-[15px]">{d.data.message}</p>
          <p className="mt-2 max-w-2xl text-sm text-muted">{d.data.interpretation}</p>
          <dl className="mt-5 grid max-w-xl grid-cols-3 gap-6">
            <Stat label="Live requests compared" value={d.data.n_current.toLocaleString()} />
            <Stat label="Reference size" value={d.data.n_reference.toLocaleString()} />
            <Stat label="Needed to judge" value={d.data.min_samples.toLocaleString()} />
          </dl>

          {d.data.features.length > 0 && (
            <>
              <Section title="How far each signal has moved" aside={<span className="text-sm text-muted">population stability index; 0.1 and 0.25 are the usual warning and alert levels</span>}>
                <div className="h-64" role="img" aria-label="Bar chart of population stability index per feature">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={d.data.features.map((f) => ({ name: f.feature.replace("_", " "), psi: f.psi, status: f.status }))}>
                      <CartesianGrid stroke={GRID} vertical={false} />
                      <XAxis dataKey="name" {...AXIS} /><YAxis {...AXIS} />
                      <Tooltip {...TOOLTIP} />
                      <ReferenceLine y={0.1} stroke={ALERT} strokeDasharray="4 4" />
                      <ReferenceLine y={0.25} stroke={ALERT} />
                      <Bar dataKey="psi" name="PSI">
                        {d.data.features.map((f) => <Cell key={f.feature} fill={f.status === "ok" ? MODEL : ALERT} />)}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <div className="mt-6 overflow-x-auto">
                  <table className="w-full min-w-[34rem] text-left text-sm">
                    <caption className="sr-only">Drift statistics per feature</caption>
                    <thead className="text-muted"><tr><th className="py-2 pr-4 font-normal">Signal</th><th className="pr-4 font-normal">Training average</th>
                      <th className="pr-4 font-normal">Live average</th><th className="pr-4 font-normal">PSI</th><th className="pr-4 font-normal">KS p-value</th><th className="font-normal">Status</th></tr></thead>
                    <tbody className="divide-y divide-line">
                      {d.data.features.map((f) => (
                        <tr key={f.feature}><td className="py-2 pr-4">{f.feature.replace("_", " ")}</td><td>{num(f.reference_mean, 2)}</td>
                          <td>{num(f.current_mean, 2)}</td><td>{num(f.psi, 3)}</td><td>{f.ks_p_value < 0.001 ? "< 0.001" : num(f.ks_p_value, 3)}</td>
                          <td className={f.status === "ok" ? "" : "text-alert"}>{f.status === "ok" ? "stable" : f.status === "warning" ? "possible shift" : "shifted"}</td></tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Section>
              <Section title="Share of requests">
                <table className="w-full max-w-xl text-left text-sm">
                  <caption className="sr-only">Prediction rate changes</caption>
                  <thead className="text-muted"><tr><th className="py-2 pr-4 font-normal">Rate</th><th className="pr-4 font-normal">Training</th><th className="pr-4 font-normal">Live</th><th className="font-normal">Status</th></tr></thead>
                  <tbody className="divide-y divide-line">
                    {d.data.rates.map((r) => (
                      <tr key={r.rate}><td className="py-2 pr-4">{r.rate.replace(/_/g, " ")}</td><td>{pct(r.reference, 1)}</td><td>{pct(r.current, 1)}</td>
                        <td className={r.status === "ok" ? "" : "text-alert"}>{r.status === "ok" ? "stable" : "shifted"}</td></tr>
                    ))}
                  </tbody>
                </table>
              </Section>
            </>
          )}
        </div>
      )}

      <Section title="Quality on reviewed cases">
        {p.error && <ErrorNote error={p.error} onRetry={p.reload} />}
        {p.data && p.data.status === "insufficient_labels" && (
          <div className="max-w-2xl">
            <p className="font-serif text-2xl">No accuracy figure yet</p>
            <p className="mt-1 text-[15px]">{p.data.message}</p>
            <p className="mt-2 text-sm text-muted">
              Reviewed so far: {p.data.reviewed_usable} of {p.data.min_labels} needed ({p.data.total_feedback} feedback records in total).
              Label records with <code>python -m ml.monitoring.review --list</code>.
            </p>
          </div>
        )}
        {p.data && p.data.metrics && (
          <>
            <dl className="grid max-w-3xl grid-cols-2 gap-6 sm:grid-cols-4">
              <Stat label="Reviewed cases" value={p.data.metrics.n} />
              <Stat label="Accuracy" value={pct(p.data.metrics.accuracy, 1)} hint={`95% interval ${pct(p.data.metrics.accuracy_ci95[0], 1)} to ${pct(p.data.metrics.accuracy_ci95[1], 1)}`} />
              <Stat label="Precision (fake)" value={pct(p.data.metrics.precision_fake, 1)} />
              <Stat label="Recall (fake)" value={pct(p.data.metrics.recall_fake, 1)} />
            </dl>
            <div className="mt-6 overflow-x-auto"><ConfusionMatrix matrix={p.data.metrics.confusion_matrix} labels={p.data.metrics.confusion_matrix_labels} /></div>
          </>
        )}
        {p.data && <p className="mt-4 max-w-2xl text-sm text-muted">{p.data.note}</p>}
      </Section>
    </div>
  );
}
