import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { Experiments } from "../api/types";
import { AXIS, GRID, MODEL, MODEL_DIM, TOOLTIP } from "../components/charts";
import { useApi } from "../components/hooks";
import { Empty, ErrorNote, PageHeader, Section, Spinner } from "../components/ui";
import { num, pct, when } from "../lib/format";

export default function ExperimentsPage() {
  const { data, error, loading, reload } = useApi<Experiments>("/api/experiments");
  return (
    <div>
      <PageHeader title="Experiments">
        Every trained model version and what it scored on its own held-out test set.
      </PageHeader>
      {loading && <Spinner label="Loading…" />}
      {error && <ErrorNote error={error} onRetry={reload} />}
      {data && data.versions.length === 0 && (
        <Empty title="No trained models found">Train one with <code>python -m ml.training.train --dataset isot --data-dir data/raw</code>.</Empty>
      )}
      {data && data.versions.length > 0 && (
        <>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[40rem] text-left text-sm">
              <caption className="sr-only">Model versions and test metrics</caption>
              <thead className="text-muted"><tr>
                <th className="py-2 pr-4 font-normal">Version</th><th className="pr-4 font-normal">Algorithm</th>
                <th className="pr-4 font-normal">Trained</th><th className="pr-4 font-normal">Accuracy</th>
                <th className="pr-4 font-normal">F1 (fake)</th><th className="pr-4 font-normal">ROC-AUC</th>
                <th className="font-normal">Calibration</th></tr></thead>
              <tbody className="divide-y divide-line">
                {data.versions.map((v) => (
                  <tr key={v.model_version}>
                    <td className="py-3 pr-4">
                      {v.model_version}
                      {v.active && <span className="ml-2 rounded-full border border-model/60 px-2 py-0.5 text-xs text-model">active</span>}
                      {v.synthetic_data && <span className="ml-2 rounded-full border border-alert/60 px-2 py-0.5 text-xs text-alert">synthetic data</span>}
                    </td>
                    <td className="pr-4">{v.selected_model?.replace("_", " ") ?? "n/a"}</td>
                    <td className="pr-4">{when(v.trained_at)}</td>
                    <td className="pr-4">{pct(v.test_metrics.accuracy, 1)}</td>
                    <td className="pr-4">{pct(v.test_metrics.f1_fake, 1)}</td>
                    <td className="pr-4">{num(v.test_metrics.roc_auc)}</td>
                    <td>{v.calibration ? (v.calibration.valid ? "validated" : "not validated") : "n/a"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <Section title="Comparing versions">
            <div className="h-64" role="img" aria-label="Bar chart comparing model versions on accuracy and F1">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={[...data.versions].reverse().map((v) => ({
                  name: v.model_version, accuracy: v.test_metrics.accuracy, f1: v.test_metrics.f1_fake }))}>
                  <CartesianGrid stroke={GRID} vertical={false} />
                  <XAxis dataKey="name" {...AXIS} /><YAxis domain={[0, 1]} {...AXIS} />
                  <Tooltip {...TOOLTIP} /><Legend />
                  <Bar dataKey="accuracy" name="Accuracy" fill={MODEL} />
                  <Bar dataKey="f1" name="F1 (fake)" fill={MODEL_DIM} />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <p className="mt-3 max-w-2xl text-sm text-muted">
              Compare versions only if they were trained on the same dataset; synthetic-data versions are not comparable to real ones.
            </p>
          </Section>

          <Section title="Full run history">
            <p className="max-w-2xl text-sm text-muted">{data.note}</p>
            <pre className="mt-3 overflow-x-auto rounded-md border border-line bg-panel p-3 text-sm">mlflow ui --backend-store-uri sqlite:///mlflow.db</pre>
          </Section>
        </>
      )}
    </div>
  );
}
