import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { MetricsReport, ModelInfo } from "../api/types";
import { AXIS, GRID, MODEL, MODEL_DIM, TOOLTIP } from "../components/charts";
import { useApi } from "../components/hooks";
import { ErrorNote, PageHeader, Section, Spinner, Stat } from "../components/ui";
import { num, pct, when } from "../lib/format";

export function ConfusionMatrix({ matrix, labels }: { matrix: number[][]; labels: string[] }) {
  return (
    <table className="text-sm" aria-label="Confusion matrix">
      <caption className="mb-2 text-left text-muted">Rows are the true label, columns what the model predicted.</caption>
      <thead><tr><th />{labels.map((l) => <th key={l} scope="col" className="px-4 pb-1 text-left font-normal text-muted">predicted {l}</th>)}</tr></thead>
      <tbody>
        {matrix.map((row, i) => {
          const total = row.reduce((a, b) => a + b, 0) || 1;
          return (
            <tr key={labels[i]}>
              <th scope="row" className="pr-4 text-left font-normal text-muted">true {labels[i]}</th>
              {row.map((c, j) => (
                <td key={j} className="h-20 w-40 border border-line p-2 align-top"
                  style={{ background: `rgba(142,162,255,${(0.08 + 0.55 * (c / total)).toFixed(2)})` }}>
                  <span className="font-serif text-2xl">{c.toLocaleString()}</span>
                  <span className="block text-xs text-text/80">{pct(c / total, 1)} of true {labels[i]}</span>
                </td>
              ))}
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

export default function ModelPage() {
  const { data: m, error, loading, reload } = useApi<ModelInfo>("/api/model-info");
  const t: MetricsReport | undefined = m?.test_metrics;
  return (
    <div>
      <PageHeader title="Model">
        How the active model was trained and how it scored on data it never saw during training.
      </PageHeader>
      {loading && <Spinner label="Loading…" />}
      {error && <ErrorNote error={error} onRetry={reload} />}
      {m && t && (
        <>
          {m.synthetic_data && (
            <p role="alert" className="mb-6 max-w-2xl rounded-lg border border-alert/60 bg-alert/10 px-4 py-3 text-sm">
              This model was trained on synthetic test data. The scores below say nothing about real-world
              performance. Train on the real dataset to get meaningful numbers.
            </p>
          )}
          <dl className="grid max-w-3xl grid-cols-2 gap-6 sm:grid-cols-4">
            <Stat label="Version" value={m.model_version} />
            <Stat label="Algorithm" value={m.selected_model.replace("_", " ")} hint="chosen on validation data" />
            <Stat label="Trained" value={when(m.trained_at)} />
            <Stat label="Articles after cleaning" value={m.dataset.rows_after_dedup.toLocaleString()} hint={`${m.dataset.rows_raw.toLocaleString()} before removing duplicates`} />
          </dl>

          <Section title="Results on the held-out test set" aside={<span className="text-sm text-muted">{t.n_samples.toLocaleString()} articles</span>}>
            <dl className="grid max-w-3xl grid-cols-2 gap-6 sm:grid-cols-5">
              <Stat label="Accuracy" value={pct(t.accuracy, 1)} />
              <Stat label="Precision (fake)" value={pct(t.precision_fake, 1)} />
              <Stat label="Recall (fake)" value={pct(t.recall_fake, 1)} />
              <Stat label="F1 (fake)" value={pct(t.f1_fake, 1)} />
              <Stat label="ROC-AUC" value={num(t.roc_auc)} />
            </dl>
            <div className="mt-8 overflow-x-auto"><ConfusionMatrix matrix={t.confusion_matrix} labels={t.confusion_matrix_labels} /></div>
          </Section>

          <Section title="Algorithms compared on validation data">
            <div className="h-64" role="img" aria-label="Bar chart comparing algorithms on validation accuracy and macro F1">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={Object.entries(m.validation_comparison).map(([name, v]) => ({
                  name: name.replace("_", " "), accuracy: v.accuracy, f1_macro: v.f1_macro }))}>
                  <CartesianGrid stroke={GRID} vertical={false} />
                  <XAxis dataKey="name" {...AXIS} /><YAxis domain={[0, 1]} {...AXIS} />
                  <Tooltip {...TOOLTIP} /><Legend />
                  <Bar dataKey="accuracy" name="Accuracy" fill={MODEL} />
                  <Bar dataKey="f1_macro" name="Macro F1" fill={MODEL_DIM} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Section>

          <Section title="Confidence and unfamiliar-text checks">
            <dl className="grid max-w-3xl grid-cols-2 gap-6 sm:grid-cols-4">
              <Stat label="Calibration error (ECE)" value={num(m.calibration.ece)} hint={m.calibration.valid ? "within the limit: probabilities shown as validated" : "above the limit: confidence shown as not validated"} />
              <Stat label="Brier score" value={num(m.calibration.brier)} />
              <Stat label="Low-confidence threshold" value={pct(m.confidence_threshold, 1)} hint="chosen on validation data" />
              <Stat label="Unfamiliar-text limit" value={num(m.novelty.threshold)} hint={`${m.novelty.percentile}th percentile of validation scores`} />
            </dl>
            <p className="mt-4 max-w-2xl text-sm text-muted">{m.novelty.note}</p>
          </Section>

          <Section title="Limits to keep in mind">
            <ul className="max-w-2xl list-disc space-y-1.5 pl-5 text-[15px]">{m.limitations.map((l) => <li key={l}>{l}</li>)}</ul>
          </Section>
        </>
      )}
    </div>
  );
}
