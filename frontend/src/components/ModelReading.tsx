import type { Prediction } from "../api/types";
import { num, pct } from "../lib/format";
import Scale from "./Scale";

export default function ModelReading({ pred }: { pred: Prediction }) {
  const thr = pred.confidence_threshold;
  const uncertain = thr > 0.5 ? { from: 1 - thr, to: thr, label: "uncertain zone" } : undefined;
  const novMax = Math.max(1, pred.novelty_score * 1.1, pred.novelty_threshold * 1.5);
  const unfamiliar = { from: pred.novelty_threshold, to: novMax, label: "unfamiliar zone" };

  return (
    <section aria-labelledby="model-h" className="rounded-xl border border-line bg-panel p-5">
      <h2 id="model-h" className="font-serif text-2xl font-medium">Model reading</h2>
      <p className="mt-1 text-sm leading-relaxed text-muted">
        A style estimate learned from one dataset. It is not a fact check and says nothing about whether
        the claims are true.
      </p>

      <p className="mt-5 text-lg">
        Writing style leans <span className="font-semibold text-model">{pred.label}</span>
        <span className="text-muted"> ({pct(pred.confidence)} confidence{pred.confidence_validated ? "" : ", not validated"})</span>
      </p>

      <div className="mt-6 space-y-6">
        <Scale
          label="Style score"
          value={pred.fake_probability}
          max={1}
          zone={uncertain}
          leftLabel="reads as real"
          rightLabel="reads as fake"
          valueText={`${num(pred.fake_probability, 2)} probability of fake`}
        />
        <Scale
          label="Familiarity"
          value={pred.novelty_score}
          max={novMax}
          zone={unfamiliar}
          leftLabel="similar to training data"
          rightLabel="unlike anything seen"
          valueText={`novelty ${num(pred.novelty_score)} (limit ${num(pred.novelty_threshold)})`}
        />
      </div>

      <dl className="mt-6 grid grid-cols-2 gap-x-4 gap-y-3 border-t border-line pt-4 text-sm">
        <div><dt className="text-muted">Confidence</dt>
          <dd>{pred.low_confidence ? `Low (below ${pct(thr)} threshold)` : "Above threshold"}</dd></div>
        <div><dt className="text-muted">Familiarity</dt>
          <dd>{pred.out_of_distribution ? "Unfamiliar text" : "Looks familiar"}</dd></div>
        <div><dt className="text-muted">Probabilities</dt>
          <dd>{pred.confidence_validated ? "Calibrated and checked" : "Calibration not validated"}</dd></div>
        <div><dt className="text-muted">Length</dt><dd>{pred.word_count} words</dd></div>
      </dl>
      <p className="mt-4 text-xs text-muted">Model version {pred.model_version}</p>
    </section>
  );
}
