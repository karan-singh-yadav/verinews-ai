import type { ClaimAssessment, EvidenceItem, Stance, VerifyResponse } from "../api/types";
import { TIER_LABEL, pct, safeHref, when } from "../lib/format";
import { Spinner, VERDICT_TEXT, VerdictBadge } from "./ui";

const STANCE: Record<Stance, { text: string; bar: string; tone: string }> = {
  supports: { text: "Supports the claim", bar: "border-l-supported", tone: "text-supported" },
  contradicts: { text: "Contradicts the claim", bar: "border-l-contradicted", tone: "text-contradicted" },
  mixed: { text: "Mixed or partly true", bar: "border-l-mixed", tone: "text-mixed" },
  neutral: { text: "Related, not counted", bar: "border-l-inconclusive", tone: "text-inconclusive" },
};

export function EvidenceCard({ item }: { item: EvidenceItem }) {
  const s = STANCE[item.stance];
  const href = safeHref(item.url);
  return (
    <li className={`border-l-4 ${s.bar} bg-raised/60 py-3 pl-4 pr-3`}>
      <p className={`text-sm font-medium ${s.tone}`}>
        {s.text}
        {item.kind === "fact_check" && item.rating && <span className="text-text"> · rated “{item.rating}”</span>}
      </p>
      <p className="mt-1 font-serif text-lg leading-snug">
        {href ? (
          <a href={href} target="_blank" rel="noopener noreferrer nofollow" className="underline decoration-line underline-offset-4 hover:decoration-model">
            {item.title}
          </a>
        ) : (
          <span>{item.title} <span className="text-xs text-muted">(link unavailable)</span></span>
        )}
      </p>
      <p className="mt-1 text-sm text-muted">
        {item.publisher || item.domain} · {TIER_LABEL[item.source_tier]} ·{" "}
        {item.kind === "fact_check" ? "fact-check" : "news report"} · {when(item.published_at)}
      </p>
      {item.reviewed_claim && <p className="mt-1 text-sm text-muted">Claim reviewed: “{item.reviewed_claim}”</p>}
      <p className="mt-1 text-xs text-muted">
        Match with claim {pct(item.relevance)} · {item.stance_basis} · retrieved {when(item.retrieved_at)}
      </p>
    </li>
  );
}

function ClaimBlock({ c, index }: { c: ClaimAssessment; index: number }) {
  return (
    <div className="mt-6">
      <div className="flex flex-wrap items-center gap-2">
        <h4 className="font-medium">Claim {index + 1}</h4>
        <VerdictBadge verdict={c.verdict} />
      </div>
      <p className="mt-1 font-serif text-lg leading-snug">“{c.claim.text}”</p>
      <p className="mt-1 text-sm text-muted">{c.reason}</p>
      {c.evidence.length > 0 && (
        <ul className="mt-3 space-y-2" aria-label={`Evidence for claim ${index + 1}`}>
          {c.evidence.map((e) => <EvidenceCard key={e.url} item={e} />)}
        </ul>
      )}
    </div>
  );
}

interface Props {
  verification: VerifyResponse["verification"];
  busy?: boolean;
  onVerify?: () => void;
}

export default function EvidencePanel({ verification, busy, onVerify }: Props) {
  const r = verification.result;
  return (
    <section aria-labelledby="evidence-h" className="rounded-xl border border-line bg-panel p-5">
      <h2 id="evidence-h" className="font-serif text-2xl font-medium">Online evidence</h2>
      <p className="mt-1 text-sm leading-relaxed text-muted">
        Fact-checks and news reports found for the main claims. Judged separately from the model.
      </p>

      {!r && (
        <div className="mt-5">
          <p className="text-[15px] leading-relaxed">{verification.note ?? "No online verification was run."}</p>
          {onVerify && (
            <button onClick={onVerify} disabled={busy}
              className="mt-3 rounded-lg border border-line px-4 py-2 text-sm font-medium hover:bg-raised disabled:opacity-50">
              Verify online
            </button>
          )}
          {busy && <div className="mt-3"><Spinner label="Searching sources…" /></div>}
        </div>
      )}

      {r && (
        <>
          <div className="mt-5">
            {r.completed ? (
              <p className={`font-serif text-4xl ${VERDICT_TEXT[r.verdict]}`} aria-label={`Verdict: ${r.verdict}`}>
                {r.verdict}
              </p>
            ) : (
              <p className="font-serif text-3xl text-inconclusive">Verification incomplete</p>
            )}
            <p className="mt-2 text-[15px] leading-relaxed">{r.summary}</p>
            {!r.completed && (
              <p className="mt-2 rounded-md border border-mixed/50 bg-mixed/10 px-3 py-2 text-sm">
                No search provider answered, so this is not a finding about the article. Nothing here
                suggests the article is true or false.
              </p>
            )}
          </div>

          <ul className="mt-4 flex flex-wrap gap-2" aria-label="Search providers">
            {r.providers.map((p) => (
              <li key={p.name} className="rounded-md border border-line px-2.5 py-1 text-xs">
                <span className="font-medium">{p.name}</span>{" "}
                {p.status === "ok" && <span className="text-muted">{p.results} results</span>}
                {p.status === "partial" && <span className="text-mixed">partial: {p.error}</span>}
                {p.status === "error" && <span className="text-contradicted">failed: {p.error}</span>}
                {p.status === "not_configured" && <span className="text-mixed">not set up: {p.error}</span>}
              </li>
            ))}
          </ul>

          {r.claims.map((c, i) => <ClaimBlock key={c.claim.text} c={c} index={i} />)}

          <details className="mt-6 text-sm text-muted">
            <summary className="cursor-pointer text-text">How this was judged and its limits</summary>
            <ul className="mt-2 list-disc space-y-1 pl-5">
              {r.limitations.map((l) => <li key={l}>{l}</li>)}
            </ul>
            <p className="mt-2">{r.discarded_irrelevant} retrieved items were dropped as not relevant to the claims.</p>
          </details>
        </>
      )}
    </section>
  );
}
