// Hand-written TEST fixtures. They are never imported by application code.
import type { EvidenceItem, Prediction, VerificationResult, VerifyResponse } from "../api/types";

export const pred = (o: Partial<Prediction> = {}): Prediction => ({
  label: "fake", fake_probability: 0.62, confidence: 0.62, confidence_validated: true, low_confidence: true,
  novelty_score: 0.31, novelty_threshold: 0.4, out_of_distribution: false, needs_verification: true,
  verification_reasons: ["Model confidence 0.62 is below the validated threshold 0.90."],
  model_version: "vtest", word_count: 120, char_length: 700, confidence_threshold: 0.9, ...o,
});

export const evidence = (o: Partial<EvidenceItem> = {}): EvidenceItem => ({
  claim_index: 0, kind: "fact_check", provider: "google_fact_check", title: "Clinics claim checked",
  url: "https://www.altnews.in/clinics", snippet: "", publisher: "Alt News", domain: "altnews.in",
  published_at: "2026-03-14T00:00:00Z", retrieved_at: "2026-10-16T10:00:00+00:00", rating: "False",
  reviewed_claim: "4,500 clinics will open", relevance: 0.82, stance: "contradicts",
  stance_basis: "Publisher rating: 'False'", source_tier: "fact_checker", stale: false, ...o,
});

export const result = (o: Partial<VerificationResult> = {}): VerificationResult => ({
  verdict: "Contradicted", completed: true, summary: "1 independent fact-checking source(s) rated the claim false or unsupported.",
  providers: [{ name: "google_fact_check", kind: "fact_check", status: "ok", results: 1, error: null }],
  claims_checked: 1,
  claims: [{ claim: { text: "4,500 clinics will open", entities: [], dates: [], figures: [], keywords: [] },
    verdict: "Contradicted", reason: "rated false", evidence: [evidence()] }],
  supporting: [], contradicting: [evidence()], other: [], evidence_count: 1, discarded_irrelevant: 0,
  limitations: ["Always open the cited sources and judge for yourself."],
  started_at: "2026-10-16T10:00:00+00:00", completed_at: "2026-10-16T10:00:05+00:00", ...o,
});

export const response = (o: Partial<VerifyResponse> = {}): VerifyResponse => ({
  request_id: "req1", ml: pred(), ml_note: null,
  verification: { triggered: true, reasons: pred().verification_reasons, result: result() },
  disclaimer: "The ML result estimates writing style. It is not proof.", ...o,
});

export const json = (body: unknown, status = 200, headers: Record<string, string> = {}) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json", ...headers } });
