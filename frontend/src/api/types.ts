// Mirrors the FastAPI responses. Every number shown in the UI comes from these payloads.
export type Verdict = "Supported" | "Contradicted" | "Mixed Evidence" | "Inconclusive";
export type Stance = "supports" | "contradicts" | "mixed" | "neutral";

export interface Prediction {
  label: "fake" | "real";
  fake_probability: number;
  confidence: number;
  confidence_validated: boolean;
  low_confidence: boolean;
  novelty_score: number;
  novelty_threshold: number;
  out_of_distribution: boolean;
  needs_verification: boolean;
  verification_reasons: string[];
  model_version: string;
  word_count: number;
  char_length: number;
  confidence_threshold: number;
}

export interface EvidenceItem {
  claim_index: number;
  kind: "fact_check" | "news";
  provider: string;
  title: string;
  url: string;
  snippet: string;
  publisher: string;
  domain: string;
  published_at: string | null;
  retrieved_at: string;
  rating: string | null;
  reviewed_claim: string | null;
  relevance: number;
  stance: Stance;
  stance_basis: string;
  source_tier: "fact_checker" | "wire_or_major" | "official" | "unrated";
  stale: boolean;
}

export interface ClaimAssessment {
  claim: { text: string; entities: string[]; dates: string[]; figures: string[]; keywords: string[] };
  verdict: Verdict;
  reason: string;
  evidence: EvidenceItem[];
}

export interface ProviderStatus {
  name: string;
  kind: string;
  status: "ok" | "partial" | "error" | "not_configured";
  results: number;
  error: string | null;
}

export interface VerificationResult {
  verdict: Verdict;
  completed: boolean;
  summary: string;
  providers: ProviderStatus[];
  claims_checked: number;
  claims: ClaimAssessment[];
  supporting: EvidenceItem[];
  contradicting: EvidenceItem[];
  other: EvidenceItem[];
  evidence_count: number;
  discarded_irrelevant: number;
  limitations: string[];
  started_at: string;
  completed_at: string;
}

export interface VerifyResponse {
  request_id: string;
  ml: Prediction | null;
  ml_note: string | null;
  verification: { triggered: boolean; reasons: string[]; result: VerificationResult | null; note?: string };
  disclaimer: string;
  source?: { url: string; title: string; published: string | null; redirects: number; fetched_at: string };
}

export interface Health {
  status: "ok" | "degraded";
  model_loaded: boolean;
  model_version: string | null;
  providers_configured: Record<string, boolean>;
  time: string;
}

export interface MetricsReport {
  accuracy: number; precision_fake: number; recall_fake: number; f1_fake: number; f1_macro: number;
  roc_auc?: number; confusion_matrix: number[][]; confusion_matrix_labels: string[]; n_samples: number;
}

export interface ModelInfo {
  model_version: string;
  selected_model: string;
  trained_at: string;
  synthetic_data: boolean;
  dataset: { type: string; rows_raw: number; rows_after_dedup: number; class_counts: Record<string, number>;
             split_sizes: Record<string, number> };
  validation_comparison: Record<string, Partial<MetricsReport>>;
  test_metrics: MetricsReport;
  calibration: { ece: number; brier: number; valid: boolean };
  confidence_threshold: number;
  novelty: { threshold: number; percentile: number; test_fraction_flagged_ood: number; note: string };
  limitations: string[];
}

export interface ExperimentVersion {
  model_version: string; active: boolean; selected_model: string | null; trained_at: string | null;
  synthetic_data: boolean; split_sizes: Record<string, number> | null;
  test_metrics: Record<string, number | null>;
  calibration: { ece: number; brier: number; valid: boolean } | null;
}
export interface Experiments { active: string | null; versions: ExperimentVersion[]; note: string }

export interface DriftFeature {
  feature: string; psi: number; ks_statistic: number; ks_p_value: number;
  status: "ok" | "warning" | "drift"; reference_mean: number; current_mean: number;
}
export interface DriftRate { rate: string; reference: number; current: number; difference: number; p_value: number; status: "ok" | "drift" }
export interface DriftReport {
  status: "ok" | "warning" | "drift" | "insufficient_data" | "no_reference";
  message: string; interpretation: string; n_current: number; n_reference: number; min_samples: number;
  features: DriftFeature[]; rates: DriftRate[]; model_version: string; reference_small?: boolean;
}

export interface Performance {
  status: "ok" | "insufficient_labels";
  total_feedback: number; reviewed_usable: number; min_labels: number; note: string; message?: string;
  metrics: null | { n: number; accuracy: number; accuracy_ci95: [number, number];
    precision_fake: number | null; recall_fake: number | null; f1_fake: number | null;
    confusion_matrix: number[][]; confusion_matrix_labels: string[] };
}

export interface Analytics {
  total: number;
  by_day: { day: string; total: number; fake: number; ood: number; verifications: number }[];
  labels: Record<string, number>;
  verdicts: Record<string, number>;
  rates: { out_of_distribution: number; low_confidence: number; verification_triggered: number };
  note?: string;
}

export interface ServiceMetrics {
  uptime_seconds: number; total_requests: number; model_version: string | null;
  routes: Record<string, { count: number; errors_4xx: number; errors_5xx: number;
                           latency_ms_p50: number | null; latency_ms_p95: number | null }>;
  predictions: Record<string, number>; verification: Record<string, number>;
  verification_provider_failures: Record<string, number>;
  feedback: { total: number; by_user_label: Record<string, number>;
              user_agreement_with_model: { agree: number; disagree: number }; note: string };
}
