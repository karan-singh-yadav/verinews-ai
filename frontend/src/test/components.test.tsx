import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import EvidencePanel from "../components/EvidencePanel";
import ModelReading from "../components/ModelReading";
import { evidence, pred, result } from "./fixtures";

describe("ModelReading", () => {
  it("shows two scales with accessible values and keeps the model separate from verdicts", () => {
    render(<ModelReading pred={pred()} />);
    const style = screen.getByRole("meter", { name: "Style score" });
    expect(style).toHaveAttribute("aria-valuenow", "0.62");
    expect(screen.getByRole("meter", { name: "Familiarity" })).toBeInTheDocument();
    expect(screen.getByText(/not a fact check/i)).toBeInTheDocument();
    expect(screen.getByText(/Low \(below 90% threshold\)/)).toBeInTheDocument();
    expect(screen.queryByText(/Contradicted|Supported/)).not.toBeInTheDocument();
  });
  it("says so when confidence is not validated or text is unfamiliar", () => {
    render(<ModelReading pred={pred({ confidence_validated: false, out_of_distribution: true })} />);
    expect(screen.getByText(/% confidence, not validated\)/)).toBeInTheDocument();
    expect(screen.getByText("Unfamiliar text")).toBeInTheDocument();
    expect(screen.getByText("Calibration not validated")).toBeInTheDocument();
  });
});

describe("EvidencePanel", () => {
  const v = (r: ReturnType<typeof result> | null, note?: string) =>
    ({ triggered: r !== null, reasons: [], result: r, note });

  it("shows the verdict, rating, source link and retrieval time", () => {
    render(<EvidencePanel verification={v(result())} />);
    expect(screen.getByLabelText("Verdict: Contradicted")).toBeInTheDocument();
    const link = screen.getByRole("link", { name: "Clinics claim checked" });
    expect(link).toHaveAttribute("href", "https://www.altnews.in/clinics");
    expect(link).toHaveAttribute("rel", expect.stringContaining("noopener"));
    expect(screen.getByText(/rated “False”/)).toBeInTheDocument();
    expect(screen.getByText(/Match with claim 82%.*retrieved/)).toBeInTheDocument();
  });
  it("never claims a verdict when verification could not be completed", () => {
    const r = result({ completed: false, verdict: "Inconclusive", summary: "Online verification could not be completed (gdelt: GDELT could not be reached).",
      providers: [{ name: "gdelt", kind: "news", status: "error", results: 0, error: "GDELT could not be reached" },
                  { name: "google_fact_check", kind: "fact_check", status: "not_configured", results: 0, error: "GOOGLE_FACT_CHECK_API_KEY is not set" }],
      claims: [], contradicting: [], evidence_count: 0 });
    render(<EvidencePanel verification={v(r)} />);
    expect(screen.getByText("Verification incomplete")).toBeInTheDocument();
    expect(screen.queryByLabelText(/Verdict:/)).not.toBeInTheDocument();
    expect(screen.getByText(/Nothing here suggests the article is true or false/)).toBeInTheDocument();
    expect(screen.getByText(/failed: GDELT could not be reached/)).toBeInTheDocument();
    expect(screen.getByText(/not set up: GOOGLE_FACT_CHECK_API_KEY is not set/)).toBeInTheDocument();
  });
  it("shows an honest Inconclusive when the search worked but found nothing", () => {
    const r = result({ verdict: "Inconclusive", summary: "Insufficient or no relevant evidence was found.", claims: [], contradicting: [], evidence_count: 0 });
    render(<EvidencePanel verification={v(r)} />);
    expect(screen.getByLabelText("Verdict: Inconclusive")).toBeInTheDocument();
  });
  it("renders unsafe links as plain text, not anchors", () => {
    const r = result({ claims: [{ claim: { text: "c", entities: [], dates: [], figures: [], keywords: [] }, verdict: "Inconclusive", reason: "r",
      evidence: [evidence({ url: "javascript:alert(1)", title: "Bad link" })] }] });
    render(<EvidencePanel verification={v(r)} />);
    expect(screen.queryByRole("link", { name: "Bad link" })).not.toBeInTheDocument();
    expect(screen.getByText(/Bad link/)).toBeInTheDocument();
    expect(screen.getByText(/link unavailable/)).toBeInTheDocument();
  });
  it("offers to verify when verification was not triggered", () => {
    render(<EvidencePanel verification={v(null, "The model was confident.")} onVerify={() => {}} />);
    const panel = screen.getByRole("region", { name: "Online evidence" });
    expect(within(panel).getByText("The model was confident.")).toBeInTheDocument();
    expect(within(panel).getByRole("button", { name: "Verify online" })).toBeInTheDocument();
  });
  it("labels who counts as which tier and stale or unrated sources honestly", () => {
    const r = result({ claims: [{ claim: { text: "c", entities: [], dates: [], figures: [], keywords: [] }, verdict: "Inconclusive", reason: "r",
      evidence: [evidence({ kind: "news", rating: null, source_tier: "unrated", stance: "neutral", stance_basis: "Older than the recency window; not counted", title: "Old story" })] }] });
    render(<EvidencePanel verification={v(r)} />);
    expect(screen.getByText(/unrated source/)).toBeInTheDocument();
    expect(screen.getByText("Related, not counted")).toBeInTheDocument();
  });
});
