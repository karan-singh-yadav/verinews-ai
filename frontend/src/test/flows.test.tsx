import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import CheckPage from "../pages/CheckPage";
import { loadHistory } from "../lib/history";
import { json, pred, response, result } from "./fixtures";

afterEach(() => vi.unstubAllGlobals());

const body = (call: unknown[]) => JSON.parse((call[1] as RequestInit).body as string);

describe("Check page", () => {
  it("disables both buttons until there is input", async () => {
    render(<CheckPage />);
    expect(screen.getByRole("button", { name: "Analyze news" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Verify online" })).toBeDisabled();
    await userEvent.type(screen.getByLabelText("Article text"), "Some news");
    expect(screen.getByRole("button", { name: "Analyze news" })).toBeEnabled();
  });

  it("Analyze runs conditional verification; Verify online forces it", async () => {
    const f = vi.fn().mockResolvedValue(json(response()));
    vi.stubGlobal("fetch", f);
    render(<CheckPage />);
    await userEvent.type(screen.getByLabelText("Article text"), "Ministry announced clinics.");
    await userEvent.click(screen.getByRole("button", { name: "Analyze news" }));
    await screen.findByRole("heading", { name: "Model reading" });
    expect(f.mock.calls[0][0]).toBe("/api/verify");
    expect(body(f.mock.calls[0])).toEqual({ text: "Ministry announced clinics.", force_verification: false });
    await userEvent.click(screen.getByRole("button", { name: "Verify online" }));
    await waitFor(() => expect(f).toHaveBeenCalledTimes(2));
    expect(body(f.mock.calls[1]).force_verification).toBe(true);
  });

  it("shows why verification was triggered, both panels, and the verdict", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json(response())));
    render(<CheckPage />);
    await userEvent.type(screen.getByLabelText("Article text"), "x");
    await userEvent.click(screen.getByRole("button", { name: "Analyze news" }));
    expect(await screen.findByText("Why we searched online")).toBeInTheDocument();
    expect(screen.getByText(/below the validated threshold/)).toBeInTheDocument();
    expect(screen.getByLabelText("Verdict: Contradicted")).toBeInTheDocument();
    expect(screen.getByText(/not proof/)).toBeInTheDocument();
  });

  it("uses the link endpoint in link mode", async () => {
    const f = vi.fn().mockResolvedValue(json(response({ source: { url: "https://news.example.com/a", title: "Headline", published: null, redirects: 0, fetched_at: "2026-10-16T10:00:00Z" } })));
    vi.stubGlobal("fetch", f);
    render(<CheckPage />);
    await userEvent.click(screen.getByRole("tab", { name: "Article link" }));
    await userEvent.type(screen.getByLabelText("Article link"), "https://news.example.com/a");
    await userEvent.click(screen.getByRole("button", { name: "Analyze news" }));
    expect(await screen.findByText(/Fetched “Headline”/)).toBeInTheDocument();
    expect(f.mock.calls[0][0]).toBe("/api/verify-url");
  });

  it("shows a clear error and no stale result when the API fails", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json({ message: "URL not allowed: Local or internal hostnames are not allowed" }, 400)));
    render(<CheckPage />);
    await userEvent.click(screen.getByRole("tab", { name: "Article link" }));
    await userEvent.type(screen.getByLabelText("Article link"), "http://localhost/");
    await userEvent.click(screen.getByRole("button", { name: "Analyze news" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("URL not allowed");
    expect(screen.queryByRole("heading", { name: "Model reading" })).not.toBeInTheDocument();
  });

  it("works when the model is missing: no ML panel numbers, evidence still shown", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json(response({ ml: null, ml_note: "The ML model is not available; online verification was run without it." }))));
    render(<CheckPage />);
    await userEvent.type(screen.getByLabelText("Article text"), "x");
    await userEvent.click(screen.getByRole("button", { name: "Verify online" }));
    expect(await screen.findByText(/model is not available/)).toBeInTheDocument();
    expect(screen.queryByRole("meter")).not.toBeInTheDocument();
  });

  it("saves a privacy-safe history record after a check", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json(response({ verification: { triggered: false, reasons: [], result: null, note: "Confident." }, ml: pred({ needs_verification: false }) }))));
    render(<CheckPage />);
    await userEvent.type(screen.getByLabelText("Article text"), "private article words");
    await userEvent.click(screen.getByRole("button", { name: "Analyze news" }));
    await screen.findByText("Confident.");
    expect(loadHistory()).toHaveLength(1);
    expect(JSON.stringify(loadHistory())).not.toContain("private");
    expect(screen.queryByText("Why we searched online")).not.toBeInTheDocument();
  });

  it("sends feedback without text unless the user consents", async () => {
    const f = vi.fn().mockImplementation((url: string) =>
      Promise.resolve(url === "/api/feedback" ? json({ id: "f1", stored_text: false, message: "ok" }, 201) : json(response())));
    vi.stubGlobal("fetch", f);
    render(<CheckPage />);
    await userEvent.type(screen.getByLabelText("Article text"), "article body");
    await userEvent.click(screen.getByRole("button", { name: "Analyze news" }));
    await userEvent.click(await screen.findByRole("button", { name: "I think it's fake" }));
    await screen.findByText(/Your feedback is saved/);
    const fb = f.mock.calls.find((c) => c[0] === "/api/feedback")!;
    expect(body(fb)).toMatchObject({ user_label: "fake", predicted_label: "fake", model_version: "vtest" });
    expect(body(fb).text).toBeUndefined();
  });
});

describe("App and pages", () => {
  const route = (path: string) => render(<MemoryRouter initialEntries={[path]}><App /></MemoryRouter>);

  it("shows API status in the sidebar, including unset providers", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json({ status: "degraded", model_loaded: false, model_version: null,
      providers_configured: { google_fact_check: false, gdelt: true }, time: "" })));
    route("/");
    expect(await screen.findByText("No model trained yet")).toBeInTheDocument();
    expect(screen.getByText(/google_fact_check: not set up/)).toBeInTheDocument();
    expect(screen.getByText(/gdelt: ready/)).toBeInTheDocument();
  });

  it("analytics shows an empty state, never sample data", async () => {
    vi.stubGlobal("fetch", vi.fn().mockImplementation((u: string) =>
      Promise.resolve(u.includes("analytics") ? json({ total: 0, by_day: [], labels: {}, verdicts: {}, rates: {} })
        : u.includes("metrics") ? json({ uptime_seconds: 60, total_requests: 0, model_version: null, routes: {}, predictions: {}, verification: {}, verification_provider_failures: {}, feedback: {} })
        : json({ status: "ok", model_loaded: true, model_version: "v", providers_configured: {}, time: "" }))));
    route("/analytics");
    expect(await screen.findByText("No requests logged yet")).toBeInTheDocument();
    expect(screen.getByText(/No sample data is shown/)).toBeInTheDocument();
  });

  it("drift page makes no claim with too little data and no accuracy figure without labels", async () => {
    vi.stubGlobal("fetch", vi.fn().mockImplementation((u: string) =>
      Promise.resolve(u.includes("drift") ? json({ status: "insufficient_data", message: "Need at least 100 logged requests to assess drift (have 3). No drift claim is made.", interpretation: "Drift is not accuracy.", n_current: 3, n_reference: 90, min_samples: 100, features: [], rates: [], model_version: "v" })
        : u.includes("performance") ? json({ status: "insufficient_labels", total_feedback: 4, reviewed_usable: 1, min_labels: 50, note: "Feedback is self-selected.", message: "Need at least 50 expert-reviewed examples. No accuracy claim is made.", metrics: null })
        : json({ status: "ok", model_loaded: true, model_version: "v", providers_configured: {}, time: "" }))));
    route("/drift");
    expect(await screen.findByText("Not enough data yet")).toBeInTheDocument();
    expect(screen.getByText("No accuracy figure yet")).toBeInTheDocument();
    expect(screen.getByText(/1 of 50 needed/)).toBeInTheDocument();
  });

  it("model page explains a missing model and flags synthetic data", async () => {
    vi.stubGlobal("fetch", vi.fn().mockImplementation((u: string) =>
      Promise.resolve(u.includes("model-info") ? json({ message: "No trained model is available." }, 503)
        : json({ status: "degraded", model_loaded: false, model_version: null, providers_configured: {}, time: "" }))));
    route("/model");
    expect(await screen.findByText("No trained model available")).toBeInTheDocument();
  });

  it("history page deletes records", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json({ status: "ok", model_loaded: true, model_version: "v", providers_configured: {}, time: "" })));
    const { recordCheck } = await import("../lib/history");
    recordCheck("url", "https://news.example.com/x", response({ verification: { triggered: true, reasons: [], result: result() } }));
    route("/history");
    expect(await screen.findByText("Link to news.example.com")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Delete all history" }));
    expect(await screen.findByText("No checks saved yet")).toBeInTheDocument();
    expect(loadHistory()).toEqual([]);
  });
});
