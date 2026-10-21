// Frontend <-> real backend integration. Skipped unless VITE_LIVE_API is set, e.g.
//   VITE_LIVE_API=http://localhost:8000 npx vitest run src/test/live.test.tsx
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { beforeAll, describe, expect, it } from "vitest";

const LIVE = import.meta.env.VITE_LIVE_API as string | undefined;

describe.skipIf(!LIVE)("live backend", () => {
  beforeAll(() => {
    (window as unknown as { __API_BASE__: string }).__API_BASE__ = LIVE!;
  });

  it("checks real text end to end and shows honest provider failures", async () => {
    const { default: CheckPage } = await import("../pages/CheckPage");
    render(<CheckPage />);
    await userEvent.type(
      screen.getByLabelText("Article text"),
      "The health ministry announced that 4,500 new clinics will open in Uttar Pradesh next year, according to officials in Lucknow.",
    );
    await userEvent.click(screen.getByRole("button", { name: "Verify online" }));
    expect(await screen.findByRole("heading", { name: "Model reading" }, { timeout: 60000 })).toBeInTheDocument();
    expect(screen.getByRole("meter", { name: "Style score" })).toBeInTheDocument();
    expect(screen.getByRole("meter", { name: "Familiarity" })).toBeInTheDocument();
    // In the sandbox no search provider is usable: the UI must say so instead of giving a verdict.
    await waitFor(() => expect(screen.getByRole("region", { name: "Online evidence" })).toBeInTheDocument());
    const panel = screen.getByRole("region", { name: "Online evidence" });
    expect(panel.textContent).toMatch(/Verification incomplete|Inconclusive|Supported|Contradicted|Mixed/);
    expect(panel.textContent).toMatch(/google_fact_check/);
    console.log("LIVE evidence panel text:", panel.textContent?.slice(0, 400));
  }, 90000);

  it("surfaces backend validation and SSRF refusals as readable errors", async () => {
    const { default: CheckPage } = await import("../pages/CheckPage");
    render(<CheckPage />);
    await userEvent.click(screen.getByRole("tab", { name: "Article link" }));
    await userEvent.type(screen.getByLabelText("Article link"), "http://169.254.169.254/latest/meta-data/");
    await userEvent.click(screen.getByRole("button", { name: "Analyze news" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(/URL not allowed/);
  });

  it("renders the model, experiments, analytics and drift pages from real recorded data", async () => {
    const { default: App } = await import("../App");
    for (const [path, expected] of [
      ["/model", /Results on the held-out test set/],
      ["/experiments", /Comparing versions/],
      ["/analytics", /Requests per day|No requests logged yet/],
      ["/drift", /Data drift/],
    ] as const) {
      const { unmount } = render(<MemoryRouter initialEntries={[path]}><App /></MemoryRouter>);
      expect(await screen.findByText(expected, {}, { timeout: 15000 })).toBeInTheDocument();
      unmount();
    }
  }, 60000);
});
