import { afterEach, describe, expect, it, vi } from "vitest";
import { api, ApiError } from "../api/client";
import { clearHistory, deleteItem, loadHistory, loadPrefs, recordCheck, savePrefs } from "../lib/history";
import { safeHref } from "../lib/format";
import { json, response } from "./fixtures";

afterEach(() => vi.unstubAllGlobals());

describe("api client errors", () => {
  it("explains a network failure in plain language", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
    await expect(api.get("/api/health")).rejects.toThrow(/Can't reach the API/);
  });
  it("uses the server's message and never leaks raw bodies", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json({ error: "http_503", message: "No trained model is available." }, 503)));
    await expect(api.get("/x")).rejects.toMatchObject({ status: 503, message: "No trained model is available." });
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("<html>Traceback...</html>", { status: 500 })));
    await expect(api.get("/x")).rejects.toThrow("The server returned an error (HTTP 500).");
  });
  it("tells the user how long to wait on 429", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json({ message: "x" }, 429, { "retry-after": "17" })));
    const err = (await api.verifyText("hello", false).catch((e: unknown) => e)) as ApiError;
    expect(err.message).toBe("Too many requests. Try again in 17 seconds.");
    expect(err.retryAfter).toBe(17);
  });
  it("summarises validation errors without echoing input", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json({ error: "validation_error", message: "Invalid request.", details: [{ field: "text", message: "text must not be empty" }] }, 422)));
    await expect(api.verifyText(" ", false)).rejects.toThrow("Invalid input: text must not be empty");
  });
  it("sends force_verification and JSON content type", async () => {
    const f = vi.fn().mockResolvedValue(json(response()));
    vi.stubGlobal("fetch", f);
    await api.verifyText("abc", true);
    const [url, init] = f.mock.calls[0];
    expect(url).toBe("/api/verify");
    expect(JSON.parse(init.body)).toEqual({ text: "abc", force_verification: true });
  });
});

describe("history privacy", () => {
  it("stores no text by default and only the hostname for links", () => {
    recordCheck("text", "my secret pasted article text", response());
    recordCheck("url", "https://news.example.com/a/very/long/path?token=abc", response({ request_id: "req2" }));
    const raw = localStorage.getItem("verinews.history.v1")!;
    expect(raw).not.toContain("secret");
    expect(raw).not.toContain("token=abc");
    expect(raw).toContain("news.example.com");
    expect(loadHistory()[1].preview).toBeNull();
  });
  it("keeps an 80 character preview only when the user opts in", () => {
    savePrefs({ saveHistory: true, keepPreview: true });
    recordCheck("text", "x".repeat(200), response());
    expect(loadHistory()[0].preview).toHaveLength(80);
  });
  it("saves nothing when history is switched off", () => {
    savePrefs({ saveHistory: false, keepPreview: false });
    expect(recordCheck("text", "abc", response())).toBeNull();
    expect(loadHistory()).toEqual([]);
  });
  it("deletes single items and everything", () => {
    recordCheck("text", "a", response({ request_id: "a" }));
    recordCheck("text", "b", response({ request_id: "b" }));
    deleteItem("a");
    expect(loadHistory().map((h) => h.id)).toEqual(["b"]);
    clearHistory();
    expect(loadHistory()).toEqual([]);
  });
  it("survives corrupted storage", () => {
    localStorage.setItem("verinews.history.v1", "{not json");
    localStorage.setItem("verinews.prefs.v1", "nope");
    expect(loadHistory()).toEqual([]);
    expect(loadPrefs().saveHistory).toBe(true);
  });
});

describe("safeHref", () => {
  it("allows only http(s)", () => {
    expect(safeHref("https://a.com/x")).toBe("https://a.com/x");
    expect(safeHref("javascript:alert(1)")).toBeUndefined();
    expect(safeHref("data:text/html,x")).toBeUndefined();
    expect(safeHref("not a url")).toBeUndefined();
  });
});
