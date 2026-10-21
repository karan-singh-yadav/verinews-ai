import { useState } from "react";
import { api, ApiError } from "../api/client";
import type { VerifyResponse } from "../api/types";
import EvidencePanel from "../components/EvidencePanel";
import ExamplePicker, { type Example } from "../components/ExamplePicker";
import FeedbackBox from "../components/FeedbackBox";
import ModelReading from "../components/ModelReading";
import { ErrorNote, PageHeader, Spinner } from "../components/ui";
import { recordCheck } from "../lib/history";
import { safeHref, when } from "../lib/format";

type Mode = "text" | "url";
const MAX = 20000;

export default function CheckPage() {
  const [mode, setMode] = useState<Mode>("text");
  const [text, setText] = useState("");
  const [url, setUrl] = useState("");
  const [busy, setBusy] = useState<null | "analyze" | "verify">(null);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<VerifyResponse | null>(null);
  const [sent, setSent] = useState<{ mode: Mode; input: string } | null>(null);
  const [example, setExample] = useState<Example | null>(null);

  const input = mode === "text" ? text.trim() : url.trim();

  async function run(force: boolean) {
    if (!input || busy) return;
    setBusy(force ? "verify" : "analyze");
    setError(null);
    try {
      const res = mode === "text" ? await api.verifyText(input, force) : await api.verifyUrl(input, force);
      setResult(res);
      setSent({ mode, input });
      recordCheck(mode, input, res);
    } catch (e) {
      setResult(null);
      setError(e instanceof ApiError ? e.message : "Unexpected error. Please try again.");
    } finally {
      setBusy(null);
    }
  }

  const tab = (m: Mode, label: string) => (
    <button
      role="tab" aria-selected={mode === m} onClick={() => setMode(m)}
      className={`px-4 py-2 text-sm font-medium ${mode === m ? "border-b-2 border-model text-text" : "text-muted hover:text-text"}`}
    >{label}</button>
  );

  return (
    <div>
      <PageHeader title="Check a news story">
        Paste an article or give a link. The model reads the writing style, and when it is unsure or the text
        looks unfamiliar, we search for fact-checks and news reports as a separate step.
      </PageHeader>

      <div role="tablist" aria-label="Input type" className="flex border-b border-line">
        {tab("text", "Paste text")}
        {tab("url", "Article link")}
      </div>

      <div className="mt-4 max-w-3xl">
        {mode === "text" ? (
          <>
            <label htmlFor="article" className="sr-only">Article text</label>
            <textarea
              id="article" value={text} maxLength={MAX} rows={9}
              onChange={(e) => {
                setText(e.target.value);
                if (example && e.target.value.trim() !== example.text.trim()) setExample(null);
              }}
              placeholder="Paste the headline and body of the story you want to check."
              className="w-full resize-y rounded-lg border border-line bg-panel p-3.5 text-[15px] leading-relaxed placeholder:text-muted"
            />
            <p className="mt-1 text-right text-xs text-muted">{text.length.toLocaleString()} / {MAX.toLocaleString()}</p>
            <ExamplePicker onPick={(ex) => { setText(ex.text); setExample(ex); setError(null); }} />
          </>
        ) : (
          <>
            <label htmlFor="link" className="sr-only">Article link</label>
            <input
              id="link" type="url" value={url} onChange={(e) => setUrl(e.target.value)} inputMode="url"
              placeholder="https://example.com/news/story"
              className="w-full rounded-lg border border-line bg-panel p-3.5 text-[15px] placeholder:text-muted"
            />
            <p className="mt-1 text-xs text-muted">
              Only public web pages are fetched. Local, private and non-web addresses are refused.
            </p>
          </>
        )}

        <div className="mt-4 flex flex-wrap items-center gap-3">
          <button
            onClick={() => run(false)} disabled={!input || !!busy}
            className="rounded-lg bg-model px-5 py-2.5 text-sm font-semibold text-ink hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-40"
          >{busy === "analyze" ? "Analyzing…" : "Analyze news"}</button>
          <button
            onClick={() => run(true)} disabled={!input || !!busy}
            className="rounded-lg border border-model/70 px-5 py-2.5 text-sm font-semibold text-model hover:bg-model/10 disabled:cursor-not-allowed disabled:opacity-40"
          >{busy === "verify" ? "Searching…" : "Verify online"}</button>
          <div aria-live="polite">
            {busy && <Spinner label={busy === "verify" ? "Searching fact-checks and news. This can take up to 40 seconds." : "Reading the text…"} />}
          </div>
        </div>
        <p className="mt-2 text-xs leading-relaxed text-muted">
          Analyze news searches online only if the model is unsure or the text looks unfamiliar. Verify online always searches.
        </p>
      </div>

      {error && <div className="mt-6 max-w-3xl"><ErrorNote error={error} /></div>}

      {result && sent && (
        <div className="mt-10" aria-live="polite">
          {result.source && (
            <p className="mb-4 text-sm text-muted">
              Fetched “{result.source.title || "untitled page"}” from{" "}
              {safeHref(result.source.url) ? (
                <a className="underline underline-offset-4" href={safeHref(result.source.url)} target="_blank" rel="noopener noreferrer nofollow">
                  {new URL(result.source.url).hostname}
                </a>
              ) : "the page"}{" "}
              · {when(result.source.fetched_at)}
            </p>
          )}

          {example && sent.mode === "text" && sent.input === example.text.trim() && result.ml && (
            <p className="mb-5 max-w-3xl rounded-lg border border-line px-4 py-3 text-sm">
              Dataset label: <span className="font-medium">{example.label}</span> · Model leaned:{" "}
              <span className="font-medium text-model">{result.ml.label}</span> ·{" "}
              {result.ml.label === example.label ? "they agree" : "they disagree"}
              <span className="mt-1 block text-muted">
                The dataset label comes from the article’s source, not from a fact-check, so a disagreement
                does not prove the model is wrong.
              </span>
            </p>
          )}

          {result.verification.triggered ? (
            <div className="mb-5 max-w-3xl rounded-lg border border-line px-4 py-3 text-sm">
              <p className="font-medium">Why we searched online</p>
              <ul className="mt-1 list-disc space-y-0.5 pl-5 text-muted">
                {result.verification.reasons.map((r) => <li key={r}>{r}</li>)}
              </ul>
            </div>
          ) : null}

          <div className="grid gap-6 xl:grid-cols-2">
            {result.ml ? <ModelReading pred={result.ml} /> : (
              <section className="rounded-xl border border-line bg-panel p-5">
                <h2 className="font-serif text-2xl font-medium">Model reading</h2>
                <p className="mt-3 text-sm text-muted">{result.ml_note ?? "The model is not available."}</p>
              </section>
            )}
            <EvidencePanel verification={result.verification} busy={busy === "verify"} onVerify={() => run(true)} />
          </div>

          <p className="mt-6 max-w-3xl text-sm leading-relaxed text-muted">{result.disclaimer}</p>
          <FeedbackBox key={result.request_id} res={result} text={sent.mode === "text" ? sent.input : null} />
        </div>
      )}
    </div>
  );
}