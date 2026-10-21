import { useState } from "react";
import { Empty, PageHeader, VerdictBadge } from "../components/ui";
import { pct, when } from "../lib/format";
import { clearHistory, deleteItem, loadHistory, loadPrefs, savePrefs, type Prefs } from "../lib/history";

export default function HistoryPage() {
  const [items, setItems] = useState(loadHistory);
  const [prefs, setPrefs] = useState<Prefs>(loadPrefs);

  const update = (p: Prefs) => { setPrefs(p); savePrefs(p); };

  return (
    <div>
      <PageHeader title="History">
        Your past checks, kept only in this browser. Nothing here is sent to or stored on the server.
      </PageHeader>

      <fieldset className="max-w-xl space-y-2 border-b border-line pb-6 text-sm">
        <legend className="mb-2 font-medium">Privacy</legend>
        <label className="flex items-start gap-2">
          <input type="checkbox" className="mt-1" checked={prefs.saveHistory} onChange={(e) => update({ ...prefs, saveHistory: e.target.checked })} />
          <span>Save a record of each check on this device</span>
        </label>
        <label className="flex items-start gap-2">
          <input type="checkbox" className="mt-1" disabled={!prefs.saveHistory} checked={prefs.keepPreview} onChange={(e) => update({ ...prefs, keepPreview: e.target.checked })} />
          <span>Keep the first 80 characters of pasted text so I can recognise each check <span className="text-muted">(off: the text is never stored)</span></span>
        </label>
        <button
          disabled={items.length === 0}
          onClick={() => { clearHistory(); setItems([]); }}
          className="mt-2 rounded-md border border-line px-3 py-1.5 hover:bg-raised disabled:opacity-40"
        >Delete all history</button>
      </fieldset>

      {items.length === 0 ? (
        <div className="mt-8"><Empty title="No checks saved yet">
          {prefs.saveHistory ? "Run a check and it will appear here." : "History saving is switched off."}
        </Empty></div>
      ) : (
        <ul className="mt-6 divide-y divide-line">
          {items.map((h) => (
            <li key={h.id} className="flex flex-wrap items-start justify-between gap-3 py-4">
              <div className="min-w-0 max-w-2xl">
                <p className="font-serif text-lg">
                  {h.mode === "url" ? `Link to ${h.host ?? "unknown site"}` : h.preview ? `“${h.preview}…”` : "Pasted text (not stored)"}
                </p>
                <p className="mt-1 text-sm text-muted">
                  {when(h.ts)}
                  {h.label && <> · model leaned <span className="text-model">{h.label}</span> ({pct(h.fakeProbability)} fake)</>}
                  {h.outOfDistribution && " · unfamiliar text"}
                  {h.lowConfidence && " · low confidence"}
                </p>
                <p className="mt-1 text-sm">
                  {!h.verified && <span className="text-muted">No online check</span>}
                  {h.verified && h.verdict && h.verificationCompleted && <>Online evidence: <VerdictBadge verdict={h.verdict} /></>}
                  {h.verified && h.verificationCompleted === false && <span className="text-mixed">Online check could not be completed</span>}
                </p>
              </div>
              <button onClick={() => { deleteItem(h.id); setItems(loadHistory()); }}
                className="rounded-md border border-line px-3 py-1.5 text-sm hover:bg-raised" aria-label={`Delete check from ${when(h.ts)}`}>
                Delete
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
