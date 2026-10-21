import { useState } from "react";
import { api } from "../api/client";
import type { VerifyResponse } from "../api/types";

export default function FeedbackBox({ res, text }: { res: VerifyResponse; text: string | null }) {
  const [consent, setConsent] = useState(false);
  const [state, setState] = useState<"idle" | "sending" | "done" | "error">("idle");

  async function send(user_label: "fake" | "real" | "unsure") {
    setState("sending");
    try {
      await api.feedback({
        user_label, request_id: res.request_id, predicted_label: res.ml?.label,
        model_version: res.ml?.model_version,
        ...(consent && text ? { text, consent_store_text: true } : {}),
      });
      setState("done");
    } catch {
      setState("error");
    }
  }

  if (state === "done") {
    return <p className="mt-8 text-sm text-muted" role="status">Thanks. Your feedback is saved for review. It does not retrain the model automatically.</p>;
  }
  return (
    <div className="mt-8 border-t border-line pt-5">
      <h3 className="font-serif text-xl">Do you know the answer?</h3>
      <p className="mt-1 max-w-xl text-sm text-muted">
        Tell us what you believe after checking the sources. Reviewers use this to find cases the model gets wrong.
      </p>
      <div className="mt-3 flex flex-wrap gap-2">
        {([["fake", "I think it's fake"], ["real", "I think it's real"], ["unsure", "I'm not sure"]] as const).map(([v, t]) => (
          <button key={v} onClick={() => send(v)} disabled={state === "sending"}
            className="rounded-lg border border-line px-3.5 py-1.5 text-sm hover:bg-raised disabled:opacity-50">{t}</button>
        ))}
      </div>
      {text && (
        <label className="mt-3 flex max-w-xl items-start gap-2 text-sm text-muted">
          <input type="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)} className="mt-1" />
          Also store the article text on the server so a reviewer can read it. Off by default; only a hash is kept otherwise.
        </label>
      )}
      {state === "error" && <p role="alert" className="mt-2 text-sm text-contradicted">Feedback could not be sent. Try again.</p>}
    </div>
  );
}
