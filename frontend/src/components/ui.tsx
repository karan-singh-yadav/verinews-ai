import type { ReactNode } from "react";
import type { ApiError } from "../api/client";
import type { Verdict } from "../api/types";

export function PageHeader({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <header className="mb-8 max-w-2xl">
      <h1 className="font-serif text-4xl font-medium tracking-tight">{title}</h1>
      {children && <p className="mt-2 text-[15px] leading-relaxed text-muted">{children}</p>}
    </header>
  );
}

export function Section({ title, children, aside }: { title: string; children: ReactNode; aside?: ReactNode }) {
  return (
    <section className="mt-10 border-t border-line pt-6">
      <div className="mb-4 flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="font-serif text-2xl font-medium">{title}</h2>
        {aside}
      </div>
      {children}
    </section>
  );
}

export function Stat({ label, value, hint }: { label: string; value: ReactNode; hint?: string }) {
  return (
    <div>
      <dt className="text-sm text-muted">{label}</dt>
      <dd className="mt-0.5 font-serif text-2xl">{value}</dd>
      {hint && <p className="mt-0.5 text-xs text-muted">{hint}</p>}
    </div>
  );
}

export function Spinner({ label }: { label: string }) {
  return (
    <span role="status" className="inline-flex items-center gap-2 text-sm text-muted">
      <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-line border-t-model" aria-hidden />
      {label}
    </span>
  );
}

export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="rounded-lg border border-dashed border-line px-5 py-8">
      <p className="font-serif text-xl">{title}</p>
      {children && <div className="mt-2 max-w-xl text-sm leading-relaxed text-muted">{children}</div>}
    </div>
  );
}

export function ErrorNote({ error, onRetry }: { error: ApiError | Error | string; onRetry?: () => void }) {
  const message = typeof error === "string" ? error : error.message;
  const status = typeof error === "object" && "status" in error ? (error as ApiError).status : 0;
  return (
    <div role="alert" className="rounded-lg border border-contradicted/50 bg-contradicted/10 px-4 py-3 text-sm">
      <p className="font-medium text-contradicted">{status === 503 ? "No trained model available" : "Something went wrong"}</p>
      <p className="mt-1 text-text/90">{message}</p>
      {status === 503 && (
        <p className="mt-1 text-muted">Train one with: python -m ml.training.train --dataset isot --data-dir data/raw</p>
      )}
      {onRetry && (
        <button onClick={onRetry} className="mt-2 rounded-md border border-line px-3 py-1 text-sm hover:bg-raised">
          Try again
        </button>
      )}
    </div>
  );
}

const VERDICT_TONE: Record<Verdict, string> = {
  Supported: "text-supported border-supported/60",
  Contradicted: "text-contradicted border-contradicted/60",
  "Mixed Evidence": "text-mixed border-mixed/60",
  Inconclusive: "text-inconclusive border-inconclusive/60",
};
export function VerdictBadge({ verdict, size = "sm" }: { verdict: Verdict; size?: "sm" | "lg" }) {
  return (
    <span
      className={`inline-block rounded-full border px-2.5 py-0.5 font-medium ${VERDICT_TONE[verdict]} ${
        size === "lg" ? "text-base" : "text-xs"
      }`}
    >
      {verdict}
    </span>
  );
}
export const VERDICT_TEXT: Record<Verdict, string> = {
  Supported: "text-supported", Contradicted: "text-contradicted",
  "Mixed Evidence": "text-mixed", Inconclusive: "text-inconclusive",
};
