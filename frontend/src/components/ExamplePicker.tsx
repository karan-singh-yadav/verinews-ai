import { useState } from "react";

export interface Example {
  id: string;
  label: "real" | "fake";
  title: string;
  text: string;
}

type State =
  | { status: "closed" }
  | { status: "loading" }
  | { status: "missing" }
  | { status: "ready"; examples: Example[] };

const PER_PAGE = 4;

function isExamples(x: unknown): x is { examples: Example[] } {
  const list = (x as { examples?: unknown })?.examples;
  return (
    Array.isArray(list) &&
    list.length > 0 &&
    list.every(
      (e) =>
        e && (e.label === "real" || e.label === "fake") && typeof e.text === "string" && typeof e.title === "string",
    )
  );
}

/** Loads examples only when asked, from /examples.json (created by scripts/make_examples.py). */
export default function ExamplePicker({ onPick }: { onPick: (e: Example) => void }) {
  const [state, setState] = useState<State>({ status: "closed" });
  const [page, setPage] = useState(0);

  async function open() {
    setState({ status: "loading" });
    try {
      const res = await fetch("/examples.json", { cache: "no-store" });
      if (!res.ok) throw new Error("missing");
      const data: unknown = await res.json(); // an HTML fallback page fails here and counts as missing
      if (!isExamples(data)) throw new Error("invalid");
      setState({ status: "ready", examples: data.examples });
    } catch {
      setState({ status: "missing" });
    }
  }

  if (state.status === "closed" || state.status === "loading") {
    return (
      <button
        type="button"
        onClick={open}
        disabled={state.status === "loading"}
        className="mt-4 text-sm text-model underline underline-offset-4 hover:brightness-125 disabled:opacity-60"
      >
        {state.status === "loading" ? "Loading examples…" : "Try an article from your dataset"}
      </button>
    );
  }

  if (state.status === "missing") {
    return (
      <p role="status" className="mt-4 max-w-3xl rounded-lg border border-line px-4 py-3 text-sm text-muted">
        No dataset examples yet. In a terminal at the project folder run{" "}
        <code className="text-text">python scripts\make_examples.py</code>, then click again.
      </p>
    );
  }

  const group = (label: "real" | "fake") => {
    const all = state.examples.filter((e) => e.label === label);
    const pages = Math.max(1, Math.ceil(all.length / PER_PAGE));
    const start = (page % pages) * PER_PAGE;
    return all.slice(start, start + PER_PAGE);
  };

  return (
    <div className="mt-4 max-w-3xl rounded-lg border border-line p-4">
      <p className="text-sm font-medium">Articles from your dataset’s held-out test set</p>
      <p className="mt-1 text-xs leading-relaxed text-muted">
        The model never trained on these. The dataset label comes from each article’s source, not from a
        fact-check, and a few examples are ones the model gets wrong.
      </p>
      <div className="mt-3 grid gap-4 sm:grid-cols-2">
        {(["real", "fake"] as const).map((label) => (
          <div key={label}>
            <h3 className="text-sm text-muted">Labelled {label} in the dataset</h3>
            <ul className="mt-2 space-y-2">
              {group(label).map((e) => (
                <li key={e.id}>
                  <button
                    type="button"
                    onClick={() => onPick(e)}
                    className="line-clamp-2 w-full rounded-md border border-line px-3 py-2 text-left text-sm hover:bg-raised"
                  >
                    {e.title}
                  </button>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
      <button
        type="button"
        onClick={() => setPage((p) => p + 1)}
        className="mt-3 text-sm text-model underline underline-offset-4"
      >
        Show other examples
      </button>
    </div>
  );
}