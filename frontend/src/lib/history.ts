import type { Verdict, VerifyResponse } from "../api/types";

const KEY = "verinews.history.v1";
const PREFS = "verinews.prefs.v1";
const MAX_ITEMS = 100;

export interface HistoryItem {
  id: string;
  ts: string;
  mode: "text" | "url";
  preview: string | null; // null unless the user opted in to keep a text preview
  host: string | null; // only the hostname of a URL, never the full link
  label: "fake" | "real" | null;
  fakeProbability: number | null;
  outOfDistribution: boolean;
  lowConfidence: boolean;
  verified: boolean;
  verdict: Verdict | null;
  verificationCompleted: boolean | null;
  modelVersion: string | null;
}

export interface Prefs { saveHistory: boolean; keepPreview: boolean }
export const DEFAULT_PREFS: Prefs = { saveHistory: true, keepPreview: false };

function read<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}
function write(key: string, value: unknown) {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* storage full/blocked: ignore */ }
}

export const loadPrefs = (): Prefs => ({ ...DEFAULT_PREFS, ...read<Partial<Prefs>>(PREFS, {}) });
export const savePrefs = (p: Prefs) => write(PREFS, p);
export const loadHistory = (): HistoryItem[] => read<HistoryItem[]>(KEY, []);
export const deleteItem = (id: string) => write(KEY, loadHistory().filter((h) => h.id !== id));
export const clearHistory = () => { try { localStorage.removeItem(KEY); } catch { /* ignore */ } };

export function hostOf(url: string): string | null {
  try { return new URL(url).hostname; } catch { return null; }
}

/** Record a completed check. Returns the stored item, or null when history is switched off. */
export function recordCheck(mode: "text" | "url", input: string, res: VerifyResponse): HistoryItem | null {
  const prefs = loadPrefs();
  if (!prefs.saveHistory) return null;
  const v = res.verification.result;
  const item: HistoryItem = {
    id: res.request_id,
    ts: new Date().toISOString(),
    mode,
    preview: prefs.keepPreview && mode === "text" ? input.trim().slice(0, 80) : null,
    host: mode === "url" ? hostOf(input) : null,
    label: res.ml?.label ?? null,
    fakeProbability: res.ml?.fake_probability ?? null,
    outOfDistribution: res.ml?.out_of_distribution ?? false,
    lowConfidence: res.ml?.low_confidence ?? false,
    verified: res.verification.triggered,
    verdict: v?.verdict ?? null,
    verificationCompleted: v ? v.completed : null,
    modelVersion: res.ml?.model_version ?? null,
  };
  write(KEY, [item, ...loadHistory().filter((h) => h.id !== item.id)].slice(0, MAX_ITEMS));
  return item;
}
