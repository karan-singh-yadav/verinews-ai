export const pct = (x: number | null | undefined, digits = 0) =>
  x == null || Number.isNaN(x) ? "n/a" : `${(x * 100).toFixed(digits)}%`;
export const num = (x: number | null | undefined, digits = 3) =>
  x == null || Number.isNaN(x) ? "n/a" : x.toFixed(digits);
export function when(iso: string | null | undefined): string {
  if (!iso) return "date unknown";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? "date unknown" : d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}
/** Only ever link to http(s) URLs. */
export const safeHref = (url: string): string | undefined => {
  try { const u = new URL(url); return u.protocol === "http:" || u.protocol === "https:" ? u.href : undefined; }
  catch { return undefined; }
};
export const TIER_LABEL: Record<string, string> = {
  fact_checker: "fact-checker", wire_or_major: "major outlet", official: "official source", unrated: "unrated source",
};
