import type { VerifyResponse } from "./types";

const BASE: string =
  (typeof window !== "undefined" && (window as unknown as { __API_BASE__?: string }).__API_BASE__) ||
  import.meta.env.VITE_API_BASE ||
  "";

export class ApiError extends Error {
  constructor(message: string, public status: number, public retryAfter?: number) {
    super(message);
    this.name = "ApiError";
  }
}

/** Turn any failure into a message that says what happened and what to do. Never shows raw bodies. */
async function toApiError(res: Response): Promise<ApiError> {
  let message = `The server returned an error (HTTP ${res.status}).`;
  try {
    const body = await res.json();
    if (typeof body?.message === "string" && body.message) message = body.message;
    if (body?.error === "validation_error" && Array.isArray(body.details) && body.details.length) {
      message = `Invalid input: ${body.details.map((d: { message: string }) => d.message).join("; ")}`;
    }
  } catch {
    /* non-JSON error body: keep the generic message */
  }
  const retry = Number(res.headers.get("retry-after"));
  if (res.status === 429) {
    message = `Too many requests. Try again in ${retry > 0 ? retry : "a few"} seconds.`;
  }
  return new ApiError(message, res.status, retry > 0 ? retry : undefined);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, init);
  } catch {
    throw new ApiError("Can't reach the API. Check that the backend is running (uvicorn on port 8000).", 0);
  }
  if (!res.ok) throw await toApiError(res);
  return (await res.json()) as T;
}

const post = <T>(path: string, body: unknown) =>
  request<T>(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

export const api = {
  get: <T>(path: string) => request<T>(path),
  verifyText: (text: string, force: boolean) =>
    post<VerifyResponse>("/api/verify", { text, force_verification: force }),
  verifyUrl: (url: string, force: boolean) =>
    post<VerifyResponse>("/api/verify-url", { url, force_verification: force }),
  feedback: (payload: {
    user_label: "fake" | "real" | "unsure"; predicted_label?: "fake" | "real"; model_version?: string;
    request_id?: string; text?: string; consent_store_text?: boolean;
  }) => post<{ id: string; stored_text: boolean; message: string }>("/api/feedback", payload),
};
