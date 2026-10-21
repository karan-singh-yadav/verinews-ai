import { useCallback, useEffect, useState } from "react";
import { api, ApiError } from "../api/client";

export function useApi<T>(path: string) {
  const [state, setState] = useState<{ data: T | null; error: ApiError | null; loading: boolean }>({
    data: null, error: null, loading: true,
  });
  const load = useCallback(async () => {
    setState((s) => ({ ...s, loading: true }));
    try {
      setState({ data: await api.get<T>(path), error: null, loading: false });
    } catch (e) {
      setState({ data: null, error: e instanceof ApiError ? e : new ApiError(String(e), 0), loading: false });
    }
  }, [path]);
  useEffect(() => { void load(); }, [load]);
  return { ...state, reload: load };
}
