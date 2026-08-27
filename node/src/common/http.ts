// Thin fetch wrapper with retry/backoff, so retry policy stays consistent across sources.
import { setTimeout as sleep } from "node:timers/promises";

const DEFAULT_RETRIES = 3;
const RETRY_STATUS = new Set([429, 500, 502, 503, 504]);
const DEFAULT_HEADERS = { "user-agent": "mtap-sourcing/0.1" };

export interface GetOptions {
  retries?: number;
  headers?: Record<string, string>;
}

export async function getJson<T = unknown>(url: string, opts: GetOptions = {}): Promise<T> {
  const retries = opts.retries ?? DEFAULT_RETRIES;
  let lastErr: unknown;
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const resp = await fetch(url, { headers: { ...DEFAULT_HEADERS, ...opts.headers } });
      if (RETRY_STATUS.has(resp.status)) {
        throw new Error(`retryable status ${resp.status}`);
      }
      if (!resp.ok) {
        throw new Error(`GET ${url} -> ${resp.status}`);
      }
      return (await resp.json()) as T;
    } catch (err) {
      lastErr = err;
      if (attempt === retries) break;
      await sleep(2 ** attempt * 1000); // 1s, 2s, 4s
    }
  }
  throw new Error(`GET ${url} failed after ${retries + 1} attempts: ${String(lastErr)}`);
}

export async function getText(url: string, opts: GetOptions = {}): Promise<string> {
  const resp = await fetch(url, { headers: { ...DEFAULT_HEADERS, ...opts.headers } });
  if (!resp.ok) throw new Error(`GET ${url} -> ${resp.status}`);
  return resp.text();
}
