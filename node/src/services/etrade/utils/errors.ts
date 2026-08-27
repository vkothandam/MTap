// Custom error types for browser and session error handling. Ported from MBin
// scrape/utils/errors.js. These classify failures so the recovery path can
// decide whether to retry, refresh, re-authenticate, or stop.

/** Browser page closed unexpectedly. Recoverable — browser can be reinitialized. */
export class BrowserClosedError extends Error {
  readonly context: string;
  readonly recoverable = true;
  readonly requiresReauth = true;

  constructor(context = "unknown") {
    super(`Browser closed during: ${context}`);
    this.name = "BrowserClosedError";
    this.context = context;
  }
}

/** Session expired (401/403 from API). Recoverable — try navigation refresh first. */
export class SessionExpiredError extends Error {
  readonly source: string;
  readonly status: number | null;
  readonly recoverable = true;
  readonly requiresReauth = false;

  constructor(source = "unknown", status: number | null = null) {
    super(`Session expired from: ${source}${status ? ` (HTTP ${status})` : ""}`);
    this.name = "SessionExpiredError";
    this.source = source;
    this.status = status;
  }
}

/** Transient network error (ECONNRESET, ETIMEDOUT, ...). Recoverable with retry. */
export class NetworkError extends Error {
  readonly context: string;
  readonly code?: string;
  readonly originalError: Error;
  readonly recoverable = true;
  readonly requiresReauth = false;

  constructor(originalError: Error & { code?: string }, context = "unknown") {
    super(`Network error during ${context}: ${originalError.message}`);
    this.name = "NetworkError";
    this.context = context;
    this.code = originalError.code;
    this.originalError = originalError;
  }
}

/** Recovery attempts exhausted. NOT recoverable — the caller should stop. */
export class UnrecoverableError extends Error {
  readonly originalError: Error | null;
  readonly recoverable = false;
  readonly requiresReauth = false;

  constructor(message: string, originalError: Error | null = null) {
    super(message);
    this.name = "UnrecoverableError";
    this.originalError = originalError;
  }
}

interface RecoverableFlags {
  recoverable?: boolean;
  requiresReauth?: boolean;
}

/** True for our custom recoverable errors. */
export function isRecoverableError(error: unknown): boolean {
  return Boolean(error && (error as RecoverableFlags).recoverable === true);
}

/** True for errors that require re-authentication. */
export function requiresReauthentication(error: unknown): boolean {
  return Boolean(error && (error as RecoverableFlags).requiresReauth === true);
}
