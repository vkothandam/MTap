// Ported from MBin config/debug.js. Controls verbose diagnostic logging for the
// scraper (cookie snapshots, SAR tracing, navigation analysis). Most flags are
// gated off the master ENABLED switch; the heavy cookie/navigation tracing is
// diagnostic and off by default in the values below except where MBin had it on.
//
// agentLog posts to an external diagnostic server and is a no-op unless
// AGENT_LOGGING.ENABLED is set — left disabled by default.

type DebugSection = Record<string, boolean | string>;

export const DEBUG_CONFIG: Record<string, boolean | DebugSection> = {
  // Master switch — set false to disable all debug logging.
  ENABLED: true,

  COOKIES: {
    ENABLED: true,
    LOG_ALL_CHANGES: false,
    LOG_TIMESTAMPS: false,
    LOG_COMPARISONS: false,
    LOG_STALE_DETECTION: false,
    LOG_MISSING_COOKIES: false,
    TRACK_XSRF: false,
    CONSOLE_ERRORS_ONLY: true,
  },

  SESSION: {
    ENABLED: true,
    LOG_REFRESH: true,
    LOG_EXPIRY: true,
    LOG_RECOVERY: true,
    CONSOLE_ERRORS_ONLY: true,
  },

  NETWORK: {
    ENABLED: true,
    LOG_REQUESTS: false,
    LOG_RESPONSES: true,
    LOG_401_ERRORS: true,
    LOG_SAR_REQUESTS: true,
  },

  PERFORMANCE: {
    ENABLED: false,
    LOG_TIMING: false,
  },

  MANUAL_NAVIGATION: {
    // Verbose navigation cookie analysis — diagnostic only, off by default.
    ENABLED: false,
    LOG_ALL_NAVIGATIONS: false,
    LOG_USER_INTERACTIONS: false,
    COMPARE_AUTO_VS_MANUAL: false,
  },

  WATCHLIST: {
    ENABLED: true,
    LOG_API_CALLS: true,
    LOG_PROGRESS: true,
    LOG_SYMBOLS: true,
    LOG_PAGINATION: true,
    LOG_SANITIZATION: true,
    LOG_ERRORS: true,
    LOG_METRICS: true,
    CONSOLE_ERRORS_ONLY: true,
  },

  PERFORMANCE_METRICS: {
    ENABLED: true,
    LOG_FLUSH: true,
    CONSOLE_ERRORS_ONLY: true,
  },

  ETRADE_SOURCER: {
    ENABLED: true,
    LOG_POLLING: true,
    LOG_SESSION: true,
    LOG_NAVIGATION: true,
    CONSOLE_ERRORS_ONLY: true,
  },

  AGENT_LOGGING: {
    ENABLED: false,
    ENDPOINT: "http://127.0.0.1:7242/ingest",
  },
};

function section(name: string): DebugSection | null {
  const s = DEBUG_CONFIG[name];
  return s && typeof s === "object" ? s : null;
}

/** Check if a debug flag is enabled (module + flag, defaulting to ENABLED). */
export function isDebugEnabled(module: string, flag = "ENABLED"): boolean {
  if (DEBUG_CONFIG.ENABLED !== true) return false;
  const s = section(module);
  return s ? s[flag] === true : false;
}

/** Run logFn only when the given debug flag is enabled. */
export function debugLog(module: string, flag: string, logFn: () => void): void {
  if (isDebugEnabled(module, flag)) logFn();
}

/**
 * Send diagnostics to an external server. No-op unless AGENT_LOGGING.ENABLED.
 * Kept for parity with MBin's hypothesis-driven debugging; fire-and-forget.
 */
export function agentLog(
  location: string,
  message: string,
  data: Record<string, unknown> = {},
  hypothesisId = "DEBUG",
): void {
  const agent = section("AGENT_LOGGING");
  if (DEBUG_CONFIG.ENABLED !== true || !agent || agent.ENABLED !== true) return;
  const endpoint = typeof agent.ENDPOINT === "string" ? agent.ENDPOINT : "";
  if (!endpoint) return;

  const payload = {
    location,
    message,
    data: { ...data, timestamp: Date.now() },
    timestamp: Date.now(),
    sessionId: "debug-session",
    hypothesisId,
  };

  void fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  }).catch(() => {});
}
