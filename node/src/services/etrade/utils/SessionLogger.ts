import { createWriteStream, existsSync, mkdirSync, type WriteStream } from "node:fs";
import { join } from "node:path";
import { REPO_ROOT } from "../../../common/config.js";
import { isDebugEnabled } from "../debug.js";
import { TimeUtils } from "./TimeUtils.js";

// Dedicated logger for session/cookie debugging + SAR (E*TRADE session
// keepalive) monitoring. Ported from MBin scrape/utils/SessionLogger.js.
// Writes to state/etrade/logs/session-YYYY-MM-DD.log.

const DEFAULT_LOG_DIR = join(REPO_ROOT, "state", "etrade", "logs");

export interface SessionLoggerOptions {
  logDir?: string;
  enabled?: boolean;
  consoleErrorsOnly?: boolean;
}

export interface LogOptions {
  isError?: boolean;
  forceConsole?: boolean;
}

// Minimal puppeteer cookie shape used for logging.
interface LoggableCookie {
  name: string;
  value: string;
  domain?: string;
  path?: string;
  expires?: number;
  httpOnly?: boolean;
  secure?: boolean;
}

type HeaderBag = Record<string, string | string[] | undefined>;

interface RequestLike {
  method(): string;
}

interface SarHistoryEntry {
  type: "request" | "response";
  timestamp: number;
  url: string;
  sarTimestamp?: number | null;
  status?: number;
  hasSetCookie?: boolean;
  count: number;
  timeSinceLastSAR?: number | null;
}

export class SessionLogger {
  private logDir: string;
  private enabled: boolean;
  private consoleErrorsOnly: boolean;
  private currentLogFile: string | null = null;
  private writeStream: WriteStream | null = null;

  private sarHistory: SarHistoryEntry[] = [];
  private lastSARTime: number | null = null;
  private sarCount = 0;
  private sarEnabled: boolean;

  constructor(options: SessionLoggerOptions = {}) {
    this.logDir = options.logDir ?? DEFAULT_LOG_DIR;
    this.enabled = options.enabled !== undefined ? options.enabled : isDebugEnabled("SESSION");
    this.consoleErrorsOnly =
      options.consoleErrorsOnly !== undefined
        ? options.consoleErrorsOnly
        : isDebugEnabled("SESSION", "CONSOLE_ERRORS_ONLY");

    this.sarEnabled = isDebugEnabled("NETWORK", "LOG_SAR_REQUESTS");

    if (this.enabled && !existsSync(this.logDir)) {
      mkdirSync(this.logDir, { recursive: true });
    }

    this.initLogFile();
  }

  private initLogFile(): void {
    if (!this.enabled) return;

    const date = TimeUtils.getLocalDateString();
    const logFilePath = join(this.logDir, `session-${date}.log`);

    if (this.currentLogFile !== logFilePath) {
      if (this.writeStream) this.writeStream.end();
      this.currentLogFile = logFilePath;
      this.writeStream = createWriteStream(logFilePath, { flags: "a" });
      this.log("=".repeat(80));
      this.log(`SESSION STARTED: ${TimeUtils.getLocalISOStringWithTZ()}`);
      this.log("=".repeat(80));
    }
  }

  log(message: string, data: unknown = null, options: LogOptions = {}): void {
    if (!this.enabled) return;
    this.initLogFile();

    const timestamp = TimeUtils.getLocalISOStringWithTZ();
    let logLine = `[${timestamp}] ${message}`;
    if (data) {
      logLine += typeof data === "object" ? "\n" + JSON.stringify(data, null, 2) : ` ${String(data)}`;
    }

    if (this.writeStream && !this.writeStream.destroyed) {
      this.writeStream.write(logLine + "\n");
    }

    const { isError = false, forceConsole = false } = options;
    if (forceConsole || isError || !this.consoleErrorsOnly) {
      if (isError) console.error(logLine);
      else console.log(logLine);
    }
  }

  logCookies(label: string, cookies: LoggableCookie[]): void {
    if (!this.enabled) return;

    this.log(`\n${"=".repeat(60)}`);
    this.log(`COOKIES: ${label}`);
    this.log(`${"=".repeat(60)}`);
    this.log(`Total cookies: ${cookies.length}`);

    const keyCookies = ["SMSESSION", "ETSESSION", "JSESSIONID", "SessionExpirationTime"];
    keyCookies.forEach((name) => {
      const cookie = cookies.find((c) => c.name === name);
      if (cookie) {
        const expires =
          cookie.expires === -1 || cookie.expires === undefined
            ? "session"
            : new Date(cookie.expires * 1000).toISOString();
        this.log(`  ${name}:`, {
          value: cookie.value.substring(0, 50) + "...",
          domain: cookie.domain,
          path: cookie.path,
          expires,
          httpOnly: cookie.httpOnly,
          secure: cookie.secure,
        });
      } else {
        this.log(`  ${name}: MISSING ⚠️`);
      }
    });

    this.log(`\nAll cookies: ${cookies.map((c) => c.name).join(", ")}`);
    this.log(`${"=".repeat(60)}\n`);
  }

  logCookieComparison(beforeCookies: LoggableCookie[], afterCookies: LoggableCookie[]): void {
    if (!this.enabled) return;

    const beforeNames = new Set(beforeCookies.map((c) => c.name));
    const afterNames = new Set(afterCookies.map((c) => c.name));
    const added = [...afterNames].filter((n) => !beforeNames.has(n));
    const removed = [...beforeNames].filter((n) => !afterNames.has(n));
    const changed = afterCookies
      .filter((ac) => {
        const bc = beforeCookies.find((c) => c.name === ac.name);
        return bc && bc.value !== ac.value;
      })
      .map((c) => c.name);

    this.log("\nCOOKIE CHANGES:");
    this.log(`  Added (${added.length}): ${added.join(", ") || "none"}`);
    this.log(`  Removed (${removed.length}): ${removed.join(", ") || "none"}`);
    this.log(`  Changed (${changed.length}): ${changed.join(", ") || "none"}`);

    if (removed.length > 0) {
      this.log("  ⚠️ WARNING: Cookies were removed - possible session issue");
    }
  }

  logError(message: string, error: (Error & { code?: string; response?: { status?: number } }) | null): void {
    if (!this.enabled) return;

    this.log(`\n${"!".repeat(60)}`, null, { isError: true });
    this.log(`ERROR: ${message}`, null, { isError: true });
    this.log(`${"!".repeat(60)}`, null, { isError: true });

    if (error) {
      this.log(
        "Error details:",
        { message: error.message, stack: error.stack, code: error.code, status: error.response?.status },
        { isError: true },
      );
    }

    this.log(`${"!".repeat(60)}\n`, null, { isError: true });
  }

  // ===================== SAR monitoring =====================

  logSarRequest(url: string, request: RequestLike | null = null): void {
    if (!this.sarEnabled) return;

    const sarMatch = url.match(/n=(\d+)/);
    const timestamp = sarMatch ? parseInt(sarMatch[1]) : null;
    const age = timestamp ? Date.now() - timestamp : null;

    this.sarCount++;
    const timeSinceLastSAR = this.lastSARTime ? (Date.now() - this.lastSARTime) / 1000 : null;

    this.log("\n🔐 SAR REQUEST DETECTED:");
    this.log(`  Count: #${this.sarCount}`);
    this.log(`  URL: ${url}`);
    this.log(`  Timestamp (n): ${timestamp}`);
    if (age !== null) {
      this.log(`  Age: ${(age / 1000).toFixed(2)}s ago`);
    }
    if (timeSinceLastSAR !== null) {
      this.log(`  Time since last SAR: ${timeSinceLastSAR.toFixed(2)}s`);
    }
    if (request) {
      this.log(`  Method: ${request.method()}`);
    }

    this.lastSARTime = Date.now();
    this.sarHistory.push({
      type: "request",
      timestamp: Date.now(),
      url,
      sarTimestamp: timestamp,
      count: this.sarCount,
      timeSinceLastSAR,
    });
  }

  logSarResponse(url: string, status: number, headers: HeaderBag): void {
    if (!this.sarEnabled) return;

    this.log("\n✅ SAR RESPONSE RECEIVED:");
    this.log(`  Count: #${this.sarCount}`);
    this.log(`  URL: ${url}`);
    this.log(`  Status: ${status}`);
    this.log(`  Has Set-Cookie: ${!!headers["set-cookie"]}`);

    this.sarHistory.push({
      type: "response",
      timestamp: Date.now(),
      url,
      status,
      hasSetCookie: !!headers["set-cookie"],
      count: this.sarCount,
    });
  }

  private getSARStats(): {
    totalRequests: number;
    totalResponses: number;
    averageInterval: string;
    lastSARTime: string;
    intervals: string[];
  } | null {
    if (this.sarHistory.length === 0) return null;

    const requests = this.sarHistory.filter((h) => h.type === "request");
    const responses = this.sarHistory.filter((h) => h.type === "response");

    const intervals: number[] = [];
    for (let i = 1; i < requests.length; i++) {
      intervals.push(requests[i].timestamp - requests[i - 1].timestamp);
    }

    const avgInterval =
      intervals.length > 0 ? intervals.reduce((a, b) => a + b, 0) / intervals.length / 1000 : null;

    return {
      totalRequests: requests.length,
      totalResponses: responses.length,
      averageInterval: avgInterval ? `${avgInterval.toFixed(2)}s` : "N/A",
      lastSARTime: this.lastSARTime ? new Date(this.lastSARTime).toISOString() : "N/A",
      intervals: intervals.map((i) => (i / 1000).toFixed(2) + "s"),
    };
  }

  logSARSummary(): void {
    if (!this.sarEnabled || this.sarHistory.length === 0) return;
    const stats = this.getSARStats();
    if (!stats) return;

    this.log("\n" + "=".repeat(60));
    this.log("SAR MONITORING SUMMARY");
    this.log("=".repeat(60));
    this.log(`Total SAR Requests: ${stats.totalRequests}`);
    this.log(`Total SAR Responses: ${stats.totalResponses}`);
    this.log(`Average Interval: ${stats.averageInterval}`);
    this.log(`Last SAR Time: ${stats.lastSARTime}`);
    if (stats.intervals.length > 0) this.log(`Intervals: ${stats.intervals.join(", ")}`);
    this.log("=".repeat(60) + "\n");
  }

  close(): void {
    if (this.writeStream) {
      this.logSARSummary();
      this.log("=".repeat(80));
      this.log(`SESSION ENDED: ${TimeUtils.getLocalISOStringWithTZ()}`);
      this.log("=".repeat(80));
      this.writeStream.end();
    }
  }
}
