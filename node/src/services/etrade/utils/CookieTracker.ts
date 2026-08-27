import { isDebugEnabled, debugLog } from "../debug.js";
import { TimeUtils } from "./TimeUtils.js";
import type { SessionLogger } from "./SessionLogger.js";

// Advanced cookie monitoring/analysis. Ported from MBin scrape/utils/CookieTracker.js.
// Every method is gated behind COOKIES.* debug flags (off by default), so this
// is inert diagnostic tooling unless those flags are turned on.

type CookieMap = Record<string, string>;

interface BrowserCookie {
  name: string;
  value: string;
}

interface Snapshot {
  context: string;
  timestamp: number;
  requestCount: number;
  cookies: CookieMap;
  analysis: SnapshotAnalysis;
}

interface TimestampInfo {
  value: number;
  age: number;
  ageMinutes: string;
  date: string;
  isStale: boolean;
}

interface SnapshotAnalysis {
  totalCookies: number;
  criticalCookiesPresent: string[];
  criticalCookiesMissing: string[];
  timestampCookies: Record<string, TimestampInfo>;
  xsrfTokenPresent: boolean;
  warnings: string[];
}

export class CookieTracker {
  private logger: SessionLogger | null;
  private lastSnapshot: Snapshot | null = null;
  private requestCount = 0;

  private criticalCookies = ["SMSESSION", "ETSESSION", "JSESSIONID", "SessionExpirationTime", "XSRF-TOKEN"];
  private timestampCookies = ["LastUpdateTime", "LastSARCheckTime", "NextSARCheckMillis"];
  private importantCookies = [...this.criticalCookies, ...this.timestampCookies];

  constructor(logger: SessionLogger | null = null) {
    this.logger = logger;
  }

  snapshot(cookies: BrowserCookie[] | CookieMap, context = "UNKNOWN"): Snapshot | null {
    if (!isDebugEnabled("COOKIES", "LOG_ALL_CHANGES")) return null;

    this.requestCount++;
    const snapshot: Snapshot = {
      context,
      timestamp: Date.now(),
      requestCount: this.requestCount,
      cookies: this._normalizeCookies(cookies),
      analysis: {
        totalCookies: 0,
        criticalCookiesPresent: [],
        criticalCookiesMissing: [],
        timestampCookies: {},
        xsrfTokenPresent: false,
        warnings: [],
      },
    };
    snapshot.analysis = this._analyzeSnapshot(snapshot);
    this.lastSnapshot = snapshot;

    debugLog("COOKIES", "LOG_ALL_CHANGES", () => this._logSnapshot(snapshot));
    return snapshot;
  }

  compare(before: Snapshot | null, after: Snapshot | null, operation = "OPERATION"): void {
    if (!before || !after || !isDebugEnabled("COOKIES", "LOG_COMPARISONS")) return;

    const added: string[] = [];
    const removed: string[] = [];
    const changed: string[] = [];

    const allKeys = new Set([...Object.keys(before.cookies), ...Object.keys(after.cookies)]);
    for (const key of allKeys) {
      const beforeVal = before.cookies[key];
      const afterVal = after.cookies[key];
      if (!beforeVal && afterVal) added.push(key);
      else if (beforeVal && !afterVal) removed.push(key);
      else if (beforeVal !== afterVal) changed.push(key);
    }

    debugLog("COOKIES", "LOG_COMPARISONS", () => {
      this.logger?.log(`\nCOOKIE COMPARISON: ${operation}`);
      this.logger?.log(`  Added (${added.length}): ${added.join(", ") || "none"}`);
      this.logger?.log(`  Removed (${removed.length}): ${removed.join(", ") || "none"}`);
      this.logger?.log(`  Changed (${changed.length}): ${changed.join(", ") || "none"}`);
    });
  }

  private _analyzeSnapshot(snapshot: Snapshot): SnapshotAnalysis {
    const analysis: SnapshotAnalysis = {
      totalCookies: Object.keys(snapshot.cookies).length,
      criticalCookiesPresent: [],
      criticalCookiesMissing: [],
      timestampCookies: {},
      xsrfTokenPresent: false,
      warnings: [],
    };

    for (const name of this.criticalCookies) {
      if (snapshot.cookies[name]) analysis.criticalCookiesPresent.push(name);
      else analysis.criticalCookiesMissing.push(name);
    }

    analysis.xsrfTokenPresent = !!snapshot.cookies["XSRF-TOKEN"];

    for (const name of this.timestampCookies) {
      const raw = snapshot.cookies[name];
      if (!raw) continue;
      const timestamp = parseInt(raw);
      if (isNaN(timestamp)) continue;
      const age = snapshot.timestamp - timestamp;
      analysis.timestampCookies[name] = {
        value: timestamp,
        age,
        ageMinutes: (age / 1000 / 60).toFixed(2),
        date: TimeUtils.getLocalISOStringWithTZ(new Date(timestamp)),
        isStale: age > 300000,
      };
      if (age > 300000) {
        analysis.warnings.push(`${name} is ${(age / 1000 / 60).toFixed(1)} minutes old (stale!)`);
      }
    }

    if (analysis.criticalCookiesMissing.length > 0) {
      analysis.warnings.push(`Missing critical cookies: ${analysis.criticalCookiesMissing.join(", ")}`);
    }
    if (!analysis.xsrfTokenPresent) {
      analysis.warnings.push("XSRF-TOKEN cookie is not present");
    }

    return analysis;
  }

  private _normalizeCookies(cookies: BrowserCookie[] | CookieMap): CookieMap {
    if (Array.isArray(cookies)) {
      const normalized: CookieMap = {};
      for (const cookie of cookies) normalized[cookie.name] = cookie.value;
      return normalized;
    } else if (typeof cookies === "object") {
      return { ...cookies };
    }
    return {};
  }

  private _logSnapshot(snapshot: Snapshot): void {
    if (!this.logger) return;
    const a = snapshot.analysis;
    this.logger.log(`\nCOOKIE SNAPSHOT: ${snapshot.context}`);
    this.logger.log(`Total Cookies: ${a.totalCookies}`);
    this.logger.log(`Critical Cookies Present: ${a.criticalCookiesPresent.join(", ") || "NONE"}`);
    if (a.criticalCookiesMissing.length > 0) {
      this.logger.log(`❌ Critical Cookies MISSING: ${a.criticalCookiesMissing.join(", ")}`);
    }
    this.logger.log(`XSRF Token Present: ${a.xsrfTokenPresent ? "✓" : "✗"}`);
    for (const warning of a.warnings) this.logger.log(`  - ${warning}`);
  }
}
