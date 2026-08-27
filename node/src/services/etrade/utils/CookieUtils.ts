// Extracts and formats E*TRADE cookies. Ported from MBin scrape/utils/CookieUtils.js.
// Kept as a static store (a process-wide cookie jar built from Set-Cookie
// headers) to match MBin's behaviour exactly.

interface CookieLogger {
  log(message: string): void;
}

interface BrowserCookie {
  name: string;
  value: string;
  domain?: string;
}

type HeaderBag = Record<string, string | string[] | undefined>;

export class CookieUtils {
  static cookieKeyVal: Record<string, string> = {};
  static logger: CookieLogger | null = null; // set by SessionLogger

  /**
   * Extract cookies from Set-Cookie headers. Handles string/array formats,
   * lenient etrade.com domain matching, and preserves important session
   * cookies regardless of domain.
   */
  static extractCookies(headers: HeaderBag): Record<string, string> {
    const setCookieHeader = headers["set-cookie"];
    if (!setCookieHeader) {
      return this.cookieKeyVal;
    }

    this.logger?.log("[CookieUtils] Extracting cookies from Set-Cookie header");

    let cookieStrings: string[] = [];
    if (Array.isArray(setCookieHeader)) {
      cookieStrings = setCookieHeader;
    } else if (typeof setCookieHeader === "string") {
      cookieStrings = setCookieHeader.split("\n").filter((s) => s.trim());
    }

    this.logger?.log(`  Found ${cookieStrings.length} cookie string(s)`);

    let extractedCount = 0;
    let skippedCount = 0;
    const importantCookies = ["SMSESSION", "ETSESSION", "JSESSIONID", "SessionExpirationTime"];

    cookieStrings.forEach((cookieString) => {
      if (!cookieString || !cookieString.trim()) return;

      const parts = cookieString.split(";").map((p) => p.trim());
      if (parts.length === 0) return;

      const [name, ...valueParts] = parts[0].split("=");
      if (!name || valueParts.length === 0) return;

      const value = valueParts.join("="); // values may contain '='

      const attributes: Record<string, string | true> = {};
      for (let i = 1; i < parts.length; i++) {
        const [attrName, ...attrValueParts] = parts[i].split("=");
        const attrKey = attrName.toLowerCase();
        attributes[attrKey] = attrValueParts.length > 0 ? attrValueParts.join("=") : true;
      }

      const domain = typeof attributes.domain === "string" ? attributes.domain : "";
      const isEtradeDomain = /\.?etrade\.com/i.test(domain);
      const isImportantCookie = importantCookies.includes(name);
      const hasNoDomain = !attributes.domain;

      if (isEtradeDomain || isImportantCookie || hasNoDomain) {
        this.cookieKeyVal[name] = value;
        extractedCount++;
        if (this.logger) {
          const reason = isImportantCookie
            ? " (important)"
            : hasNoDomain
              ? " (session)"
              : isEtradeDomain
                ? " (etrade domain)"
                : "";
          this.logger.log(`  ✓ Extracted: ${name} = ${value.substring(0, 30)}...${reason}`);
        }
      } else {
        skippedCount++;
        this.logger?.log(`  ✗ Skipped: ${name} (domain: ${domain || "none"})`);
      }
    });

    if (this.logger) {
      this.logger.log(`  Summary: ${extractedCount} extracted, ${skippedCount} skipped`);
      this.logger.log(`  Total cookies in store: ${Object.keys(this.cookieKeyVal).length}`);
    }

    return this.cookieKeyVal;
  }

  /** All stored cookies. */
  static getCookies(): Record<string, string> {
    return this.cookieKeyVal;
  }

  /** Clear the store (testing/debugging). */
  static clearCookies(): void {
    this.cookieKeyVal = {};
    this.logger?.log("[CookieUtils] Cleared all cookies");
  }

  /** Format cookies for an HTTP Cookie header, from an array or the internal store. */
  static formatCookiesForHeader(cookies: BrowserCookie[] | Record<string, string>): string {
    if (Array.isArray(cookies)) {
      return cookies.map((c) => `${c.name}=${c.value}`).join(";");
    } else if (typeof cookies === "object") {
      return Object.entries(cookies)
        .map(([name, value]) => `${name}=${value}`)
        .join(";");
    }
    return "";
  }

  /** Specific cookie by name. */
  static getCookie(name: string): string | undefined {
    return this.cookieKeyVal[name];
  }

  /** Whether important session cookies are present. */
  static hasValidSession(): boolean {
    return Boolean(this.cookieKeyVal.SMSESSION || this.cookieKeyVal.ETSESSION);
  }

  /**
   * Sync the store with the full browser cookie jar (page.cookies()), so the
   * store isn't limited to what appeared in response headers.
   */
  static syncWithBrowserCookies(browserCookies: BrowserCookie[]): void {
    if (!Array.isArray(browserCookies)) return;

    this.logger?.log(`[CookieUtils] Syncing with ${browserCookies.length} browser cookies`);

    let syncedCount = 0;
    let skippedCount = 0;

    browserCookies.forEach((cookie) => {
      const { name, value, domain } = cookie;
      const isEtradeDomain = domain && /\.?etrade\.com/i.test(domain);
      if (isEtradeDomain) {
        this.cookieKeyVal[name] = value;
        syncedCount++;
      } else {
        skippedCount++;
      }
    });

    if (this.logger) {
      this.logger.log(`  Synced: ${syncedCount}, Skipped: ${skippedCount}`);
      this.logger.log(`  Total cookies in store: ${Object.keys(this.cookieKeyVal).length}`);
    }
  }
}
