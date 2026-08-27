import type {
  Browser,
  Cookie,
  CookieParam,
  ClickOptions,
  ElementHandle,
  Frame,
  GoToOptions,
  HTTPResponse,
  KeyboardTypeOptions,
  Page,
  WaitForOptions,
  WaitForSelectorOptions,
} from "puppeteer";
import { EVENTS } from "../../../common/bus/events.js";
import type { Bus } from "../../../common/bus/Bus.js";
import type { StoredCookie } from "../../../common/session.js";
import { BrowserClosedError, NetworkError } from "./errors.js";

// Centralized wrapper for Puppeteer page operations. Ported from MBin
// scrape/utils/BrowserGuard.js. Differences from MBin: it holds an MTap `Bus`
// (not the in-process EventBroker) and publishes EVENTS.SESSION_EXPIRED on the
// recovery path. Provides page-validity checks, error classification
// (browser-closed vs transient-network), and session-restore helpers.

interface GuardLogger {
  log(message: string): void;
}

interface ErrorLike {
  name?: string;
  message?: string;
  code?: string;
}

export interface SessionRestoreResult {
  success: boolean;
  cookies: Cookie[] | null;
  message: string;
}

export class BrowserGuard {
  // Errors that indicate the browser/page has been closed or is unusable.
  static BROWSER_CLOSED_PATTERNS = [
    "TargetCloseError",
    "Session closed",
    "Target closed",
    "Protocol error",
    "Connection closed",
    "Execution context destroyed",
    "page has been closed",
    "browser has disconnected",
    "Navigation failed because page crashed",
    // Frame detachment errors - common in headless mode during redirects.
    "frame was detached",
    "detached Frame",
    "Navigating frame was detached",
  ];

  // Transient network errors that should trigger retry.
  static TRANSIENT_NETWORK_ERRORS = [
    "ECONNRESET",
    "ETIMEDOUT",
    "ECONNREFUSED",
    "ENOTFOUND",
    "ENETUNREACH",
    "EAI_AGAIN",
    "EPIPE",
    "ECONNABORTED",
    "socket hang up",
    "network error",
    "timeout",
  ];

  private bus: Bus | null;
  private logger: GuardLogger | null;
  private page: Page | null = null;
  private browser: Browser | null = null;

  constructor(bus: Bus | null = null, logger: GuardLogger | null = null) {
    this.bus = bus;
    this.logger = logger;
  }

  /** Set the page instance to guard. */
  setPage(page: Page): void {
    this.page = page;
  }

  /** Set the browser instance (for reinitialization). */
  setBrowser(browser: Browser): void {
    this.browser = browser;
  }

  /** Check if the page is still valid and usable. */
  isPageValid(): boolean {
    try {
      return Boolean(this.page && typeof this.page.isClosed === "function" && !this.page.isClosed());
    } catch {
      // If checking throws, the page is definitely invalid.
      return false;
    }
  }

  /** Check if an error indicates the browser/page was closed. */
  isBrowserClosedError(error: unknown): boolean {
    if (!error) return false;
    const e = error as ErrorLike;
    const errorName = e.name ?? "";
    const errorMessage = e.message ?? "";
    return BrowserGuard.BROWSER_CLOSED_PATTERNS.some(
      (pattern) => errorName.includes(pattern) || errorMessage.includes(pattern),
    );
  }

  /** Check if an error is a transient network error. */
  isTransientNetworkError(error: unknown): boolean {
    if (!error) return false;
    const e = error as ErrorLike;
    const errorMessage = (e.message ?? "").toLowerCase();
    const errorCode = e.code ?? "";
    return BrowserGuard.TRANSIENT_NETWORK_ERRORS.some(
      (pattern) => errorMessage.includes(pattern.toLowerCase()) || errorCode === pattern,
    );
  }

  /** Emit a recovery event on the bus so the service can reinitialize/re-auth. */
  emitRecoveryEvent(reason: string, context: string, error: ErrorLike | null = null): void {
    this.bus?.publish(EVENTS.SESSION_EXPIRED, {
      timestamp: Date.now(),
      source: context,
      reason,
      errorMessage: error?.message,
      errorName: error?.name,
    });
    this.logger?.log(`[BrowserGuard] Recovery event emitted: ${reason} from ${context}`);
  }

  /**
   * Classify an error and throw the appropriate custom error (or re-throw).
   * Always throws — never returns.
   */
  handleError(error: unknown, context: string): never {
    const e = error as ErrorLike;
    this.logger?.log(`[BrowserGuard] Error in ${context}: ${e?.message}`);

    if (this.isBrowserClosedError(error)) {
      this.emitRecoveryEvent("BROWSER_CLOSED", context, e);
      throw new BrowserClosedError(context);
    }

    if (this.isTransientNetworkError(error)) {
      throw new NetworkError(error as Error & { code?: string }, context);
    }

    throw error;
  }

  /**
   * Safely execute a page operation with error handling. Core method that all
   * async convenience methods delegate to.
   */
  async safeExecute<T>(operation: () => Promise<T>, context = "unknown"): Promise<T> {
    if (!this.isPageValid()) {
      this.logger?.log(`[BrowserGuard] Page invalid before ${context}`);
      this.emitRecoveryEvent("BROWSER_CLOSED", context);
      throw new BrowserClosedError(context);
    }

    try {
      return await operation();
    } catch (error) {
      return this.handleError(error, context);
    }
  }

  // ==========================================================================
  // Convenience Methods - wrap common page operations
  // ==========================================================================

  /** Get all cookies from the page. */
  async cookies(): Promise<Cookie[]> {
    return this.safeExecute(() => this.page!.cookies(), "cookies");
  }

  /** Navigate to a URL. */
  async goto(url: string, options: GoToOptions = {}): Promise<HTTPResponse | null> {
    return this.safeExecute(() => this.page!.goto(url, options), `goto:${url}`);
  }

  /** Reload the current page. */
  async reload(options: WaitForOptions = {}): Promise<HTTPResponse | null> {
    return this.safeExecute(() => this.page!.reload(options), "reload");
  }

  /** Get the current page URL (synchronous; validity-checked). */
  url(): string {
    if (!this.isPageValid()) throw new BrowserClosedError("url");
    return this.page!.url();
  }

  /** Get the main frame (synchronous; validity-checked). */
  mainFrame(): Frame {
    if (!this.isPageValid()) throw new BrowserClosedError("mainFrame");
    return this.page!.mainFrame();
  }

  /** Type text into an element. */
  async type(selector: string, text: string, options?: Readonly<KeyboardTypeOptions>): Promise<void> {
    return this.safeExecute(() => this.page!.type(selector, text, options), `type:${selector}`);
  }

  /** Click on an element. */
  async click(selector: string, options?: Readonly<ClickOptions>): Promise<void> {
    return this.safeExecute(() => this.page!.click(selector, options), `click:${selector}`);
  }

  /** Wait for a selector to appear. */
  async waitForSelector(
    selector: string,
    options?: WaitForSelectorOptions,
  ): Promise<ElementHandle<Element> | null> {
    return this.safeExecute(() => this.page!.waitForSelector(selector, options), `waitForSelector:${selector}`);
  }

  /** Wait for a navigation to complete. */
  async waitForNavigation(options?: WaitForOptions): Promise<HTTPResponse | null> {
    return this.safeExecute(() => this.page!.waitForNavigation(options), "waitForNavigation");
  }

  /** Evaluate a function in the page context. */
  async evaluate(pageFunction: any, ...args: any[]): Promise<any> {
    return this.safeExecute(() => this.page!.evaluate(pageFunction, ...args), "evaluate");
  }

  /** Query a selector and evaluate against it. */
  async $eval(selector: string, pageFunction: any, ...args: any[]): Promise<any> {
    return this.safeExecute(() => this.page!.$eval(selector, pageFunction, ...args), `$eval:${selector}`);
  }

  /** Query all matching selectors and evaluate against them. */
  async $$eval(selector: string, pageFunction: any, ...args: any[]): Promise<any> {
    return this.safeExecute(() => this.page!.$$eval(selector, pageFunction, ...args), `$$eval:${selector}`);
  }

  /** Register an event handler on the page (no safeExecute — just registers). */
  on(event: string, handler: (...args: any[]) => void): void {
    if (!this.isPageValid()) {
      this.logger?.log(`[BrowserGuard] Cannot register event ${event} - page invalid`);
      return;
    }
    // page.on is strongly typed by event map; this wrapper is intentionally generic.
    (this.page as Page).on(event as never, handler as never);
  }

  /** Enable/disable request interception. */
  async setRequestInterception(value: boolean): Promise<void> {
    return this.safeExecute(() => this.page!.setRequestInterception(value), "setRequestInterception");
  }

  // ==========================================================================
  // Session Recovery Methods
  // ==========================================================================

  /**
   * Inject cookies into the page to restore a session after a browser crash.
   * Returns the number of cookies successfully injected.
   */
  async injectCookies(cookies: StoredCookie[] | undefined): Promise<number> {
    if (!this.isPageValid()) throw new BrowserClosedError("injectCookies");
    if (!cookies || cookies.length === 0) return 0;

    let injectedCount = 0;
    for (const cookie of cookies) {
      try {
        const param: CookieParam = {
          name: cookie.name,
          value: cookie.value,
          domain: cookie.domain ?? ".etrade.com",
          path: cookie.path ?? "/",
          httpOnly: cookie.httpOnly ?? true,
          secure: cookie.secure ?? true,
        };
        await this.page!.setCookie(param);
        injectedCount++;
      } catch (e) {
        // Log but continue - some cookies may fail (e.g. expired).
        this.logger?.log(`[BrowserGuard] Failed to inject cookie ${cookie.name}: ${(e as Error).message}`);
      }
    }

    this.logger?.log(`[BrowserGuard] Injected ${injectedCount}/${cookies.length} cookies`);
    return injectedCount;
  }

  /**
   * Try to restore a session using saved cookies: inject them, navigate to the
   * target URL, and confirm we landed on an authenticated page (not login).
   */
  async tryRestoreSession(
    cookies: StoredCookie[] | undefined,
    targetUrl: string,
    options: GoToOptions = { waitUntil: "networkidle2" },
  ): Promise<SessionRestoreResult> {
    if (!cookies || cookies.length === 0) {
      return { success: false, cookies: null, message: "No cookies to restore" };
    }
    if (!this.isPageValid()) {
      return { success: false, cookies: null, message: "Page is not valid" };
    }

    try {
      this.logger?.log(`[BrowserGuard] Attempting session restore with ${cookies.length} cookies...`);

      const injectedCount = await this.injectCookies(cookies);
      if (injectedCount === 0) {
        return { success: false, cookies: null, message: "Failed to inject any cookies" };
      }

      await this.goto(targetUrl, options);

      const currentUrl = this.url();
      const isOnTarget = currentUrl.includes("watchlist") || currentUrl.includes("portfolio");
      const redirectedToLogin = currentUrl.includes("login") || currentUrl.includes("authenticate");

      if (isOnTarget && !redirectedToLogin) {
        const newCookies = await this.cookies();
        this.logger?.log("[BrowserGuard] Session RESTORED successfully - no re-auth needed!");
        return { success: true, cookies: newCookies, message: "Session restored from saved cookies" };
      }

      this.logger?.log(`[BrowserGuard] Session restore failed - redirected to: ${currentUrl}`);
      return { success: false, cookies: null, message: `Redirected to login: ${currentUrl}` };
    } catch (error) {
      this.logger?.log(`[BrowserGuard] Session restore error: ${(error as Error).message}`);
      return { success: false, cookies: null, message: `Error during restore: ${(error as Error).message}` };
    }
  }
}
