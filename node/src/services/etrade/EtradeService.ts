import type { Cookie, HTTPRequest, HTTPResponse, Page } from "puppeteer";
import { Service } from "../../common/service.js";
import { SessionStore, type StoredCookie } from "../../common/session.js";
import { EVENTS } from "../../common/bus/events.js";
import type { BusMessage } from "../../common/bus/types.js";
import { Browser } from "./Browser.js";
import { Authentication } from "./Authentication.js";
import { ETPageScraper } from "./ETPageScraper.js";
import { BrowserGuard } from "./utils/BrowserGuard.js";
import { CookieTracker } from "./utils/CookieTracker.js";
import { CookieUtils } from "./utils/CookieUtils.js";
import { SessionLogger } from "./utils/SessionLogger.js";
import { FileLogger } from "./utils/FileLogger.js";
import { TimeUtils } from "./utils/TimeUtils.js";
import { BrowserClosedError } from "./utils/errors.js";
import { URLS, INTERVALS, WATCHLISTS, SERVER_CONFIG } from "./constants.js";
import { ENV, hasLoginCredentials } from "./env.js";
import { OtpCoordinator } from "./otp.js";
import { OtpServer } from "./otpServer.js";
import { isDebugEnabled, agentLog } from "./debug.js";

// The long-running E*TRADE scraper, ported from MBin scrape/index.js
// (ETradeSourcer) to an MTap Service. It owns the only E*TRADE browser session:
// login (incl. the in-process OTP flow), multi-watchlist polling, periodic page
// refresh to keep the session alive, and session-expiry recovery. It publishes
// raw POLLED_DATA on the bus (stamped with idempotency keys) and reacts to
// SESSION_EXPIRED.
//
// Changes from MBin:
//   - extends Service; the EventBroker is replaced by the MTap Bus.
//   - The whole OTP flow lives here (OtpCoordinator + OtpServer), no MBin hop.
//   - Session persistence is MTap's file-backed SessionStore (not MBin's DB), and
//     a saved session is reused on startup to skip re-login when still valid.
//   - Idempotency: a per-process runId + a per-scraper-generation suffix means
//     each publish gets a unique, stable key (see buildScrapers()).
//   - Dead code dropped: the disabled proactive re-auth and the WSService import.

interface SessionExpiredEvent {
  reason?: string;
  source?: string;
}

interface NavigationCookieChanges {
  added: string[];
  removed: string[];
  changed: string[];
}

export class EtradeService extends Service {
  readonly name = "etrade";

  // Alternate page for navigation-based refresh (extends the session better than a reload).
  static POSITIONS_PAGE_URL = "https://us.etrade.com/etx/pxy/portfolios/positions?rta=true";

  private requestHeaders: Record<string, string> = {};
  private responseHeaders: Record<string, string> = {};
  private cookies: StoredCookie[] | null = null;

  private browser: Browser | null = null;
  private page: Page | null = null;
  private auth: Authentication | null = null;

  private readonly sessionLogger: SessionLogger;
  private readonly cookieTracker: CookieTracker;
  private readonly logger: FileLogger;
  private readonly browserGuard: BrowserGuard;
  private readonly sessionStore: SessionStore;
  private readonly otpCoordinator: OtpCoordinator;
  private readonly otpServer: OtpServer;

  // Idempotency: stable per-process id + a generation bumped on every scraper rebuild.
  private readonly runId: string;
  private scraperGeneration = 0;

  private dataScrapers: Record<string, ETPageScraper> = {};
  private dataScraper: ETPageScraper | null = null;

  // Polling / refresh state.
  private dataIntervals: Record<string, ReturnType<typeof setInterval>> = {};
  private pageInterval: ReturnType<typeof setInterval> | null = null;
  private moversInterval: ReturnType<typeof setInterval> | null = null;
  private refreshCount = 0;
  private isRefreshInProgress = false;
  private isRefreshingSession = false;
  private loginTimestamp: number | null = null;

  // Watchlist poll mutex — only one watchlist is polled at a time.
  private watchlistPollLock: Promise<void> | null = null;
  private pollLockResolve: (() => void) | null = null;
  private pollLockHolder: string | null = null;

  // Navigation tracking (diagnostic; MANUAL_NAVIGATION debug flag, off by default).
  private isAutomatedNavigation = false;
  private lastNavigationCookies: Cookie[] | null = null;
  private navigationCount = { automated: 0, manual: 0 };

  private unsubscribeSessionExpired: (() => void) | null = null;
  private started = false;

  constructor() {
    super();
    this.runId = `${process.pid}-${Date.now()}`;
    this.sessionLogger = new SessionLogger();
    this.cookieTracker = new CookieTracker(this.sessionLogger);
    this.logger = new FileLogger({ moduleName: "ETradeSourcer", configSection: "ETRADE_SOURCER" });
    this.browserGuard = new BrowserGuard(this.bus, this.sessionLogger);
    this.sessionStore = new SessionStore("etrade");
    this.otpCoordinator = new OtpCoordinator();
    this.otpServer = new OtpServer(this.otpCoordinator);

    // Route CookieUtils' internal logging into the session log.
    CookieUtils.logger = this.sessionLogger;

    // Placeholder scrapers (generation 0) with empty headers; rebuilt with real
    // request headers after login captures stk1.
    this.buildScrapers();
  }

  // ==========================================================================
  // Service lifecycle
  // ==========================================================================

  async start(): Promise<void> {
    if (this.started) return;
    this.started = true;

    if (!ENV.FEATURES.SCRAPING) {
      this.logger.warn("FEATURE_SCRAPING is off — E*TRADE service will not start a browser.");
      return;
    }
    if (!hasLoginCredentials() && !this.sessionStore.hasSession()) {
      throw new Error(
        "No E*TRADE credentials (ET_USERNAME/ET_PASSWORD) and no saved session. Set them in the environment; see .env.example.",
      );
    }

    // Bring up the OTP entry page before login so a challenge can be answered.
    await this.otpServer.start();

    // React to session-expiry (published by BrowserGuard / ETPageScraper).
    this.unsubscribeSessionExpired = this.bus.subscribe(
      EVENTS.SESSION_EXPIRED,
      (msg: BusMessage) => void this.handleSessionExpired(msg as SessionExpiredEvent),
      { label: "etrade:session" },
    );

    await this.initialize();
  }

  async stop(): Promise<void> {
    this.started = false;
    if (this.unsubscribeSessionExpired) {
      this.unsubscribeSessionExpired();
      this.unsubscribeSessionExpired = null;
    }
    this.stopMonitoring();
    this.otpCoordinator.cancel("service stopping");
    await this.otpServer.stop();
    if (this.browser) {
      await this.browser.close();
      this.browser = null;
    }
  }

  // ==========================================================================
  // Initialization
  // ==========================================================================

  private async initialize(): Promise<void> {
    this.browser = new Browser();
    this.page = await this.browser.initialize(false);
    if (!this.page) {
      this.logger.warn("Browser did not produce a page (scraping disabled?); aborting initialize.");
      return;
    }

    // Wire the guard to the live page/browser.
    this.browserGuard.setPage(this.page);
    const instance = this.browser.instance;
    if (instance) this.browserGuard.setBrowser(instance);

    // Authentication registers the sole request.continue() handler.
    this.auth = this.makeAuthentication();
    this.logger.info("Setting up Authentication (in-process OTP)");
    await this.auth.initialize();

    // Register the service's own request/response listeners BEFORE any watchlist
    // navigation so stk1 and Set-Cookie are captured on the first XHRs. (These
    // handlers never continue the request — Authentication does that.)
    this.browser.setupRequestHandlers(
      (request: HTTPRequest) => this.handleRequest(request),
      (response: HTTPResponse) => this.handleResponse(response),
    );
    this.setupNavigationTracking();

    // Fast path: reuse a saved session if it is still valid.
    const restored = await this.tryReuseSavedSession();

    if (!restored) {
      await this.auth.login(ENV.ET_USERNAME, ENV.ET_PASSWORD);
      this.loginTimestamp = Date.now();
      agentLog(
        "EtradeService:initialize",
        "LOGIN_COMPLETE",
        { loginTimestamp: this.loginTimestamp },
        "H1",
      );

      this.isAutomatedNavigation = true;
      await this.browserGuard.goto(URLS.WATCHLIST_PAGE, { waitUntil: "networkidle2" });
      this.isAutomatedNavigation = false;

      this.cookies = await this.browserGuard.cookies();
    }

    if (!this.cookies || this.cookies.length === 0) {
      throw new Error("Failed to get cookies after login");
    }
    this.persistSession();

    // Rebuild scrapers with the captured request headers (new generation).
    this.buildScrapers();

    // Initial fetch for every watchlist.
    const cookies = this.cookies;
    this.logger.info(
      `Initializing ${Object.keys(WATCHLISTS).length} watchlists: ${Object.keys(WATCHLISTS).join(", ")}`,
    );
    for (const [key, scraper] of Object.entries(this.dataScrapers)) {
      const config = WATCHLISTS[key];
      this.logger.info(
        `Fetching initial data for ${key} (ID: ${config?.id}, interval: ${config?.intervalMs}ms, prefix: '${config?.csvPrefix}')`,
      );
      await scraper.getWatchlist(cookies);
    }

    this.startMonitoring();
  }

  /** Build an Authentication wired to publish the optional AUTH_OTP_REQUIRED mirror. */
  private makeAuthentication(): Authentication {
    return new Authentication(this.browserGuard, this.otpCoordinator, {
      onOtpRequired: () => {
        // Read-only "login required" signal for MBin's dashboard; not on the login path.
        this.bus.publish(EVENTS.AUTH_OTP_REQUIRED, {
          timestamp: Date.now(),
          otpUrl: `${SERVER_CONFIG.BASE_URL}${SERVER_CONFIG.OTP_PATH}`,
        });
      },
    });
  }

  /** Try to restore a saved session (inject cookies + navigate). Returns true on success. */
  private async tryReuseSavedSession(): Promise<boolean> {
    if (!this.sessionStore.hasSession()) return false;

    const stored = this.sessionStore.get();
    if (!stored?.cookies?.length) return false;

    // Seed any saved request headers (e.g. stk1) as a fallback; live capture overrides.
    for (const [key, value] of Object.entries(stored.requestHeaders ?? {})) {
      if (typeof value === "string") this.requestHeaders[key] = value;
    }

    this.logger.info(`Found saved session (${stored.cookies.length} cookies); attempting reuse...`);
    this.isAutomatedNavigation = true;
    const result = await this.browserGuard.tryRestoreSession(stored.cookies, URLS.WATCHLIST_PAGE);
    this.isAutomatedNavigation = false;

    if (result.success && result.cookies) {
      this.cookies = result.cookies;
      this.loginTimestamp = Date.now();
      this.logger.info("Saved session reused — skipped re-login.");
      return true;
    }

    this.logger.info(`Saved session unusable (${result.message}); falling back to login.`);
    return false;
  }

  /**
   * (Re)build one ETPageScraper per configured watchlist. Each rebuild uses a
   * fresh runId generation so a scraper's seq counter resetting to 0 can never
   * collide with idempotency keys emitted by an earlier generation.
   */
  private buildScrapers(): void {
    const genRunId = `${this.runId}.${this.scraperGeneration++}`;
    this.dataScrapers = {};
    for (const [key, config] of Object.entries(WATCHLISTS)) {
      this.dataScrapers[key] = new ETPageScraper(this.requestHeaders, this.bus, genRunId, {
        id: config.id,
        name: config.name,
        csvPrefix: config.csvPrefix,
      });
    }
    this.dataScraper = this.dataScrapers.MAIN ?? null;
  }

  // ==========================================================================
  // Request / response handlers (accumulate headers + cookies; never continue)
  // ==========================================================================

  private handleRequest(request: HTTPRequest): void {
    const url = request.url();
    this.requestHeaders = { ...this.requestHeaders, ...request.headers() };

    if (url.includes("/login/sar")) {
      this.sessionLogger.logSarRequest(url, request);
    }
    // NOTE: we do NOT call request.continue() here — Authentication owns that.
  }

  private handleResponse(response: HTTPResponse): void {
    const url = response.url();
    const status = response.status();
    const headers = response.headers();

    this.responseHeaders = { ...this.responseHeaders, ...headers };

    if (url.includes("/login/sar")) {
      this.sessionLogger.logSarResponse(url, status, headers);
      const setCookieStr = headers["set-cookie"] ?? "";
      agentLog(
        "EtradeService:handleResponse",
        "SAR_RESPONSE",
        {
          status,
          hasSetCookie: Boolean(headers["set-cookie"]),
          containsSessionExp: setCookieStr.includes("SessionExpirationTime"),
          containsXSRF: setCookieStr.includes("XSRF-TOKEN"),
        },
        "H2",
      );
    }

    const beforeSnapshot = this.cookieTracker.snapshot(CookieUtils.getCookies(), `BEFORE_RESPONSE:${url}`);

    const beforeCount = Object.keys(CookieUtils.getCookies()).length;
    CookieUtils.extractCookies(this.responseHeaders);
    const afterCount = Object.keys(CookieUtils.getCookies()).length;

    const afterSnapshot = this.cookieTracker.snapshot(CookieUtils.getCookies(), `AFTER_RESPONSE:${url}`);
    this.cookieTracker.compare(beforeSnapshot, afterSnapshot, `RESPONSE:${url}`);

    if (afterCount > beforeCount) {
      this.sessionLogger.log(`CookieUtils: ${beforeCount} → ${afterCount} cookies (+${afterCount - beforeCount})`);
    } else if (headers["set-cookie"]) {
      this.sessionLogger.log("⚠️ WARNING: Set-Cookie present but CookieUtils extracted nothing!");
    }
  }

  // ==========================================================================
  // Navigation tracking (diagnostic — gated behind MANUAL_NAVIGATION, off by default)
  // ==========================================================================

  private setupNavigationTracking(): void {
    if (!isDebugEnabled("MANUAL_NAVIGATION", "ENABLED")) return;

    this.browserGuard.on("framenavigated", (frame: { url(): string } & unknown) => {
      try {
        if (frame !== this.browserGuard.mainFrame()) return;
      } catch {
        return; // page may be closed
      }
      const isManual = !this.isAutomatedNavigation;
      if (isManual) this.navigationCount.manual++;
      else this.navigationCount.automated++;

      if (isDebugEnabled("MANUAL_NAVIGATION", "LOG_ALL_NAVIGATIONS")) {
        const sessionAge = this.loginTimestamp
          ? ((Date.now() - this.loginTimestamp) / 1000 / 60).toFixed(2)
          : "unknown";
        this.sessionLogger.log(
          `NAVIGATION ${isManual ? "MANUAL" : "AUTOMATED"}: ${frame.url()} (session age ${sessionAge}m)`,
        );
      }
    });

    this.browserGuard.on("load", () => {
      if (!isDebugEnabled("MANUAL_NAVIGATION", "COMPARE_AUTO_VS_MANUAL")) return;
      void (async () => {
        let currentCookies: Cookie[];
        try {
          currentCookies = await this.browserGuard.cookies();
        } catch {
          return;
        }
        if (this.lastNavigationCookies) {
          const changes = this.compareNavigationCookies(this.lastNavigationCookies, currentCookies);
          this.sessionLogger.log(
            `Cookies after navigation — added: ${changes.added.join(",") || "none"}, ` +
              `removed: ${changes.removed.join(",") || "none"}, changed: ${changes.changed.join(",") || "none"}`,
          );
        }
        this.lastNavigationCookies = currentCookies;
      })();
    });

    this.sessionLogger.log("[EtradeService] Navigation tracking enabled");
  }

  private compareNavigationCookies(before: Cookie[], after: Cookie[]): NavigationCookieChanges {
    const beforeMap = new Map(before.map((c) => [c.name, c.value]));
    const afterMap = new Map(after.map((c) => [c.name, c.value]));
    const added: string[] = [];
    const removed: string[] = [];
    const changed: string[] = [];

    for (const [name, value] of afterMap) {
      if (!beforeMap.has(name)) added.push(name);
      else if (beforeMap.get(name) !== value) changed.push(name);
    }
    for (const [name] of beforeMap) {
      if (!afterMap.has(name)) removed.push(name);
    }
    return { added, removed, changed };
  }

  // ==========================================================================
  // Monitoring loops
  // ==========================================================================

  private startMonitoring(): void {
    const endOfTradeTodaysEpoch = TimeUtils.calculateEndOfTradeTodaysEpoch();
    const now = Date.now();
    this.logger.info(`EndOfTradeTodaysEpoch: ${endOfTradeTodaysEpoch}`);
    this.logger.info(`Time until market close: ${Math.round((endOfTradeTodaysEpoch - now) / 1000 / 60)} minutes`);
    this.logger.info(`Page refresh interval: ${INTERVALS.REFRESH_PAGE}ms`);

    this.pageInterval = setInterval(() => {
      void this.refreshPage();
    }, INTERVALS.REFRESH_PAGE);

    this.logger.info("Setting up watchlist polling intervals:");
    for (const [key, config] of Object.entries(WATCHLISTS)) {
      this.logger.info(`  - ${key}: ${config.intervalMs}ms, ID: ${config.id}, prefix: '${config.csvPrefix}'`);
      this.dataIntervals[key] = setInterval(() => {
        void this.refreshWatchlist(key);
      }, config.intervalMs);
    }

    // Market-movers polling is disabled (parity with MBin); refreshMovers()
    // remains available for manual/re-enabled use.
  }

  /**
   * Refresh a specific watchlist by key. Uses a mutex so only one watchlist is
   * polled at a time (avoids concurrent API calls fighting over cookies).
   */
  private async refreshWatchlist(watchlistKey: string): Promise<void> {
    if (this.isRefreshInProgress) {
      this.sessionLogger.log(`[refreshWatchlist:${watchlistKey}] Skipping — page refresh in progress`);
      return;
    }

    const scraper = this.dataScrapers[watchlistKey];
    if (!scraper) {
      this.logger.warn(`[refreshWatchlist] Unknown watchlist key: ${watchlistKey}`);
      return;
    }
    if (!this.cookies) return;

    await this.acquireWatchlistPollLock(watchlistKey);
    try {
      if (watchlistKey === "MAIN") {
        this.cookieTracker.snapshot(this.cookies, "BEFORE_WATCHLIST_API");
      }

      await scraper.getWatchlist(this.cookies);

      const afterCookies = await this.browserGuard.cookies();
      if (watchlistKey === "MAIN") {
        this.cookieTracker.snapshot(afterCookies, "AFTER_WATCHLIST_API");
      }

      // Keep this.cookies in sync with the browser, merging any header-extracted cookies.
      this.cookies = afterCookies;
      const extracted = CookieUtils.getCookies();
      if (Object.keys(extracted).length > 0) {
        this.cookies = this.mergeCookies(this.cookies, extracted);
      }
      this.persistSession();
    } catch (error) {
      if (error instanceof BrowserClosedError) {
        this.sessionLogger.log(`[refreshWatchlist:${watchlistKey}] Browser closed: ${error.message}`);
      } else {
        this.sessionLogger.logError(`[refreshWatchlist:${watchlistKey}] Error`, error as Error);
      }
    } finally {
      this.releaseWatchlistPollLock();
    }
  }

  private async acquireWatchlistPollLock(watchlistKey: string): Promise<void> {
    while (this.watchlistPollLock) {
      this.sessionLogger.log(`[refreshWatchlist:${watchlistKey}] Waiting for ongoing poll...`);
      await this.watchlistPollLock;
    }
    this.watchlistPollLock = new Promise<void>((resolve) => {
      this.pollLockResolve = resolve;
    });
    this.pollLockHolder = watchlistKey;
  }

  private releaseWatchlistPollLock(): void {
    const resolve = this.pollLockResolve;
    this.watchlistPollLock = null;
    this.pollLockResolve = null;
    this.pollLockHolder = null;
    if (resolve) resolve();
  }

  private async refreshMovers(endOfTradeTodaysEpoch: number): Promise<void> {
    if (this.isRefreshInProgress) {
      this.sessionLogger.log("[refreshMovers] Skipping — page refresh in progress");
      return;
    }
    if (Date.now() >= endOfTradeTodaysEpoch) {
      this.stopMonitoring();
      return;
    }
    if (!this.dataScraper || !this.cookies) return;

    try {
      await this.dataScraper.getMarketMoversUp(this.cookies);
      await this.dataScraper.getMarketMoversDown(this.cookies);

      this.cookies = await this.browserGuard.cookies();
      const extracted = CookieUtils.getCookies();
      if (Object.keys(extracted).length > 0) {
        this.cookies = this.mergeCookies(this.cookies, extracted);
      }
      this.persistSession();
    } catch (error) {
      if (error instanceof BrowserClosedError) {
        this.sessionLogger.log(`[refreshMovers] Browser closed: ${error.message}`);
      } else {
        this.sessionLogger.logError("[refreshMovers] Error", error as Error);
      }
    }
  }

  /**
   * Refresh the page to keep the session alive. Alternates between a simple
   * reload (odd counts) and a navigation round-trip via the positions page
   * (even counts), which extends the session more effectively.
   */
  private async refreshPage(): Promise<void> {
    // Session refresh continues even after market close to keep the browser alive.
    this.isRefreshInProgress = true;
    this.refreshCount++;

    const useNavigation = this.refreshCount % 2 === 0;
    const refreshMethod = useNavigation ? "NAVIGATION" : "RELOAD";
    this.sessionLogger.log(`PAGE REFRESH STARTED (${refreshMethod}, count ${this.refreshCount})`);

    try {
      const beforeCookies = await this.browserGuard.cookies();
      const beforeSnapshot = this.cookieTracker.snapshot(beforeCookies, "BEFORE_PAGE_REFRESH");
      this.sessionLogger.logCookies("BEFORE PAGE REFRESH", beforeCookies);

      let response: HTTPResponse | null = null;
      if (useNavigation) {
        this.isAutomatedNavigation = true;
        await this.browserGuard.goto(EtradeService.POSITIONS_PAGE_URL, { waitUntil: "networkidle2" });
        response = await this.browserGuard.goto(URLS.WATCHLIST_PAGE, { waitUntil: "networkidle2" });
        this.isAutomatedNavigation = false;
      } else {
        this.isAutomatedNavigation = true;
        response = await this.browserGuard.reload({ waitUntil: "networkidle2" });
        this.isAutomatedNavigation = false;
      }

      const afterCookies = await this.browserGuard.cookies();
      const finalUrl = this.browserGuard.url();
      const responseStatus = response ? response.status() : "unknown";
      this.sessionLogger.log(
        `Navigation complete (${refreshMethod}): ${finalUrl} status=${responseStatus} ` +
          `login=${finalUrl.includes("/login")} auth=${finalUrl.includes("/authenticate")}`,
      );

      const afterSnapshot = this.cookieTracker.snapshot(afterCookies, "AFTER_PAGE_REFRESH");
      this.cookieTracker.compare(beforeSnapshot, afterSnapshot, "PAGE_REFRESH");
      this.sessionLogger.logCookies("AFTER PAGE REFRESH", afterCookies);
      this.sessionLogger.logCookieComparison(beforeCookies, afterCookies);

      const hasValidSession = afterCookies.some((c) => c.name === "SMSESSION" || c.name === "ETSESSION");
      if (!hasValidSession) {
        this.sessionLogger.logError("No valid session cookies found after refresh", new Error("SESSION_INVALID"));
      }

      this.cookies = afterCookies;
      const extracted = CookieUtils.getCookies();
      if (Object.keys(extracted).length > 0) {
        this.cookies = this.mergeCookies(this.cookies, extracted);
      }
      this.persistSession();

      this.bus.publish(EVENTS.SESSION_REFRESHED, {
        timestamp: Date.now(),
        cookieCount: this.cookies.length,
        hasValidSession,
      });

      this.sessionLogger.log(`Page refresh (${refreshMethod}) ${hasValidSession ? "SUCCESSFUL" : "FAILED"}`);
    } catch (error) {
      this.sessionLogger.logError("Error during page refresh", error as Error);
    } finally {
      this.isRefreshInProgress = false;
    }
  }

  // ==========================================================================
  // Session persistence + recovery
  // ==========================================================================

  /** Write cookies + request headers to the MTap session store (survives restart). */
  private persistSession(): void {
    try {
      this.sessionStore.put({
        cookies: this.cookies ?? undefined,
        requestHeaders: this.requestHeaders,
      });
    } catch (err) {
      this.logger.warn(`SessionStore.put failed: ${(err as Error).message}`);
    }
  }

  /** Merge header-extracted cookies into the browser cookie list (by name). */
  private mergeCookies(pageCookies: StoredCookie[], extractedCookies: Record<string, string>): StoredCookie[] {
    const cookieMap = new Map<string, StoredCookie>(pageCookies.map((c) => [c.name, c]));
    for (const [name, value] of Object.entries(extractedCookies)) {
      cookieMap.set(name, {
        name,
        value,
        domain: ".etrade.com",
        path: "/",
        expires: -1,
        httpOnly: true,
        secure: true,
      });
    }
    return Array.from(cookieMap.values());
  }

  /**
   * Recover from a SESSION_EXPIRED event: reinitialize the browser if it crashed,
   * try a cookie restore, then a navigation refresh, then a full re-login.
   */
  private async handleSessionExpired(eventData: SessionExpiredEvent): Promise<void> {
    if (this.isRefreshingSession) {
      this.sessionLogger.log("Session refresh already in progress, skipping");
      return;
    }

    this.isRefreshInProgress = true;
    this.isRefreshingSession = true;

    const sessionAge = this.loginTimestamp ? ((Date.now() - this.loginTimestamp) / 1000 / 60).toFixed(2) : "unknown";
    this.sessionLogger.log(`SESSION RECOVERY INITIATED (reason: ${eventData.reason ?? "UNKNOWN"}, age: ${sessionAge}m)`);

    try {
      // Step 0: reinitialize the browser if the page is gone.
      if (!this.browserGuard.isPageValid()) {
        this.sessionLogger.log("Step 0: Browser/page closed, reinitializing...");
        if (this.browser) await this.browser.close();

        this.browser = new Browser();
        this.page = await this.browser.initialize(false);
        if (!this.page) throw new Error("Failed to reinitialize browser page");

        this.browserGuard.setPage(this.page);
        const instance = this.browser.instance;
        if (instance) this.browserGuard.setBrowser(instance);

        this.browser.setupRequestHandlers(
          (request: HTTPRequest) => this.handleRequest(request),
          (response: HTTPResponse) => this.handleResponse(response),
        );
        this.setupNavigationTracking();

        this.auth = this.makeAuthentication();
        await this.auth.initialize();
        this.sessionLogger.log("Browser reinitialized.");

        // Step 0.5: try to restore the previous session from saved cookies.
        if (this.cookies && this.cookies.length > 0) {
          this.sessionLogger.log("Step 0.5: Attempting session restore from saved cookies...");
          this.isAutomatedNavigation = true;
          const restoreResult = await this.browserGuard.tryRestoreSession(this.cookies, URLS.WATCHLIST_PAGE);
          this.isAutomatedNavigation = false;

          if (restoreResult.success && restoreResult.cookies) {
            this.cookies = restoreResult.cookies;
            this.loginTimestamp = Date.now();
            this.persistSession();
            this.buildScrapers();

            this.bus.publish(EVENTS.SESSION_REFRESHED, {
              timestamp: Date.now(),
              recoveredFrom: eventData.source,
              method: "COOKIE_RESTORE",
              cookieCount: this.cookies.length,
            });
            this.sessionLogger.log("Session RESTORED from saved cookies — no re-auth needed!");
            return;
          }
          this.sessionLogger.log(`Cookie restore failed: ${restoreResult.message}`);
        }
        this.sessionLogger.log("Proceeding to re-authentication...");
      }

      // For non-crash expiry, try a navigation-based refresh before a full re-login.
      const browserWasClosed = eventData.reason === "BROWSER_CLOSED";
      let hasValidSession = false;

      if (!browserWasClosed) {
        this.sessionLogger.log("Step 1: Attempting navigation-based refresh...");
        this.isAutomatedNavigation = true;
        await this.browserGuard.goto(EtradeService.POSITIONS_PAGE_URL, { waitUntil: "networkidle2" });
        await this.browserGuard.goto(URLS.WATCHLIST_PAGE, { waitUntil: "networkidle2" });
        this.isAutomatedNavigation = false;

        this.cookies = await this.browserGuard.cookies();
        hasValidSession = this.cookies.some((c) => c.name === "SMSESSION" || c.name === "ETSESSION");
      }

      if (hasValidSession) {
        this.sessionLogger.log("Navigation refresh SUCCESSFUL — session restored");
        const extracted = CookieUtils.getCookies();
        if (this.cookies && Object.keys(extracted).length > 0) {
          this.cookies = this.mergeCookies(this.cookies, extracted);
        }
        this.persistSession();
        this.buildScrapers();
        this.loginTimestamp = Date.now();

        this.bus.publish(EVENTS.SESSION_REFRESHED, {
          timestamp: Date.now(),
          recoveredFrom: eventData.source,
          method: "NAVIGATION",
          cookieCount: this.cookies?.length ?? 0,
        });
      } else {
        // Step 2: full re-authentication.
        this.sessionLogger.log("Step 2: Attempting FULL RE-AUTHENTICATION...");
        if (!this.auth) this.auth = this.makeAuthentication();
        await this.auth.login(ENV.ET_USERNAME, ENV.ET_PASSWORD);

        this.isAutomatedNavigation = true;
        await this.browserGuard.goto(URLS.WATCHLIST_PAGE, { waitUntil: "networkidle2" });
        this.isAutomatedNavigation = false;

        this.cookies = await this.browserGuard.cookies();
        this.loginTimestamp = Date.now();
        this.persistSession();
        this.buildScrapers();

        this.sessionLogger.logCookies("AFTER RE-AUTHENTICATION", this.cookies);
        this.sessionLogger.log("Full re-authentication SUCCESSFUL");

        this.bus.publish(EVENTS.SESSION_REFRESHED, {
          timestamp: Date.now(),
          recoveredFrom: eventData.source,
          method: "REAUTH",
          cookieCount: this.cookies.length,
        });
      }
    } catch (error) {
      this.sessionLogger.logError("CRITICAL: Session recovery FAILED", error as Error);
      this.sessionLogger.log("Stopping monitoring due to unrecoverable session failure");
      this.stopMonitoring();
    } finally {
      this.isRefreshingSession = false;
      this.isRefreshInProgress = false;
    }
  }

  private stopMonitoring(): void {
    for (const intervalId of Object.values(this.dataIntervals)) {
      if (intervalId) clearInterval(intervalId);
    }
    this.dataIntervals = {};

    if (this.pageInterval) {
      clearInterval(this.pageInterval);
      this.pageInterval = null;
    }
    if (this.moversInterval) {
      clearInterval(this.moversInterval);
      this.moversInterval = null;
    }

    this.sessionLogger.close();

    if (this.browser) {
      this.browser.close().catch((err: unknown) => {
        this.logger.error(`Error closing browser during stopMonitoring: ${(err as Error).message}`);
      });
    }
  }
}

export default EtradeService;
