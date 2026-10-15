import vanillaPuppeteer from "puppeteer";
import { addExtra } from "puppeteer-extra";
import StealthPlugin from "puppeteer-extra-plugin-stealth";
import type { Browser as PuppeteerBrowser, BrowserContext, Page, PuppeteerLaunchOptions } from "puppeteer";
import { join } from "node:path";
import { ENV } from "./env.js";
import { stateRoot } from "../../common/config.js";

// puppeteer-extra ships a CommonJS `.d.ts` with no `exports`/`type` field, so a
// default import resolves to the module namespace (not the PuppeteerExtra
// instance) under NodeNext. addExtra(vanillaPuppeteer) is the officially
// recommended way to obtain a plugin-capable puppeteer. The cast bridges
// puppeteer-extra's stale VanillaPuppeteer type (it still requires the removed
// createBrowserFetcher); addExtra only proxies launch/connect at runtime.
const puppeteer = addExtra(vanillaPuppeteer as unknown as Parameters<typeof addExtra>[0]);

// Puppeteer browser lifecycle. Ported from MBin scrape/modules/Browser.js
// (puppeteer-extra + stealth). Differences: headless/executable-path come from
// MTap's env (ENV), and initialize() no-ops when scraping is disabled.

type RequestHandler = (...args: any[]) => void;

export class Browser {
  private browser: PuppeteerBrowser | null = null;
  private context: BrowserContext | null = null;
  page: Page | null = null;

  constructor() {
    puppeteer.use(StealthPlugin());
  }

  /** Launch the browser and open the working page. Returns null if scraping is disabled. */
  async initialize(incognito = false): Promise<Page | null> {
    if (!ENV.FEATURES.SCRAPING) return null;

    const launchOptions: PuppeteerLaunchOptions = {
      // puppeteer 22: headless=true is the (new) headless mode; false shows the GUI.
      headless: ENV.HEADLESS,
      protocolTimeout: 180000, // 3 minutes — long-running sessions
      timeout: 60000,
      // A persistent profile keeps E*TRADE's device-trust and bot-manager cookies across
      // runs, so a login looks like a returning browser instead of a fresh one each time.
      userDataDir: ENV.CHROME_PROFILE_DIR ?? join(stateRoot(), "etrade", "chrome-profile"),
      // --enable-automation sets navigator.webdriver and the "controlled by automated
      // software" bar, both of which bot detection reads.
      ignoreDefaultArgs: ["--enable-automation"],
      args: [
        // Security sandbox settings
        "--no-sandbox",
        "--disable-setuid-sandbox",
        "--disable-dev-shm-usage",
        // Memory & performance
        "--disable-gpu",
        "--disable-software-rasterizer",
        "--disable-extensions",
        "--disable-background-networking",
        "--disable-sync",
        "--disable-translate",
        "--disable-features=TranslateUI",
        "--no-first-run",
        "--no-default-browser-check",
        // Stability
        "--disable-hang-monitor",
        "--disable-popup-blocking",
        "--disable-prompt-on-repost",
        "--disable-backgrounding-occluded-windows",
        "--disable-renderer-backgrounding",
        "--disable-ipc-flooding-protection",
        // Memory limits (prevent OOM on long sessions)
        "--js-flags=--max-old-space-size=512",
        "--memory-pressure-off",
        // Network stability
        "--disable-features=NetworkService,NetworkServiceInProcess",
      ],
    };

    if (ENV.PUPPETEER_EXECUTABLE_PATH) {
      launchOptions.executablePath = ENV.PUPPETEER_EXECUTABLE_PATH;
    }

    console.log(`[Browser] launching (headless=${ENV.HEADLESS}, profile=${launchOptions.userDataDir})...`);
    const launchStart = Date.now();
    this.browser = await puppeteer.launch(launchOptions);
    console.log(`[Browser] launched in ${Date.now() - launchStart}ms`);

    this.page = await this.createPage(incognito);
    await this.setupPage();
    return this.page;
  }

  private async createPage(incognito: boolean): Promise<Page> {
    if (!this.browser) throw new Error("Browser not launched");
    if (incognito) {
      this.context = await this.browser.createBrowserContext();
      return this.context.newPage();
    }
    return this.browser.newPage();
  }

  private async setupPage(): Promise<void> {
    if (!this.page) throw new Error("Page not created");
    await this.page.setViewport({ width: 1280, height: 800 });
    // The real browser's UA (minus "Headless"), so it matches the version in the client
    // hints; a fixed UA string from another Chrome version is itself a bot signal.
    const userAgent = (await this.browser!.userAgent()).replace("HeadlessChrome", "Chrome");
    await this.page.setUserAgent(userAgent);
    await this.page.setRequestInterception(true);
  }

  /** The underlying puppeteer Browser (for BrowserGuard / cleanup). */
  get instance(): PuppeteerBrowser | null {
    return this.browser;
  }

  setupRequestHandlers(requestCallback: RequestHandler, responseCallback: RequestHandler): void {
    if (!this.page) return;
    this.page.on("request", requestCallback as never);
    this.page.on("response", responseCallback as never);
  }

  isConnected(): boolean {
    return Boolean(this.browser && this.browser.connected);
  }

  isPageValid(): boolean {
    try {
      return Boolean(this.page && !this.page.isClosed());
    } catch {
      return false;
    }
  }

  /** Cleanly close the page and browser. */
  async close(): Promise<void> {
    try {
      if (this.page && !this.page.isClosed()) {
        await this.page.close();
      }
      if (this.browser && this.browser.connected) {
        await this.browser.close();
      }
    } catch (error) {
      console.error("[Browser] Error closing browser:", (error as Error).message);
    } finally {
      this.page = null;
      this.browser = null;
      this.context = null;
    }
  }
}
