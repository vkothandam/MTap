import { join } from "node:path";
import type { HTTPRequest } from "puppeteer";
import { Browser } from "../Browser.js";
import { Authentication } from "../Authentication.js";
import { OtpCoordinator } from "../otp.js";
import { OtpServer } from "../otpServer.js";
import { BrowserGuard } from "../utils/BrowserGuard.js";
import { URLS } from "../constants.js";
import type { StoredCookie } from "../../../common/session.js";
import type { RequestHeaders } from "../ETPageScraper.js";
import { accountSessionStore, type EtradeAccount } from "./account.js";

// A one-off E*TRADE browser login for the watchlist utility: the login steps of
// EtradeService.initialize (saved session first, else username/password and the OTP step),
// without polling or the bus, in the account's own state folder (account.ts).

export interface EtradeSession {
  cookies: StoredCookie[];
  requestHeaders: RequestHeaders;
  close(): Promise<void>;
}

/** Log in (or reuse the account's saved session) and return the cookies and request headers. */
export async function openEtradeSession(account: EtradeAccount, log: (line: string) => void = console.log): Promise<EtradeSession> {
  const browser = new Browser(join(account.stateDir, "chrome-profile"));
  const guard = new BrowserGuard(null);
  const otpCoordinator = new OtpCoordinator();
  const otpServer = new OtpServer(otpCoordinator);
  const store = accountSessionStore(account);
  let requestHeaders: RequestHeaders = {};

  const close = async (): Promise<void> => {
    await browser.close();
    await otpServer.stop();
  };

  try {
    const page = await browser.initialize(false);
    if (!page) throw new Error("no browser page (FEATURE_SCRAPING=false?)");
    guard.setPage(page);
    if (browser.instance) guard.setBrowser(browser.instance);

    // Authentication owns request.continue(); this listener only collects headers (stk1).
    const auth = new Authentication(guard, otpCoordinator, { bearerPath: join(account.stateDir, "bearer.json") });
    await auth.initialize();
    browser.setupRequestHandlers(
      (request: HTTPRequest) => {
        requestHeaders = { ...requestHeaders, ...request.headers() };
      },
      () => {},
    );

    let cookies: StoredCookie[] | null = null;
    const stored = store.get();
    if (stored?.cookies?.length) {
      for (const [key, value] of Object.entries(stored.requestHeaders ?? {})) {
        if (typeof value === "string" && requestHeaders[key] === undefined) requestHeaders[key] = value;
      }
      const restored = await guard.tryRestoreSession(stored.cookies, URLS.WATCHLIST_PAGE);
      log(`[${account.name}] saved session: ${restored.message}`);
      if (restored.success) cookies = restored.cookies;
    }

    if (!cookies) {
      // The OTP page is optional here (the code can be typed in the window), e.g. when a
      // collector on this machine already holds the port.
      await otpServer.start().catch((error: Error) => log(`[${account.name}] no OTP page: ${error.message}`));
      log(`[${account.name}] logging in; if E*TRADE asks for a code, type it in the Chrome window and tick "save device"`);
      await auth.login(account.username, account.password);
      await guard.goto(URLS.WATCHLIST_PAGE, { waitUntil: "networkidle2" });
      cookies = await guard.cookies();
    }

    if (!cookies?.length) throw new Error("no cookies after login");
    if (!requestHeaders.stk1) throw new Error("no stk1 header captured from the watchlist page");
    store.put({ cookies, requestHeaders });
    return { cookies, requestHeaders, close };
  } catch (error) {
    await close();
    throw error;
  }
}
