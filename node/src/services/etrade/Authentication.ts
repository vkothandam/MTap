import { writeFileSync } from "node:fs";
import { join } from "node:path";
import type { HTTPRequest } from "puppeteer";
import { URLS, SERVER_CONFIG, PHONE_CONFIG } from "./constants.js";
import { ENV } from "./env.js";
import { SMSNotifier } from "./utils/SMSNotifier.js";
import { BrowserGuard } from "./utils/BrowserGuard.js";
import { BrowserClosedError } from "./utils/errors.js";
import type { OtpCoordinator } from "./otp.js";
import { stateRoot } from "../../common/config.js";

// E*TRADE browser-login flow. Ported from MBin scrape/modules/Authentication.js.
// Changes for MTap:
//   - The dead WSService import is dropped.
//   - The OTP wait no longer goes over the EventBroker; it resolves in-process
//     via OtpCoordinator (the human submits on MTap's own /auth/otp page).
//   - All page work goes through BrowserGuard (always present), so MBin's
//     "page fallback" branches are gone.
//   - Credentials come from ENV; the SMS gateway account too.

// These identifiers exist in the browser context where the evaluate() callbacks
// run; DOM lib provides their types at compile time.

export interface AuthenticationOptions {
  /** Optional hook fired when an OTP challenge is detected (e.g. to mirror AUTH_OTP_REQUIRED on the bus). */
  onOtpRequired?: () => void;
}

export class Authentication {
  /** How long to wait after "Log on" for the verification step or the logged-in page. */
  static LOGIN_OUTCOME_TIMEOUT_MS = 60000;

  private readonly guard: BrowserGuard;
  private readonly otpCoordinator: OtpCoordinator;
  private readonly onOtpRequired?: () => void;
  private readonly smsNotifier: SMSNotifier;
  private readonly maxRetries = 3;
  private bearerToken: string | null = null;

  constructor(guard: BrowserGuard, otpCoordinator: OtpCoordinator, options: AuthenticationOptions = {}) {
    this.guard = guard;
    this.otpCoordinator = otpCoordinator;
    this.onOtpRequired = options.onOtpRequired;
    this.smsNotifier = new SMSNotifier({
      email: ENV.SMS_EMAIL,
      emailPassword: ENV.SMS_EMAIL_PASSWORD,
      phoneNumber: PHONE_CONFIG.PERSONAL_NUMBER,
      carrier: PHONE_CONFIG.CARRIER,
    });
  }

  setBearerToken(token: string): void {
    this.bearerToken = token;
  }

  getBearerToken(): string | null {
    return this.bearerToken;
  }

  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  /** Enable request interception and capture any Bearer token. This handler is the sole `request.continue()` caller. */
  async initialize(): Promise<void> {
    await this.guard.setRequestInterception(true);
    this.guard.on("request", (request: HTTPRequest) => {
      const headers = request.headers();
      if (headers.authorization && headers.authorization.startsWith("Bearer ")) {
        this.setBearerToken(headers.authorization);
      }
      // Interception is on; exactly one handler must resolve the request.
      void request.continue().catch(() => {});
    });
  }

  async login(username: string, password: string): Promise<boolean> {
    let attempts = 0;

    while (attempts < this.maxRetries) {
      try {
        attempts++;
        console.log(`[Auth] Login attempt ${attempts} of ${this.maxRetries}`);

        await this.guard.goto(URLS.LOGIN_PAGE, { waitUntil: "networkidle2", timeout: 60000 });

        // Let the page settle before interacting.
        await this.sleep(2000);
        await this.guard.waitForSelector("#USER", { timeout: 30000 });

        await this.typeInto("#USER", username);
        await this.typeInto("#password", password);

        // Click first, then wait for navigation separately (Promise.all can detach the
        // frame headless). The step-up page may not navigate at all, so a timeout here is
        // not an error: handlePhoneVerification waits for the actual outcome.
        await this.guard.click("#mfaLogonButton");
        try {
          await this.guard.waitForNavigation({ waitUntil: "networkidle2", timeout: 30000 });
        } catch (error) {
          if (error instanceof BrowserClosedError) throw error;
          await this.sleep(3000);
        }

        const shouldRetry = await this.handleLoginErrors(attempts);
        if (shouldRetry) continue;

        await this.handlePhoneVerification();
        return true;
      } catch (error) {
        if (error instanceof BrowserClosedError) {
          console.error(`[Auth] Login attempt ${attempts} failed: browser closed`);
          throw error;
        }

        const message = (error as Error).message ?? "";
        if (message.includes("detached")) {
          console.error(`[Auth] Login attempt ${attempts} failed: frame detached — retrying with fresh state`);
          await this.sleep(3000);
        } else {
          console.error(`[Auth] Login attempt ${attempts} failed:`, message);
        }

        if (attempts === this.maxRetries) {
          throw new Error(`Failed to login after ${this.maxRetries} attempts: ${message}`);
        }
        await this.sleep(2000);
      }
    }
    return false;
  }

  /**
   * Type into an input after clearing it. Headed, Chrome fills a login it has saved (one
   * typed by hand in the window), and typing appends to it: the ID and password doubled.
   * A filled password reads as empty to the page until the first keystroke commits it, so
   * the first clear can miss it; then the field is cleared and typed once more.
   */
  private async typeInto(selector: string, text: string): Promise<void> {
    let length = -1;
    for (let round = 1; round <= 2; round++) {
      await this.guard.evaluate((sel: string) => {
        const input = document.querySelector(sel) as HTMLInputElement | null;
        if (!input) return;
        input.value = "";
        input.dispatchEvent(new Event("input", { bubbles: true }));
      }, selector);
      await this.guard.type(selector, text, { delay: 100 });
      length = await this.guard.evaluate(
        (sel: string) => (document.querySelector(sel) as HTMLInputElement | null)?.value.length ?? -1,
        selector,
      );
      if (length === text.length) return;
      console.log(`[Auth] ${selector} held ${length} characters after typing ${text.length} (a saved login); typing again`);
    }
    throw new Error(`${selector} holds ${length} characters after typing ${text.length}`);
  }

  /** Returns true if the caller should retry the login attempt. */
  async handleLoginErrors(attempts: number): Promise<boolean> {
    try {
      const bodyText: string = await this.guard.$eval("body", (el: HTMLElement) => el.innerText);
      if (bodyText.includes("Help us confirm your identity")) {
        // Phone verification screen, not an error.
        return false;
      }

      const errorElements: string[] = await this.guard.$$eval(".error-message, .alert-error", (elements: Element[]) =>
        elements.map((el: Element) => (el.textContent ?? "").trim()),
      );

      if (errorElements.length > 0) {
        const errorText = errorElements.join(" ");
        console.error("[Auth] Login error:", errorText);

        if (errorText.includes("942")) {
          console.log(`[Auth] Error 942 detected on attempt ${attempts}`);
          if (attempts === this.maxRetries) {
            throw new Error("Maximum login attempts reached with error 942");
          }
          await this.sleep(2000);
          return true;
        }

        throw new Error(`E*TRADE Login Error: ${errorText}`);
      }

      const currentUrl = this.guard.url();
      if (!currentUrl.includes("login")) {
        console.log("[Auth] Login successful — redirected to a new page");
      }
      return false;
    } catch (error) {
      if (error instanceof BrowserClosedError) throw error;
      console.error("[Auth] Error during login:", error);
      throw error;
    }
  }

  /** Detect the SMS challenge, dispatch the OTP link, and wait for the code (in-process). */
  async handlePhoneVerification(): Promise<void> {
    let verificationDetected = false;
    try {
      // After "Log on" the page can spin for a while (longer headless) before it shows the
      // verification step or leaves the login page; decide only once one of them happens.
      if (!(await this.waitForLoginOutcome())) {
        console.log("[Auth] No phone verification needed");
        return;
      }

      verificationDetected = true;
      console.log("[Auth] Phone verification detected...");
      this.onOtpRequired?.();

      await this.selectPhoneNumberByPriority();

      const otpUrl = `${SERVER_CONFIG.BASE_URL}${SERVER_CONFIG.OTP_PATH}`;
      await this.smsNotifier.sendNotification(`ETrade verification required. Enter OTP at: ${otpUrl}`);

      await this.sleep(4000);

      // #send-otp-btn is the current page; #sendOTPCodeBtn the older one.
      const sendButton = "#send-otp-btn, #sendOTPCodeBtn";
      await this.guard.waitForSelector(sendButton);
      await this.guard.click(sendButton);
      console.log("[Auth] Clicked Send Code button");

      const otp = await this.waitForCodeOrManualEntry();
      if (otp === null) {
        console.log("[Auth] Verification completed in the browser window");
      } else {
        await this.enterOTPAndRemember(otp);
      }
    } catch (error) {
      if (error instanceof BrowserClosedError) throw error;
      console.error(
        verificationDetected
          ? "[Auth] Error during phone verification process:"
          : "[Auth] Error checking for phone verification:",
        error,
      );
      throw error;
    }
  }

  /**
   * Wait until the login either reaches the phone verification step (true) or leaves the
   * login pages (false). Throws if it is still on the login page after LOGIN_OUTCOME_TIMEOUT_MS.
   */
  private async waitForLoginOutcome(): Promise<boolean> {
    const deadline = Date.now() + Authentication.LOGIN_OUTCOME_TIMEOUT_MS;
    while (Date.now() < deadline) {
      const bodyText: string = await this.guard
        .$eval("body", (el: HTMLElement) => el.innerText)
        .catch((error: unknown) => {
          if (error instanceof BrowserClosedError) throw error;
          return ""; // mid-navigation
        });
      if (bodyText.includes("Help us confirm your identity")) return true;
      if (!this.guard.url().includes("/etx/pxy/login")) return false;
      await this.sleep(1000);
    }
    throw new Error(`still on the login page ${Authentication.LOGIN_OUTCOME_TIMEOUT_MS / 1000}s after "Log on"`);
  }

  /**
   * Wait for the code from the /auth/otp page, or for someone to enter it directly in the
   * (non-headless) browser window, seen as the page leaving the login pages. Returns the
   * code, or null when it was entered in the window.
   */
  private async waitForCodeOrManualEntry(): Promise<string | null> {
    const fromOtpPage = this.otpCoordinator.waitForCode(PHONE_CONFIG.OTP_TIMEOUT);
    let done = false;
    const inWindow = (async (): Promise<null> => {
      while (!done) {
        await this.sleep(2000);
        if (!done && !this.guard.url().includes("/etx/pxy/login")) return null;
      }
      return null;
    })();
    try {
      const result = await Promise.race([fromOtpPage, inWindow]);
      if (result === null) {
        this.otpCoordinator.cancel("code entered in the browser window");
        fromOtpPage.catch(() => {}); // its rejection is expected now
      }
      return result;
    } finally {
      done = true;
    }
  }

  /** Select the registered phone by priority (default 9699 > 5646 > 5647; overridable via env). */
  async selectPhoneNumberByPriority(): Promise<void> {
    try {
      const phoneNumbers: Array<{ id: string; phoneNumber: string }> = await this.guard.$$eval(
        'input[name="phoneNumbers"]',
        (radios: Element[]) =>
          radios.map((radio: Element) => {
            const label = document.querySelector(`label[for="${radio.id}"]`);
            return { id: radio.id, phoneNumber: label ? (label.textContent ?? "").trim() : "" };
          }),
      );

      console.log(
        "[Auth] Available phone numbers:",
        phoneNumbers.map((p) => p.phoneNumber),
      );

      // With a single number on the account the page just shows it, with no radio list.
      if (phoneNumbers.length === 0) {
        console.log("[Auth] No phone number choice on the page (single number); nothing to select");
        return;
      }

      let selectedPhone: { id: string; phoneNumber: string } | undefined;
      for (const suffix of PHONE_CONFIG.SELECT_PRIORITY) {
        selectedPhone = phoneNumbers.find((p) => p.phoneNumber.endsWith(suffix));
        if (selectedPhone) {
          console.log(`[Auth] Selected phone ending with ${suffix}: ${selectedPhone.phoneNumber}`);
          break;
        }
      }

      const targetId = selectedPhone?.id ?? (phoneNumbers.length > 0 ? phoneNumbers[0].id : null);
      if (!targetId) {
        throw new Error("No phone number options available");
      }

      const clicked: boolean = await this.guard.evaluate((id: string) => {
        const radio = document.getElementById(id) as HTMLInputElement | null;
        if (radio) {
          radio.click();
          radio.checked = true;
          radio.dispatchEvent(new Event("change", { bubbles: true }));
          return true;
        }
        return false;
      }, targetId);

      if (!clicked) {
        throw new Error(`Radio button with id ${targetId} not found`);
      }
      console.log("[Auth] Phone number radio button selected successfully");
      await this.sleep(500);
    } catch (error) {
      if (error instanceof BrowserClosedError) throw error;
      console.error("[Auth] Error selecting phone number:", (error as Error).message);
      throw error;
    }
  }

  async enterOTPAndRemember(otp: string): Promise<void> {
    // Keep only digits, max length 6.
    const formattedOTP = otp.replace(/\D/g, "").slice(0, 6);

    console.log("[Auth] Waiting for OTP input field...");
    // #verificationCode on the older page; the others match a standard one-time-code field.
    const codeInput = '#verificationCode, input[autocomplete="one-time-code"], input[name*="code" i], input[id*="code" i]';
    try {
      await this.guard.waitForSelector(codeInput, { timeout: 30000 });
    } catch (error) {
      if (error instanceof BrowserClosedError) throw error;
      const file = join(stateRoot(), "logs", `otp-page-${Date.now()}.html`);
      writeFileSync(file, await this.guard.evaluate(() => document.documentElement.outerHTML));
      console.error(`[Auth] No OTP input field found; page saved to ${file}`);
      throw error;
    }

    // Clear the field first.
    await this.guard.evaluate((selector: string) => {
      const input = document.querySelector(selector) as HTMLInputElement | null;
      if (input) input.value = "";
    }, codeInput);

    await this.guard.type(codeInput, formattedOTP, { delay: 100 });
    // "Remember this device", when the page offers it, so later logins skip the code.
    const remembered: boolean = await this.guard.evaluate(() => {
      const box = document.getElementById("saveDevice") as HTMLInputElement | null;
      if (box && !box.checked) box.click();
      return box !== null;
    });
    if (!remembered) console.log("[Auth] No 'remember this device' option on the OTP page");

    // Small delay to let form validation settle, then submit.
    await this.sleep(500);
    try {
      await this.guard.click('button.btn-primary[type="button"]');
    } catch (error) {
      if (error instanceof BrowserClosedError) throw error;
      console.log("[Auth] Direct submit click failed, trying alternative method...");
      await this.guard.evaluate(() => {
        const submitBtn = Array.from(document.querySelectorAll("button")).find(
          (button) => (button.textContent ?? "").trim() === "Submit",
        );
        if (submitBtn) {
          submitBtn.focus();
          submitBtn.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true, view: window }));
        }
      });
    }

    await this.guard.waitForNavigation();
  }
}
