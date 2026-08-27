import type { HTTPRequest } from "puppeteer";
import { URLS, SERVER_CONFIG, PHONE_CONFIG } from "./constants.js";
import { ENV } from "./env.js";
import { SMSNotifier } from "./utils/SMSNotifier.js";
import { BrowserGuard } from "./utils/BrowserGuard.js";
import { BrowserClosedError } from "./utils/errors.js";
import type { OtpCoordinator } from "./otp.js";

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

        await this.guard.type("#USER", username, { delay: 100 });
        await this.guard.type("#password", password, { delay: 100 });

        // In headless mode, Promise.all with navigation can cause frame detachment;
        // click first, then wait for navigation separately.
        if (ENV.HEADLESS) {
          await this.guard.click("#mfaLogonButton");
          try {
            await this.guard.waitForNavigation({ waitUntil: "networkidle2", timeout: 30000 });
          } catch {
            // Navigation may have completed via a fast redirect; give it a moment.
            await this.sleep(3000);
          }
        } else {
          await Promise.all([
            this.guard.click("#mfaLogonButton"),
            this.guard.waitForNavigation({ waitUntil: "networkidle2" }),
          ]);
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
      const bodyText: string = await this.guard.$eval("body", (el: HTMLElement) => el.innerText);
      if (!bodyText.includes("Help us confirm your identity")) {
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

      await this.guard.waitForSelector("#sendOTPCodeBtn");
      await this.guard.click("#sendOTPCodeBtn");
      console.log("[Auth] Clicked Send Code button");

      const otp = await this.otpCoordinator.waitForCode(PHONE_CONFIG.OTP_TIMEOUT);
      await this.enterOTPAndRemember(otp);
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
    await this.guard.waitForSelector("#verificationCode", { timeout: 30000 });

    // Clear the field first.
    await this.guard.evaluate(() => {
      const input = document.getElementById("verificationCode") as HTMLInputElement | null;
      if (input) input.value = "";
    });

    await this.guard.type("#verificationCode", formattedOTP, { delay: 100 });
    await this.guard.click("#saveDevice");

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
