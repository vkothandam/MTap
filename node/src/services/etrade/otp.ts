// In-process coordinator for the human-in-the-loop OTP step. The whole login
// flow lives inside MTap: Authentication calls waitForCode() when E*TRADE
// presents the SMS challenge, and otpServer.ts calls submitCode() when the
// human posts the code to /auth/otp. Deliberately transport-independent (no bus
// hop) — the code never leaves this process, matching the plan's OTP design.

interface Waiter {
  resolve: (code: string) => void;
  reject: (err: Error) => void;
  timer: ReturnType<typeof setTimeout>;
}

export class OtpCoordinator {
  private waiter: Waiter | null = null;

  /** True while a login is blocked waiting for a code (drives the /auth/otp page state). */
  isPending(): boolean {
    return this.waiter !== null;
  }

  /**
   * Wait for the human to submit an OTP code. Rejects if none arrives within
   * timeoutMs, or if a newer wait supersedes this one.
   */
  waitForCode(timeoutMs: number): Promise<string> {
    // A new challenge supersedes any stale pending wait.
    this.cancel("superseded by a new OTP challenge");
    return new Promise<string>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.waiter = null;
        reject(new Error(`OTP not submitted within ${timeoutMs}ms`));
      }, timeoutMs);
      this.waiter = { resolve, reject, timer };
    });
  }

  /**
   * Submit a code from the /auth/otp endpoint. Returns false if no login is
   * currently waiting (stale submission), true if it resolved a pending wait.
   */
  submitCode(code: string): boolean {
    const w = this.waiter;
    if (!w) return false;
    clearTimeout(w.timer);
    this.waiter = null;
    w.resolve(code);
    return true;
  }

  /** Abort a pending wait (shutdown, or a superseding challenge). */
  cancel(reason = "cancelled"): void {
    const w = this.waiter;
    if (!w) return;
    clearTimeout(w.timer);
    this.waiter = null;
    w.reject(new Error(`OTP wait ${reason}`));
  }
}
