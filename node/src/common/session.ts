import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { REPO_ROOT } from "./config.js";

// MTap-internal session store: persists the E*TRADE browser cookies and the
// stk1/stk2 request headers to state/<scope>/session.json so a login survives
// restarts. This replaces the `sourcing` scope of MBin's DB-backed SessionStore
// — MTap now owns the live browser session end to end. state/ is gitignored
// (the file holds live session material).

// A single browser cookie. Structurally compatible with Puppeteer's
// Protocol.Network.Cookie (which carries extra fields we ignore) so the
// scraper can persist `page.cookies()` directly and re-inject on restart.
export interface StoredCookie {
  name: string;
  value: string;
  domain?: string;
  path?: string;
  expires?: number;
  httpOnly?: boolean;
  secure?: boolean;
}

export interface EtradeSession {
  cookies?: StoredCookie[];
  requestHeaders?: {
    stk1?: string;
    stk2?: string;
    [key: string]: string | undefined;
  };
  updatedAt?: string;
}

export class SessionStore {
  private path: string;
  private cache: EtradeSession | null = null;

  constructor(scope = "etrade") {
    this.path = process.env.SESSION_PATH ?? join(REPO_ROOT, "state", scope, "session.json");
  }

  /** Last persisted session, or null if none has been written yet. */
  get(): EtradeSession | null {
    if (this.cache) return this.cache;
    if (!existsSync(this.path)) return null;
    try {
      this.cache = JSON.parse(readFileSync(this.path, "utf-8")) as EtradeSession;
    } catch (err) {
      console.error(`[SessionStore] failed to read ${this.path}:`, (err as Error).message);
      this.cache = null;
    }
    return this.cache;
  }

  /** Persist a session (stamped with updatedAt), creating state/ as needed. */
  put(session: EtradeSession): void {
    this.cache = { ...session, updatedAt: new Date().toISOString() };
    mkdirSync(dirname(this.path), { recursive: true });
    writeFileSync(this.path, JSON.stringify(this.cache, null, 2), "utf-8");
  }

  /** True when a session with cookies is on file. */
  hasSession(): boolean {
    const s = this.get();
    return Boolean(s?.cookies);
  }
}
