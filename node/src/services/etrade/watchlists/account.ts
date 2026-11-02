import { readFileSync } from "node:fs";
import { join } from "node:path";
import { parseDotEnv } from "../../../common/dotenv.js";
import { stateRoot } from "../../../common/config.js";
import { SessionStore, type StoredCookie } from "../../../common/session.js";
import type { RequestHeaders } from "../ETPageScraper.js";

// Which E*TRADE login the watchlist utility uses. Each account has its own state folder, so a
// second login never touches the collector's Chrome profile, session.json or bearer.json:
//   <state root>/etrade-<account>/{chrome-profile, session.json, bearer.json}

export interface EtradeAccount {
  name: string;
  username: string;
  password: string;
  stateDir: string;
}

/** The env names for an account: main = ET_USERNAME/ET_PASSWORD, else <NAME>_ET_*. */
export function accountEnvNames(account: string): { user: string; pass: string } {
  const prefix = account === "main" ? "" : `${account.toUpperCase().replace(/[^A-Z0-9]/g, "_")}_`;
  return { user: `${prefix}ET_USERNAME`, pass: `${prefix}ET_PASSWORD` };
}

/**
 * An account's credentials, from `secretsFile` (KEY=VALUE, as the NAS's ~/.mtap.env) when
 * given, else from `env`. Only the two credential keys are read.
 */
export function resolveAccount(account: string, secretsFile?: string, env: NodeJS.ProcessEnv = process.env): EtradeAccount {
  const names = accountEnvNames(account);
  const source = secretsFile ? parseDotEnv(readFileSync(secretsFile, "utf-8")) : env;
  const username = source[names.user] ?? "";
  const password = source[names.pass] ?? "";
  if (!username || !password) {
    throw new Error(`no ${names.user}/${names.pass} in ${secretsFile ?? "the environment"}`);
  }
  return { name: account, username, password, stateDir: join(stateRoot(), `etrade-${account}`) };
}

/** The account's session store (cookies and stk1 from its last login). */
export function accountSessionStore(account: EtradeAccount): SessionStore {
  return new SessionStore(`etrade-${account.name}`, join(account.stateDir, "session.json"));
}

/** The account's saved cookies and headers, for API calls with no browser; null if none. */
export function savedSession(account: EtradeAccount): { cookies: StoredCookie[]; requestHeaders: RequestHeaders } | null {
  const stored = accountSessionStore(account).get();
  if (!stored?.cookies?.length || !stored.requestHeaders?.stk1) return null;
  return { cookies: stored.cookies, requestHeaders: stored.requestHeaders as RequestHeaders };
}
