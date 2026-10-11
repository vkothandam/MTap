import { existsSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

// Load the repo-root .env into process.env, never overriding a variable that is
// already set. Python (common/config.py) and the schedule scripts read the same
// file, so one .env per machine configures every runtime. Imported for its side
// effect first thing in cli.ts and by config.ts, so it runs before any module
// reads process.env at load time (e.g. services/etrade/env.ts).

const __dirname = dirname(fileURLToPath(import.meta.url));
const DOTENV = resolve(__dirname, "..", "..", "..", ".env");

export function parseDotEnv(text: string): Record<string, string> {
  const out: Record<string, string> = {};
  for (const raw of text.split(/\r?\n/)) {
    let line = raw.trim();
    if (!line || line.startsWith("#") || !line.includes("=")) continue;
    if (line.startsWith("export ")) line = line.slice("export ".length);
    const eq = line.indexOf("=");
    const key = line.slice(0, eq).trim();
    let value = line.slice(eq + 1).trim();
    const q = value[0];
    if (value.length >= 2 && (q === '"' || q === "'") && value.endsWith(q)) {
      value = value.slice(1, -1);
    } else if (value.includes(" #")) {
      value = value.slice(0, value.indexOf(" #")).trimEnd();
    }
    out[key] = value;
  }
  return out;
}

export function loadDotEnv(path: string = DOTENV): void {
  if (!existsSync(path)) return;
  for (const [key, value] of Object.entries(parseDotEnv(readFileSync(path, "utf-8")))) {
    if (process.env[key] === undefined) process.env[key] = value;
  }
}

if (!process.env.MTAP_SKIP_DOTENV) loadDotEnv();
