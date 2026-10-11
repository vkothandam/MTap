// Load the shared source registry and resolve repo paths.
import "./dotenv.js";
import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, isAbsolute, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
// node/src/common/config.ts -> repo root is four parents up.
export const REPO_ROOT = resolve(__dirname, "..", "..", "..");
const REGISTRY = join(REPO_ROOT, "shared", "config", "sources.json");

export type OutputFormat = "jsonl" | "parquet";

export interface SourceConfig {
  name: string;
  runtime: string;
  format: OutputFormat;
  schemaPath: string;
  description: string;
}

function rootFromEnv(envVar: string, fallback: string): string {
  const raw = process.env[envVar];
  if (!raw) return fallback;
  const expanded = raw.startsWith("~/") ? join(homedir(), raw.slice(2)) : raw;
  return isAbsolute(expanded) ? expanded : join(REPO_ROOT, expanded);
}

// The two storage roots, read from the same env vars (via .env) as Python's
// common/config.py. Node only reads the env vars, not settings.toml [storage].
/** Large write-once data (NAS). MTAP_DATA_ROOT, default the repo root. */
export function dataRoot(): string {
  return rootFromEnv("MTAP_DATA_ROOT", REPO_ROOT);
}

/** Small local state (sessions, logs, heartbeat). MTAP_STATE_ROOT, default ./state. */
export function stateRoot(): string {
  return rootFromEnv("MTAP_STATE_ROOT", join(REPO_ROOT, "state"));
}

/** OUT_ROOT, default <data root>/out; a relative OUT_ROOT resolves under the data root (as in Python). */
export function outRoot(): string {
  const raw = process.env.OUT_ROOT;
  if (!raw) return join(dataRoot(), "out");
  return isAbsolute(raw) ? raw : join(dataRoot(), raw);
}

/** Intraday feeds to run (env INTRADAY_FEEDS, comma list; default the E*TRADE watchlists). */
export function intradayFeeds(): string[] {
  const raw = process.env.INTRADAY_FEEDS ?? "etrade_watchlist";
  return raw.split(",").map((s) => s.trim()).filter(Boolean);
}

export function loadSource(name: string): SourceConfig {
  const registry = JSON.parse(readFileSync(REGISTRY, "utf-8"));
  const sources = registry.sources ?? {};
  const entry = sources[name];
  if (!entry) {
    const available = Object.keys(sources).sort().join(", ") || "(none)";
    throw new Error(`Unknown source '${name}'. Registered: ${available}`);
  }
  return {
    name,
    runtime: entry.runtime,
    format: entry.format,
    schemaPath: join(REPO_ROOT, entry.schema),
    description: entry.description ?? "",
  };
}
