// Load the shared source registry and resolve repo paths.
import { readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
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

export function outRoot(): string {
  return process.env.OUT_ROOT ?? join(REPO_ROOT, "out");
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
