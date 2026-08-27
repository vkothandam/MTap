// Output writer: injects provenance, applies naming conventions, emits jsonl/parquet.
// Implements schemas/conventions.md so no source reinvents it.
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { outRoot, type OutputFormat } from "./config.js";

export type Record_ = Record<string, unknown>;

function runId(): string {
  // Timestamp-based, UTC, filename-safe: YYYYMMDDTHHMMSSZ
  return new Date().toISOString().replace(/[-:]/g, "").replace(/\.\d+Z$/, "Z");
}

function utcDate(): string {
  return new Date().toISOString().slice(0, 10); // YYYY-MM-DD
}

function withProvenance(records: Record_[], source: string, id: string): Record_[] {
  const ingestedAt = new Date().toISOString();
  return records.map((r) => ({
    ...r,
    _source: source,
    _ingested_at: ingestedAt,
    _run_id: id,
  }));
}

function outputPath(source: string, fmt: OutputFormat, id: string, dt: string): string {
  const ext = fmt === "jsonl" ? "jsonl" : "parquet";
  return join(outRoot(), source, `dt=${dt}`, `${source}-${id}.${ext}`);
}

export function write(
  records: Record_[],
  opts: { source: string; format: OutputFormat },
): string {
  const id = runId();
  const dt = utcDate();
  const enriched = withProvenance(records, opts.source, id);
  const path = outputPath(opts.source, opts.format, id, dt);
  mkdirSync(dirname(path), { recursive: true });

  if (opts.format === "jsonl") {
    writeJsonl(enriched, path);
  } else {
    writeParquet(enriched, path);
  }
  return path;
}

function writeJsonl(records: Record_[], path: string): void {
  const body = records.map((r) => JSON.stringify(r)).join("\n") + "\n";
  writeFileSync(path, body, "utf-8");
}

function writeParquet(_records: Record_[], _path: string): void {
  // TODO: install parquetjs / parquet-wasm / apache-arrow and implement.
  // Kept as an explicit failure so parquet sources can't silently emit nothing.
  throw new Error(
    "parquet output not yet implemented for the Node stack — add a parquet lib and fill in writeParquet()",
  );
}
