import { mkdirSync, renameSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { stateRoot } from "../../../common/config.js";

/** Where the captured bearer token is mirrored for the Python premarket job. */
export function bearerFilePath(): string {
  return join(stateRoot(), "etrade", "bearer.json");
}

/**
 * Write {"token", "captured_at"} atomically (tmp + rename, mode 0600). Never throws:
 * a failed write must not affect the collector. Returns true when written.
 */
export function writeBearerFile(token: string, path: string = bearerFilePath(), now: Date = new Date()): boolean {
  try {
    mkdirSync(dirname(path), { recursive: true });
    const tmp = `${path}.${process.pid}.tmp`;
    writeFileSync(tmp, JSON.stringify({ token, captured_at: now.toISOString() }), { mode: 0o600 });
    renameSync(tmp, path);
    return true;
  } catch (error) {
    console.error("[Auth] could not write the bearer token file:", (error as Error).message);
    return false;
  }
}
