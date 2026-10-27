import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, readdirSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { writeBearerFile } from "./bearerFile.js";

test("writes token JSON atomically with mode 600", () => {
  const dir = mkdtempSync(join(tmpdir(), "bearer-"));
  const path = join(dir, "etrade", "bearer.json");
  assert.equal(writeBearerFile("Bearer abc", path, new Date("2026-10-27T10:00:00Z")), true);
  assert.deepEqual(JSON.parse(readFileSync(path, "utf8")), {
    token: "Bearer abc",
    captured_at: "2026-10-27T10:00:00.000Z",
  });
  assert.equal(statSync(path).mode & 0o777, 0o600);
  assert.deepEqual(readdirSync(join(dir, "etrade")), ["bearer.json"]);
});

test("a write failure returns false and does not throw", () => {
  const dir = mkdtempSync(join(tmpdir(), "bearer-"));
  const blocker = join(dir, "file");
  writeBearerFile("x", blocker); // makes `file` a regular file
  assert.equal(writeBearerFile("Bearer abc", join(blocker, "sub", "bearer.json")), false);
});
