import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { loadDotEnv, parseDotEnv } from "./dotenv.js";
import { REPO_ROOT, dataRoot, outRoot, stateRoot } from "./config.js";

test("parseDotEnv: same rules as python common/config.py and scripts/schedule/lib.sh", () => {
  const text = "# c\nA=1\nexport B=\"two words\"\nC='x=y'\n\nbad line\nD=  spaced  \nE=v # trailing comment\n";
  assert.deepEqual(parseDotEnv(text), { A: "1", B: "two words", C: "x=y", D: "spaced", E: "v" });
});

test("loadDotEnv never overrides a variable that is already set", () => {
  const file = join(mkdtempSync(join(tmpdir(), "mtap-dotenv-")), ".env");
  writeFileSync(file, "MTAP_T_SET=from-file\nMTAP_T_NEW=from-file\n");
  process.env.MTAP_T_SET = "from-env";
  delete process.env.MTAP_T_NEW;
  try {
    loadDotEnv(file);
    assert.equal(process.env.MTAP_T_SET, "from-env");
    assert.equal(process.env.MTAP_T_NEW, "from-file");
  } finally {
    delete process.env.MTAP_T_SET;
    delete process.env.MTAP_T_NEW;
  }
});

test("storage roots: defaults, absolute, relative and ~ paths", () => {
  const saved = { d: process.env.MTAP_DATA_ROOT, s: process.env.MTAP_STATE_ROOT, o: process.env.OUT_ROOT };
  try {
    delete process.env.MTAP_DATA_ROOT;
    delete process.env.MTAP_STATE_ROOT;
    delete process.env.OUT_ROOT;
    assert.equal(dataRoot(), REPO_ROOT);
    assert.equal(stateRoot(), join(REPO_ROOT, "state"));
    assert.equal(outRoot(), join(REPO_ROOT, "out"));
    process.env.MTAP_DATA_ROOT = "/Volumes/mtap/mtap";
    process.env.MTAP_STATE_ROOT = "~/mtap-state";
    process.env.OUT_ROOT = "alt-out";
    assert.equal(outRoot(), "/Volumes/mtap/mtap/alt-out");
    assert.ok(stateRoot().endsWith("/mtap-state") && !stateRoot().includes("~"));
    process.env.MTAP_DATA_ROOT = "nas";
    assert.equal(dataRoot(), join(REPO_ROOT, "nas"));
  } finally {
    for (const [k, v] of [["MTAP_DATA_ROOT", saved.d], ["MTAP_STATE_ROOT", saved.s], ["OUT_ROOT", saved.o]] as const) {
      if (v === undefined) delete process.env[k];
      else process.env[k] = v;
    }
  }
});
