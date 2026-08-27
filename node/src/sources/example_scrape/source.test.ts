// Schema-conformance test — the Node half of shared contract enforcement.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadSource } from "../../common/config.js";
import { loadSchema, validateRecords } from "../../common/schema.js";
import { ExampleScrapeSource } from "./source.js";

test("example_scrape records conform to schema", async () => {
  const source = new ExampleScrapeSource();
  const records = await source.extract(await source.fetch());
  const schema = loadSchema(loadSource(source.name).schemaPath);
  // Throws with details if any record fails.
  validateRecords(records, schema);
});

test("example_scrape produces records", async () => {
  const source = new ExampleScrapeSource();
  const records = await source.extract(await source.fetch());
  assert.ok(records.length > 0);
  assert.ok(records.every((r) => "id" in r));
});
