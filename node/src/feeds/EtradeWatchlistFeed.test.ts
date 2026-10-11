import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Bus } from "../common/bus/Bus.js";
import { InProcBus } from "../common/bus/InProcBus.js";
import { EVENTS } from "../common/bus/events.js";
import { REPO_ROOT } from "../common/config.js";
import { loadSchema, validateRecords } from "../common/schema.js";
import { EtradeWatchlistFeed, parseNum, toRows } from "./EtradeWatchlistFeed.js";

const fixture = (name: string): any => JSON.parse(readFileSync(new URL(`./fixtures/${name}`, import.meta.url), "utf-8"));
const SCHEMA = loadSchema(join(REPO_ROOT, "schemas", "intraday_feed.schema.json"));
// The cross-stack golden batch: Python's storage consumer tests read the same file.
const GOLDEN = JSON.parse(readFileSync(join(REPO_ROOT, "schemas", "fixtures", "intraday_feed_batch.etrade.json"), "utf-8"));

function harness() {
  const bus = new Bus(new InProcBus());
  const dir = mkdtempSync(join(tmpdir(), "mtap-feed-"));
  const heartbeatPath = join(dir, "collector.heartbeat");
  const feed = new EtradeWatchlistFeed(bus, { heartbeatPath }).start();
  const batches: any[] = [];
  bus.subscribe(EVENTS.INTRADAY_FEED, (m) => batches.push(m), { label: "test" });
  return { bus, feed, batches, heartbeatPath };
}

test("parseNum: strips separators, rejects placeholders", () => {
  assert.equal(parseNum("1,581,211"), 1581211);
  assert.equal(parseNum("-0.3%"), -0.3);
  assert.equal(parseNum("$12.5"), 12.5);
  assert.equal(parseNum("--"), undefined);
  assert.equal(parseNum(""), undefined);
  assert.equal(parseNum("n/a"), undefined);
  assert.equal(parseNum(7), 7);
});

test("toRows: MAIN maps by header name and matches the golden batch", () => {
  const msg = fixture("polled_watchlist_main.json");
  const rows = toRows(msg, new Date(msg._ts).toISOString(), 0);
  assert.deepEqual(rows, GOLDEN.rows);
  validateRecords(rows, SCHEMA);
});

test("toRows: MidCap (different headers, entryDetails only) maps Last Price and Volume", () => {
  const msg = fixture("polled_watchlist_midcap.json");
  const [row, ...rest] = toRows(msg, "2026-10-14T14:31:10.000Z", 3);
  assert.equal(rest.length, 0);
  assert.equal(row.symbol, "AAL");
  assert.equal(row.price, 15.14);
  assert.equal(row.volume_cum, 36630);
  assert.equal(row.bid, undefined); // MidCap has no quotes
  assert.deepEqual(row.source_meta, { watchlist_id: "215584661306", watchlist: "MidCap" });
  validateRecords([row], SCHEMA);
});

test("feed: one POLLED_DATA watchlist poll -> one INTRADAY_FEED batch, keyed off the poll", () => {
  const { bus, feed, batches, heartbeatPath } = harness();
  const msg = fixture("polled_watchlist_main.json");
  const { _key, _ts, ...payload } = msg;
  // Publish through the Bus like ETPageScraper does; the Bus stamps _key/_ts.
  bus.publish(EVENTS.POLLED_DATA, payload, { idempotencyKey: _key });
  bus.publish(EVENTS.POLLED_DATA, { type: "marketMovers", direction: "up", data: {} });
  assert.equal(batches.length, 1);
  const b = batches[0];
  assert.equal(b.schema_version, 1);
  assert.equal(b.source, "etrade_watchlist");
  assert.equal(b.rows.length, 2);
  assert.equal(b._key, `intraday:${_key}`);
  assert.equal(b.rows[0].ts_utc, b.ts_utc);
  feed.stop();
  const hb = JSON.parse(readFileSync(heartbeatPath, "utf-8"));
  assert.equal(hb.watchlists.Experimental.polls, 1);
  assert.equal(hb.watchlists.Experimental.last_rows, 2);
});

test("feed: a redelivered poll (same _key) is dropped", () => {
  const { bus, batches } = harness();
  const transport = bus.transport;
  const msg = fixture("polled_watchlist_main.json");
  transport.publish(EVENTS.POLLED_DATA, msg);
  transport.publish(EVENTS.POLLED_DATA, msg);
  assert.equal(batches.length, 1);
});

test("feed: a payload without a watchListView publishes nothing", () => {
  const { bus, batches } = harness();
  bus.publish(EVENTS.POLLED_DATA, { type: "watchlist", data: { error: "x" }, watchlistId: "1", watchlistName: "W", csvPrefix: "" });
  assert.equal(batches.length, 0);
});
