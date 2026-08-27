import { test } from "node:test";
import assert from "node:assert/strict";
import { Bus } from "./Bus.js";
import { InProcBus } from "./InProcBus.js";
import { RedisBus } from "./RedisBus.js";

const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms));

test("InProcBus: publish/subscribe delivers payloads", () => {
  const bus = new Bus(new InProcBus());
  const got: any[] = [];
  bus.subscribe("POLLED_DATA", (m) => got.push(m), { label: "c1" });
  bus.publish("POLLED_DATA", { symbol: "AAPL" });
  assert.equal(got.length, 1);
  assert.equal(got[0].symbol, "AAPL");
});

test("InProcBus: request/reply round-trips in-process", async () => {
  const bus = new Bus(new InProcBus());
  bus.subscribe("WATCHLIST_CMD", (m) => bus.reply(m._correlationId, { ok: true, echo: m.action }), {
    label: "wl",
  });
  const res = await bus.request("WATCHLIST_CMD", { action: "add" }, { timeoutMs: 1000 });
  assert.deepEqual(res, { ok: true, echo: "add" });
});

test("Bus: subscribe-side idempotency dedups by _key per label", () => {
  const transport = new InProcBus();
  const bus = new Bus(transport);
  let calls = 0;
  bus.subscribe("POLLED_DATA", () => calls++, { label: "dedup" });
  // Simulate at-least-once delivery: the same _key arrives twice from the transport.
  transport.publish("POLLED_DATA", { _key: "k1", v: 1 });
  transport.publish("POLLED_DATA", { _key: "k1", v: 1 });
  assert.equal(calls, 1);
});

// Cross-instance test over a real Redis (two Bus instances = two processes on
// the wire). Skipped unless REDIS_URL is set. Passing here is the cross-repo
// interop guarantee: MTap's RedisBus uses the same wire constants as MBin's.
test(
  "RedisBus: cross-instance delivery + request/reply + dedup",
  { skip: !process.env.REDIS_URL },
  async () => {
    const a = new RedisBus();
    const b = new RedisBus();
    const busA = new Bus(a);
    const busB = new Bus(b);
    try {
      const got: any[] = [];
      busB.subscribe("POLLED_DATA", (m) => got.push(m), { label: "consumerB" });
      busB.subscribe("WATCHLIST_CMD", (m) => busB.reply(m._correlationId, { ok: true, echo: m.action }), {
        label: "wl",
      });
      await sleep(600); // let consumer groups get created

      busA.publish("POLLED_DATA", { symbol: "AAPL" }, { idempotencyKey: "k1" });
      // A raw duplicate carrying the same _key must be dropped subscribe-side on B.
      a.publish("POLLED_DATA", { _key: "k1", symbol: "AAPL", _ts: Date.now() });
      await sleep(700);
      assert.equal(got.length, 1, "exactly one delivery after subscribe-side dedup");
      assert.equal(got[0].symbol, "AAPL");

      const res = await busA.request("WATCHLIST_CMD", { action: "add" }, { timeoutMs: 3000 });
      assert.deepEqual(res, { ok: true, echo: "add" });
    } finally {
      await a.close();
      await b.close();
    }
  },
);
