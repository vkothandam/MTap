import { createClient } from "redis";
import type { Transport, Handler, SubscribeOptions, BusMessage } from "./types.js";

// Cross-process transport over Redis Streams + consumer groups. This is the
// TypeScript counterpart of MBin's core/bus/RedisBus.js and MUST stay wire-
// compatible with it: one stream per event (`mtap:evt:<EVENT>`), the JSON
// envelope carried in a `data` field, one consumer group per subscriber label
// (each logical consumer sees every message once), and a broadcast `__reply__`
// channel for request/reply. The public surface is synchronous to satisfy the
// Transport contract; the async redis client is driven behind `this.ready`.
//
// Delivery is live-tail by default: a fresh consumer group is pointed at the
// stream head so restarts don't replay a backlog. The reply channel and any
// `replay: true` subscriber instead resume from their last ack (durable).

type Client = ReturnType<typeof createClient>;

// The subset of an XREADGROUP reply we consume. Declared locally so the loop
// stays readable regardless of redis's generic reply typing.
type StreamEntry = { id: string; message: Record<string, string> };
type StreamRead = Array<{ name: string; messages: StreamEntry[] }> | null;

const STREAM_PREFIX = "mtap:evt:";
const REPLY_EVENT = "__reply__";
const DEFAULT_MAXLEN = 5000;
const READ_COUNT = 50;
const BLOCK_MS = 5000;
const RECONNECT_BACKOFF_MS = 500;

const streamKey = (event: string): string => `${STREAM_PREFIX}${event}`;
const sleep = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));

interface SubState {
  stopped: boolean;
  conn: Client | null;
}

interface LoopOpts {
  key: string;
  group: string;
  consumer: string;
  handler: Handler;
  broadcast: boolean;
  replay: boolean;
  event: string;
}

export class RedisBus implements Transport {
  private url: string;
  private maxLen: number;
  private instanceId: string;
  private stopped = false;
  private subCounter = 0;
  private subscriptions = new Set<SubState>();
  private pub: Client;
  private ready: Promise<unknown>;

  constructor(options: { url?: string; maxLen?: number } = {}) {
    this.url = options.url ?? process.env.REDIS_URL ?? "redis://127.0.0.1:6379";
    this.maxLen = options.maxLen ?? DEFAULT_MAXLEN;
    // Unique per process so broadcast groups and consumer names never collide.
    this.instanceId = `${process.pid}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
    this.pub = createClient({ url: this.url });
    this.pub.on("error", (err: Error) => console.error("[RedisBus] publisher error:", err.message));
    this.ready = this.pub.connect().catch((err: Error) => {
      console.error("[RedisBus] initial connect failed:", err.message);
      throw err;
    });
  }

  publish(event: string, payload: BusMessage): void {
    const key = streamKey(event);
    this.ready
      .then(() =>
        this.pub.xAdd(
          key,
          "*",
          { data: JSON.stringify(payload) },
          { TRIM: { strategy: "MAXLEN", strategyModifier: "~", threshold: this.maxLen } },
        ),
      )
      .catch((err: Error) => console.error(`[RedisBus] publish ${event} failed:`, err.message));
  }

  subscribe(event: string, handler: Handler, options: SubscribeOptions = {}): () => void {
    const label = options.label ?? "anonymous";
    // The reply channel fans out to every instance; a broadcast subscriber gets
    // its own per-instance group so it isn't load-balanced with peers.
    const broadcast = event === REPLY_EVENT || options.broadcast === true;
    const group = broadcast ? `${label}:${this.instanceId}` : label;
    const key = streamKey(event);
    const consumer = `${this.instanceId}:${this.subCounter++}`;

    const sub: SubState = { stopped: false, conn: null };
    this.subscriptions.add(sub);

    this.runLoop(sub, { key, group, consumer, handler, broadcast, replay: options.replay === true, event }).catch(
      (err: Error) => console.error(`[RedisBus] subscribe ${event} loop crashed:`, err.message),
    );

    return () => {
      sub.stopped = true;
      this.subscriptions.delete(sub);
    };
  }

  private async runLoop(sub: SubState, opts: LoopOpts): Promise<void> {
    const { key, group, consumer, handler, broadcast, replay, event } = opts;
    await this.ready;
    // Blocking XREADGROUP needs a dedicated connection so it never stalls publishes.
    const conn = this.pub.duplicate();
    conn.on("error", (err: Error) => console.error(`[RedisBus] reader error (${event}):`, err.message));
    await conn.connect();
    sub.conn = conn;

    await this.ensureGroup(conn, key, group, { broadcast, replay });

    while (!sub.stopped && !this.stopped) {
      let res: StreamRead;
      try {
        res = (await conn.xReadGroup(
          group,
          consumer,
          { key, id: ">" },
          { COUNT: READ_COUNT, BLOCK: BLOCK_MS },
        )) as unknown as StreamRead;
      } catch (err) {
        if (sub.stopped || this.stopped) break;
        console.error(`[RedisBus] read ${event} failed:`, (err as Error).message);
        await sleep(RECONNECT_BACKOFF_MS);
        continue;
      }
      if (!res) continue;

      for (const stream of res) {
        for (const entry of stream.messages) {
          try {
            handler(JSON.parse(entry.message.data));
          } catch (err) {
            console.error(`[RedisBus] handler for ${event} threw:`, (err as Error).message);
          } finally {
            conn.xAck(key, group, entry.id).catch((ackErr: Error) => {
              if (!sub.stopped && !this.stopped) {
                console.error(`[RedisBus] ack ${event} failed:`, ackErr.message);
              }
            });
          }
        }
      }
    }

    await conn.quit().catch(() => {});
  }

  private async ensureGroup(
    conn: Client,
    key: string,
    group: string,
    { broadcast, replay }: { broadcast: boolean; replay: boolean },
  ): Promise<void> {
    // Broadcast (reply) groups start at 0 so a reply published during the
    // requester's setup gap isn't missed — replies are correlationId-filtered
    // and low-volume, so a tiny one-time replay is harmless. Streaming groups
    // start at the tail ($).
    const startId = broadcast ? "0" : "$";
    try {
      await conn.xGroupCreate(key, group, startId, { MKSTREAM: true });
    } catch (err) {
      if (!String((err as Error).message).includes("BUSYGROUP")) throw err;
    }
    // For a pre-existing streaming group (no replay), skip whatever backlog
    // accrued while we were down and resume at the live tail.
    if (!broadcast && !replay) {
      await conn
        .xGroupSetId(key, group, "$")
        .catch((err: Error) => console.error(`[RedisBus] setId ${key}/${group} failed:`, err.message));
    }
  }

  async close(): Promise<void> {
    this.stopped = true;
    const closing: Promise<void>[] = [];
    for (const sub of this.subscriptions) {
      sub.stopped = true;
      if (sub.conn) closing.push(sub.conn.disconnect().catch(() => {}));
    }
    await Promise.all(closing);
    await this.pub.quit().catch(() => {});
  }
}
