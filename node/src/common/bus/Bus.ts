import type { Transport, Handler, SubscribeOptions, BusMessage } from "./types.js";
import { InProcBus } from "./InProcBus.js";
import { RedisBus } from "./RedisBus.js";

// Faithful TypeScript port of MBin's core/bus/Bus.js. The Bus wraps a Transport
// and adds two things the transport doesn't provide:
//   1. idempotency dedup — a publish may carry an `idempotencyKey`; the same key
//      is delivered to a given subscriber label at most once (defends against
//      at-least-once Redis delivery).
//   2. request/reply — request() publishes with a `_correlationId`; a responder
//      calls reply(), which publishes `{ correlationId, payload }` on the
//      broadcast `__reply__` channel. Field names match MBin exactly so a
//      request from one repo can be answered by the other.

const REPLY_EVENT = "__reply__";
const DEDUP_TTL_MS = 5 * 60 * 1000;

let correlationCounter = 0;

export interface PublishOptions {
  idempotencyKey?: string;
}

export interface RequestOptions {
  timeoutMs?: number;
}

interface ReplyWaiter {
  resolve: (value: any) => void;
}

export class Bus {
  private seen = new Map<string, number>();
  private replyWaiters = new Map<string, ReplyWaiter>();

  constructor(public readonly transport: Transport) {
    this.transport.subscribe(
      REPLY_EVENT,
      (msg) => {
        const waiter = this.replyWaiters.get(msg.correlationId);
        if (waiter) {
          this.replyWaiters.delete(msg.correlationId);
          waiter.resolve(msg.payload);
        }
      },
      { label: "bus:reply" },
    );
  }

  publish(event: string, payload: BusMessage, options: PublishOptions = {}): { deduped: boolean; key?: string } {
    const { idempotencyKey } = options;
    if (idempotencyKey && this.hasSeenRecently(idempotencyKey)) {
      return { deduped: true, key: idempotencyKey };
    }
    if (idempotencyKey) this.markSeen(idempotencyKey);
    this.transport.publish(event, { ...payload, _key: idempotencyKey, _ts: Date.now() });
    return { deduped: false, key: idempotencyKey };
  }

  subscribe(event: string, handler: Handler, options: SubscribeOptions = {}): () => void {
    const label = options.label ?? "anonymous";
    const wrapped: Handler = (msg) => {
      // Per-label dedup: an event carrying `_key` is handled once per subscriber.
      if (msg && msg._key) {
        const consumedKey = `consumed:${label}:${msg._key}`;
        if (this.hasSeenRecently(consumedKey)) return;
        this.markSeen(consumedKey);
      }
      handler(msg);
    };
    return this.transport.subscribe(event, wrapped, options);
  }

  request(event: string, payload: BusMessage, { timeoutMs = 2000 }: RequestOptions = {}): Promise<any> {
    const correlationId = `${event}:${Date.now()}:${correlationCounter++}`;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.replyWaiters.delete(correlationId);
        reject(new Error(`bus.request(${event}) timed out after ${timeoutMs}ms`));
      }, timeoutMs);
      this.replyWaiters.set(correlationId, {
        resolve: (value) => {
          clearTimeout(timer);
          resolve(value);
        },
      });
      this.transport.publish(event, { ...payload, _correlationId: correlationId });
    });
  }

  reply(correlationId: string, payload: BusMessage): void {
    this.transport.publish(REPLY_EVENT, { correlationId, payload });
  }

  private hasSeenRecently(key: string): boolean {
    this.evictExpired();
    return this.seen.has(key);
  }

  private markSeen(key: string): void {
    this.seen.set(key, Date.now() + DEDUP_TTL_MS);
  }

  private evictExpired(): void {
    const now = Date.now();
    for (const [key, expiry] of this.seen) {
      if (expiry <= now) this.seen.delete(key);
    }
  }
}

function createTransport(): Transport {
  if (process.env.BUS_TRANSPORT === "redis") {
    console.log(`[Bus] transport=redis (${process.env.REDIS_URL ?? "redis://127.0.0.1:6379"})`);
    return new RedisBus();
  }
  return new InProcBus();
}

let shared: Bus | null = null;

/** Process-wide Bus; transport selected by BUS_TRANSPORT (inproc default / redis). */
export function getBus(): Bus {
  shared ??= new Bus(createTransport());
  return shared;
}

/** Override the shared Bus (tests / explicit wiring). */
export function setBus(bus: Bus): void {
  shared = bus;
}
