// Shared transport contract for the Bus. A transport moves messages between
// publishers and subscribers; the Bus (Bus.ts) layers idempotency dedup and
// request/reply on top. Two transports implement this: InProcBus (local/tests)
// and RedisBus (cross-process, interoperates with MBin's core/bus).
export type BusMessage = Record<string, any>;

export type Handler = (msg: BusMessage) => void;

export interface SubscribeOptions {
  /** Consumer group name (Redis) / subscriber label; distinct labels each receive every message. */
  label?: string;
  /** Higher runs first (in-process transport only). */
  priority?: number;
  /** Redis: durable consumer that resumes from its last ack instead of the live tail. */
  replay?: boolean;
  /** Redis: fan out to every instance rather than load-balance (used for the reply channel). */
  broadcast?: boolean;
}

export interface Transport {
  publish(event: string, payload: BusMessage): void;
  subscribe(event: string, handler: Handler, options?: SubscribeOptions): () => void;
}
