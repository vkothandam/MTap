import type { Transport, Handler, SubscribeOptions, BusMessage } from "./types.js";

interface Sub {
  handler: Handler;
  priority: number;
  label: string;
}

/**
 * In-process transport for local runs and tests. Synchronous fan-out to
 * subscribers, highest priority first — mirrors MBin's EventBroker closely
 * enough for single-process use. RedisBus is the real cross-process transport.
 */
export class InProcBus implements Transport {
  private subs = new Map<string, Sub[]>();

  publish(event: string, payload: BusMessage): void {
    const list = [...(this.subs.get(event) ?? [])].sort((a, b) => b.priority - a.priority);
    for (const sub of list) {
      try {
        sub.handler(payload);
      } catch (err) {
        console.error(`[InProcBus] handler ${sub.label} for ${event} threw:`, err);
      }
    }
  }

  subscribe(event: string, handler: Handler, options: SubscribeOptions = {}): () => void {
    const sub: Sub = {
      handler,
      priority: options.priority ?? 0,
      label: options.label ?? "anonymous",
    };
    const list = this.subs.get(event) ?? [];
    list.push(sub);
    this.subs.set(event, list);

    return () => {
      const arr = this.subs.get(event);
      if (!arr) return;
      const i = arr.indexOf(sub);
      if (i >= 0) arr.splice(i, 1);
    };
  }
}
