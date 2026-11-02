import type { Transport, Handler, SubscribeOptions, BusMessage, TransportPublishOptions } from "./types.js";

const DEFAULT_PREFIX = "mtap:evt:";
const subKey = (event: string, prefix: string = DEFAULT_PREFIX): string => `${prefix}${event}`;

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

  // Subscriptions are keyed by prefix + event, so `mbin:evt:X` and `mtap:evt:X` stay
  // apart in-process as they do on Redis (an absent prefix means the default).
  publish(event: string, payload: BusMessage, options: TransportPublishOptions = {}): void {
    const list = [...(this.subs.get(subKey(event, options.prefix)) ?? [])].sort((a, b) => b.priority - a.priority);
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
    const key = subKey(event, options.prefix);
    const list = this.subs.get(key) ?? [];
    list.push(sub);
    this.subs.set(key, list);

    return () => {
      const arr = this.subs.get(key);
      if (!arr) return;
      const i = arr.indexOf(sub);
      if (i >= 0) arr.splice(i, 1);
    };
  }
}
