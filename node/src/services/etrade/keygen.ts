import { createHash } from "node:crypto";

/**
 * Deterministic idempotency keys for POLLED_DATA. Same inputs → same key, so a
 * redelivery of the *same* publish (Redis is at-least-once) collapses to a
 * no-op at the subscriber, while distinct ticks get distinct keys and all flow
 * through. Mirrors MBin's core/idempotency/keygen.js.
 *
 * A per-process `runId` is woven in so that after a restart the sequence
 * counter resetting to 0 cannot collide with keys a still-running MBin already
 * saw (its dedup cache has a few-minute TTL).
 */
function hash(parts: Array<string | number | undefined>): string {
  const h = createHash("sha1");
  h.update(parts.map((p) => String(p ?? "")).join("|"));
  return h.digest("hex").slice(0, 24);
}

export function polledDataKey(parts: {
  runId: string;
  seq: number;
  type: string;
  scope?: string;
}): string {
  return `poll:${hash([parts.runId, parts.seq, parts.type, parts.scope])}`;
}
