import { mkdirSync, renameSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import type { Bus } from "../common/bus/Bus.js";
import { EVENTS } from "../common/bus/events.js";
import type { BusMessage } from "../common/bus/types.js";
import { stateRoot } from "../common/config.js";

// The E*TRADE watchlist feed adapter (schemas/intraday_feed.md). It listens to the
// raw POLLED_DATA the E*TRADE service already publishes, turns each watchlist poll
// into canonical intraday rows, and publishes them as one batch on INTRADAY_FEED.
// POLLED_DATA itself is untouched (frozen contract), so MBin can keep consuming it.
//
// Columns are looked up by header name, never by position: the MAIN and MidCap
// watchlists have different column sets and orders.

export const SOURCE = "etrade_watchlist";
const HEARTBEAT_EVERY_MS = 5000;

export interface FeedRow {
  symbol: string;
  ts_utc: string;
  source: string;
  kind: "snapshot";
  feed_seq?: number;
  price?: number;
  volume_cum?: number;
  day_open?: number;
  prev_close?: number;
  bid?: number;
  ask?: number;
  bid_size?: number;
  ask_size?: number;
  vol_10d_src?: number;
  source_meta?: Record<string, string>;
}

export interface FeedBatch {
  schema_version: 1;
  source: string;
  kind: "snapshot";
  ts_utc: string;
  feed_seq: number;
  rows: FeedRow[];
}

type NumericField = "price" | "volume_cum" | "day_open" | "prev_close" | "bid" | "ask" | "bid_size" | "ask_size" | "vol_10d_src";

// Header name -> canonical field. The first header present wins for `price`.
const PRICE_HEADERS = ["Last Trade", "Last Price", "Last", "Price"];
const FIELD_HEADERS: Array<[NumericField, string]> = [
  ["volume_cum", "Volume"],
  ["day_open", "Open"],
  ["prev_close", "Previous Close"],
  ["bid", "Bid"],
  ["ask", "Ask"],
  ["bid_size", "Bid Size"],
  ["ask_size", "Ask Size"],
  ["vol_10d_src", "10 Day Volume"],
];
// A 0 here means "not set yet" (e.g. Open before 09:30), never a real value.
const ZERO_IS_NULL = new Set<NumericField>(["price", "day_open", "prev_close", "bid", "ask"]);

/** Lenient number parse: strips , % $ and spaces; "--", "" and junk become undefined. */
export function parseNum(v: unknown): number | undefined {
  if (typeof v === "number") return Number.isFinite(v) ? v : undefined;
  if (typeof v !== "string") return undefined;
  const s = v.replace(/[,%$\s]/g, "");
  if (!s || s === "--" || s === "-") return undefined;
  const n = Number(s);
  return Number.isFinite(n) ? n : undefined;
}

interface WatchListView {
  columnHeaders?: string[];
  columnValues?: Array<{ entryValuesList?: unknown[] }>;
  entryDetails?: Array<{ entryValuesList?: unknown[] }>;
}

/** The watchListView inside a POLLED_DATA `data`, wherever the API nested it. */
function findView(data: any): WatchListView | null {
  return data?.watchListView ?? data?.data?.watchListView ?? null;
}

/** Map one POLLED_DATA watchlist message to canonical rows (pure; exported for tests). */
export function toRows(msg: BusMessage, tsUtc: string, feedSeq: number): FeedRow[] {
  const view = findView(msg.data);
  const headers = view?.columnHeaders;
  const entries = view?.columnValues ?? view?.entryDetails;
  if (!Array.isArray(headers) || !Array.isArray(entries)) return [];

  const at = (name: string): number => headers.indexOf(name);
  const symbolIdx = at("Symbol");
  if (symbolIdx < 0) return [];
  const priceIdx = PRICE_HEADERS.map(at).find((i) => i >= 0) ?? -1;
  const fieldIdx = FIELD_HEADERS.map(([field, header]) => [field, at(header)] as const).filter(([, i]) => i >= 0);
  const meta = { watchlist_id: String(msg.watchlistId ?? ""), watchlist: String(msg.watchlistName ?? "") };

  const rows: FeedRow[] = [];
  for (const entry of entries) {
    const values = entry?.entryValuesList;
    if (!Array.isArray(values)) continue;
    const symbol = typeof values[symbolIdx] === "string" ? (values[symbolIdx] as string).trim() : "";
    if (!symbol) continue;
    const row: FeedRow = { symbol, ts_utc: tsUtc, source: SOURCE, kind: "snapshot", feed_seq: feedSeq, source_meta: meta };
    const set = (field: NumericField, idx: number): void => {
      const n = parseNum(values[idx]);
      if (n === undefined || (n === 0 && ZERO_IS_NULL.has(field))) return;
      row[field] = n;
    };
    if (priceIdx >= 0) set("price", priceIdx);
    for (const [field, idx] of fieldIdx) set(field, idx);
    rows.push(row);
  }
  return rows;
}

interface WatchlistStats {
  polls: number;
  rows: number;
  last_poll_utc: string;
  last_rows: number;
}

export interface EtradeWatchlistFeedOptions {
  heartbeatPath?: string;
  now?: () => number;
}

export class EtradeWatchlistFeed {
  private unsubscribe: (() => void) | null = null;
  private seq = 0;
  private stats: Record<string, WatchlistStats> = {};
  private lastHeartbeat = 0;
  private readonly heartbeatPath: string;
  private readonly now: () => number;
  private readonly startedAt: string;

  constructor(private readonly bus: Bus, options: EtradeWatchlistFeedOptions = {}) {
    this.heartbeatPath = options.heartbeatPath ?? join(stateRoot(), "etrade", "collector.heartbeat");
    this.now = options.now ?? Date.now;
    this.startedAt = new Date(this.now()).toISOString();
  }

  start(): this {
    if (this.unsubscribe) return this;
    this.unsubscribe = this.bus.subscribe(EVENTS.POLLED_DATA, (msg) => this.onPolled(msg), {
      label: "intraday-feed:etrade",
    });
    return this;
  }

  stop(): void {
    this.unsubscribe?.();
    this.unsubscribe = null;
    this.writeHeartbeat(true);
  }

  private onPolled(msg: BusMessage): void {
    if (msg?.type !== "watchlist") return;
    // _ts is the scraper's publish time, i.e. when the poll returned.
    const tsMs = typeof msg._ts === "number" ? msg._ts : this.now();
    const tsUtc = new Date(tsMs).toISOString();
    const feedSeq = this.seq++;
    const rows = toRows(msg, tsUtc, feedSeq);
    if (rows.length === 0) return;

    const batch: FeedBatch = { schema_version: 1, source: SOURCE, kind: "snapshot", ts_utc: tsUtc, feed_seq: feedSeq, rows };
    // Re-key on the POLLED_DATA key so a redelivered poll maps to the same batch key.
    const key = msg._key ? `intraday:${msg._key}` : undefined;
    this.bus.publish(EVENTS.INTRADAY_FEED, batch as unknown as BusMessage, { idempotencyKey: key });

    const name = String(msg.watchlistName ?? msg.watchlistId ?? "unknown");
    const s = (this.stats[name] ??= { polls: 0, rows: 0, last_poll_utc: tsUtc, last_rows: 0 });
    s.polls += 1;
    s.rows += rows.length;
    s.last_poll_utc = tsUtc;
    s.last_rows = rows.length;
    this.writeHeartbeat(false);
  }

  /** Atomic JSON heartbeat: the schedule's status/shutdown summary and `storage check` read it. */
  private writeHeartbeat(force: boolean): void {
    const now = this.now();
    if (!force && now - this.lastHeartbeat < HEARTBEAT_EVERY_MS) return;
    this.lastHeartbeat = now;
    const body = {
      updated_at: new Date(now).toISOString(),
      started_at: this.startedAt,
      pid: process.pid,
      source: SOURCE,
      watchlists: this.stats,
    };
    try {
      mkdirSync(dirname(this.heartbeatPath), { recursive: true });
      const tmp = `${this.heartbeatPath}.tmp`;
      writeFileSync(tmp, JSON.stringify(body, null, 2));
      renameSync(tmp, this.heartbeatPath);
    } catch (err) {
      console.error("[EtradeWatchlistFeed] heartbeat write failed:", (err as Error).message);
    }
  }
}
