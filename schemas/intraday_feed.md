# intraday_feed: the canonical intraday record (v1)

Every intraday feed is turned into these rows before anything else sees it. Storage, bars,
the ML panel and live features read only this shape, so a feed can be swapped without changing
anything downstream. The Massive per-second bars are too many for this stream and have their
own table and pipeline: see [agg_1s](#agg_1s-massive-per-second-bars-v1) below.
Machine-checkable form: [intraday_feed.schema.json](intraday_feed.schema.json), checked by a
conformance test in each stack.

## Transport

- Redis stream `mtap:evt:INTRADAY_FEED`. Each entry has one field, `data`, holding the JSON
  batch envelope, the same wire format as every other MTap/MBin bus event (`RedisBus.ts`).
- One entry per poll (E*TRADE: one watchlist, every 2 s or 20 s):
  ```jsonc
  { "schema_version": 1, "source": "etrade_watchlist", "kind": "snapshot",
    "ts_utc": "2026-10-12T14:31:02.118Z", "feed_seq": 812,
    "rows": [ { "symbol": "AAPL", "ts_utc": "…", "source": "…", "kind": "snapshot", "price": 254.1, … } ],
    "_key": "intraday:…", "_ts": 1791988262118 }
  ```
- Fields that are null are left out of a row to keep the stream small; readers treat a
  missing field as null.
- The stream is trimmed by length (about the last hour). Redis has no persistence; the
  durable copy is the storage consumer's parquet segments (`sourcing-py intraday store`).
- Consumer groups: `storage` (durable; resumes from its last ack after a restart) and later
  `live-bars`. Each group gets every entry.

## Field mapping per feed

| field | E*TRADE watchlist (`etrade_watchlist`) |
| --- | --- |
| `symbol` | Symbol |
| `ts_utc` | poll time |
| `kind` | `snapshot` |
| `price` | Last Trade (MAIN) / Last Price (MidCap) |
| `open_i`, `high_i`, `low_i` | null |
| `volume_cum` | Volume |
| `volume_i` | null (bars take the change in `volume_cum`) |
| `vwap_session` | null (recomputed in bars) |
| `vwap_i` | null |
| `day_open` | Open (null while it reads 0, before the open) |
| `prev_close` | Previous Close (MAIN only) |
| `trades_i` | null |
| `bid`, `ask`, `bid_size`, `ask_size` | Bid, Ask, Bid Size, Ask Size (MAIN only) |
| `vol_10d_src` | 10 Day Volume (MAIN only) |
| `source_meta` | `{watchlist_id, watchlist}` |

E*TRADE columns deliberately dropped: MBin's computed VWAP (it resets when the scraper
restarts), Qty, Price Paid, Date Acquired, Change since Added %, Change $/%, Day's Gain %
(all derivable or position-specific).

Values are parsed leniently: thousands separators, `%` and `$` are stripped; `--`, empty and
non-numeric values become null; a price or open of 0 becomes null.

## Storage layout (written by `sourcing-py intraday store`)

- Local, while open: `STATE_ROOT/segments/<source>/date=<D>/h<HH>-<opened>.arrows`, an Arrow
  IPC stream fsynced after every write (readable up to its last complete batch after a
  crash). `D` and `HH` are the New York date and hour of the batch's `ts_utc`; `opened` is
  the UTC time the segment was opened (`20261014T143000Z`).
- Closed (new hour, `idle_close_s` without rows, or shutdown): converted to zstd parquet
  `h<HH>-<opened>.parquet` next to it, with the row count checked.
- On the NAS, once copied and read back with the same row count:
  `DATA_ROOT/intraday_feed/date=<D>/source=<source>/h<HH>-<opened>.parquet`; the local copy is
  then deleted. A restart mid-hour opens a new segment, so one hour can have several files
  and none is ever overwritten.
- Delivery is at least once: a crash after a write but before its ack means the entries are
  read again on restart. Readers dedup on (`stream_id`, `symbol`).
- Parquet columns are the fields above, with `ts_utc` as a UTC timestamp (ms),
  `source_meta` as a JSON string, plus `stream_id` (the Redis entry id) for audit.

## agg_1s: Massive per-second bars (v1)

`sourcing-py massive-ws follow` (launchd `com.mtap.massive`) subscribes to `A.*` on the
Massive WebSocket (`wss://delayed.massive.com/stocks`, 15 min delayed) and writes the bars
itself: no Redis, no storage consumer. One row per symbol per second that traded.

- Local: `STATE_ROOT/segments_massive/massive_ws/date=D/hHH-mMM-<stamp>.arrows`, one per
  5 minutes of bar time, closed to zstd parquet sorted by (`symbol`, `ts_utc`).
- NAS: `DATA_ROOT/agg_1s/date=D/source=massive_ws/hHH-mMM-<stamp>.parquet`; the local copy is
  deleted once the NAS row count matches. `D` is the bar's NY date.
- After the window closes, `massive-ws compact` (the evening sleep job) combines each day's
  files into `day-D.parquet`: same columns and values, duplicates (`symbol`, `ts_utc`)
  dropped, sorted, zstd 19, about 40% smaller. Read every `*.parquet` in the folder: a day
  can hold `day-D.parquet` plus 5-minute files uploaded after it ran.
- Not under `intraday_feed/`, so `export-tft` and `intraday summary` do not read it.
- Parquet metadata: `agg_1s_schema_version=1`, `source=massive_ws`.

| column | type | Massive field |
| --- | --- | --- |
| `symbol` | string | `sym` |
| `ts_utc` | timestamp(ms, UTC) | `s` (bar start; the bar ends at `e`) |
| `price` | float64 | `c` (close of the second) |
| `open_i` / `high_i` / `low_i` | float64 | `o` / `h` / `l` |
| `volume_i` | int64 | `v` |
| `volume_cum` | int64 | `av` (day's volume so far) |
| `vwap_i` | float64 | `vw` (the second's VWAP) |
| `vwap_session` | float64 | `a` (day's VWAP so far) |
| `day_open` | float64 | `op` (null before the open) |
| `avg_trade_size` | int64 | `z` (average trade size, not a trade count) |
| `otc` | bool | `otc` (absent means false) |

A disconnect loses the bars it spans (the feed has no replay); the collector logs each one in
its heartbeat (`disconnects`, `last_error`, `dropped_frames`), and `compact` reports the
longest regular-session stretch with no bar at all (normally 1 s).

Volumes are whole shares (`v`, `av`); the fractional-share volumes `dv`, `dav` are not kept.
Massive's REST minute bars use the fractional volume, so the per-second volumes summed to a
minute read a little lower (median 0.1% for AAPL, about 3% for a retail-heavy small cap);
prices match exactly. `av` also counts trades that never update a bar (odd lots and other
conditions), so it rises faster than the sum of `v`.

