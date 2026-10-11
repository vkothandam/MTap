# intraday_feed: the canonical intraday record (v1)

Every intraday feed is turned into these rows before anything else sees it. Storage, bars,
the ML panel and live features read only this shape, so a feed can be swapped (E*TRADE
watchlist now, Massive WebSocket later) without changing anything downstream.
Machine-checkable form: [intraday_feed.schema.json](intraday_feed.schema.json), checked by a
conformance test in each stack.

## Transport

- Redis stream `mtap:evt:INTRADAY_FEED`. Each entry has one field, `data`, holding the JSON
  batch envelope, the same wire format as every other MTap/MBin bus event (`RedisBus.ts`).
- One entry per poll (E*TRADE: one watchlist, every 2 s or 20 s) or per second (Massive):
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

| field | E*TRADE watchlist (`etrade_watchlist`) | Massive WS (`massive_ws`, later) |
| --- | --- | --- |
| `symbol` | Symbol | `sym` |
| `ts_utc` | poll time | `s` (interval start) |
| `kind` | `snapshot` | `agg_1s` (channel `A`) / `agg_1m` (`AM`) |
| `price` | Last Trade (MAIN) / Last Price (MidCap) | `c` |
| `open_i`, `high_i`, `low_i` | null | `o`, `h`, `l` |
| `volume_cum` | Volume | `av` |
| `volume_i` | null (bars take the change in `volume_cum`) | `v` |
| `vwap_session` | null (recomputed in bars) | `a` |
| `vwap_i` | null | `vw` |
| `day_open` | Open (null while it reads 0, before the open) | `op` |
| `prev_close` | Previous Close (MAIN only) | REST snapshot at startup |
| `trades_i` | null | `z` if present |
| `bid`, `ask`, `bid_size`, `ask_size` | Bid, Ask, Bid Size, Ask Size (MAIN only) | null unless `Q` is subscribed |
| `vol_10d_src` | 10 Day Volume (MAIN only) | null |
| `source_meta` | `{watchlist_id, watchlist}` | `{}` |

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
