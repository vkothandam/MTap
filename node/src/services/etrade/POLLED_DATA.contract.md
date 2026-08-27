# POLLED_DATA — inter-repo contract (MTap → MBin)

**Version: 1 (frozen).** Direction: MTap (producer) → MBin (consumer), over the
`Bus` (`BUS_TRANSPORT=redis` in production, `inproc` in tests). Event name:
`POLLED_DATA` (must match MBin `core/events.js` exactly).

This is the payload MTap's E*TRADE scraper emits and MBin's `DataTransformer`
consumes. It is frozen: any breaking change bumps the version here, and only then
may the wire shape change. Machine-checkable form: `POLLED_DATA.schema.json`
(co-located).

## Envelope

Every message is a JSON object with a `type` discriminator. The `Bus` adds two
fields at publish time (see `common/bus/Bus.ts`):

| Field  | Type    | Added by | Meaning |
|--------|---------|----------|---------|
| `_key` | string  | Bus      | Idempotency key (see below). Present whenever the publish carried an `idempotencyKey`. |
| `_ts`  | integer | Bus      | Publish time, ms since epoch. |

Consumers **must** ignore unknown fields (forward-compatibility) and **must**
dedup on `_key` (delivery is at-least-once).

## Variant: `watchlist`

Emitted by `ETPageScraper.getWatchlist()`, once per successful poll of a watchlist.

```jsonc
{
  "type": "watchlist",
  "data": { /* opaque: raw E*TRADE watchlist API response, see below */ },
  "watchlistId": "215565674306",
  "watchlistName": "Experimental",
  "csvPrefix": "",
  "_key": "poll:…",
  "_ts": 1730000000000
}
```

- `data` is an **opaque passthrough** of the upstream E*TRADE response — MTap does
  not reshape it (transformation stays in MBin's `DataTransformer`). Informative,
  not normative: MBin currently reads `data.data.watchListView.{entryDetails,
  columnValues,columnHeaders}`. MTap normalizes so `entryDetails` and
  `columnValues` are both present (one is copied from the other when only one is
  returned), and the paginated-fallback path returns the same `watchListView`
  shape.
- `watchlistId` / `watchlistName` / `csvPrefix` identify which watchlist and drive
  MBin's per-watchlist CSV naming.

## Variant: `marketMovers`

Emitted by `ETPageScraper.getMarketMovers{Up,Down}()`.

```jsonc
{
  "type": "marketMovers",
  "direction": "up",          // "up" | "down"
  "data": { /* opaque: raw markitdigital movers response */ },
  "_key": "poll:…",
  "_ts": 1730000000000
}
```

> Market-movers polling is currently disabled in `EtradeService` (parity with
> MBin), but the payload shape is part of the frozen contract.

## Idempotency key

`_key` is computed by `keygen.ts#polledDataKey`:

```
poll:<sha1( runId | seq | type | scope ).hex[:24]>
```

- `runId` — per-process id **plus a scraper-generation suffix** (`"<pid>-<ts>.<gen>"`).
  The generation is bumped every time the scrapers are rebuilt (login, re-auth,
  recovery), so a scraper's `seq` resetting to 0 can never collide with keys from
  an earlier generation.
- `seq` — monotonic per `ETPageScraper` instance, incremented on every publish.
- `type` — `"watchlist"` or `"marketMovers"`.
- `scope` — `watchlistId` for `watchlist`, `direction` for `marketMovers`.

Property: a **redelivery of the same publish** carries the same `_key` (collapses
to a no-op at the consumer); **distinct ticks** get distinct keys (all flow
through). MBin dedups with the same `core/idempotency/keygen.js` scheme and a
few-minute TTL.

## Related events (not POLLED_DATA)

- `SESSION_EXPIRED` / `SESSION_REFRESHED` — session lifecycle (internal to MTap;
  optional health mirror to MBin).
- `AUTH_OTP_REQUIRED` — optional read-only "login required" banner signal to MBin;
  the OTP flow itself is entirely in-process in MTap. Carries `{ timestamp, otpUrl }`.
