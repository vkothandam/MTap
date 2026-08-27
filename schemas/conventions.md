# Output conventions

These rules apply to **every** source, regardless of language. Both the Python and
Node writers implement them; the conformance tests enforce them.

## Formats

- **jsonl** — one JSON object per line, UTF-8, `\n`-terminated. Default for small or
  irregular payloads and for scraped data still being shaped.
- **parquet** — columnar, snappy-compressed. Default for large, stable, tabular data
  consumed by analytics.

A source declares its format in `shared/config/sources.json` (`format` field).

## Schema

Each source has `schemas/<source>.schema.json`, a JSON Schema describing one record
(one jsonl line / one parquet row). Rules:

- Field names are `snake_case`.
- Types map to parquet as: `string→string`, `integer→int64`, `number→double`,
  `boolean→bool`, `object`/`array`→ serialized json string (keep nested data in jsonl).
- `required` lists fields that must be non-null on every record.

## File naming

Output path (flat — one folder per info-type, date in the filename):

```
<out_root>/<dataset>/<dataset>-<YYYY-MM-DD>.<ext>
```

- `<out_root>` comes from config (`[destination].out_root`) or env `OUT_ROOT`; defaults to `./out`.
- `<dataset>` is the info-type label from the registry (`dataset` field, e.g. `daily_summary`);
  defaults to the source name if unset. This keeps everything for one dataset in a single
  directory, easy to list and glob.
- `<YYYY-MM-DD>` is the logical date the record set describes. Re-running a date overwrites
  its file (idempotent). Sources with no logical date use a run timestamp instead.
- `<ext>` is `jsonl` or `parquet`.

## Rate limiting (cadence)

Provider request caps (e.g. free-tier `N`/minute) are **per account**, shared across every
source hitting that provider. The module enforces a minimum interval between requests with a
cross-process file lock (`state/rate/<provider>.lock`), so running several programs against the
same provider queues them rather than bursting past the limit. Configure the cap in settings
(e.g. `[massive].rate_limit_per_min`).

## Every record carries provenance

Writers inject these fields automatically (do not hand-populate them):

- `_source` — the source name
- `_ingested_at` — ISO-8601 UTC timestamp of write
- `_run_id` — the run identifier

These are part of the contract and are appended to every declared schema.
