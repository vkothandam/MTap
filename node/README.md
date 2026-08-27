# sourcing-node

Node.js sourcers for MTap (TypeScript, ESM). This is where you reuse your existing
Node libraries.

## Setup

```bash
cd node
npm install
```

## Run a source

```bash
npm run run-source -- run example_scrape
# output -> ./out/example_scrape/dt=YYYY-MM-DD/example_scrape-<run_id>.jsonl
```

## Run a service (long-running)

A **service** is the long-running counterpart to a source: it holds a persistent
session and publishes onto the message bus instead of writing files (e.g. the
E*TRADE scraper). Start one with the `serve` verb:

```bash
npm run serve -- etrade          # -> tsx src/cli.ts serve etrade
```

It runs until Ctrl-C (graceful `stop()`). See
[src/services/etrade/README.md](src/services/etrade/README.md) for the E*TRADE
login/OTP flow and configuration.

## Test (includes schema conformance)

```bash
npm test
```

## Adding a source

Create `src/sources/<name>/source.ts` exporting a class that extends `Source` and
implements `fetch()` and `extract()`. Register it in `../shared/config/sources.json`
and add it to the switch in `src/registry.ts`. The base class handles the writer,
provenance fields, and naming per `../schemas/conventions.md`.

## Adding a service

Create `src/services/<name>/` with a class that extends `Service` (`src/common/service.ts`)
and implements `start()` / `stop()`. Register it in `src/serviceRegistry.ts` (parallels
`registry.ts`). The base class wires up the `Bus`; use `src/common/session.ts` for
session persistence. Launch it with `npm run serve -- <name>`.

## Parquet

The writer emits jsonl natively. For parquet, install a lib (`parquetjs`,
`parquet-wasm`, or `apache-arrow`) and fill in `writeParquet` in `src/common/writer.ts`.
Prefer jsonl for scraped/irregular data; use parquet for large stable tables.
