# E*TRADE scraping service

A long-running MTap **service** that owns a headless E*TRADE browser session: it
logs in (handling the SMS/OTP challenge), polls watchlists and market movers, and
publishes raw `POLLED_DATA` onto the message bus for downstream consumers (MBin's
trading system). Unlike a batch source (see [node/README](../../../README.md)), a
service holds a persistent session and **streams events** rather than writing files.

> **Status.** The service platform and the core scraper are in place: the `serve`
> CLI verb, the `Service` base class, the Redis-backed `Bus`, the file-backed
> session store, the `/auth/otp` server, and the ported browser/login/page-scraper
> modules — `etrade` is registered, so `serve etrade` runs. The `POLLED_DATA`
> contract is frozen (`POLLED_DATA.contract.md` / `.schema.json`). Still to come:
> the input-driven `WatchlistManager` and the `WATCHLIST_CMD` command path
> (watchlist CRUD from MBin's UI). This guide describes the operational flow so new
> sourcing use cases can follow the same shape.

## Prerequisites

- Node ≥ 20 with deps installed (`cd node && npm install`). The scraper port adds
  `puppeteer`, which downloads a Chromium on first install.
- **Redis** reachable at `REDIS_URL` (the bus transport, shared with MBin). Locally:
  `redis-server`.
- Credentials configured (below).

## Configure

Copy the template at the repo root and fill it in — never commit real secrets:

```bash
cp .env.example .env
```

| Variable | Purpose |
|---|---|
| `BUS_TRANSPORT` | `redis` to talk to MBin cross-process (`inproc` = local only) |
| `REDIS_URL` | Redis endpoint for the bus |
| `ET_USERNAME` / `ET_PASSWORD` | E*TRADE browser-login credentials |
| `SMS_EMAIL` / `SMS_EMAIL_PASSWORD` | email→SMS gateway account used to text the OTP link |
| `OTP_HOST` / `OTP_PORT` | where MTap serves the `/auth/otp` code-entry page (the SMS links here) |

> ⚠️ **Rotate before first use.** The values previously committed in MBin's
> `config/environment.js` are exposed and must be treated as compromised — rotate
> the E*TRADE password and the SMS-gateway app-password. MBin's OAuth *trading*
> credentials (consumer key/secret) stay in MBin and are **not** needed here.

Secrets are read from the process environment — export them, or launch with Node's
env-file support: `node --env-file=.env dist/cli.js serve etrade`.

## Start a session

Dev (TypeScript, no build):

```bash
cd node
npm run serve -- etrade          # -> tsx src/cli.ts serve etrade
```

Built:

```bash
cd node && npm run build
node --env-file=../.env dist/cli.js serve etrade
```

The process runs until you stop it. Ctrl-C (SIGINT/SIGTERM) triggers a graceful
`stop()` — it closes the browser and the bus connections.

## What happens on start

1. The service connects to the bus (Redis) and launches a headless browser.
2. It loads any persisted session from `state/etrade/session.json`
   (cookies + `stk1`/`stk2` request headers).
3. **If that session is still valid** → login and OTP are **skipped**; polling
   starts immediately.
4. **If there is no session, or it has expired** → the login flow runs (next
   section), which will trigger OTP unless E*TRADE still recognises the device.
5. Once authenticated it polls on timers — the main watchlist roughly every 2 s,
   midcap roughly every 20 s — refreshes the page about every 5 min, and
   republishes each tick as a `POLLED_DATA` event.

## Login + OTP flow

The first login on a machine (and any time the session can't be reused) needs a
human to enter an OTP code. The **entire flow lives in MTap** — no MBin process is
involved.

```
serve etrade
   │
   ▼
launch browser ──▶ valid session on disk? ──yes──▶ polling
   │
   no
   ▼
navigate to E*TRADE login, submit ET_USERNAME / ET_PASSWORD
   │
   ▼
E*TRADE presents a device-verification (OTP) challenge
   │
   ▼
scraper selects a phone number and requests an SMS code
   │
   ▼
SMSNotifier texts a link:  http://OTP_HOST:OTP_PORT/auth/otp
   │
   ▼
human opens that link, reads the code from the SMS, submits it
   │
   ▼
MTap's /auth/otp server resolves the pending login in-process;
the scraper types the code into E*TRADE ──▶ login completes
   │
   ▼
session (cookies + stk1/stk2) saved to state/etrade/session.json ──▶ polling
```

Notes:

- **When OTP triggers:** first login on a machine, an expired/cleared session, or
  E*TRADE no longer recognising the device. A reused valid session skips it.
- **Where you enter the code:** MTap serves its *own* minimal page at `/auth/otp`
  (`OTP_HOST:OTP_PORT`); the SMS contains the link. Open it from any browser that
  can reach that host. There is no MBin UI in this path.
- **Timeout:** if the code isn't submitted within the login window the attempt
  fails — re-run `serve etrade` to retry.
- **Mid-run expiry:** while polling, a session expiry is detected automatically and
  re-authentication kicks off, which may prompt a fresh OTP. Keep the `/auth/otp`
  host reachable for the life of the service.

## Data + commands on the bus

- **Out — `POLLED_DATA`** (MTap → MBin): raw watchlist and market-mover payloads,
  each stamped with an idempotency key so consumers dedup at-least-once delivery.
- **In — `WATCHLIST_CMD`** (MBin → MTap, request/reply): watchlist create/delete
  issued from MBin's UI; the service runs it in the live session and replies.
- **Optional — `AUTH_OTP_REQUIRED`** (MTap → MBin): a read-only "login required"
  signal so MBin's dashboard can show a banner. Informational; not on the login path.

The transport and envelope are defined in [`../../common/bus`](../../common/bus) and
are wire-compatible with MBin's `core/bus` (same `mtap:evt:<EVENT>` streams,
`__reply__` channel, and `{ ...payload, _key, _ts }` envelope).

## Session persistence & reuse

`state/etrade/session.json` (git-ignored) holds the live session so restarts avoid a
fresh login/OTP. Delete it to force a clean login. Override the path with
`SESSION_PATH`.

## Downstream reuse: Python fundamentals scraper

The same `state/etrade/session.json` (the `cookies` string + `stk1`/`stk2` request
headers written by this service) is **consumed by the Python E*TRADE fundamentals
collector** to call the authenticated wsod REST API (balance sheet, income statement,
SEC filings) — see [python/sourcing_py/etrade/README.md](../../../../python/sourcing_py/etrade/README.md).

That collector does **not** log in itself: it reads the session this service produces.
So its Phase 2 (`sourcing-py etrade fetch`) is **gated on the login here being merged and
having written `session.json`** — until then it fails fast with a clear message. Its
Phase 1 (`build-symbols`, from local `daily_summary` files) needs no session and runs
independently. The login/OTP flow above is therefore a prerequisite for the fundamentals
scrape, not just for the streaming poller.

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `No service registered for 'etrade'` | Scraper port not landed yet (see Status), or not registered in `src/serviceRegistry.ts`. |
| Hangs at login, no SMS arrives | Check `SMS_EMAIL` / `SMS_EMAIL_PASSWORD` and the gateway account. |
| SMS link unreachable on your phone | `OTP_HOST` must be reachable from the device opening it — use the machine's LAN IP, not `127.0.0.1`. |
| `POLLED_DATA` never reaches MBin | Both sides need `BUS_TRANSPORT=redis` and the same `REDIS_URL`; confirm Redis is up. |
| Repeated OTP prompts every start | Session not persisting — check `state/etrade/` is writable and `SESSION_PATH`. |
| Browser fails to launch | Chromium missing/incompatible — set `PUPPETEER_EXECUTABLE_PATH` or reinstall `puppeteer`. |
