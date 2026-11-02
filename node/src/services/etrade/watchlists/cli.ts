import { readFileSync, writeFileSync } from "node:fs";
import { parseArgs } from "node:util";
import { WATCHLISTS } from "../constants.js";
import { FIELD_HEADERS, PRICE_HEADERS } from "../../../feeds/EtradeWatchlistFeed.js";
import { resolveAccount, savedSession, type EtradeAccount } from "./account.js";
import { WatchlistClient, addColumnCodes, axiosPost, parseSymbolList, type AddResult, type WatchlistView } from "./WatchlistClient.js";
import type { StoredCookie } from "../../../common/session.js";
import type { RequestHeaders } from "../ETPageScraper.js";

// `npm run watchlist -- <command> ... [--account main|alt] [--secrets FILE]`: manage an
// E*TRADE account's watchlists. Commands call the watchlist API with the account's saved
// session, with no browser; only when there is none, or it no longer works, a browser login
// runs first (the code typed in the Chrome window) and the window closes once it is done.
// --account main reads ET_USERNAME/ET_PASSWORD, any other name <NAME>_ET_*, from
// --secrets FILE when given, else from the environment.

export const USAGE = `usage: npm run watchlist -- <command> [options]
  login                                   browser login (saves the session the other commands use)
  list                                    the account's watchlists (id, name)
  show   <id> [--view NAME] [--out FILE]  columns, feed columns missing, symbol count; --out writes the symbols
  create <name> --from FILE [--view NAME] new list with FILE's symbols (refuses an existing name)
  add    <id> (--from FILE | SYM...)      add symbols
  remove <id> (--from FILE | SYM...)      delete symbols
  sync   <id> --from FILE [--prune] [--dry-run]  add the missing symbols; --prune also deletes the rest
  rename <id> <name>
  view   [NAME] [--columns "A,B,..." | --add-columns "A,B,..."]  column views; NAME's columns;
         --columns sets exactly these (in order), --add-columns appends; both create NAME if missing
  delete <id> [--force]                   delete a list (--force for a list the collector polls)
options: --account main|<name> (default main)  --secrets FILE  --view NAME (default: the collector's view, else the list's name)  --delay MS`;

const OPTIONS = {
  account: { type: "string", default: "main" },
  secrets: { type: "string" },
  view: { type: "string" },
  from: { type: "string" },
  out: { type: "string" },
  delay: { type: "string" },
  prune: { type: "boolean", default: false },
  "dry-run": { type: "boolean", default: false },
  force: { type: "boolean", default: false },
  "add-columns": { type: "string" },
  columns: { type: "string" },
  help: { type: "boolean", default: false },
} as const;

type Values = ReturnType<typeof parseArgs<{ options: typeof OPTIONS; allowPositionals: true }>>["values"];

/** The feed's columns (EtradeWatchlistFeed) a view does not have. */
export function missingFeedColumns(columnHeaders: string[]): string[] {
  const have = new Set(columnHeaders);
  const missing: string[] = [];
  if (!have.has("Symbol")) missing.push("Symbol");
  if (!PRICE_HEADERS.some((h) => have.has(h))) missing.push(`price (${PRICE_HEADERS.join(" | ")})`);
  for (const [, header] of FIELD_HEADERS) if (!have.has(header)) missing.push(header);
  return missing;
}

/** Symbols from --from FILE plus any given on the command line. */
function symbolsFrom(values: Values, positionals: string[]): string[] {
  const text = [values.from ? readFileSync(values.from, "utf-8") : "", ...positionals].join("\n");
  return parseSymbolList(text);
}

function printView(id: string, view: string, wl: WatchlistView): void {
  console.log(`watchlist ${id}, view "${view}": ${wl.entries.length} symbols`);
  console.log(`columns: ${wl.columnHeaders.join(", ") || "(none)"}`);
  const missing = missingFeedColumns(wl.columnHeaders);
  console.log(missing.length ? `feed columns MISSING: ${missing.join(", ")}` : "feed columns: all present");
}

/**
 * Report an add, then check it against the list as E*TRADE now has it: a call can time out
 * after E*TRADE has added the symbol. True when every symbol is in the list.
 */
async function reportAdd(client: WatchlistClient, id: string, view: string, symbols: string[], result: AddResult): Promise<boolean> {
  console.log(`added ${result.added.length}, rejected ${result.rejected.length}, failed ${result.failed.length}`);
  if (result.rejected.length) console.log(`rejected (E*TRADE created nothing; an unknown symbol can also come back as a "Service failure"): ${result.rejected.join(" ")}`);
  for (const f of result.failed) console.log(`failed ${f.symbol}: ${f.error}`);
  if (result.aborted) console.log(`ABORTED: ${result.aborted}`);
  const wl = await client.getEntries(id, view);
  printView(id, view, wl);
  const have = new Set(wl.entries.map((e) => e.symbol));
  const missing = symbols.filter((s) => !have.has(s));
  console.log(missing.length ? `NOT in the list: ${missing.join(" ")}` : `all ${symbols.length} symbols are in the list`);
  return missing.length === 0;
}

async function run(client: WatchlistClient, command: string, args: string[], values: Values): Promise<number> {
  const lists = await client.listWatchlists();
  const need = (n: number): void => {
    if (args.length < n) throw new Error(`${command}: missing argument\n${USAGE}`);
  };
  /** The default view: the collector's for a list it polls, else the list's own name. */
  const viewOf = (id: string): string => {
    const list = lists.find((w) => w.id === id);
    if (!list) throw new Error(`no watchlist ${id} on this account (see: list)`);
    const polled = Object.values(WATCHLISTS).find((w) => w.id === id);
    return values.view ?? (polled ? (polled.view ?? polled.name) : list.name);
  };

  switch (command) {
    case "list":
      for (const w of lists) console.log(`${w.id}\t${w.name}`);
      console.log(`${lists.length} watchlists`);
      return 0;

    case "show": {
      need(1);
      const view = viewOf(args[0]);
      const wl = await client.getEntries(args[0], view);
      printView(args[0], view, wl);
      if (values.out) {
        const symbols = wl.entries.map((e) => e.symbol).sort();
        const header = `# E*TRADE watchlist ${args[0]} (view "${view}"), ${symbols.length} symbols, read ${new Date().toISOString()}.\n`;
        writeFileSync(values.out, header + symbols.join("\n") + "\n");
        console.log(`wrote ${values.out}`);
      }
      return 0;
    }

    case "create": {
      need(1);
      const name = args[0];
      const symbols = symbolsFrom(values, args.slice(1));
      const clash = lists.find((w) => w.name.toLowerCase() === name.toLowerCase());
      if (clash) throw new Error(`a watchlist named "${clash.name}" exists already (${clash.id})`);
      const id = await client.createWatchlist(name);
      console.log(`created "${name}": ${id}`);
      const view = values.view ?? name;
      return (await reportAdd(client, id, view, symbols, await client.addSymbols(id, symbols, view))) ? 0 : 1;
    }

    case "add": {
      need(1);
      const view = viewOf(args[0]);
      const symbols = symbolsFrom(values, args.slice(1));
      return (await reportAdd(client, args[0], view, symbols, await client.addSymbols(args[0], symbols, view))) ? 0 : 1;
    }

    case "remove": {
      need(1);
      const { removed, notFound } = await client.removeSymbols(args[0], symbolsFrom(values, args.slice(1)), viewOf(args[0]));
      console.log(`removed ${removed.length}${notFound.length ? `; not in the list: ${notFound.join(" ")}` : ""}`);
      return 0;
    }

    case "sync": {
      need(1);
      if (!values.from) throw new Error("sync: --from FILE is required");
      const result = await client.sync(args[0], symbolsFrom(values, []), viewOf(args[0]), {
        prune: values.prune,
        dryRun: values["dry-run"],
      });
      console.log(`to add ${result.add.length}: ${result.add.join(" ")}`);
      console.log(`to remove ${result.remove.length}: ${result.remove.join(" ")}`);
      if (!result.result) return 0;
      const ok = await reportAdd(client, args[0], viewOf(args[0]), result.add, result.result);
      return ok ? 0 : 1;
    }

    case "rename":
      need(2);
      viewOf(args[0]);
      await client.renameWatchlist(args[0], args[1]);
      console.log(`renamed ${args[0]} to "${args[1]}"`);
      return 0;

    case "delete": {
      need(1);
      viewOf(args[0]);
      const polled = Object.values(WATCHLISTS).find((w) => w.id === args[0]);
      if (polled && !values.force) throw new Error(`${args[0]} is the collector's ${polled.name} list; add --force to delete it`);
      await client.deleteWatchlist(args[0]);
      console.log(`deleted ${args[0]}`);
      return 0;
    }

    case "view": {
      if (!args[0]) {
        for (const v of await client.listViews()) console.log(`${v.custom ? "custom " : "built-in"}\t${v.name}`);
        return 0;
      }
      const name = args[0];
      let view = await client.getView(name);
      const exact = values.columns !== undefined;
      const wanted = (exact ? values.columns : values["add-columns"])?.split(",").map((c) => c.trim()).filter(Boolean);
      if (wanted) {
        const catalog = await client.allColumns();
        if (view) console.log(`before: ${view.columnHeaders.join(", ")} (codes ${view.columnHeaderCodes})`);
        const codes = addColumnCodes(exact || !view ? "" : view.columnHeaderCodes, wanted, catalog);
        if (view) await client.setViewColumns(view, codes);
        else {
          await client.createView(name, codes);
          console.log(`created view "${name}"`);
        }
        view = await client.getView(name);
      }
      if (!view) {
        console.log(`no custom view "${name}" (lists read with it get E*TRADE's built-in Market View)`);
        return 1;
      }
      console.log(`view "${name}": ${view.columnHeaders.join(", ")} (codes ${view.columnHeaderCodes})`);
      const missing = missingFeedColumns(view.columnHeaders);
      console.log(missing.length ? `feed columns MISSING: ${missing.join(", ")}` : "feed columns: all present");
      return 0;
    }

    default:
      console.error(USAGE);
      return 2;
  }
}

/** A browser login (the saved session if it still works, else username/password and code). */
async function browserLogin(account: EtradeAccount): Promise<{ cookies: StoredCookie[]; requestHeaders: RequestHeaders }> {
  // Loaded here so the usage text and tests never pull in puppeteer.
  const { openEtradeSession } = await import("./session.js");
  const session = await openEtradeSession(account);
  await session.close(); // the API calls need only the cookies and stk1
  return session;
}

/** A client on the saved session when it still answers, else after a browser login. */
async function connect(account: EtradeAccount, makeClient: (s: { cookies: StoredCookie[]; requestHeaders: RequestHeaders }) => WatchlistClient): Promise<WatchlistClient> {
  const saved = savedSession(account);
  if (saved) {
    const client = makeClient(saved);
    try {
      await client.listWatchlists();
      return client;
    } catch (error) {
      console.log(`[${account.name}] saved session no longer works (${(error as Error).message}); logging in`);
    }
  }
  return makeClient(await browserLogin(account));
}

export async function watchlistMain(argv: string[]): Promise<number> {
  const { values, positionals } = parseArgs({ args: argv, options: OPTIONS, allowPositionals: true });
  const [command, ...args] = positionals;
  if (!command || values.help) {
    console.error(USAGE);
    return command ? 0 : 2;
  }
  const account = resolveAccount(values.account, values.secrets);
  const makeClient = (s: { cookies: StoredCookie[]; requestHeaders: RequestHeaders }): WatchlistClient =>
    new WatchlistClient(axiosPost(s.cookies, s.requestHeaders), {
      delayMs: values.delay === undefined ? undefined : Number(values.delay),
      log: console.log,
    });
  if (command === "login") {
    const lists = await makeClient(await browserLogin(account)).listWatchlists();
    console.log(`[${account.name}] logged in; ${lists.length} watchlists`);
    return 0;
  }
  return run(await connect(account, makeClient), command, args, values);
}
