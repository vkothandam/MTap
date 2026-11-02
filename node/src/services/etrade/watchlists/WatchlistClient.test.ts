import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { WatchlistApiError, WatchlistClient, addColumnCodes, parseSymbolList, planSync, unwrap, type Post } from "./WatchlistClient.js";
import { accountEnvNames, resolveAccount } from "./account.js";
import { missingFeedColumns } from "./cli.js";

const API = "https://us.etrade.com/webapipf/watchlist";

/** A fake post that records calls and answers from `reply(endpoint, value)`. */
function fakePost(reply: (endpoint: string, value: any) => unknown): { post: Post; calls: { endpoint: string; body: any }[] } {
  const calls: { endpoint: string; body: any }[] = [];
  const post: Post = async (url, body) => {
    assert.ok(url.startsWith(`${API}/`), url);
    const endpoint = url.slice(API.length + 1);
    calls.push({ endpoint, body });
    return reply(endpoint, (body as { value?: unknown } | undefined)?.value);
  };
  return { post, calls };
}

const client = (post: Post): WatchlistClient => new WatchlistClient(post, { delayMs: 0 });

test("unwrap returns data, or the body when there is no data", () => {
  assert.deepEqual(unwrap({ data: { a: 1 } }), { a: 1 });
  assert.deepEqual(unwrap({ a: 2 }), { a: 2 });
  assert.deepEqual(unwrap('{"data":{"b":3}}'), { b: 3 });
});

test("unwrap throws on an error in an HTTP 200 body", () => {
  assert.throws(() => unwrap({ error: { message: "bad session" } }), /bad session/);
  assert.throws(() => unwrap({ success: false }), WatchlistApiError);
  assert.throws(() => unwrap({ data: { Messages: [{ type: "ERROR", text: "no such list" }] } }), /no such list/);
  assert.deepEqual(unwrap({ data: { Messages: [{ type: "INFO", text: "ok" }], x: 1 } }), {
    Messages: [{ type: "INFO", text: "ok" }],
    x: 1,
  });
});

test("listWatchlists posts no body and maps id/name", async () => {
  const { post, calls } = fakePost(() => ({ data: { watchlists: [{ id: 217428640306, name: "Experimental" }] } }));
  assert.deepEqual(await client(post).listWatchlists(), [{ id: "217428640306", name: "Experimental" }]);
  assert.deepEqual(calls, [{ endpoint: "getAllWatchlists.json", body: undefined }]);
});

test("createWatchlist wraps the name in {value} and returns the new id", async () => {
  const { post, calls } = fakePost(() => ({ data: { watchlist: { watchlistID: 123 } } }));
  assert.equal(await client(post).createWatchlist("MidCap"), "123");
  assert.deepEqual(calls[0], { endpoint: "createWL.json", body: { value: { watchlistName: "MidCap" } } });
});

test("createWatchlist throws when no id comes back", async () => {
  const { post } = fakePost(() => ({ data: { watchlist: {} } }));
  await assert.rejects(client(post).createWatchlist("x"), /no watchlistID/);
});

test("rename and delete send the list id", async () => {
  const { post, calls } = fakePost(() => ({ data: {} }));
  await client(post).renameWatchlist("9", "New");
  await client(post).deleteWatchlist("9");
  assert.deepEqual(calls, [
    { endpoint: "renameWL.json", body: { value: { watchlistID: "9", watchlistName: "New" } } },
    { endpoint: "deleteWL.json", body: { value: { watchlistID: "9" } } },
  ]);
});

test("getEntries sends the collector's payload and reads entryDetails", async () => {
  const { post, calls } = fakePost(() => ({
    data: {
      watchListView: {
        columnHeaders: ["Symbol", "Last Price"],
        entryDetails: [
          { symbol: " aapl ", entryID: 11 },
          { symbol: "MSFT", entryID: 12 },
        ],
      },
    },
  }));
  const view = await client(post).getEntries("77", "MidCap");
  assert.deepEqual(view, {
    columnHeaders: ["Symbol", "Last Price"],
    entries: [
      { symbol: "AAPL", entryID: "11" },
      { symbol: "MSFT", entryID: "12" },
    ],
  });
  const req = calls[0].body.value;
  assert.equal(calls[0].endpoint, "getTSPWatchlistDetails.json");
  assert.equal(req.watchListRequest.watchlistID, "77");
  assert.equal(req.watchListRequest.viewName, "MidCap");
  assert.equal(req.watchListRequest.isCustomView, true);
  assert.deepEqual(req.watchListRequest.pagination, { startPosNum: 1, posPerPage: WatchlistClient.PAGE_SIZE });
  assert.deepEqual(req.noteJSON, { noteType: "WATCHLIST", watchlistId: "77" });
});

test("getEntries falls back to the Symbol column of columnValues", async () => {
  const { post } = fakePost(() => ({
    data: { watchListView: { columnHeaders: ["Last", "Symbol"], columnValues: [{ entryValuesList: ["1.5", "ibm"] }] } },
  }));
  assert.deepEqual((await client(post).getEntries("1", "v")).entries, [{ symbol: "IBM", entryID: "" }]);
});

test("getEntries pages until a short page", async () => {
  const saved = WatchlistClient.PAGE_SIZE;
  WatchlistClient.PAGE_SIZE = 2;
  try {
    const pages = [
      [{ symbol: "A", entryID: 1 }, { symbol: "B", entryID: 2 }],
      [{ symbol: "C", entryID: 3 }],
    ];
    const { post, calls } = fakePost(() => ({ data: { watchListView: { columnHeaders: ["Symbol"], entryDetails: pages.shift() } } }));
    const view = await client(post).getEntries("1", "v");
    assert.deepEqual(view.entries.map((e) => e.symbol), ["A", "B", "C"]);
    assert.deepEqual(calls.map((c) => c.body.value.watchListRequest.pagination.startPosNum), [1, 3]);
  } finally {
    WatchlistClient.PAGE_SIZE = saved;
  }
});

test("addSymbols: one call per symbol with MBin's payload; empty entries = rejected", async () => {
  const { post, calls } = fakePost((_endpoint, value) => ({
    data: { watchlistEntry: { entries: value.symbol === "ZZZZ" ? [] : [{ entryID: 1 }] } },
  }));
  const result = await client(post).addSymbols("5", ["aapl", "ZZZZ"], "MidCap");
  assert.deepEqual(result, { added: ["AAPL"], rejected: ["ZZZZ"], failed: [] });
  assert.deepEqual(calls[0], {
    endpoint: "createWatchListEntry.json",
    body: {
      value: { isTSP: true, newIndexID: 1, positionType: 1, quantity: 100, symbol: "AAPL", viewName: "MidCap", watchlistID: "5" },
    },
  });
});

test("addSymbols records a failure and carries on", async () => {
  const { post } = fakePost((_endpoint, value) => {
    if (value.symbol === "BAD") return { error: "boom" };
    return { data: { watchlistEntry: { entries: [{}] } } };
  });
  const result = await client(post).addSymbols("5", ["A", "BAD", "C"], "v");
  assert.deepEqual(result.added, ["A", "C"]);
  assert.deepEqual(result.failed, [{ symbol: "BAD", error: "boom" }]);
  assert.equal(result.aborted, undefined);
});

test("addSymbols stops on a 401", async () => {
  let n = 0;
  const post: Post = async () => {
    if (++n === 2) throw new WatchlistApiError("Unauthorized", 401);
    return { data: { watchlistEntry: { entries: [{}] } } };
  };
  const result = await client(post).addSymbols("5", ["A", "B", "C"], "v");
  assert.deepEqual(result.added, ["A"]);
  assert.match(result.aborted ?? "", /session expired \(HTTP 401\) at B/);
  assert.equal(n, 2);
});

test("addSymbols stops after too many failures in a row", async () => {
  let n = 0;
  const post: Post = async () => {
    n++;
    throw new WatchlistApiError("timeout");
  };
  const symbols = Array.from({ length: 10 }, (_, i) => `S${i}`);
  const result = await client(post).addSymbols("5", symbols, "v");
  assert.equal(n, WatchlistClient.MAX_CONSECUTIVE_FAILURES);
  assert.equal(result.failed.length, WatchlistClient.MAX_CONSECUTIVE_FAILURES);
  assert.match(result.aborted ?? "", /failures in a row/);
});

test("deleteEntries joins ids with commas in batches", async () => {
  const { post, calls } = fakePost(() => ({ data: {} }));
  const ids = Array.from({ length: WatchlistClient.DELETE_BATCH + 2 }, (_, i) => String(i));
  await client(post).deleteEntries("5", ids);
  assert.equal(calls.length, 2);
  assert.equal(calls[0].endpoint, "deleteWatchListEntry.json");
  assert.equal(calls[0].body.value.entryID, ids.slice(0, WatchlistClient.DELETE_BATCH).join(","));
  assert.deepEqual(calls[1].body.value, { entryID: `${WatchlistClient.DELETE_BATCH},${WatchlistClient.DELETE_BATCH + 1}`, watchlistID: "5" });
});

test("removeSymbols deletes by entry id and reports symbols not in the list", async () => {
  const { post, calls } = fakePost((endpoint) =>
    endpoint === "getTSPWatchlistDetails.json"
      ? { data: { watchListView: { columnHeaders: ["Symbol"], entryDetails: [{ symbol: "A", entryID: 1 }, { symbol: "B", entryID: 2 }] } } }
      : { data: {} },
  );
  assert.deepEqual(await client(post).removeSymbols("5", ["b", "Q"], "v"), { removed: ["B"], notFound: ["Q"] });
  assert.deepEqual(calls[1].body.value, { entryID: "2", watchlistID: "5" });
});

test("planSync compares trimmed upper-case symbols", () => {
  assert.deepEqual(planSync(["AAPL", "msft", "IBM"], [" aapl", "GOOG", "", "ibm"]), { add: ["GOOG"], remove: ["MSFT"] });
});

test("sync adds the missing symbols and prunes only with prune", async () => {
  const reply = (endpoint: string): unknown =>
    endpoint === "getTSPWatchlistDetails.json"
      ? { data: { watchListView: { columnHeaders: ["Symbol"], entryDetails: [{ symbol: "A", entryID: 1 }, { symbol: "X", entryID: 9 }] } } }
      : { data: { watchlistEntry: { entries: [{}] } } };

  const dry = fakePost(reply);
  assert.deepEqual(await client(dry.post).sync("5", ["A", "B"], "v", { prune: true, dryRun: true }), { add: ["B"], remove: ["X"] });
  assert.equal(dry.calls.length, 1);

  const noPrune = fakePost(reply);
  const kept = await client(noPrune.post).sync("5", ["A", "B"], "v");
  assert.deepEqual(kept.remove, []);
  assert.deepEqual(noPrune.calls.map((c) => c.endpoint), ["getTSPWatchlistDetails.json", "createWatchListEntry.json"]);

  const pruned = fakePost(reply);
  await client(pruned.post).sync("5", ["A", "B"], "v", { prune: true });
  assert.deepEqual(pruned.calls.at(-1), { endpoint: "deleteWatchListEntry.json", body: { value: { entryID: "9", watchlistID: "5" } } });
});

test("parseSymbolList: comments, separators, case and duplicates", () => {
  assert.deepEqual(parseSymbolList("# header\naapl\nMSFT, ibm  # note\n\nAAPL\r\nbrk.b\n"), ["AAPL", "MSFT", "IBM", "BRK.B"]);
});

test("account names: main = ET_*, others <NAME>_ET_*", () => {
  assert.deepEqual(accountEnvNames("main"), { user: "ET_USERNAME", pass: "ET_PASSWORD" });
  assert.deepEqual(accountEnvNames("alt"), { user: "ALT_ET_USERNAME", pass: "ALT_ET_PASSWORD" });
  assert.deepEqual(accountEnvNames("old-1"), { user: "OLD_1_ET_USERNAME", pass: "OLD_1_ET_PASSWORD" });
});

test("resolveAccount reads only the account's keys, from the secrets file or env", () => {
  const file = join(mkdtempSync(join(tmpdir(), "mtap-wl-")), ".mtap.env");
  writeFileSync(file, "ET_USERNAME=u1\nET_PASSWORD=p1\nALT_ET_USERNAME=u2\nALT_ET_PASSWORD='p 2'\n");
  const alt = resolveAccount("alt", file, {});
  assert.equal(alt.username, "u2");
  assert.equal(alt.password, "p 2");
  assert.match(alt.stateDir, /etrade-alt$/);
  assert.equal(resolveAccount("main", undefined, { ET_USERNAME: "e", ET_PASSWORD: "f" }).username, "e");
  assert.throws(() => resolveAccount("other", file, {}), /OTHER_ET_USERNAME/);
});

test("missingFeedColumns names the feed columns a view lacks", () => {
  const full = ["Symbol", "Last Trade", "Volume", "Open", "Previous Close", "Bid", "Ask", "Bid Size", "Ask Size", "10 Day Volume"];
  assert.deepEqual(missingFeedColumns(full), []);
  assert.deepEqual(missingFeedColumns(["Symbol", "Last Price", "Volume"]), [
    "Open",
    "Previous Close",
    "Bid",
    "Ask",
    "Bid Size",
    "Ask Size",
    "10 Day Volume",
  ]);
  assert.deepEqual(missingFeedColumns([]).slice(0, 2), ["Symbol", "price (Last Trade | Last Price | Last | Price)"]);
});

const VIEW_API = "https://us.etrade.com/webapipf/view";

/** A fake post for the view endpoints, recording url, body and method. */
function fakeViewPost(reply: (url: string, body: any) => unknown): { post: Post; calls: { url: string; body: any; method?: string }[] } {
  const calls: { url: string; body: any; method?: string }[] = [];
  const post: Post = async (url, body, method) => {
    calls.push({ url, body, method });
    return reply(url, body);
  };
  return { post, calls };
}

test("listViews GETs the view list", async () => {
  const { post, calls } = fakeViewPost(() => ({
    data: { views: [{ name: "Experimental", customView: true }, { name: "Market View", customView: false }] },
  }));
  assert.deepEqual(await client(post).listViews(), [
    { name: "Experimental", custom: true },
    { name: "Market View", custom: false },
  ]);
  assert.deepEqual(calls, [{ url: `${VIEW_API}/getViewList/tspWatchlist.json`, body: undefined, method: "GET" }]);
});

test("getView returns the view, or null when E*TRADE falls back to Market View", async () => {
  const view = { viewName: "Experimental", isCustomView: true, columnHeaderCodes: "1,47", columnHeaders: ["Symbol", "Last Trade"], baseView: "", viewSecType: "" };
  const { post, calls } = fakeViewPost((_url, body) =>
    body.value.viewName === "Experimental" ? { data: { viewResponse: view } } : { data: { viewResponse: { ...view, viewName: "Market View", isCustomView: false } } },
  );
  const got = await client(post).getView("Experimental");
  assert.deepEqual(got, { name: "Experimental", custom: true, columnHeaderCodes: "1,47", columnHeaders: ["Symbol", "Last Trade"], baseView: "", viewSecType: "" });
  assert.equal(await client(post).getView("MidCap"), null);
  assert.deepEqual(calls[0], {
    url: `${VIEW_API}/getView.json`,
    body: { value: { isCustomView: true, viewName: "Experimental", viewSrcType: "watchlist" } },
    method: undefined,
  });
});

test("allColumns maps every group's column names to codes, plus those only custom views hold", async () => {
  const { post } = fakeViewPost((url) => {
    if (url.endsWith("getAllColumns/tspWatchlist.json")) {
      return { data: { Columns: { basicQuotes: [{ columnCode: 1, columnName: "Symbol" }], detailedQuotes: [{ columnCode: 33, columnName: "Open" }] } } };
    }
    if (url.endsWith("getViewList/tspWatchlist.json")) return { data: { views: [{ name: "Mine", customView: true }, { name: "Market View", customView: false }] } };
    return { data: { viewResponse: { viewName: "Mine", columnHeaderCodes: "1,1000", columnHeaders: ["Symbol", "Actions"] } } };
  });
  assert.deepEqual([...(await client(post).allColumns())], [["Symbol", 1], ["Open", 33], ["Actions", 1000]]);
});

test("setViewColumns and createView post the page's payloads", async () => {
  const { post, calls } = fakeViewPost(() => ({ data: {}, success: true }));
  const view = { name: "Experimental", custom: true, columnHeaderCodes: "1,47", columnHeaders: [], baseView: "", viewSecType: "" };
  await client(post).setViewColumns(view, "1,47,33");
  await client(post).createView("MidCap", "1,47");
  assert.deepEqual(calls[0], {
    url: `${VIEW_API}/updateView.json`,
    body: { value: { baseView: "", columnHeaderCodes: "1,47,33", isCustomView: true, viewName: "Experimental", viewSecType: "", viewSrcType: "watchlist" } },
    method: undefined,
  });
  assert.equal(calls[1].url, `${VIEW_API}/createView.json`);
  assert.equal(calls[1].body.value.viewName, "MidCap");
});

test("a view of more than 16 columns is refused before any call", async () => {
  const { post, calls } = fakeViewPost(() => ({ data: {} }));
  const codes = Array.from({ length: WatchlistClient.MAX_VIEW_COLUMNS + 1 }, (_, i) => String(i + 1)).join(",");
  await assert.rejects(client(post).createView("Big", codes), /at most 16 columns, not 17/);
  assert.equal(calls.length, 0);
});

test("addColumnCodes appends missing codes in order and rejects unknown names", () => {
  const catalog = new Map([["Symbol", 1], ["Open", 33], ["Bid", 13], ["Last Trade", 47]]);
  assert.equal(addColumnCodes("1,47", ["Open", "Bid", "Last Trade"], catalog), "1,47,33,13");
  assert.equal(addColumnCodes("", ["Symbol", "Bid"], catalog), "1,13");
  assert.throws(() => addColumnCodes("1", ["Nope"], catalog), /no column "Nope"/);
});
