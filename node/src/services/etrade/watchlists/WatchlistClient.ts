import axios from "axios";
import { CookieUtils } from "../utils/CookieUtils.js";
import { etradeRequestConfig, type RequestHeaders } from "../ETPageScraper.js";
import type { StoredCookie } from "../../../common/session.js";

// E*TRADE watchlist management over the website's own JSON API (the calls the watchlist page
// makes): list, create, rename and delete whole lists; read, add and delete entries; read and
// change the account's column views. Calls send {value: {...}} with the page's cookies and
// stk1 header (etradeRequestConfig). Entry add/delete payloads come from MBin's
// WatchlistManager.js; the list-level and view endpoints and the comma-joined entry delete
// from E*TRADE's watchlist page bundle. No bus: nothing here reaches Redis or MBin.

const API = "https://us.etrade.com/webapipf/watchlist";
const VIEW_API = "https://us.etrade.com/webapipf/view";
/** viewSrcType in view requests (the page's constant). */
const VIEW_SOURCE = "watchlist";

export interface Watchlist {
  id: string;
  name: string;
}

export interface WatchlistEntry {
  symbol: string;
  entryID: string;
}

export interface WatchlistView {
  columnHeaders: string[];
  entries: WatchlistEntry[];
}

export interface AddResult {
  added: string[];
  /** E*TRADE answered but created nothing (unknown or invalid symbol). */
  rejected: string[];
  /** The call failed (timeout, network, error in the body). */
  failed: { symbol: string; error: string }[];
  /** Set when the run stopped early (session expired, too many failures in a row). */
  aborted?: string;
}

/** A column view: which columns getTSPWatchlistDetails returns when it names the view. */
export interface ColumnView {
  name: string;
  custom: boolean;
  /** Comma-joined column codes, in display order (getAllColumns names them). */
  columnHeaderCodes: string;
  columnHeaders: string[];
  baseView: string;
  viewSecType: string;
}

/** Send json to url (POST, or GET for the few read-only GET endpoints), resolving to the parsed body; injectable for tests. */
export type Post = (url: string, body: unknown, method?: "GET" | "POST") => Promise<unknown>;

export interface WatchlistClientOptions {
  post?: Post;
  /** Pause between single-entry calls (E*TRADE's page sends one entry per call). */
  delayMs?: number;
  log?: (line: string) => void;
}

/** An E*TRADE answer that is an error (HTTP 200 with an error body, or a non-2xx status). */
export class WatchlistApiError extends Error {
  constructor(
    message: string,
    readonly status = 0,
  ) {
    super(message);
    this.name = "WatchlistApiError";
  }

  get sessionExpired(): boolean {
    return this.status === 401 || this.status === 403;
  }
}

/** Trimmed, upper-case symbol, as E*TRADE stores it. */
export function normalizeSymbol(symbol: string): string {
  return symbol.trim().toUpperCase();
}

/**
 * Unwrap a response body as the watchlist page does: throw on `error`, `success: false` or an
 * ERROR message, then return `data` (or the body itself).
 */
export function unwrap(body: unknown): Record<string, any> {
  const b = (typeof body === "string" ? JSON.parse(body) : body ?? {}) as Record<string, any>;
  if (b.error != null) throw new WatchlistApiError(messageOf(b.error));
  if (b.success === false) throw new WatchlistApiError(messageOf(b.message_type ?? "success: false"));
  const messages = b.Messages ?? b.data?.Messages;
  const first = Array.isArray(messages) ? messages[0] : messages;
  if (first?.type === "ERROR") throw new WatchlistApiError(messageOf(first));
  return (b.data ?? b) as Record<string, any>;
}

function messageOf(error: unknown): string {
  if (typeof error === "string") return error;
  const e = error as { message?: string; text?: string; code?: string } | null;
  return e?.message ?? e?.text ?? JSON.stringify(error);
}

/** What to add to and remove from `current` to make it `target` (symbols, normalized). */
export function planSync(current: string[], target: string[]): { add: string[]; remove: string[] } {
  const have = new Set(current.map(normalizeSymbol));
  const want = new Set(target.map(normalizeSymbol).filter(Boolean));
  return {
    add: [...want].filter((s) => !have.has(s)),
    remove: [...have].filter((s) => !want.has(s)),
  };
}

/** Read a symbol file: one per line (or comma/space separated); `#` starts a comment. */
export function parseSymbolList(text: string): string[] {
  const out: string[] = [];
  const seen = new Set<string>();
  for (const line of text.split(/\r?\n/)) {
    for (const token of line.replace(/#.*/, "").split(/[\s,]+/)) {
      const symbol = normalizeSymbol(token);
      if (symbol && !seen.has(symbol)) {
        seen.add(symbol);
        out.push(symbol);
      }
    }
  }
  return out;
}

/** An axios POST with the page's cookies and headers; non-2xx statuses become WatchlistApiError. */
export function axiosPost(cookies: StoredCookie[], requestHeaders: RequestHeaders): Post {
  const config = etradeRequestConfig(CookieUtils.formatCookiesForHeader(cookies), requestHeaders);
  return async (url, body, method = "POST") => {
    try {
      const response = method === "GET" ? await axios.get(url, config) : await axios.post(url, body, config);
      return response.data;
    } catch (error) {
      const e = error as { message?: string; response?: { status?: number; data?: unknown } };
      const status = e.response?.status ?? 0;
      const detail = e.response?.data ? ` ${JSON.stringify(e.response.data).slice(0, 300)}` : "";
      throw new WatchlistApiError(`${e.message ?? "request failed"}${detail}`, status);
    }
  };
}

export class WatchlistClient {
  static PAGE_SIZE = 1000;
  static DELETE_BATCH = 50;
  static MAX_CONSECUTIVE_FAILURES = 5;
  /** E*TRADE refuses a view with more columns ("cannot be more than 16"). */
  static MAX_VIEW_COLUMNS = 16;

  private readonly post: Post;
  private readonly delayMs: number;
  private readonly log: (line: string) => void;

  constructor(post: Post, options: Omit<WatchlistClientOptions, "post"> = {}) {
    this.post = post;
    this.delayMs = options.delayMs ?? 400;
    this.log = options.log ?? (() => {});
  }

  private async call(endpoint: string, value?: Record<string, unknown>, base = API): Promise<Record<string, any>> {
    const body = value === undefined ? undefined : { value };
    return unwrap(await this.post(`${base}/${endpoint}`, body));
  }

  private sleep(ms: number): Promise<void> {
    return ms > 0 ? new Promise((resolve) => setTimeout(resolve, ms)) : Promise.resolve();
  }

  /** The account's watchlists (id and name). */
  async listWatchlists(): Promise<Watchlist[]> {
    const data = await this.call("getAllWatchlists.json");
    const lists = (data.watchlists ?? []) as { id?: unknown; name?: unknown }[];
    return lists.map((w) => ({ id: String(w.id ?? ""), name: String(w.name ?? "") }));
  }

  /** Create an empty watchlist; returns its id. */
  async createWatchlist(name: string): Promise<string> {
    const data = await this.call("createWL.json", { watchlistName: name });
    const id = data.watchlist?.watchlistID;
    if (id == null || id === "") throw new WatchlistApiError(`createWL returned no watchlistID: ${JSON.stringify(data)}`);
    return String(id);
  }

  async renameWatchlist(id: string, name: string): Promise<void> {
    await this.call("renameWL.json", { watchlistID: id, watchlistName: name });
  }

  async deleteWatchlist(id: string): Promise<void> {
    await this.call("deleteWL.json", { watchlistID: id });
  }

  /**
   * A list's column headers and entries, read with the collector's payload. `view` is the
   * account's custom column view (the collector sends its watchlist name).
   */
  async getEntries(id: string, view: string): Promise<WatchlistView> {
    let columnHeaders: string[] = [];
    const entries: WatchlistEntry[] = [];
    for (let start = 1; ; start += WatchlistClient.PAGE_SIZE) {
      const data = await this.call("getTSPWatchlistDetails.json", detailsRequest(id, view, start, WatchlistClient.PAGE_SIZE));
      const wl = data.watchListView ?? {};
      if (start === 1) columnHeaders = (wl.columnHeaders ?? []) as string[];
      const page = (wl.entryDetails ?? wl.columnValues ?? []) as Record<string, unknown>[];
      const symbolIdx = columnHeaders.indexOf("Symbol");
      for (const e of page) {
        // Entries carry symbol and entryID; else take the symbol from the Symbol column.
        const values = e.entryValuesList as unknown[] | undefined;
        const symbol = e.symbol ?? (symbolIdx >= 0 ? values?.[symbolIdx] : undefined);
        if (symbol == null || String(symbol).trim() === "") continue;
        entries.push({ symbol: normalizeSymbol(String(symbol)), entryID: String(e.entryID ?? "") });
      }
      if (page.length < WatchlistClient.PAGE_SIZE) break;
    }
    return { columnHeaders, entries };
  }

  /** Add symbols one call each (as the page does); skips none, so filter with planSync first. */
  async addSymbols(id: string, symbols: string[], view: string): Promise<AddResult> {
    const result: AddResult = { added: [], rejected: [], failed: [] };
    let failuresInARow = 0;
    for (const [i, raw] of symbols.entries()) {
      const symbol = normalizeSymbol(raw);
      try {
        const data = await this.call("createWatchListEntry.json", {
          isTSP: true,
          newIndexID: 1,
          positionType: 1,
          quantity: 100,
          symbol,
          viewName: view,
          watchlistID: id,
        });
        const created = data.watchlistEntry?.entries;
        if (Array.isArray(created) ? created.length > 0 : Boolean(created)) result.added.push(symbol);
        else result.rejected.push(symbol);
        failuresInARow = 0;
      } catch (error) {
        const e = error as WatchlistApiError;
        if (e.sessionExpired) {
          result.aborted = `session expired (HTTP ${e.status}) at ${symbol}`;
          break;
        }
        result.failed.push({ symbol, error: e.message });
        if (++failuresInARow >= WatchlistClient.MAX_CONSECUTIVE_FAILURES) {
          result.aborted = `${failuresInARow} failures in a row (last: ${e.message})`;
          break;
        }
      }
      if ((i + 1) % 50 === 0) this.log(`  ${i + 1}/${symbols.length} (added ${result.added.length})`);
      await this.sleep(this.delayMs);
    }
    return result;
  }

  /** Delete entries by id, several per call (comma-joined, as the page does). */
  async deleteEntries(id: string, entryIds: string[]): Promise<void> {
    for (let i = 0; i < entryIds.length; i += WatchlistClient.DELETE_BATCH) {
      const batch = entryIds.slice(i, i + WatchlistClient.DELETE_BATCH);
      await this.call("deleteWatchListEntry.json", { entryID: batch.join(","), watchlistID: id });
      await this.sleep(this.delayMs);
    }
  }

  /** Delete the given symbols from a list; returns the symbols that were not in it. */
  async removeSymbols(id: string, symbols: string[], view: string): Promise<{ removed: string[]; notFound: string[] }> {
    const { entries } = await this.getEntries(id, view);
    const bySymbol = new Map(entries.map((e) => [e.symbol, e.entryID]));
    const removed: string[] = [];
    const notFound: string[] = [];
    for (const s of symbols.map(normalizeSymbol)) (bySymbol.has(s) ? removed : notFound).push(s);
    await this.deleteEntries(id, removed.map((s) => bySymbol.get(s)!));
    return { removed, notFound };
  }

  /** Make a list hold `symbols`: add the missing ones and, with prune, delete the rest. */
  async sync(
    id: string,
    symbols: string[],
    view: string,
    { prune = false, dryRun = false } = {},
  ): Promise<{ add: string[]; remove: string[]; result?: AddResult }> {
    const { entries } = await this.getEntries(id, view);
    const plan = planSync(
      entries.map((e) => e.symbol),
      symbols,
    );
    const remove = prune ? plan.remove : [];
    if (dryRun) return { add: plan.add, remove };
    const result = await this.addSymbols(id, plan.add, view);
    if (remove.length && !result.aborted) {
      const bySymbol = new Map(entries.map((e) => [e.symbol, e.entryID]));
      await this.deleteEntries(id, remove.map((s) => bySymbol.get(s)!));
    }
    return { add: plan.add, remove, result };
  }

  /** The account's column views (name, and whether the account made it or it is built in). */
  async listViews(): Promise<{ name: string; custom: boolean }[]> {
    const data = unwrap(await this.post(`${VIEW_API}/getViewList/tspWatchlist.json`, undefined, "GET"));
    const views = (data.views ?? []) as { name?: unknown; customView?: unknown }[];
    return views.map((v) => ({ name: String(v.name ?? ""), custom: v.customView === true }));
  }

  /**
   * A custom view's columns, or null when the account has no view of that name: E*TRADE then
   * answers with its built-in "Market View" (as getTSPWatchlistDetails does for that name).
   */
  async getView(name: string): Promise<ColumnView | null> {
    const data = await this.call("getView.json", { isCustomView: true, viewName: name, viewSrcType: VIEW_SOURCE }, VIEW_API);
    const v = data.viewResponse ?? {};
    if (String(v.viewName ?? "") !== name) return null;
    return {
      name,
      custom: v.isCustomView === true,
      columnHeaderCodes: String(v.columnHeaderCodes ?? ""),
      columnHeaders: (v.columnHeaders ?? []) as string[],
      baseView: String(v.baseView ?? ""),
      viewSecType: String(v.viewSecType ?? ""),
    };
  }

  /**
   * Every column a view can hold: column name -> code. getAllColumns leaves some out (Actions,
   * 1000), so the account's custom views add the names and codes they hold.
   */
  async allColumns(): Promise<Map<string, number>> {
    const data = unwrap(await this.post(`${VIEW_API}/getAllColumns/tspWatchlist.json`, undefined, "GET"));
    const out = new Map<string, number>();
    for (const group of Object.values((data.Columns ?? {}) as Record<string, { columnCode?: unknown; columnName?: unknown }[]>)) {
      for (const c of group ?? []) if (c.columnName != null && c.columnCode != null) out.set(String(c.columnName), Number(c.columnCode));
    }
    for (const { name, custom } of await this.listViews()) {
      const view = custom ? await this.getView(name) : null;
      const codes = view?.columnHeaderCodes.split(",") ?? [];
      view?.columnHeaders.forEach((header, i) => {
        if (!out.has(header) && codes[i]) out.set(header, Number(codes[i]));
      });
    }
    return out;
  }

  /** Replace a custom view's columns (codes in display order), as the page's "customize view" does. */
  async setViewColumns(view: ColumnView, columnHeaderCodes: string): Promise<void> {
    checkColumnCount(columnHeaderCodes);
    await this.call(
      "updateView.json",
      {
        baseView: view.baseView,
        columnHeaderCodes,
        isCustomView: true,
        viewName: view.name,
        viewSecType: view.viewSecType,
        viewSrcType: VIEW_SOURCE,
      },
      VIEW_API,
    );
  }

  /** Create a custom view with these columns (codes in display order). */
  async createView(name: string, columnHeaderCodes: string): Promise<void> {
    checkColumnCount(columnHeaderCodes);
    await this.call(
      "createView.json",
      { baseView: "", columnHeaderCodes, isCustomView: true, viewName: name, viewSecType: "", viewSrcType: VIEW_SOURCE },
      VIEW_API,
    );
  }
}

function checkColumnCount(columnHeaderCodes: string): void {
  const n = columnHeaderCodes.split(",").filter(Boolean).length;
  if (n > WatchlistClient.MAX_VIEW_COLUMNS) {
    throw new WatchlistApiError(`a view holds at most ${WatchlistClient.MAX_VIEW_COLUMNS} columns, not ${n}`);
  }
}

/** Codes to append to `codes` so the view also holds `names`; throws on an unknown column name. */
export function addColumnCodes(codes: string, names: string[], catalog: Map<string, number>): string {
  const have = codes.split(",").map((c) => c.trim()).filter(Boolean);
  for (const name of names) {
    const code = catalog.get(name);
    if (code === undefined) throw new Error(`no column "${name}"; known: ${[...catalog.keys()].join(", ")}`);
    if (!have.includes(String(code))) have.push(String(code));
  }
  return have.join(",");
}

/** The getTSPWatchlistDetails request the collector sends (ETPageScraper.getWatchlistPayload). */
function detailsRequest(id: string, view: string, start: number, size: number): Record<string, unknown> {
  return {
    noteJSON: { noteType: "WATCHLIST", watchlistId: id },
    viewType: "portfolio",
    watchListRequest: {
      fromReactWatchLists: true,
      isCustomView: true,
      pagination: { startPosNum: start, posPerPage: size },
      viewName: view,
      sortBy: "0",
      sortOrder: "0",
      watchlistID: id,
    },
  };
}
