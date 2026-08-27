import axios, { type AxiosRequestConfig } from "axios";
import { URLS } from "./constants.js";
import { CookieUtils } from "./utils/CookieUtils.js";
import { BrowserGuard } from "./utils/BrowserGuard.js";
import { EVENTS } from "../../common/bus/events.js";
import type { Bus } from "../../common/bus/Bus.js";
import type { StoredCookie } from "../../common/session.js";
import { polledDataKey } from "./keygen.js";
import { agentLog } from "./debug.js";

// API client for watchlist + market-mover data. Ported from MBin
// scrape/modules/ETPageScraper.js. Changes for MTap:
//   - Publishes POLLED_DATA on the MTap Bus (not the EventBroker), stamped with
//     a per-(runId, seq) idempotency key so MBin dedups at-least-once Redis
//     redelivery. The payload shape is unchanged — it is the frozen inter-repo
//     contract (see docs).
//   - Reuses BrowserGuard's transient-error patterns for retry classification.

export interface WatchlistConfig {
  id: string;
  name: string;
  csvPrefix: string;
}

type RequestHeaders = Record<string, string | undefined>;

interface ApiError {
  message?: string;
  code?: string;
  response?: { status?: number; statusText?: string };
}

function asApiError(error: unknown): ApiError {
  return (error ?? {}) as ApiError;
}

export class ETPageScraper {
  static MAX_RETRIES = 3;
  static RETRY_DELAY_MS = 2000;
  static DEFAULT_WATCHLIST_ID = "215565674306";
  static DEFAULT_WATCHLIST_NAME = "Experimental";
  static MAX_ENTRIES_PER_PAGE = 1000; // handle 619+ entries in a single call

  private requestHeaders: RequestHeaders;
  private bus: Bus | null;
  private readonly runId: string;
  private seq = 0;
  private consecutiveFailures = 0;
  private readonly watchlistId: string;
  private readonly watchlistName: string;
  private readonly csvPrefix: string;

  constructor(
    requestHeaders: RequestHeaders,
    bus: Bus | null,
    runId: string,
    watchlistConfig?: WatchlistConfig,
  ) {
    this.requestHeaders = requestHeaders;
    this.bus = bus;
    this.runId = runId;
    this.watchlistId = watchlistConfig?.id ?? ETPageScraper.DEFAULT_WATCHLIST_ID;
    this.watchlistName = watchlistConfig?.name ?? ETPageScraper.DEFAULT_WATCHLIST_NAME;
    this.csvPrefix = watchlistConfig?.csvPrefix ?? "";
  }

  /** Deterministic idempotency key for the next publish (monotonic seq per instance). */
  private nextKey(type: string, scope?: string): string {
    return polledDataKey({ runId: this.runId, seq: ++this.seq, type, scope });
  }

  isTransientError(error: unknown): boolean {
    const e = asApiError(error);
    const errorMessage = (e.message ?? "").toLowerCase();
    const errorCode = e.code ?? "";
    return BrowserGuard.TRANSIENT_NETWORK_ERRORS.some(
      (pattern) => errorMessage.includes(pattern.toLowerCase()) || errorCode === pattern,
    );
  }

  isSessionError(error: unknown): boolean {
    const e = asApiError(error);
    return e.response?.status === 401 || e.response?.status === 403 || e.message === "SESSION_EXPIRED";
  }

  private delay(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  private emitSessionExpired(source: string, reason: string, error: unknown): void {
    const e = asApiError(error);
    this.bus?.publish(EVENTS.SESSION_EXPIRED, {
      timestamp: Date.now(),
      source,
      reason,
      errorMessage: e.message,
      status: e.response?.status,
      consecutiveFailures: this.consecutiveFailures,
    });
  }

  /** Execute an API call, retrying transient network errors with backoff. */
  async executeWithRetry<T>(apiCall: () => Promise<T>, context: string): Promise<T> {
    let lastError: unknown;

    for (let attempt = 1; attempt <= ETPageScraper.MAX_RETRIES; attempt++) {
      try {
        const result = await apiCall();
        this.consecutiveFailures = 0;
        return result;
      } catch (error) {
        lastError = error;

        // Session errors propagate immediately (not retried).
        if (this.isSessionError(error)) throw error;

        if (this.isTransientError(error) && attempt < ETPageScraper.MAX_RETRIES) {
          const delayMs = ETPageScraper.RETRY_DELAY_MS * attempt;
          console.log(
            `[ETPageScraper] ${context} - transient error (${asApiError(error).message}), retrying in ${delayMs}ms (attempt ${attempt}/${ETPageScraper.MAX_RETRIES})`,
          );
          await this.delay(delayMs);
          continue;
        }

        throw error;
      }
    }

    throw lastError;
  }

  private handleApiError(error: unknown, source: string): void {
    this.consecutiveFailures++;
    const e = asApiError(error);

    agentLog(
      `ETPageScraper:${source}`,
      "API_CALL_FAILED",
      {
        status: e.response?.status,
        statusText: e.response?.statusText,
        errorMessage: e.message,
        consecutiveFailures: this.consecutiveFailures,
      },
      "H1,H2,H3,H4,H5",
    );

    if (this.isSessionError(error)) {
      console.error(`[ETPageScraper] Session expired from ${source}`);
      this.emitSessionExpired(source, "AUTH_ERROR", error);
    } else {
      console.error(`[ETPageScraper] API call failed from ${source}: ${e.message}`);
      this.emitSessionExpired(source, "NETWORK_ERROR", error);
    }
  }

  async getWatchlist(cookies: StoredCookie[]): Promise<unknown> {
    const cookie = CookieUtils.formatCookiesForHeader(cookies);

    try {
      const result = await this.executeWithRetry(async () => {
        const response = await axios.post(URLS.WATCHLIST_API, this.getWatchlistPayload(), this.getRequestConfig(cookie));
        if (response.status === 401 || response.status === 403) {
          throw new Error("SESSION_EXPIRED");
        }
        return response.data;
      }, "getWatchlist");

      // Normalize: expose both entryDetails and columnValues for downstream consumers.
      const watchListView = result?.data?.watchListView;
      if (watchListView) {
        if (watchListView.entryDetails && !watchListView.columnValues) {
          watchListView.columnValues = watchListView.entryDetails;
        }
        if (watchListView.columnValues && !watchListView.entryDetails) {
          watchListView.entryDetails = watchListView.columnValues;
        }
      }

      this.publishWatchlist(result.data);
      return result.data;
    } catch (error) {
      // A 400 usually means the single large page was rejected; fall back to pagination.
      if (asApiError(error).response?.status === 400) {
        console.log("[ETPageScraper] Single call failed with 400, trying paginated fallback...");
        try {
          const paginatedResult = await this.getWatchlistPaginated(cookie);
          this.publishWatchlist(paginatedResult);
          return paginatedResult;
        } catch (paginationError) {
          this.handleApiError(paginationError, "watchlist-paginated");
          throw paginationError;
        }
      }

      this.handleApiError(error, "watchlist");
      throw error;
    }
  }

  private publishWatchlist(data: unknown): void {
    this.bus?.publish(
      EVENTS.POLLED_DATA,
      {
        type: "watchlist",
        data,
        watchlistId: this.watchlistId,
        watchlistName: this.watchlistName,
        csvPrefix: this.csvPrefix,
      },
      { idempotencyKey: this.nextKey("watchlist", this.watchlistId) },
    );
  }

  /** Fallback: fetch the watchlist page-by-page when a single large call fails. */
  async getWatchlistPaginated(cookie: string, pageSize = 200): Promise<Record<string, unknown>> {
    const allEntries: unknown[] = [];
    let startPos = 1;
    let hasMore = true;
    let pageCount = 0;
    let columnHeaders: unknown = null;

    while (hasMore) {
      try {
        pageCount++;
        const response = await axios.post(
          URLS.WATCHLIST_API,
          this.getWatchlistPayloadPaginated(startPos, pageSize),
          this.getRequestConfig(cookie),
        );

        if (response.status === 401 || response.status === 403) {
          throw new Error("SESSION_EXPIRED");
        }

        const watchListView = response.data?.data?.watchListView;
        const entries: unknown[] = watchListView?.entryDetails ?? [];
        allEntries.push(...entries);

        if (pageCount === 1 && watchListView?.columnHeaders) {
          columnHeaders = watchListView.columnHeaders;
        }

        if (entries.length < pageSize) {
          hasMore = false;
        } else {
          startPos += pageSize;
          await this.delay(100);
        }
      } catch (error) {
        if (this.isSessionError(error)) throw error;
        console.error(`[ETPageScraper] Pagination page ${pageCount} failed:`, asApiError(error).message);
        hasMore = false;
      }
    }

    return {
      watchListView: {
        columnHeaders,
        entryDetails: allEntries,
        columnValues: allEntries,
      },
    };
  }

  getWatchlistPayloadPaginated(startPos: number, pageSize: number): Record<string, unknown> {
    return {
      value: {
        noteJSON: { noteType: "WATCHLIST", watchlistId: this.watchlistId },
        viewType: "portfolio",
        watchListRequest: {
          fromReactWatchLists: true,
          isCustomView: true,
          pagination: { startPosNum: startPos, posPerPage: pageSize },
          viewName: this.watchlistName,
          sortBy: "0",
          sortOrder: "0",
          watchlistID: this.watchlistId,
        },
      },
    };
  }

  async getMarketMoversUp(cookies: StoredCookie[]): Promise<unknown> {
    return this.getMarketMovers(cookies, "up", URLS.MARKET_POSITIVE, "marketMoversUp");
  }

  async getMarketMoversDown(cookies: StoredCookie[]): Promise<unknown> {
    return this.getMarketMovers(cookies, "down", URLS.MARKET_NEGATIVE, "marketMoversDown");
  }

  private async getMarketMovers(
    cookies: StoredCookie[],
    direction: "up" | "down",
    url: string,
    source: string,
  ): Promise<unknown> {
    // Format cookies into a header (MBin passed the raw array here — a latent bug;
    // fixed to format consistently with getWatchlist).
    const cookie = CookieUtils.formatCookiesForHeader(cookies);
    try {
      const result = await this.executeWithRetry(async () => {
        const response = await axios.get(url, {
          headers: {
            ...(this.getRequestConfig(cookie).headers as Record<string, string>),
            Authorization: this.requestHeaders.Authorization ?? "",
          },
        });
        if (response.status === 401 || response.status === 403) {
          throw new Error("SESSION_EXPIRED");
        }
        return response.data;
      }, source);

      this.bus?.publish(
        EVENTS.POLLED_DATA,
        { type: "marketMovers", direction, data: result },
        { idempotencyKey: this.nextKey("marketMovers", direction) },
      );
      return result;
    } catch (error) {
      this.handleApiError(error, source);
      throw error;
    }
  }

  getWatchlistPayload(): Record<string, unknown> {
    return {
      value: {
        noteJSON: { noteType: "WATCHLIST", watchlistId: this.watchlistId },
        viewType: "portfolio",
        watchListRequest: {
          fromReactWatchLists: true,
          isCustomView: true,
          pagination: { startPosNum: 1, posPerPage: ETPageScraper.MAX_ENTRIES_PER_PAGE },
          viewName: this.watchlistName,
          sortBy: "0",
          sortOrder: "0",
          watchlistID: this.watchlistId,
        },
      },
    };
  }

  getRequestConfig(cookie: string): AxiosRequestConfig {
    return {
      headers: {
        "Content-Type": "application/json",
        Cookie: cookie,
        stk1: this.requestHeaders.stk1 ?? "",
        referer: "https://us.etrade.com/etx/pxy/watchlists",
        origin: "https://us.etrade.com",
        host: "us.etrade.com",
        "x-requested-with": "XMLHttpRequest",
      },
    };
  }
}
