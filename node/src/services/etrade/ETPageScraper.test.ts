import { test } from "node:test";
import assert from "node:assert/strict";
import { ETPageScraper } from "./ETPageScraper.js";

const request = (s: ETPageScraper): Record<string, unknown> =>
  (s.getWatchlistPayload().value as { watchListRequest: Record<string, unknown> }).watchListRequest;

test("the poll asks for the configured view, else the watchlist name", () => {
  const withView = new ETPageScraper({}, null, "r", { id: "1", name: "Experimental", view: "MTapFeed", csvPrefix: "" });
  assert.equal(request(withView).viewName, "MTapFeed");
  assert.equal(request(withView).watchlistID, "1");
  const paged = withView.getWatchlistPayloadPaginated(201, 200).value as { watchListRequest: Record<string, unknown> };
  assert.equal(paged.watchListRequest.viewName, "MTapFeed");

  const plain = new ETPageScraper({}, null, "r", { id: "2", name: "MidCap", csvPrefix: "MC" });
  assert.equal(request(plain).viewName, "MidCap");
});

test("published rows keep the watchlist name, not the view", async () => {
  const published: Record<string, unknown>[] = [];
  const bus = { publish: (_e: string, payload: Record<string, unknown>) => published.push(payload) };
  const s = new ETPageScraper({}, bus as never, "r", { id: "1", name: "Experimental", view: "MTapFeed", csvPrefix: "" });
  (s as unknown as { publishWatchlist(d: unknown): void }).publishWatchlist({});
  assert.equal(published[0].watchlistName, "Experimental");
});
