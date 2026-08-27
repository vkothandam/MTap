// Ported from MBin scrape/config/constants.js. E*TRADE endpoints, trading
// hours, poll intervals, watchlist config, and OTP delivery settings. Values
// that are account- or operator-specific (SMS destination, OTP host/port) are
// read from the environment rather than hardcoded; see .env.example.

export const URLS = {
  WATCHLIST_API: "https://us.etrade.com/webapipf/watchlist/getTSPWatchlistDetails.json",
  WATCHLIST_PAGE: "https://us.etrade.com/etx/pxy/watchlists#/wl",
  LOGIN_PAGE: "https://us.etrade.com/etx/pxy/login",
  WATCHLIST_CREATE: "https://us.etrade.com/webapipf/watchlist/createWatchListEntry.json",
  WATCHLIST_DELETE: "https://us.etrade.com/webapipf/watchlist/deleteWatchListEntry.json",
  MARKET_POSITIVE:
    "https://api.markitdigital.com/etrade-api/1.0/getNewsTable?rows=20&rankedType=Epctchg+&set=US",
  MARKET_NEGATIVE:
    "https://api.markitdigital.com/etrade-api/1.0/getNewsTable?rows=20&rankedType=Epctchg-&set=US",
  MARKET_MOVERS_PAGE: "https://www.etrade.wallst.com/etrade-web/markets/movers",
  QUOTE_URL: "https://api.etrade.com/v1/market/quote/",
} as const;

export const TRADING_HOURS = {
  START: { HOURS: 9, MINUTES: 30, SECONDS: 0 },
  END: { HOURS: 16, MINUTES: 0, SECONDS: 0 },
} as const;

export const INTERVALS = {
  REFRESH_DATA: 2000, // 2 seconds (main watchlist)
  REFRESH_PAGE: 300000, // 5 minutes
  REFRESH_MOVERS: 5000,
} as const;

/**
 * Watchlist configuration. Each watchlist has an ID, display name, polling
 * interval, and CSV prefix. Account-specific IDs; Step 4 (WatchlistManager)
 * makes these command/input-driven, at which point they can move to config.
 */
export interface WatchlistConfig {
  id: string;
  name: string;
  intervalMs: number;
  csvPrefix: string;
}

export const WATCHLISTS: Record<string, WatchlistConfig> = {
  // Main watchlist - polled every 2 seconds
  MAIN: { id: "215565674306", name: "Experimental", intervalMs: 2000, csvPrefix: "" },
  // Mid-cap watchlist - polled every 20 seconds
  MIDCAP: { id: "215584661306", name: "MidCap", intervalMs: 20000, csvPrefix: "MC" },
};

/**
 * OTP delivery. The destination number/carrier (where the OTP link is texted)
 * is operator- and account-specific, so it comes from the environment. The
 * selection priority is the last-4 order used to pick which E*TRADE-registered
 * phone to challenge; overridable via OTP_PHONE_PRIORITY (comma-separated).
 */
export const PHONE_CONFIG = {
  PERSONAL_NUMBER: process.env.SMS_PHONE_NUMBER ?? "",
  CARRIER: process.env.SMS_CARRIER ?? "verizon",
  OTP_TIMEOUT: Number(process.env.OTP_TIMEOUT ?? 300000), // 5 minutes
  SELECT_PRIORITY:
    process.env.OTP_PHONE_PRIORITY?.split(",")
      .map((s) => s.trim())
      .filter(Boolean) ?? ["9699", "5646", "5647"],
};

/**
 * The OTP entry page is served by MTap itself (otpServer.ts). OTP_HOST is
 * embedded in the SMS link, so it must be reachable from the device that opens
 * it — use the machine's LAN IP (not 127.0.0.1) when reading it on a phone.
 */
const OTP_HOST = process.env.OTP_HOST ?? "127.0.0.1";
const OTP_PORT = Number(process.env.OTP_PORT ?? 8787);

export const SERVER_CONFIG = {
  HOST: OTP_HOST,
  PORT: OTP_PORT,
  BASE_URL: `http://${OTP_HOST}:${OTP_PORT}`,
  OTP_PATH: "/auth/otp",
} as const;
