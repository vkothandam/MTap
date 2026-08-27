// Event names shared on the wire with MBin (core/events.js). The string VALUES
// must match MBin exactly for cross-repo delivery; the keys are for local ergonomics.
export const EVENTS = {
  POLLED_DATA: "POLLED_DATA", // MTap -> MBin: raw watchlist + market-movers payloads
  WATCHLIST_CMD: "WATCHLIST_CMD", // MBin -> MTap (+ reply): watchlist CRUD
  AUTH_OTP_REQUIRED: "authOtpRequired", // internal to MTap (optional read-only mirror to MBin)
  AUTH_OTP_SUBMITTED: "authOtpSubmitted", // internal to MTap
  SESSION_EXPIRED: "sessionExpired",
  SESSION_REFRESHED: "sessionRefreshed",
} as const;

export type EventName = (typeof EVENTS)[keyof typeof EVENTS];
