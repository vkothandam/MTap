// E*TRADE scraper runtime config, read from the process environment. Replaces
// MBin's config/environment.js `ENV` (which hardcoded secrets). Only the
// scraper-relevant subset lives here: browser-login credentials, the SMS
// gateway account, and browser runtime knobs. MBin's OAuth trading credentials
// (consumer key/secret) stay in MBin — the scraper authenticates via browser
// login, not OAuth. Never commit real values; see .env.example.

const flag = (value: string | undefined, dflt: boolean): boolean =>
  value === undefined ? dflt : value === "true";

export const ENV = {
  // E*TRADE browser-login credentials (username/password typed into the login form).
  ET_USERNAME: process.env.ET_USERNAME ?? "",
  ET_PASSWORD: process.env.ET_PASSWORD ?? "",

  // Email->SMS gateway account used to text the OTP entry link.
  SMS_EMAIL: process.env.SMS_EMAIL ?? "",
  SMS_EMAIL_PASSWORD: process.env.SMS_EMAIL_PASSWORD ?? "",

  // Feature flags (mirror MBin; SCRAPING defaults on for a service that exists to scrape).
  FEATURES: {
    SCRAPING: flag(process.env.FEATURE_SCRAPING, true),
  },

  // Browser runtime. HEADLESS also honours MBin's legacy BROWSER_HEADLESS name.
  HEADLESS: flag(process.env.HEADLESS, false) || process.env.BROWSER_HEADLESS === "true",
  PUPPETEER_EXECUTABLE_PATH: process.env.PUPPETEER_EXECUTABLE_PATH,
};

/** True when the minimum credentials for a browser login are present. */
export function hasLoginCredentials(): boolean {
  return Boolean(ENV.ET_USERNAME && ENV.ET_PASSWORD);
}
