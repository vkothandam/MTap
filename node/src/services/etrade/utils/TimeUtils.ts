import { TRADING_HOURS } from "../constants.js";

/**
 * Centralized local-time handling. Ported from MBin scrape/utils/TimeUtils.js.
 *
 * IMPORTANT: use these instead of toISOString()/split — toISOString() returns
 * UTC, which lands on the wrong calendar day when local time differs from UTC.
 */
export class TimeUtils {
  // ==================== LOCAL DATE/TIME FORMATTING ====================

  /** Local date as YYYY-MM-DD. */
  static getLocalDateString(date: Date = new Date()): string {
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, "0");
    const day = String(date.getDate()).padStart(2, "0");
    return `${year}-${month}-${day}`;
  }

  /** Local time as HH-MM-SS (filename-safe). */
  static getLocalTimeString(date: Date = new Date()): string {
    const hours = String(date.getHours()).padStart(2, "0");
    const minutes = String(date.getMinutes()).padStart(2, "0");
    const seconds = String(date.getSeconds()).padStart(2, "0");
    return `${hours}-${minutes}-${seconds}`;
  }

  /** Local datetime as YYYY-MM-DD HH:MM:SS. */
  static getLocalDateTimeString(date: Date = new Date()): string {
    const dateStr = this.getLocalDateString(date);
    const hours = String(date.getHours()).padStart(2, "0");
    const minutes = String(date.getMinutes()).padStart(2, "0");
    const seconds = String(date.getSeconds()).padStart(2, "0");
    return `${dateStr} ${hours}:${minutes}:${seconds}`;
  }

  /** ISO-like timestamp in local time (YYYY-MM-DDTHH:MM:SS.sss). */
  static getLocalISOString(date: Date = new Date()): string {
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, "0");
    const day = String(date.getDate()).padStart(2, "0");
    const hours = String(date.getHours()).padStart(2, "0");
    const minutes = String(date.getMinutes()).padStart(2, "0");
    const seconds = String(date.getSeconds()).padStart(2, "0");
    const ms = String(date.getMilliseconds()).padStart(3, "0");
    return `${year}-${month}-${day}T${hours}:${minutes}:${seconds}.${ms}`;
  }

  /** Timezone offset string, e.g. "-05:00". */
  static getTimezoneOffset(date: Date = new Date()): string {
    const offset = -date.getTimezoneOffset();
    const sign = offset >= 0 ? "+" : "-";
    const hours = String(Math.floor(Math.abs(offset) / 60)).padStart(2, "0");
    const minutes = String(Math.abs(offset) % 60).padStart(2, "0");
    return `${sign}${hours}:${minutes}`;
  }

  /** Full local ISO string with timezone offset. */
  static getLocalISOStringWithTZ(date: Date = new Date()): string {
    return `${this.getLocalISOString(date)}${this.getTimezoneOffset(date)}`;
  }

  // ==================== TRADING HOURS CALCULATIONS ====================

  static calculateStartOfTradeEpoch(startTime: number | string | Date): number {
    const dateObj = new Date(startTime);
    const options = { timeZone: "America/New_York" };
    const formattedDate = dateObj.toLocaleString("en-US", options);
    const nyDate = new Date(formattedDate);

    const deltaNY = nyDate.getTime();
    nyDate.setHours(
      TRADING_HOURS.START.HOURS,
      TRADING_HOURS.START.MINUTES,
      TRADING_HOURS.START.SECONDS,
    );

    return dateObj.getTime() - (deltaNY - nyDate.getTime());
  }

  /**
   * Epoch (ms) for today's market close (4:00 PM ET), expressed against local
   * time. Ported verbatim from MBin.
   */
  static calculateEndOfTradeTodaysEpoch(): number {
    const localDate = new Date();
    localDate.setHours(0, 0, 0, 0);

    const targetET = new Date(localDate);
    targetET.setHours(TRADING_HOURS.END.HOURS, TRADING_HOURS.END.MINUTES, TRADING_HOURS.END.SECONDS);

    const etOptions = { timeZone: "America/New_York" };
    const etTimeString = targetET.toLocaleString("en-US", etOptions);
    const etTime = new Date(etTimeString);

    const offsetHours = etTime.getHours() - targetET.getHours();
    targetET.setHours(TRADING_HOURS.END.HOURS - offsetHours);

    return targetET.getTime();
  }
}
