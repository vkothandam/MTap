import { createWriteStream, existsSync, mkdirSync, type WriteStream } from "node:fs";
import { join } from "node:path";
import { REPO_ROOT } from "../../../common/config.js";
import { isDebugEnabled } from "../debug.js";
import { TimeUtils } from "./TimeUtils.js";

// Unified file-based logging. Ported from MBin scrape/utils/FileLogger.js.
// Logs land under state/etrade/logs (gitignored); console output is limited to
// warnings/errors unless consoleErrorsOnly is off.

const DEFAULT_LOG_DIR = join(REPO_ROOT, "state", "etrade", "logs");

export interface FileLoggerOptions {
  moduleName?: string;
  configSection?: string | null;
  logDir?: string;
  consoleErrorsOnly?: boolean;
}

type LogLevel = "INFO" | "WARN" | "ERROR" | "DEBUG" | "FATAL";

export class FileLogger {
  private moduleName: string;
  private configSection: string | null;
  private logDir: string;
  private enabled: boolean;
  private consoleErrorsOnly: boolean;
  private currentLogFile: string | null = null;
  private writeStream: WriteStream | null = null;

  constructor(options: FileLoggerOptions = {}) {
    this.moduleName = options.moduleName ?? "App";
    this.configSection = options.configSection ?? null;
    this.logDir = options.logDir ?? DEFAULT_LOG_DIR;

    this.enabled = this.configSection ? isDebugEnabled(this.configSection) : true;
    this.consoleErrorsOnly =
      options.consoleErrorsOnly !== undefined
        ? options.consoleErrorsOnly
        : this.configSection
          ? isDebugEnabled(this.configSection, "CONSOLE_ERRORS_ONLY")
          : true;

    if (this.enabled && !existsSync(this.logDir)) {
      mkdirSync(this.logDir, { recursive: true });
    }

    this.initLogFile();
  }

  private initLogFile(): void {
    if (!this.enabled) return;

    const date = TimeUtils.getLocalDateString();
    const logFileName = `${this.moduleName.toLowerCase()}-${date}.log`;
    const logFilePath = join(this.logDir, logFileName);

    if (this.currentLogFile !== logFilePath) {
      if (this.writeStream) this.writeStream.end();
      this.currentLogFile = logFilePath;
      this.writeStream = createWriteStream(logFilePath, { flags: "a" });
      this.writeToFile("=".repeat(80));
      this.writeToFile(`[${this.moduleName}] SESSION STARTED: ${TimeUtils.getLocalISOStringWithTZ()}`);
      this.writeToFile("=".repeat(80));
    }
  }

  private writeToFile(message: string): void {
    if (this.writeStream) this.writeStream.write(message + "\n");
  }

  log(level: LogLevel, message: string, data: unknown = null): void {
    if (!this.enabled) return;
    this.initLogFile();

    const timestamp = TimeUtils.getLocalISOStringWithTZ();
    let logLine = `[${timestamp}][${level}] ${message}`;
    if (data !== null) {
      logLine += typeof data === "object" ? " " + JSON.stringify(data) : ` ${String(data)}`;
    }

    this.writeToFile(logLine);

    const isError = level === "ERROR" || level === "WARN" || level === "FATAL";
    if (isError || !this.consoleErrorsOnly) {
      const prefix = `[${this.moduleName}]`;
      if (level === "ERROR" || level === "FATAL") console.error(`${prefix} ${message}`);
      else if (level === "WARN") console.warn(`${prefix} ${message}`);
      else console.log(`${prefix} ${message}`);
    }
  }

  info(message: string, data: unknown = null): void {
    this.log("INFO", message, data);
  }

  debug(message: string, data: unknown = null): void {
    this.log("DEBUG", message, data);
  }

  warn(message: string, data: unknown = null): void {
    this.log("WARN", message, data);
  }

  error(message: string, data: unknown = null): void {
    this.log("ERROR", message, data);
  }

  logError(context: string, error: Error, additionalData: Record<string, unknown> = {}): void {
    this.log("ERROR", context, {
      context,
      message: error.message,
      name: error.name,
      stack: error.stack,
      ...additionalData,
    });
  }

  close(): void {
    if (this.writeStream) {
      this.writeToFile("=".repeat(80));
      this.writeToFile(`[${this.moduleName}] SESSION ENDED: ${TimeUtils.getLocalISOStringWithTZ()}`);
      this.writeToFile("=".repeat(80));
      this.writeStream.end();
      this.writeStream = null;
    }
  }
}

export default FileLogger;
