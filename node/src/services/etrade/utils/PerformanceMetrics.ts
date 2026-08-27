import { existsSync, mkdirSync, renameSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { REPO_ROOT } from "../../../common/config.js";
import { FileLogger } from "./FileLogger.js";
import { TimeUtils } from "./TimeUtils.js";

// Tracks and periodically persists operation timings. Ported from MBin
// scrape/utils/PerformanceMetrics.js. Not on the core scraper path (used by the
// industry scraper / transformers); ported for parity.

const DEFAULT_OUTPUT_DIR = join(REPO_ROOT, "state", "etrade", "metrics");

export interface PerformanceMetricsOptions {
  moduleName?: string;
  outputPath?: string;
  flushInterval?: number;
  enabled?: boolean;
}

interface OperationStats {
  count: number;
  totalTime: number;
  minTime: number;
  maxTime: number;
  avgTime: number;
  errors: number;
}

interface ErrorRecord {
  operation: string;
  message: string;
  stack?: string;
  timestamp: string;
}

export class PerformanceMetrics {
  private moduleName: string;
  private outputPath: string;
  private flushInterval: number;
  private enabled: boolean;
  private logger: FileLogger;
  private flushTimer?: ReturnType<typeof setInterval>;

  private metrics: {
    operations: Map<string, OperationStats>;
    errors: ErrorRecord[];
    startTime: number;
    lastFlush: number;
  };

  constructor(options: PerformanceMetricsOptions = {}) {
    this.moduleName = options.moduleName ?? "UnknownModule";
    this.outputPath = options.outputPath ?? DEFAULT_OUTPUT_DIR;
    this.flushInterval = options.flushInterval ?? 30000;
    this.enabled = options.enabled !== false;

    this.logger = new FileLogger({
      moduleName: "PerformanceMetrics",
      configSection: "PERFORMANCE_METRICS",
    });

    this.metrics = {
      operations: new Map(),
      errors: [],
      startTime: Date.now(),
      lastFlush: Date.now(),
    };

    if (this.enabled && !existsSync(this.outputPath)) {
      mkdirSync(this.outputPath, { recursive: true });
    }

    if (this.enabled) {
      this.flushTimer = setInterval(() => this.flush(), this.flushInterval);
    }
  }

  async measure<T>(operationName: string, fn: () => Promise<T>): Promise<T> {
    if (!this.enabled) return fn();

    const startTime = process.hrtime.bigint();
    let error: Error | null = null;
    try {
      return await fn();
    } catch (err) {
      error = err as Error;
      this.recordError(operationName, error);
      throw err;
    } finally {
      const durationMs = Number(process.hrtime.bigint() - startTime) / 1_000_000;
      this.recordMetric(operationName, durationMs, error);
    }
  }

  recordMetric(operationName: string, durationMs: number, error: Error | null = null): void {
    if (!this.enabled) return;

    let metric = this.metrics.operations.get(operationName);
    if (!metric) {
      metric = { count: 0, totalTime: 0, minTime: Infinity, maxTime: -Infinity, avgTime: 0, errors: 0 };
      this.metrics.operations.set(operationName, metric);
    }

    metric.count++;
    metric.totalTime += durationMs;
    metric.minTime = Math.min(metric.minTime, durationMs);
    metric.maxTime = Math.max(metric.maxTime, durationMs);
    metric.avgTime = metric.totalTime / metric.count;
    if (error) metric.errors++;
  }

  recordError(operationName: string, error: Error): void {
    if (!this.enabled) return;

    this.metrics.errors.push({
      operation: operationName,
      message: error.message,
      stack: error.stack,
      timestamp: TimeUtils.getLocalISOStringWithTZ(),
    });

    if (this.metrics.errors.length > 100) {
      this.metrics.errors = this.metrics.errors.slice(-100);
    }
  }

  getMetrics(): Record<string, unknown> {
    const uptimeMs = Date.now() - this.metrics.startTime;
    const operations: Record<string, unknown> = {};
    for (const [name, data] of this.metrics.operations) {
      operations[name] = {
        count: data.count,
        totalTimeMs: parseFloat(data.totalTime.toFixed(3)),
        avgTimeMs: parseFloat(data.avgTime.toFixed(3)),
        minTimeMs: parseFloat(data.minTime.toFixed(3)),
        maxTimeMs: parseFloat(data.maxTime.toFixed(3)),
        errors: data.errors,
        throughput: data.count / (uptimeMs / 1000),
      };
    }

    return {
      module: this.moduleName,
      uptime: { ms: uptimeMs, seconds: parseFloat((uptimeMs / 1000).toFixed(2)) },
      operations,
      totalErrors: this.metrics.errors.length,
      lastFlush: TimeUtils.getLocalISOStringWithTZ(new Date(this.metrics.lastFlush)),
      timestamp: TimeUtils.getLocalISOStringWithTZ(),
    };
  }

  flush(): void {
    if (!this.enabled) return;
    try {
      const metricsData = this.getMetrics();
      const filename = `${this.moduleName}_metrics_${TimeUtils.getLocalDateString()}.json`;
      const filepath = join(this.outputPath, filename);
      const tempPath = filepath + ".tmp";
      writeFileSync(tempPath, JSON.stringify(metricsData, null, 2));
      renameSync(tempPath, filepath);
      this.metrics.lastFlush = Date.now();
      this.logger.info(`Flushed metrics for ${this.moduleName} to ${filename}`);
    } catch (error) {
      this.logger.logError("Error flushing metrics", error as Error);
    }
  }

  reset(): void {
    this.metrics = { operations: new Map(), errors: [], startTime: Date.now(), lastFlush: Date.now() };
  }

  destroy(): void {
    if (this.flushTimer) clearInterval(this.flushTimer);
    this.flush();
    this.logger.info(`${this.moduleName} metrics stopped`);
    this.logger.close();
  }
}

export default PerformanceMetrics;
