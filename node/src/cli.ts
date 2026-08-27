#!/usr/bin/env node
// CLI entry point:
//   sourcing-node run <source>    one-shot batch source (fetch -> extract -> write)
//   sourcing-node serve <service> long-running service (e.g. the E*TRADE scraper)
import { loadSource } from "./common/config.js";
import { createSource } from "./registry.js";
import { createService } from "./serviceRegistry.js";

async function runSource(name: string): Promise<number> {
  loadSource(name); // validate it's registered before constructing
  const source = createSource(name);
  const path = await source.run();
  console.log(`wrote ${path}`);
  return 0;
}

async function serveService(name: string): Promise<number> {
  const service = createService(name);
  let stopping = false;
  const shutdown = async (signal: string): Promise<void> => {
    if (stopping) return;
    stopping = true;
    console.log(`\n[${name}] ${signal} — shutting down`);
    try {
      await service.stop();
    } catch (err) {
      console.error(err);
    }
    process.exit(0);
  };
  process.on("SIGINT", () => void shutdown("SIGINT"));
  process.on("SIGTERM", () => void shutdown("SIGTERM"));

  await service.start();
  console.log(`[${name}] serving — Ctrl-C to stop`);
  await new Promise<never>(() => {}); // run until a signal triggers shutdown()
  return 0; // unreachable
}

async function main(argv: string[]): Promise<number> {
  const [verb, name] = argv;
  if (argv.length === 2 && verb === "run") return runSource(name);
  if (argv.length === 2 && verb === "serve") return serveService(name);
  console.error("usage: sourcing-node run <source> | serve <service>");
  return 2;
}

main(process.argv.slice(2)).then(
  (code) => process.exit(code),
  (err) => {
    console.error(err);
    process.exit(1);
  },
);
