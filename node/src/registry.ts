// Maps a source name to its Source class. Add new sources here.
import type { Source } from "./common/source.js";
import { ExampleScrapeSource } from "./sources/example_scrape/source.js";

const SOURCES: Record<string, new () => Source> = {
  example_scrape: ExampleScrapeSource,
};

export function createSource(name: string): Source {
  const Ctor = SOURCES[name];
  if (!Ctor) {
    const available = Object.keys(SOURCES).sort().join(", ") || "(none)";
    throw new Error(`No Node source registered for '${name}'. Available: ${available}`);
  }
  return new Ctor();
}
