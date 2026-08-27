// Maps a service name to its Service class. Parallels registry.ts (for batch
// Sources). Long-running services are added here as they land.
import type { Service } from "./common/service.js";
import { EtradeService } from "./services/etrade/EtradeService.js";

const SERVICES: Record<string, new () => Service> = {
  etrade: EtradeService,
};

export function createService(name: string): Service {
  const Ctor = SERVICES[name];
  if (!Ctor) {
    const available = Object.keys(SERVICES).sort().join(", ") || "(none)";
    throw new Error(`No service registered for '${name}'. Available: ${available}`);
  }
  return new Ctor();
}
