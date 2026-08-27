import { Bus, getBus } from "./bus/Bus.js";

// Long-running counterpart to Source. Where a Source is a one-shot batch
// (fetch -> extract -> write) driven by `sourcing-node run`, a Service owns a
// persistent resource (e.g. a browser session), reacts to bus events, and
// publishes on the bus. Registered in serviceRegistry.ts and launched with
// `sourcing-node serve <name>`. This is additive: the batch Source path is
// untouched.
export abstract class Service {
  /** Registry name; must match a key in serviceRegistry.ts. */
  abstract readonly name: string;

  protected bus: Bus;

  constructor(bus?: Bus) {
    this.bus = bus ?? getBus();
  }

  /** Acquire resources and begin work (subscribe to events, start polling). */
  abstract start(): Promise<void>;

  /** Release resources for a graceful shutdown. */
  abstract stop(): Promise<void>;
}
