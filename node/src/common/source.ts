// Base class for a Node source. Subclasses implement fetch() and extract().
import { loadSource, type SourceConfig } from "./config.js";
import { loadSchema, validateRecords } from "./schema.js";
import { write, type Record_ } from "./writer.js";

export abstract class Source {
  /** Registry name; must match a key in shared/config/sources.json. */
  abstract readonly name: string;

  protected _config?: SourceConfig;

  get config(): SourceConfig {
    this._config ??= loadSource(this.name);
    return this._config;
  }

  /** Retrieve raw data (call an API, download a page). */
  abstract fetch(): Promise<unknown>;

  /** Turn the raw payload into records conforming to the schema. */
  abstract extract(raw: unknown): Promise<Record_[]> | Record_[];

  async run({ validate = true }: { validate?: boolean } = {}): Promise<string> {
    const raw = await this.fetch();
    const records = await this.extract(raw);
    if (validate) {
      validateRecords(records, loadSchema(this.config.schemaPath));
    }
    return write(records, { source: this.name, format: this.config.format });
  }
}
