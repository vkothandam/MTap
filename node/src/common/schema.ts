// Schema loading and record conformance — the Node half of shared contract enforcement.
import { readFileSync } from "node:fs";
import { Ajv } from "ajv";
import * as ajvFormats from "ajv-formats";

// ajv-formats is CJS with only a default export; under module: NodeNext its
// ESM-style .d.ts paired with the CJS .js types the plugin as the module
// namespace, hiding its callability. It IS callable at runtime
// (module.exports = formatsPlugin), so cast to its call signature.
const addFormats = ajvFormats.default as unknown as (ajv: Ajv, opts?: unknown) => unknown;

// Provenance fields the writer injects on every record (see schemas/conventions.md).
export const PROVENANCE_FIELDS = ["_source", "_ingested_at", "_run_id"] as const;

export type JsonSchema = Record<string, unknown>;

export function loadSchema(schemaPath: string): JsonSchema {
  return JSON.parse(readFileSync(schemaPath, "utf-8"));
}

export function validateRecords(records: unknown[], schema: JsonSchema): void {
  const ajv = new Ajv({ allErrors: true, strict: false });
  addFormats(ajv);
  const validate = ajv.compile(allowProvenance(schema));

  const errors: string[] = [];
  records.forEach((record, i) => {
    if (!validate(record)) {
      for (const err of validate.errors ?? []) {
        errors.push(`record[${i}]: ${err.instancePath || "/"} ${err.message}`);
      }
    }
  });
  if (errors.length > 0) {
    throw new Error("Schema conformance failed:\n" + errors.join("\n"));
  }
}

function allowProvenance(schema: JsonSchema): JsonSchema {
  if (schema.additionalProperties === false && schema.properties) {
    const props = { ...(schema.properties as Record<string, unknown>) };
    for (const field of PROVENANCE_FIELDS) {
      props[field] ??= { type: "string" };
    }
    return { ...schema, properties: props };
  }
  return schema;
}
