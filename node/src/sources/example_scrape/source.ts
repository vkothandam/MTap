// Example scrape source. Copy this as the template for a real one.
// Replace fetch() with a real download + cheerio parse, extract() with real mapping.
import * as cheerio from "cheerio";
import { Source } from "../../common/source.js";
import type { Record_ } from "../../common/writer.js";

export class ExampleScrapeSource extends Source {
  readonly name = "example_scrape";

  async fetch(): Promise<string> {
    // Real version:
    //   import { getText } from "../../common/http.js";
    //   return getText("https://example.com/listing");
    return `
      <ul>
        <li data-id="1" data-score="9.5" class="active">alpha</li>
        <li data-id="2" data-score="3.2">beta</li>
      </ul>`;
  }

  extract(raw: unknown): Record_[] {
    const $ = cheerio.load(raw as string);
    const now = new Date().toISOString();
    return $("li")
      .toArray()
      .map((el) => {
        const $el = $(el);
        return {
          id: Number($el.attr("data-id")),
          name: $el.text().trim(),
          score: Number($el.attr("data-score")),
          active: $el.hasClass("active"),
          fetched_at: now,
        };
      });
  }
}
