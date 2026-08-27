import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import { SERVER_CONFIG } from "./constants.js";
import type { OtpCoordinator } from "./otp.js";

// Minimal HTTP surface for the human-in-the-loop OTP step. The whole login flow
// lives inside MTap: when E*TRADE challenges, Authentication texts a link to
// this page (SMSNotifier), the human opens it and submits the code, and
// OtpCoordinator resolves the pending login in-process. No bus hop, no MBin.
//
// Deliberately tiny (no framework): one route, GET renders a form, POST submits
// the code. It binds to OTP_HOST/OTP_PORT (see constants.ts / .env.example).

const MAX_BODY_BYTES = 4 * 1024;

export class OtpServer {
  private server: Server | null = null;
  private readonly coordinator: OtpCoordinator;
  private readonly path: string;

  constructor(coordinator: OtpCoordinator, path: string = SERVER_CONFIG.OTP_PATH) {
    this.coordinator = coordinator;
    this.path = path;
  }

  /** Start listening. Idempotent — a second call while running resolves immediately. */
  start(port: number = SERVER_CONFIG.PORT, host: string = SERVER_CONFIG.HOST): Promise<void> {
    if (this.server) return Promise.resolve();
    return new Promise((resolve, reject) => {
      const server = createServer((req, res) => this.handle(req, res));
      server.on("error", reject);
      server.listen(port, host, () => {
        server.off("error", reject);
        this.server = server;
        console.log(`[OtpServer] OTP entry page at http://${host}:${port}${this.path}`);
        resolve();
      });
    });
  }

  /** Stop listening (graceful shutdown). */
  stop(): Promise<void> {
    return new Promise((resolve) => {
      if (!this.server) return resolve();
      this.server.close(() => {
        this.server = null;
        resolve();
      });
    });
  }

  private handle(req: IncomingMessage, res: ServerResponse): void {
    const url = new URL(req.url ?? "/", `http://${req.headers.host ?? "localhost"}`);
    if (url.pathname !== this.path) {
      res.writeHead(404, { "Content-Type": "text/plain" });
      res.end("Not found");
      return;
    }

    if (req.method === "GET") {
      this.sendHtml(res, 200, this.renderForm());
      return;
    }

    if (req.method === "POST") {
      this.handleSubmit(req, res);
      return;
    }

    res.writeHead(405, { "Content-Type": "text/plain", Allow: "GET, POST" });
    res.end("Method not allowed");
  }

  private handleSubmit(req: IncomingMessage, res: ServerResponse): void {
    let body = "";
    let aborted = false;

    req.on("data", (chunk: Buffer) => {
      if (aborted) return;
      body += chunk.toString("utf-8");
      if (body.length > MAX_BODY_BYTES) {
        aborted = true;
        res.writeHead(413, { "Content-Type": "text/plain" });
        res.end("Payload too large");
        req.destroy();
      }
    });

    req.on("end", () => {
      if (aborted) return;
      const code = (new URLSearchParams(body).get("code") ?? "").trim();

      if (!code) {
        this.sendHtml(res, 400, this.renderPage("Missing code", "Please enter the code from your SMS.", true));
        return;
      }

      const accepted = this.coordinator.submitCode(code);
      if (accepted) {
        this.sendHtml(res, 200, this.renderPage("Code submitted", "Login is continuing. You can close this page."));
      } else {
        this.sendHtml(
          res,
          409,
          this.renderPage("No login waiting", "No login is currently waiting for a code (it may have timed out).", true),
        );
      }
    });
  }

  private renderForm(): string {
    if (!this.coordinator.isPending()) {
      return this.renderPage("No login in progress", "There is no login waiting for a code right now.");
    }
    return this.renderPage(
      "Enter verification code",
      `<form method="POST" action="${this.path}">
         <input name="code" inputmode="numeric" autocomplete="one-time-code"
                pattern="[0-9]*" placeholder="Code" autofocus
                style="font-size:1.4rem;padding:.5rem;letter-spacing:.2rem;width:12rem;" />
         <button type="submit" style="font-size:1.1rem;padding:.55rem 1rem;margin-left:.5rem;">Submit</button>
       </form>`,
      false,
      true,
    );
  }

  private renderPage(title: string, body: string, isError = false, bodyIsHtml = false): string {
    const color = isError ? "#b00020" : "#111";
    const content = bodyIsHtml ? body : `<p style="color:${color};">${body}</p>`;
    return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>E*TRADE OTP</title>
</head>
<body style="font-family:system-ui,sans-serif;max-width:32rem;margin:3rem auto;padding:0 1rem;">
  <h2 style="color:${color};">${title}</h2>
  ${content}
</body>
</html>`;
  }

  private sendHtml(res: ServerResponse, status: number, html: string): void {
    res.writeHead(status, { "Content-Type": "text/html; charset=utf-8" });
    res.end(html);
  }
}
