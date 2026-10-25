import { test } from "node:test";
import assert from "node:assert/strict";

// constants.ts reads the environment when it is imported, so each case imports a fresh copy.
async function serverConfig(env: Record<string, string | undefined>) {
  const saved = { host: process.env.OTP_HOST, bind: process.env.OTP_BIND_HOST };
  for (const [key, value] of Object.entries(env)) {
    if (value === undefined) delete process.env[key];
    else process.env[key] = value;
  }
  try {
    const mod = await import(`./constants.js?case=${Math.random()}`);
    return mod.SERVER_CONFIG;
  } finally {
    if (saved.host === undefined) delete process.env.OTP_HOST; else process.env.OTP_HOST = saved.host;
    if (saved.bind === undefined) delete process.env.OTP_BIND_HOST; else process.env.OTP_BIND_HOST = saved.bind;
  }
}

test("OTP page listens on OTP_HOST unless OTP_BIND_HOST is set", async () => {
  const plain = await serverConfig({ OTP_HOST: "192.168.1.20", OTP_BIND_HOST: undefined });
  assert.equal(plain.BIND_HOST, "192.168.1.20");
  assert.equal(plain.BASE_URL.startsWith("http://192.168.1.20:"), true);

  const container = await serverConfig({ OTP_HOST: "192.168.1.100", OTP_BIND_HOST: "0.0.0.0" });
  assert.equal(container.BIND_HOST, "0.0.0.0");
  assert.equal(container.BASE_URL.startsWith("http://192.168.1.100:"), true); // the SMS link keeps the LAN IP
});
