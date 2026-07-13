// Diagnostic probe: drive the LOCAL preview through a real-model chat turn
// with FULL telemetry — every console message (page + workers), every
// chat/status/download line, and a per-5s progress snapshot — so a hang or
// crash shows WHERE it stopped, not just that it stopped.
//
//   HAI_PROBE_MODEL=Qwen/Qwen2.5-0.5B-Instruct node bdd/probe-live-debug.mjs
//
// Requires a built dist/ (pnpm build) + built wasm bindings. Serves dist/ via
// `vite preview` with the repo's headers (COOP/COEP), so the threaded pool
// engages exactly as deployed.
import { chromium } from "playwright";
import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { startFixtureServer, DEEP_FIXTURE_REPO } from "./fixture-server.mjs";

const WEB = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
// HAI_PROBE_DEEP=1 drives the HERMETIC deep fixture (head_dim 128, 12 layers,
// int8) served locally — no network, reproduces the deployed real-model hang.
const DEEP = process.env.HAI_PROBE_DEEP === "1";
const fixtureServer = DEEP ? await startFixtureServer() : null;
const MODEL = DEEP ? DEEP_FIXTURE_REPO : (process.env.HAI_PROBE_MODEL || "HuggingFaceTB/SmolLM2-135M-Instruct");
const PROMPT = process.env.HAI_PROBE_PROMPT || "Hello there!";
const TURN_TIMEOUT_MS = Number(process.env.HAI_PROBE_TURN_TIMEOUT_MS || 300_000);
const stamp = () => new Date().toISOString().slice(11, 19);
const log = (tag, line) => console.log(`${stamp()} [${tag}] ${line}`);

// ── vite preview over the built dist ────────────────────────────────────────
const preview = spawn("npx", ["vite", "preview", "--port", "4199", "--strictPort"], {
  cwd: WEB,
  stdio: ["ignore", "pipe", "pipe"],
});
await new Promise((resolve, reject) => {
  const t = setTimeout(() => reject(new Error("vite preview did not start")), 20_000);
  preview.stdout.on("data", (d) => {
    if (String(d).includes("Local:")) {
      clearTimeout(t);
      resolve();
    }
  });
  preview.stderr.on("data", (d) => log("vite", String(d).trim()));
});
const APP = "http://localhost:4199/hologram-ai/";

const browser = await chromium.launch({
  args: ["--disable-features=PrivateNetworkAccessChecks,LocalNetworkAccessChecks"],
});

try {
  const page = await browser.newContext().then((c) => c.newPage());
  page.on("console", (m) => log(`console.${m.type()}`, m.text()));
  page.on("pageerror", (e) => log("pageerror", String(e)));
  page.on("response", (r) => {
    if (r.status() >= 400 && !r.url().includes("huggingface")) log("http", `${r.status()} ${r.url()}`);
  });

  // Optional single-threaded run (HAI_PROBE_THREADS=0): discriminates a
  // pooled-kernel hang from everything else.
  if (process.env.HAI_PROBE_THREADS === "0") {
    await page.addInitScript(() => localStorage.setItem("hologram_threads", "0"));
    log("probe", "threaded pool DISABLED for this run");
  }
  // A custom catalogue entry for the probe model (the app ships none).
  if (DEEP) {
    await page.addInitScript((base) => localStorage.setItem("hologram_hf_base", base), fixtureServer.base);
  }
  await page.addInitScript(
    (entry) => localStorage.setItem("hologram_catalogue_custom", entry),
    JSON.stringify([
      {
        id: "probe-model",
        hfId: MODEL,
        displayName: `probe: ${MODEL}`,
        description: "live debug probe",
        modality: "text-chat",
        size: "?",
        approxArchiveMb: 0,
        quantize: "int8",
        promptTemplate: DEEP ? "User:\n{prompt}\nAssistant:\n" : null,
        stop: [],
        chatTurnSeparator: null,
      },
    ]),
  );

  log("probe", `app ${APP} model ${MODEL}`);
  await page.goto(APP, { waitUntil: "networkidle" });
  await page.goto(`${APP}#/models`);
  await page.waitForSelector("h1:has-text('Models')");

  // Mirror every app log line as it lands.
  await page.exposeFunction("__probe_log", (tag, line) => log(tag, line));
  await page.evaluate(() => {
    const g = globalThis;
    const status = (g.__hologram_status ??= []);
    const push = status.push.bind(status);
    status.push = (line) => {
      g.__probe_log("status", String(line));
      return push(line);
    };
  });

  const row = page.locator(".list-item", { hasText: "probe:" });
  await row.first().waitFor({ timeout: 20_000 });
  await row.locator("button", { hasText: "Download" }).click();
  log("probe", "download clicked");
  await row.locator("button", { hasText: "Ready" }).waitFor({ timeout: 1_800_000 });
  log("probe", "model Ready");

  await page.goto(`${APP}#/chat`);
  await page.waitForSelector("h1:has-text('Chat')");
  await page.locator("input[type=number]").first().fill("0");
  await page.locator(".composer textarea").fill(PROMPT);
  await page.locator(".composer button", { hasText: "Send" }).click();
  log("probe", "turn sent — watching");

  const t0 = Date.now();
  let lastSnapshot = "";
  while (Date.now() - t0 < TURN_TIMEOUT_MS) {
    const send = page.locator(".composer button", { hasText: "Send" });
    const done = (await send.count()) > 0 && (await send.first().isVisible().catch(() => false));
    const bubbles = await page.locator(".bubble.assistant").allInnerTexts().catch(() => []);
    const snapshot = JSON.stringify(bubbles);
    if (snapshot !== lastSnapshot) {
      lastSnapshot = snapshot;
      log("bubble", snapshot.slice(0, 400));
    }
    if (done && bubbles.length > 0) {
      const completions = await page.evaluate(() => globalThis.__hologram_completions ?? []);
      if (completions.length > 0) {
        log("probe", `TURN COMPLETE: ${JSON.stringify(completions[0]).slice(0, 300)}`);
        process.exitCode = 0;
        break;
      }
    }
    await new Promise((r) => setTimeout(r, 5000));
    log("tick", `${Math.round((Date.now() - t0) / 1000)}s elapsed`);
  }
  if (Date.now() - t0 >= TURN_TIMEOUT_MS || process.exitCode === 1) {
    const bubbles = await page.locator(".bubble.assistant").allInnerTexts().catch(() => []);
    log("probe", `FINAL bubbles: ${JSON.stringify(bubbles)}`);
    const status = await page.evaluate(() => globalThis.__hologram_status ?? []);
    log("probe", `FINAL status tail: ${JSON.stringify(status.slice(-6))}`);
  }
} finally {
  await browser.close();
  preview.kill();
  if (fixtureServer) await fixtureServer.close();
}
