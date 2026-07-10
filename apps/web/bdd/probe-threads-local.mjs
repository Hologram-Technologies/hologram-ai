// Local V&V of the multi-threaded decode pool (ADR-0018), against the BUILT
// bundle served by `vite preview` — which sends COOP/COEP directly, so the page
// is cross-origin-isolated without the service-worker shim (the shim is what the
// DEPLOY probe exercises; here we isolate the pool itself).
//
// Asserts, in one run:
//   1. `crossOriginIsolated === true` (SharedArrayBuffer available → pool can start);
//   2. the generate worker logs the pool ACTIVE with N > 0 workers (a silent
//      fallback to single-threaded cannot masquerade as success — [[dark-gates]]);
//   3. the hermetic fixture handshake is BYTE-EXACT to the committed reference
//      (the fixture is fetched cross-origin from 127.0.0.1 with NO CORP header —
//      the in-repo canary that COEP:credentialless does not break downloads;
//      and that threaded init keeps decode correct).
//
// NOTE on the pool floor: the fixture's GEMVs are below the substrate's 256 KiB
// pool floor, so the pool REGISTERS here but does not fan out for the fixture.
// The supra-floor fan-out + speedup is witnessed by `probe-threads-live.mjs`.
import { chromium } from "playwright";
import { spawn } from "node:child_process";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { startFixtureServer, FIXTURE_REPO } from "./fixture-server.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../..");
const WEB = path.join(ROOT, "apps/web");
const PORT = Number(process.env.HAI_PREVIEW_PORT || 4174);
const APP = `http://127.0.0.1:${PORT}/hologram-ai/`;
const FIXTURE_DISPLAY = "handshake-tiny (hermetic fixture)";
const HANDSHAKE = ["Hello there!", "How are you today?", "Say goodbye."];
const reference = JSON.parse(
  readFileSync(path.join(ROOT, "oracles/fixture/reference-transcript.json"), "utf8"),
);

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
function ok(m) { console.log(`  ✓ ${m}`); }
function fail(m) { console.error(`  ✗ ${m}`); process.exitCode = 1; }

const preview = spawn(
  "pnpm",
  ["exec", "vite", "preview", "--host", "127.0.0.1", "--port", String(PORT), "--strictPort"],
  { cwd: WEB, stdio: "ignore" },
);
async function waitPort() {
  for (let i = 0; i < 150; i++) {
    try { const r = await fetch(APP); if (r.status < 500) return; } catch { /* not up yet */ }
    await sleep(200);
  }
  throw new Error("vite preview did not start");
}

const fixture = await startFixtureServer();
const browser = await chromium.launch({
  args: ["--disable-features=PrivateNetworkAccessChecks,LocalNetworkAccessChecks"],
});
const consoleLines = [];
const pageErrors = [];
try {
  await waitPort();
  console.log(`Local threaded-pool probe: ${APP}`);
  const context = await browser.newContext();
  const page = await context.newPage();
  page.on("console", (m) => consoleLines.push(m.text()));
  page.on("pageerror", (e) => pageErrors.push(String(e)));
  page.on("worker", (w) => consoleLines.push(`[worker spawned] ${w.url()}`));

  const entry = {
    id: "handshake-tiny",
    hfId: FIXTURE_REPO,
    displayName: FIXTURE_DISPLAY,
    description: "The committed journey fixture.",
    modality: "text-chat",
    size: "tiny",
    approxArchiveMb: 1,
    quantize: "none",
    promptTemplate: reference.template,
    stop: ["\nUser:"],
    chatTurnSeparator: reference.separator,
    maxTokens: reference.max_tokens,
  };
  await page.addInitScript(
    ([base, entryJson]) => {
      localStorage.setItem("hologram_hf_base", base);
      localStorage.setItem("hologram_catalogue_custom", entryJson);
    },
    [fixture.base, JSON.stringify([entry])],
  );

  await page.goto(APP, { waitUntil: "networkidle" });
  const isolated = await page.evaluate(() => globalThis.crossOriginIsolated === true);
  if (isolated) ok("crossOriginIsolated === true (SharedArrayBuffer available)");
  else fail("crossOriginIsolated is false — COOP/COEP headers missing");

  // Download → Ready.
  await page.goto(`${APP}#/models`);
  await page.waitForSelector("h1:has-text('Models')");
  const row = page.locator(".list-item", { hasText: FIXTURE_DISPLAY });
  await row.locator("button", { hasText: "Download" }).click();
  await row.locator("button", { hasText: "Ready" }).waitFor({ timeout: 150_000 });
  ok("fixture downloaded + compiled → Ready (credentialless did not block the cross-origin fetch)");

  // Chat: greedy, the 3-message handshake, byte-exact.
  await page.goto(`${APP}#/chat`);
  await page.waitForSelector("h1:has-text('Chat')");
  const temp = page.locator("input[type=number]");
  if (await temp.count()) await temp.first().fill("0");
  for (let i = 0; i < HANDSHAKE.length; i++) {
    await page.locator(".composer textarea").fill(HANDSHAKE[i]);
    await page.locator(".composer button", { hasText: "Send" }).click();
    await page.locator(".composer button", { hasText: "Send" }).waitFor({ timeout: 150_000 });
    const bubbles = page.locator(".bubble.assistant .md");
    const got = (await bubbles.nth((await bubbles.count()) - 1).innerText()).trim();
    const want = reference.turns[i].completion.trim();
    if (got === want) ok(`turn ${i} byte-exact to reference`);
    else fail(`turn ${i}: "${got}" ≠ reference "${want}"`);
  }

  // The pool must have ENGAGED in the generate worker (not silently fallen back).
  // The generate worker records a `pool: …` progress line in `__hologram_status`
  // (dedicated-worker console is not surfaced to the page).
  const status = await page.evaluate(
    () => (globalThis.__hologram_status || []),
  );
  const poolLine = status.find((l) => /^pool: /.test(l));
  const mw = poolLine && poolLine.match(/multi-threaded decode active \((\d+) workers\)/);
  if (mw && Number(mw[1]) > 0) ok(`decode pool engaged: ${poolLine}`);
  else {
    fail(`pool did not engage (fell back to single-threaded?): "${poolLine || "(no pool line)"}"`);
    console.error("  --- __hologram_status ---");
    status.forEach((l) => console.error(`    | ${l}`));
  }

  if (pageErrors.length === 0) ok("no uncaught page errors");
  else fail(`page errors: ${JSON.stringify(pageErrors.slice(0, 5))}`);
} finally {
  await browser.close();
  await fixture.close();
  preview.kill("SIGTERM");
}
console.log(process.exitCode ? "\nLOCAL THREADED PROBE: FAILED" : "\nLOCAL THREADED PROBE: PASS");
