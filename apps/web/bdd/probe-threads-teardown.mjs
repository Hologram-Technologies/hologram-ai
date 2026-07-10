// Teardown witness for the decode pool (ADR-0018 C1). The pool workers are
// spawned + owned by the MAIN thread precisely so that terminating the execute
// worker on cancel also tears the pool down — otherwise each cancel orphans N
// workers, each pinning the whole (model-sized) shared memory, until the tab OOMs.
//
// This drives real cancels and asserts the invariant via `__hologram_pool_live`
// (the count of live pool workers the main thread holds):
//   1. a turn brings the pool up to N (> 0);
//   2. Cancel takes it to 0 (no orphans);
//   3. across several cancel→regenerate cycles it NEVER exceeds N (no leak).
// Fails-without: drop the `terminatePool()` from the cancel path and step 2/3 fail.
//
// Uses SmolLM2-135M (int8) because its decode is slow enough (seconds) to cancel
// mid-generation deterministically; the sub-floor hermetic fixture finishes too
// fast to catch running.
import { chromium } from "playwright";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../..");
const WEB = path.join(ROOT, "apps/web");
const PORT = Number(process.env.HAI_PREVIEW_PORT || 4176);
const APP = `http://127.0.0.1:${PORT}/hologram-ai/`;
const MODEL = {
  id: "smollm2-135m-instruct",
  hfId: "HuggingFaceTB/SmolLM2-135M-Instruct",
  displayName: "SmolLM2 135M Instruct",
  description: "Pool teardown witness.",
  modality: "text-chat",
  size: "135M",
  approxArchiveMb: 0,
  quantize: "int8",
  promptTemplate: null,
  stop: [],
  chatTurnSeparator: null,
};
const LONG_PROMPT = "Write a long, detailed story about a lighthouse keeper.";
const CYCLES = 3;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
function ok(m) { console.log(`  ✓ ${m}`); }
function fail(m) { console.error(`  ✗ ${m}`); process.exitCode = 1; }
const poolLive = (page) =>
  page.evaluate(() => globalThis.__hologram_pool_live ?? -1);
async function waitFor(page, pred, ms, label) {
  const t = Date.now();
  while (Date.now() - t < ms) {
    if (pred(await poolLive(page))) return true;
    await sleep(50);
  }
  return false;
}

const preview = spawn(
  "pnpm",
  ["exec", "vite", "preview", "--host", "127.0.0.1", "--port", String(PORT), "--strictPort"],
  { cwd: WEB, stdio: "ignore" },
);
async function waitPort() {
  for (let i = 0; i < 150; i++) {
    try { const r = await fetch(APP); if (r.status < 500) return; } catch { /* not up */ }
    await sleep(200);
  }
  throw new Error("vite preview did not start");
}

const browser = await chromium.launch();
let peak = 0;
try {
  await waitPort();
  console.log(`Pool teardown witness: ${APP}`);
  const context = await browser.newContext();
  const page = await context.newPage();
  await page.addInitScript(
    (entryJson) => localStorage.setItem("hologram_catalogue_custom", entryJson),
    JSON.stringify([MODEL]),
  );
  await page.goto(APP, { waitUntil: "networkidle" });
  if (await page.evaluate(() => globalThis.crossOriginIsolated === true)) ok("cross-origin-isolated");
  else fail("not isolated");

  await page.goto(`${APP}#/models`);
  await page.waitForSelector("h1:has-text('Models')");
  const row = page.locator(".list-item", { hasText: MODEL.displayName });
  await row.locator("button", { hasText: "Download" }).click();
  await row.locator("button", { hasText: "Ready" }).waitFor({ timeout: 600_000 });
  ok("model ready");

  await page.goto(`${APP}#/chat`);
  await page.waitForSelector("h1:has-text('Chat')");
  const composer = page.locator(".composer textarea");
  const send = page.locator(".composer button", { hasText: "Send" });
  const cancel = page.locator("button", { hasText: "Cancel" });

  const N = null; // discovered on the first cycle
  let n = 0;
  for (let c = 0; c < CYCLES; c++) {
    await composer.fill(LONG_PROMPT);
    await send.click();
    // Pool comes up to N > 0 during the turn.
    if (!(await waitFor(page, (v) => v > 0, 60_000, "pool up"))) {
      fail(`cycle ${c}: pool never came up`);
      break;
    }
    const up = await poolLive(page);
    if (c === 0) n = up;
    peak = Math.max(peak, up);
    // Cancel mid-generation.
    await cancel.click();
    if (await waitFor(page, (v) => v === 0, 15_000, "pool down")) {
      ok(`cycle ${c}: pool ${up} → 0 on cancel`);
    } else {
      fail(`cycle ${c}: pool did NOT tear down on cancel (live=${await poolLive(page)})`);
    }
    await sleep(200);
  }

  if (n > 0 && peak === n) ok(`no orphan accumulation across ${CYCLES} cancels (peak live = ${peak} = N)`);
  else fail(`orphan accumulation: peak live ${peak} vs N ${n}`);
} finally {
  await browser.close();
  preview.kill("SIGTERM");
}
console.log(process.exitCode ? "\nTEARDOWN WITNESS: FAILED" : "\nTEARDOWN WITNESS: PASS");
