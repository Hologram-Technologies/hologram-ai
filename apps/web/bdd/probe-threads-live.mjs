// Live byte-identity + speedup witness for the multi-threaded decode pool
// (ADR-0018), against the BUILT bundle served by `vite preview` (cross-origin
// isolated). Uses a REAL model — SmolLM2-135M int8 W8A8, whose decode GEMVs
// (hidden 576) are well above the substrate's 256 KiB pool floor, so the pool
// actually FANS OUT (unlike the sub-floor hermetic fixture).
//
// It runs the SAME model + prompt + greedy decode TWICE in the same session:
//   A) threaded  (the pool, N workers over shared memory);
//   B) single-threaded (`hologram_threads=0`).
// and asserts:
//   1. A is cross-origin-isolated and the pool engaged with N > 0 workers;
//   2. B ran single-threaded;
//   3. A's completion === B's completion, BYTE-FOR-BYTE — the pool's fan-out is
//      output-identical to serial (the substrate guarantees the kernel is
//      bit-identical; this witnesses our integration preserves it end-to-end);
//   4. reports A vs B warm decode throughput (the payoff — measured, not asserted;
//      it scales with the host's cores).
import { chromium } from "playwright";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../..");
const WEB = path.join(ROOT, "apps/web");
const PORT = Number(process.env.HAI_PREVIEW_PORT || 4175);
const APP = `http://127.0.0.1:${PORT}/hologram-ai/`;
const MODEL = {
  id: "byteid-model",
  hfId: process.env.HAI_PROBE_HF || "HuggingFaceTB/SmolLM2-135M-Instruct",
  displayName: process.env.HAI_PROBE_MODEL || "SmolLM2 135M Instruct",
  description: "Threaded-pool byte-identity witness.",
  modality: "text-chat",
  size: process.env.HAI_PROBE_SIZE || "135M",
  approxArchiveMb: 0,
  quantize: "int8",
  promptTemplate: null,
  stop: [],
  chatTurnSeparator: null,
};
const PROMPT = "The capital of France is";
const WARM = "List three colors.";

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
    try { const r = await fetch(APP); if (r.status < 500) return; } catch { /* not up */ }
    await sleep(200);
  }
  throw new Error("vite preview did not start");
}

// Run the model once in a fresh context; return the greedy completion, the pool
// status line, and a warm-turn throughput estimate.
async function runOnce(browser, { threaded }) {
  const context = await browser.newContext();
  const page = await context.newPage();
  await page.addInitScript(
    ([entryJson, threadsOff]) => {
      localStorage.setItem("hologram_catalogue_custom", entryJson);
      if (threadsOff) localStorage.setItem("hologram_threads", "0");
    },
    [JSON.stringify([MODEL]), threaded ? "" : "1"],
  );
  await page.goto(APP, { waitUntil: "networkidle" });
  const isolated = await page.evaluate(() => globalThis.crossOriginIsolated === true);

  await page.goto(`${APP}#/models`);
  await page.waitForSelector("h1:has-text('Models')");
  const row = page.locator(".list-item", { hasText: MODEL.displayName });
  await row.locator("button", { hasText: "Download" }).click();
  await row.locator("button", { hasText: "Ready" }).waitFor({ timeout: 600_000 });

  await page.goto(`${APP}#/chat`);
  await page.waitForSelector("h1:has-text('Chat')");
  const temp = page.locator("input[type=number]");
  if (await temp.count()) await temp.first().fill("0");

  async function send(text) {
    const t = Date.now();
    await page.locator(".composer textarea").fill(text);
    await page.locator(".composer button", { hasText: "Send" }).click();
    await page.locator(".composer button", { hasText: "Send" }).waitFor({ timeout: 300_000 });
    const b = page.locator(".bubble.assistant .md");
    const out = (await b.nth((await b.count()) - 1).innerText()).trim();
    return { out, ms: Date.now() - t };
  }
  const first = await send(PROMPT);
  const warm = await send(WARM);
  const status = await page.evaluate(() => globalThis.__hologram_status || []);
  const poolLine = status.find((l) => /^pool: /.test(l)) || "(none)";
  const toks = Math.max(1, Math.round(warm.out.split(/\s+/).filter(Boolean).length * 1.3));
  await context.close();
  return { isolated, completion: first.out, poolLine, tokPerSec: toks / (warm.ms / 1000) };
}

const browser = await chromium.launch();
try {
  await waitPort();
  console.log(`Live threaded byte-identity probe: ${APP}`);

  console.log("  … run A: threaded (pool)");
  const A = await runOnce(browser, { threaded: true });
  console.log("  … run B: single-threaded");
  const B = await runOnce(browser, { threaded: false });

  if (A.isolated) ok("run A cross-origin-isolated");
  else fail("run A not isolated");

  const mA = A.poolLine.match(/multi-threaded decode active \((\d+) workers\)/);
  if (mA && Number(mA[1]) > 0) ok(`run A pool engaged: ${A.poolLine}`);
  else fail(`run A pool did NOT engage: "${A.poolLine}"`);

  if (/single-threaded/.test(B.poolLine)) ok(`run B single-threaded: ${B.poolLine}`);
  else fail(`run B was not single-threaded: "${B.poolLine}"`);

  console.log(`  [A threaded]        "${PROMPT}" → "${A.completion}"`);
  console.log(`  [B single-threaded] "${PROMPT}" → "${B.completion}"`);
  if (A.completion.length > 0 && A.completion === B.completion) {
    ok("threaded completion is BYTE-IDENTICAL to single-threaded (pool fan-out preserves output)");
  } else {
    fail(`completions DIFFER — threaded pool changed the output (A="${A.completion}" B="${B.completion}")`);
  }
  if (/paris/i.test(A.completion)) ok("completion is coherent (contains 'Paris')");
  else fail(`completion not coherent: "${A.completion}"`);

  const speedup = A.tokPerSec / B.tokPerSec;
  console.log(
    `  warm decode: threaded ${A.tokPerSec.toFixed(2)} tok/s vs single ${B.tokPerSec.toFixed(2)} tok/s ` +
      `→ ${speedup.toFixed(2)}× on ${(await browser.newPage().then((p) => p.evaluate(() => navigator.hardwareConcurrency)))} logical cores`,
  );
} finally {
  await browser.close();
  preview.kill("SIGTERM");
}
console.log(process.exitCode ? "\nLIVE THREADED PROBE: FAILED" : "\nLIVE THREADED PROBE: PASS");
