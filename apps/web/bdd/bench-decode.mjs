// Real end-to-end browser DECODE throughput (ADR-0018) — the metric that matters
// for the deployment target, not the GEMV microbenchmark in isolation. It polls
// the streaming completion and fits the STEADY-STATE slope (excludes TTFT/prefill),
// so it measures decode tok/s, not first-token latency.
//
// One download (persistent context), three configs in it: single-threaded,
// threaded @ logical cores, threaded @ a smaller pool — to see whether the pool
// helps end-to-end and whether over-subscription (pool + execute + browser on the
// host's physical cores) erodes it. NOTE the codespace has 4 physical cores and
// is running vite + Chromium + the pool, so absolute numbers are contended; the
// RATIO single→threaded is the signal.
import { chromium } from "playwright";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../..");
const WEB = path.join(ROOT, "apps/web");
const PORT = Number(process.env.HAI_PREVIEW_PORT || 4180);
const APP = `http://127.0.0.1:${PORT}/hologram-ai/`;
const MODEL = {
  id: "bench-decode", hfId: process.env.HAI_BENCH_HF || "Qwen/Qwen2.5-0.5B-Instruct",
  displayName: process.env.HAI_BENCH_MODEL || "Qwen2.5 0.5B Instruct",
  description: "decode bench", modality: "text-chat", size: "0.5B", approxArchiveMb: 0,
  quantize: "int8", promptTemplate: null, stop: [], chatTurnSeparator: null, maxTokens: 80,
};
const PROMPT = "Write a detailed paragraph explaining how photosynthesis works, step by step.";

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
function ok(m) { console.log(`  ✓ ${m}`); }
function fail(m) { console.error(`  ✗ ${m}`); process.exitCode = 1; }

// Steady-state slope of the (t, words) stream, words in [lo, hi]; words/sec.
function marginalRate(samples) {
  const maxW = samples.length ? samples[samples.length - 1][1] : 0;
  const lo = 6, hi = Math.max(lo + 4, maxW - 4);
  const pts = samples.filter(([, w]) => w >= lo && w <= hi);
  if (pts.length < 3) return { rate: NaN, words: maxW };
  // least-squares slope
  const n = pts.length;
  const sx = pts.reduce((a, [t]) => a + t, 0), sy = pts.reduce((a, [, w]) => a + w, 0);
  const sxx = pts.reduce((a, [t]) => a + t * t, 0), sxy = pts.reduce((a, [t, w]) => a + t * w, 0);
  const slope = (n * sxy - sx * sy) / (n * sxx - sx * sx); // words per ms
  return { rate: slope * 1000, words: maxW };
}

async function measure(browser, cfg) {
  // Fresh context per config (isolated OPFS + localStorage): the persistent-context
  // sharing broke the threaded pool init. Costs a re-download; worth the reliability.
  const context = await browser.newContext();
  const page = await context.newPage();
  await page.addInitScript(
    ([entryJson, ls]) => {
      localStorage.setItem("hologram_catalogue_custom", entryJson);
      for (const [k, v] of Object.entries(ls)) localStorage.setItem(k, v);
    },
    [JSON.stringify([MODEL]), cfg.ls],
  );
  await page.goto(`${APP}#/models`, { waitUntil: "networkidle" });
  await page.waitForSelector("h1:has-text('Models')");
  const row = page.locator(".list-item", { hasText: MODEL.displayName });
  if (await row.locator("button", { hasText: "Download" }).count())
    await row.locator("button", { hasText: "Download" }).click().catch(() => {});
  await row.locator("button", { hasText: "Ready" }).waitFor({ timeout: 900_000 });
  await page.goto(`${APP}#/chat`);
  await page.waitForSelector("h1:has-text('Chat')");
  const temp = page.locator("input[type=number]");
  if (await temp.count()) await temp.first().fill("0");
  const composer = page.locator(".composer textarea");
  const send = page.locator(".composer button", { hasText: "Send" });
  const bubble = page.locator(".bubble.assistant .md").last();

  await composer.fill(PROMPT);
  const samples = [];
  const t0 = Date.now();
  await send.click();
  while (Date.now() - t0 < 300_000) {
    const done = (await send.count()) > 0;
    const words = await bubble.innerText().then((t) => t.trim().split(/\s+/).filter(Boolean).length).catch(() => 0);
    samples.push([Date.now() - t0, words]);
    if (done && words > 0) break;
    await sleep(250);
  }
  const status = await page.evaluate(() => globalThis.__hologram_status || []);
  const poolLine = status.find((l) => /^pool: /.test(l)) || "(none)";
  const { rate, words } = marginalRate(samples);
  await context.close();
  return { rate, words, poolLine };
}

const preview = spawn("pnpm", ["exec", "vite", "preview", "--host", "127.0.0.1", "--port", String(PORT), "--strictPort"], { cwd: WEB, stdio: "ignore" });
async function waitPort() { for (let i = 0; i < 150; i++) { try { const r = await fetch(APP); if (r.status < 500) return; } catch { /* */ } await sleep(200); } throw new Error("preview down"); }

const browser = await chromium.launch();
try {
  await waitPort();
  const hc = await browser.newPage().then(async (p) => { await p.goto(APP); const c = await p.evaluate(() => navigator.hardwareConcurrency); await p.close(); return c; });
  console.log(`Browser decode benchmark: ${MODEL.displayName} on ${APP} (host ${hc} logical cores)\n`);

  const configs = [
    { label: "single-threaded", ls: { hologram_threads: "0" } },
    { label: `threaded (${hc - 1}w=logical)`, ls: { hologram_threads: "1" } },
    { label: "threaded (3w=physical)", ls: { hologram_threads: "1", hologram_pool_workers: "3" } },
  ];
  const res = [];
  for (const cfg of configs) {
    const r = await measure(browser, cfg);
    res.push({ ...cfg, ...r });
    console.log(`  ${cfg.label.padEnd(26)} steady decode ${r.rate.toFixed(2)} words/s over ${r.words} words | ${r.poolLine}`);
  }
  const base = res[0].rate;
  console.log("");
  for (const r of res.slice(1)) {
    const sp = r.rate / base;
    console.log(`  ${r.label.padEnd(26)} → ${sp.toFixed(2)}× vs single-threaded`);
  }
} finally {
  await browser.close();
  preview.kill("SIGTERM");
}
console.log(process.exitCode ? "\nDECODE BENCH: FAILED" : "\nDECODE BENCH: DONE");
