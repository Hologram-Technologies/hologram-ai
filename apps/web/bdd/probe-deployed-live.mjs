// Live real-model probe of the DEPLOYED instance — exercises the W8A8 int8 decode
// path (the fixture is f32 and cannot). Adds a real model as a user would, lets
// the deployed app download from real HuggingFace, quantize to int8 in-worker,
// compile weightless, and decode with the fused output-major W8A8 GEMV (the
// v0.8.1 optimization). Asserts a coherent, non-empty, error-free completion and
// reports browser decode throughput.
//
// Small model on purpose: SmolLM2-135M int8 is ~270 MB — well under the codespace
// headless OPFS ceiling — so this witnesses the optimization's correctness and
// that it runs without errors on the live build, rather than stressing residency.
import { chromium } from "playwright";

const DEPLOY_URL =
  process.env.HAI_DEPLOY_URL || "https://hologram-technologies.github.io/hologram-ai/";

// PARAMETRIC — probe ANY model, no hard-coded architecture. Set HAI_PROBE_MODEL
// to a JSON catalogue entry (at minimum `hfId`), and HAI_PROBE_PROMPT / _EXPECT
// to a factual prompt and a regex its greedy completion must contain. Default:
// SmolLM2-135M (a fast integration SMOKE — not a usable model). Popular usable
// architectures (Qwen2.5, Llama-3.2, …) are probed by passing their entry here;
// the app derives staging/residency/quant automatically for whatever it is.
const MODEL = process.env.HAI_PROBE_MODEL
  ? JSON.parse(process.env.HAI_PROBE_MODEL)
  : {
      hfId: "HuggingFaceTB/SmolLM2-135M-Instruct",
      displayName: "SmolLM2 135M Instruct",
    };
// Sensible catalogue-entry defaults the app expects, without overriding a
// caller-supplied field. The app is parametric — it needs only `hfId` + int8.
MODEL.id ??= MODEL.hfId.split("/").pop().toLowerCase();
MODEL.displayName ??= MODEL.hfId.split("/").pop();
MODEL.description ??= "Live real-model probe (user-added).";
MODEL.modality ??= "text-chat";
MODEL.size ??= "";
MODEL.approxArchiveMb ??= 0;
MODEL.quantize ??= "int8";
MODEL.promptTemplate ??= null;
MODEL.stop ??= [];
MODEL.chatTurnSeparator ??= null;
// BOUND generation: a factual coherence check needs only the first few tokens,
// and some small models ramble without emitting a stop token — an unbounded
// turn can run many minutes single-threaded. Cap it (env-tunable) so the probe
// is robust for ANY model, not just the well-behaved ones.
MODEL.maxTokens ??= Number(process.env.HAI_PROBE_MAX_TOKENS || 40);
const PROMPT = process.env.HAI_PROBE_PROMPT || "The capital of France is";
const EXPECT = new RegExp(process.env.HAI_PROBE_EXPECT || "paris", "i");

function ok(m) {
  console.log(`  ✓ ${m}`);
}
function fail(m) {
  console.error(`  ✗ ${m}`);
  process.exitCode = 1;
}

const browser = await chromium.launch();
const pageErrors = [];
const consoleErrors = [];
try {
  console.log(`Live W8A8 probe (real model) on: ${DEPLOY_URL}`);
  const context = await browser.newContext();
  const page = await context.newPage();
  page.on("pageerror", (e) => pageErrors.push(String(e)));
  page.on("console", (m) => {
    if (m.type() === "error") consoleErrors.push(m.text());
  });

  // Real HF (no hologram_hf_base override), model added as a user entry.
  await page.addInitScript(
    (entryJson) => localStorage.setItem("hologram_catalogue_custom", entryJson),
    JSON.stringify([MODEL]),
  );
  await page.goto(DEPLOY_URL, { waitUntil: "networkidle" });
  ok(`loaded (${await page.title()})`);

  await page.goto(`${DEPLOY_URL}#/models`);
  await page.waitForSelector("h1:has-text('Models')");
  const row = page.locator(".list-item", { hasText: MODEL.displayName });
  const tDl = Date.now();
  // A usable model may be gigabytes: allow generous download+compile time.
  const DL_TIMEOUT = Number(process.env.HAI_PROBE_DL_TIMEOUT_MS || 1_800_000);
  await row.locator("button", { hasText: "Download" }).click();
  await row.locator("button", { hasText: "Ready" }).waitFor({ timeout: DL_TIMEOUT });
  ok(`downloaded + int8-quantized + compiled → Ready (${((Date.now() - tDl) / 1000).toFixed(0)}s)`);

  await page.goto(`${DEPLOY_URL}#/chat`);
  await page.waitForSelector("h1:has-text('Chat')");
  const temp = page.locator("input[type=number]");
  if (await temp.count()) await temp.first().fill("0");

  const tGen = Date.now();
  const GEN_TIMEOUT = Number(process.env.HAI_PROBE_GEN_TIMEOUT_MS || 600_000);
  await page.locator(".composer textarea").fill(PROMPT);
  await page.locator(".composer button", { hasText: "Send" }).click();
  await page.locator(".composer button", { hasText: "Send" }).waitFor({ timeout: GEN_TIMEOUT });
  const genMs = Date.now() - tGen;
  const bubbles = page.locator(".bubble.assistant .md");
  const out = (await bubbles.nth((await bubbles.count()) - 1).innerText()).trim();

  console.log(`  [gen] "${PROMPT}" → "${out}"`);
  if (out.length > 0) ok("completion is non-empty");
  else fail("completion is empty");
  // Coherence: the model greedily answers this factual prompt with EXPECT.
  if (EXPECT.test(out)) ok(`completion is coherent (matches ${EXPECT})`);
  else fail(`completion not coherent for the prompt: "${out}"`);

  const words = out.split(/\s+/).filter(Boolean).length;
  console.log(`  W8A8 browser decode: ~${words} words in ${(genMs / 1000).toFixed(1)}s`);

  // MEASURE, do not assume: read the real cross-origin-isolation state and the
  // pool status the worker recorded (`pool: multi-threaded decode active (N
  // workers)` / `pool: single-threaded decode`). On GitHub Pages isolation comes
  // from the coi-serviceworker; on a local preview it comes from vite headers —
  // either way this reports what ACTUALLY ran, not a hard-coded label.
  const isolated = await page.evaluate(() => globalThis.crossOriginIsolated === true);
  const poolLine =
    (await page.evaluate(
      () => (globalThis.__hologram_status ?? []).filter((l) => /^pool:/.test(l)).at(-1) ?? "",
    )) || "(no pool status recorded)";
  const threaded = /multi-threaded/.test(poolLine);
  const label = threaded ? `multi-threaded (${poolLine})` : "single-threaded";
  console.log(`  crossOriginIsolated=${isolated}; decode ran ${label}`);
  if (isolated) ok("page is cross-origin-isolated (SharedArrayBuffer / threads available)");
  else console.log("  (note: not isolated — single-threaded fallback; expected on first load or Safari)");

  // Warm turn: window already compiled/resident, so this rate is closer to
  // steady-state decode than the TTFT-inclusive first turn.
  {
    const t = Date.now();
    await page.locator(".composer textarea").fill("Tell me a short fact about the sun.");
    await page.locator(".composer button", { hasText: "Send" }).click();
    await page.locator(".composer button", { hasText: "Send" }).waitFor({ timeout: GEN_TIMEOUT });
    const b = page.locator(".bubble.assistant");
    const warmOut = (await b.nth((await b.count()) - 1).innerText()).trim();
    const toks = Math.round(warmOut.split(/\s+/).filter(Boolean).length * 1.3);
    const secs = (Date.now() - t) / 1000;
    console.log(`  W8A8 warm decode: ~${toks} tok in ${secs.toFixed(1)}s → ${(toks / secs).toFixed(1)} tok/s (${label})`);
  }

  if (pageErrors.length === 0) ok("no uncaught page errors");
  else fail(`page errors: ${JSON.stringify(pageErrors.slice(0, 5))}`);
  const realErrs = consoleErrors.filter((e) => !/favicon|404.*\.png/i.test(e));
  if (realErrs.length === 0) ok("no console errors");
  else fail(`console errors: ${JSON.stringify(realErrs.slice(0, 5))}`);
} finally {
  await browser.close();
}
console.log(process.exitCode ? "\nLIVE W8A8 PROBE: FAILED" : "\nLIVE W8A8 PROBE: PASS");
