// Step definitions for the browser-executor dictionary rows (S1 + S4).
// Every step drives the real app in real Chromium — no mocks between the UI
// and the substrate. κ verification runs the same wasm binding node-side.
import { Given, When, Then } from "@cucumber/cucumber";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { referenceTranscript } from "./world.mjs";
import { BAD_CONFIG_REPO, FIXTURE_REPO, MISSING_REPO, ROPE_EXOTIC_REPO, ROPE_SCALED_REPO, SEARCH_UNSUPPORTED_REPO, SECOND_FIXTURE_REPO, TOO_LARGE_REPO, UNSUPPORTED_FAMILY_REPO } from "./fixture-server.mjs";

const WEB_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const ROOT = path.resolve(WEB_DIR, "../..");
const FIXTURE_DISPLAY = "handshake-tiny (hermetic fixture)";
const HANDSHAKE = ["Hello there!", "How are you today?", "Say goodbye."];

// ── the wasm binding, node-side (κ hashing + κ-map parsing) ────────────────
let wasmApi = null;
async function wasm() {
  if (wasmApi) return wasmApi;
  const pkg = await import(path.join(WEB_DIR, "src/wasm/hologram_ai_wasm.js"));
  const bytes = readFileSync(path.join(WEB_DIR, "src/wasm/hologram_ai_wasm_bg.wasm"));
  await pkg.default(bytes);
  wasmApi = pkg;
  return pkg;
}

/** Parse the committed fixture safetensors: name → tensor byte slice. */
function fixtureTensors() {
  const bytes = readFileSync(path.join(ROOT, "oracles/fixture/model.safetensors"));
  const headerLen = Number(new DataView(bytes.buffer, bytes.byteOffset, 8).getBigUint64(0, true));
  const header = JSON.parse(bytes.subarray(8, 8 + headerLen).toString("utf8"));
  const tensors = new Map();
  for (const [name, meta] of Object.entries(header)) {
    if (name === "__metadata__") continue;
    const [start, end] = meta.data_offsets;
    tensors.set(name, bytes.subarray(8 + headerLen + start, 8 + headerLen + end));
  }
  return tensors;
}

/** The browser's assistant-text cleaning, restricted to the fixture alphabet
 * (mirrors xtask::fixture::clean_assistant_text). */
function cleanCompletion(text) {
  return text.replaceAll("</s>", "").replaceAll("<s>", "").replaceAll("<unk>", "").trimEnd();
}

async function installCustomEntry(world, entry) {
  // As an init script: the world's own init script re-writes the custom
  // catalogue on every navigation, so this must run after it, appending.
  await world.page.addInitScript((json) => {
    const current = JSON.parse(localStorage.getItem("hologram_catalogue_custom") ?? "[]");
    current.push(JSON.parse(json));
    localStorage.setItem("hologram_catalogue_custom", JSON.stringify(current));
  }, JSON.stringify(entry));
  await world.page.reload({ waitUntil: "networkidle" });
}

// ── shared givens ───────────────────────────────────────────────────────────

Given("the app is open in the browser against the hermetic model server", async function () {
  await this.openApp();
});

Given("the app is open in the browser against the live HuggingFace Hub", async function () {
  await this.openApp({ live: true });
});

// ── S1 — download ───────────────────────────────────────────────────────────

// Matched keyword-agnostically (used as both When and background Given).
When("the fixture model is downloaded", async function () {
  await this.downloadModel(FIXTURE_DISPLAY);
});

const SECOND_FIXTURE_DISPLAY = "handshake-tiny B (switch target)";
const ROPE_SCALED_DISPLAY = "handshake-tiny (llama3-scaled rope)";

/** The fixture catalogue entry under another identity (same chat contract). */
async function installFixtureAlias(world, id, hfId, displayName) {
  const reference = referenceTranscript();
  await installCustomEntry(world, {
    id,
    hfId,
    displayName,
    description: "Fixture alias.",
    modality: "text-chat",
    size: "tiny",
    approxArchiveMb: 1,
    quantize: "none",
    promptTemplate: reference.template,
    stop: ["\nUser:"],
    chatTurnSeparator: reference.separator,
    maxTokens: reference.max_tokens,
  });
}

// ── Chat model menu: compiled-only invariant ────────────────────────────────

Given("a fresh instance with no compiled models", async function () {
  // The hermetic world starts with an empty OPFS; assert it explicitly so a
  // leaked/persisted model would fail here, not silently pass downstream.
  await this.gotoChat();
});

Then(/^the chat model menu (?:still )?offers no models to select$/, async function () {
  // Self-contained: navigate to the chat page (which waits for its own
  // readiness) rather than assume a prior step left us there.
  await this.gotoChat();
  const options = await this.page
    .locator("select")
    .first()
    .locator("option")
    .evaluateAll((os) => os.map((o) => ({ value: o.value, text: o.textContent })));
  // The only option is the disabled "No compiled archives" placeholder (empty
  // value) — nothing selectable, nothing hard-coded.
  const selectable = options.filter((o) => o.value && o.value.length > 0);
  assert.equal(
    selectable.length,
    0,
    `the chat menu must offer no models on a fresh instance, got ${JSON.stringify(options)}`,
  );
});

When("a model is added to the catalogue but not downloaded", async function () {
  // A user-added entry (catalogue DATA) that is NOT downloaded/compiled must
  // never reach the chat menu.
  await installCustomEntry(this, {
    ...BARE_ENTRY,
    id: "added-not-downloaded",
    hfId: "hologram-fixture/added-not-downloaded",
    displayName: "added but not downloaded",
  });
  await this.gotoChat();
});

Then("the chat model menu lists exactly the compiled fixture model", async function () {
  await this.gotoChat();
  const select = this.page.locator("select").first();
  // A closed <select>'s <option>s are `hidden`, so poll their VALUES (present
  // in the DOM) rather than wait for visibility.
  await this.page.waitForFunction(
    () =>
      Array.from(document.querySelector("select")?.options ?? []).some((o) =>
        o.value.startsWith("models/handshake-tiny/"),
      ),
    { timeout: 10_000 },
  );
  const options = await select
    .locator("option")
    .evaluateAll((os) => os.map((o) => ({ value: o.value, text: o.textContent })));
  const selectable = options.filter((o) => o.value && o.value.length > 0);
  assert.equal(
    selectable.length,
    1,
    `exactly the one compiled model must be offered, got ${JSON.stringify(options)}`,
  );
  // Its value is an on-disk archive path — never a bare model id or a
  // catalogue name.
  assert.match(
    selectable[0].value,
    /^models\/handshake-tiny\/.+/,
    `the menu entry must be a compiled archive path, got ${JSON.stringify(selectable[0])}`,
  );
});

Then("the stored-models list shows the fixture model", async function () {
  await this.gotoModels();
  const row = this.page.locator("[data-stored-model='handshake-tiny']");
  await row.first().waitFor({ timeout: 10_000 });
});

When("the user removes the fixture model from this device", async function () {
  await this.gotoModels();
  const row = this.page.locator("[data-stored-model='handshake-tiny']");
  await row.first().waitFor({ timeout: 10_000 });
  await row.locator("button", { hasText: "Remove" }).click();
  // The removal awaits an OPFS recursive delete + refresh; wait for the row
  // to disappear so the subsequent assertions see the settled state.
  await row.first().waitFor({ state: "detached", timeout: 10_000 });
});

Then("the stored-models list is empty", async function () {
  await this.gotoModels();
  await this.page.waitForSelector("h2:has-text('Stored on this device')");
  const rows = await this.page.locator("[data-stored-model]").count();
  assert.equal(rows, 0, `storage must be empty after removal, ${rows} model(s) remain`);
});

const DEEP_FIXTURE_DISPLAY = "deep-tiny (head_dim-128 hermetic fixture)";

Given("the deep hermetic fixture model is available", async function () {
  await this.installDeepFixture();
});

When("the deep fixture model is downloaded", { timeout: 600_000 }, async function () {
  await this.downloadModel(DEEP_FIXTURE_DISPLAY, { timeoutMs: 540_000 });
});

When("the user sends a chat message on the deep fixture model", { timeout: 420_000 }, async function () {
  await this.gotoChat();
  await this.page.locator("input[type=number]").first().fill("0");
  await selectArchiveByDir(this, "deep-tiny");
  // Inline send + wait so a FAILED turn surfaces the real diagnostics (the
  // app's status tail + captured console errors) instead of a bare selector
  // timeout — the deployed "[object Event]" was undiagnosable for exactly
  // this reason. The completion may be EMPTY (the fixture's deterministic
  // noise weights emit end-of-sequence immediately) — that is the point: a
  // real-shape turn that produces no text must COMPLETE and be handled
  // honestly, never hang or vanish.
  await this.page.locator(".composer textarea").fill(HANDSHAKE[0]);
  await this.page.locator(".composer button", { hasText: "Send" }).click();
  await this.page
    .locator(".composer button", { hasText: "Send" })
    .waitFor({ timeout: 300_000 });
  const completions = await this.page.evaluate(() => globalThis.__hologram_completions ?? []);
  const crash = (this.consoleErrors ?? []).find((e) => /worker (failed|crashed)|\[object Event\]/i.test(e));
  if (crash) throw new Error(`deep-fixture turn hit a worker crash: ${crash}`);
  if (completions.length === 0) {
    const status = await this.page.evaluate(() => (globalThis.__hologram_status ?? []).slice(-8));
    const bubbles = await this.page.locator(".bubble.assistant").allInnerTexts();
    throw new Error(
      `deep-fixture turn did not complete (no worker crash surfaced).\n` +
        `  bubbles: ${JSON.stringify(bubbles)}\n` +
        `  status tail: ${JSON.stringify(status)}\n` +
        `  console errors: ${JSON.stringify((this.consoleErrors ?? []).slice(-12))}`,
    );
  }
});

Then("the real-shape turn completes and its assistant reply is committed honestly", async function () {
  // The real-model SHAPE (head_dim 128, many stages, int8) runs end to end in
  // the browser: the turn completes (asserted above — no crash, a completion
  // recorded) and the assistant turn is COMMITTED, never silently dropped. An
  // empty completion shows the honest no-output notice; a non-empty one shows
  // its text. Either way exactly one assistant bubble joins the transcript.
  const completions = await this.page.evaluate(() => globalThis.__hologram_completions ?? []);
  assert.ok(completions.length >= 1, "the turn must record a completion");
  const bubbles = await this.page.locator(".bubble.assistant").allInnerTexts();
  assert.equal(bubbles.length, 1, `exactly one assistant bubble must be committed, got ${bubbles.length}`);
  const text = cleanCompletion(completions.at(-1).text);
  if (text.length === 0) {
    assert.match(
      bubbles[0],
      /no output/i,
      `an empty completion must render the honest no-output notice, got: ${JSON.stringify(bubbles[0])}`,
    );
  } else {
    assert.ok(bubbles[0].includes(text.slice(0, 8)), "a non-empty completion must render its text");
  }
});

When("the second fixture model is downloaded", async function () {
  await installFixtureAlias(this, "handshake-tiny-b", SECOND_FIXTURE_REPO, SECOND_FIXTURE_DISPLAY);
  await this.downloadModel(SECOND_FIXTURE_DISPLAY);
});

When("the llama3-scaled fixture model is downloaded", async function () {
  await installFixtureAlias(this, "handshake-tiny-llama3rope", ROPE_SCALED_REPO, ROPE_SCALED_DISPLAY);
  await this.downloadModel(ROPE_SCALED_DISPLAY);
});

/** Select the compiled archive whose OPFS dir is exactly `dirId` (option
 * values are `models/<dir>/<artifact>`; a label match would collide on the
 * shared `handshake-tiny` prefix). */
async function selectArchiveByDir(world, dirId) {
  const select = world.page.locator("select").first();
  const values = await select.locator("option").evaluateAll((os) => os.map((o) => o.value));
  const value = values.find((v) => v.startsWith(`models/${dirId}/`));
  assert.ok(value, `no compiled archive for ${dirId}; options: ${JSON.stringify(values)}`);
  await select.selectOption(value);
}

When("the user chats on the llama3-scaled fixture model", async function () {
  await this.gotoChat();
  await this.page.locator("input[type=number]").fill("0");
  await selectArchiveByDir(this, "handshake-tiny-llama3rope");
  await this.sendChat(HANDSHAKE[0]);
});

Then("the llama3-scaled turn streams a non-empty completion", async function () {
  const completions = await this.page.evaluate(
    () => globalThis.__hologram_completions ?? [],
  );
  if (completions.length === 0) {
    const bubbles = await this.page.locator(".bubble.assistant").allInnerTexts();
    assert.fail(`the scaled-rope generation did not complete; bubbles: ${JSON.stringify(bubbles)}`);
  }
  assert.ok(
    cleanCompletion(completions.at(-1).text).length > 0,
    `the scaled-rope turn must stream non-empty text: ${JSON.stringify(completions)}`,
  );
});

When("the user switches the chat to the second fixture model", async function () {
  await selectArchiveByDir(this, "handshake-tiny-b");
});

When("the user switches the chat back to the first fixture model", async function () {
  await selectArchiveByDir(this, "handshake-tiny");
});

When("the user sends a chat message on the switched model", async function () {
  await this.sendChat(HANDSHAKE[0]);
});

Then("no download requests were repeated for the first model", async function () {
  const log = await this.fixtureRequests();
  const firstModelShards = log.filter(
    (p) => p.includes(`/${FIXTURE_REPO}/resolve/`) && p.includes("model.safetensors") && !p.includes("#"),
  );
  assert.ok(
    firstModelShards.length <= 1,
    `switching back must not re-download the first model's shard: ${JSON.stringify(firstModelShards)}`,
  );
});

Then("every tensor in the fixture manifest is persisted under its κ in OPFS", async function () {
  const { compute_kappa } = await wasm();
  const names = await this.opfsTensorNames();
  for (const [tensor, bytes] of fixtureTensors()) {
    const kappa = compute_kappa(bytes);
    assert.ok(
      names.includes(`${kappa}.bin`),
      `tensor ${tensor} (κ ${kappa}) missing from the OPFS κ-store`,
    );
  }
});

Then("each persisted blob re-hashes to its κ", async function () {
  const { compute_kappa } = await wasm();
  const names = await this.opfsTensorNames();
  for (const name of names) {
    const bytes = await this.page.evaluate(async (n) => {
      const root = await navigator.storage.getDirectory();
      const dir = await root.getDirectoryHandle("tensors");
      const fh = await dir.getFileHandle(n);
      const file = await fh.getFile();
      return Array.from(new Uint8Array(await file.arrayBuffer()));
    }, name);
    const kappa = compute_kappa(new Uint8Array(bytes));
    assert.equal(`${kappa}.bin`, name, `κ-store blob ${name} does not re-hash to its label`);
  }
});

Then("identical tensors are stored once", async function () {
  const { compute_kappa } = await wasm();
  const distinct = new Set();
  for (const [, bytes] of fixtureTensors()) distinct.add(compute_kappa(bytes));
  const names = await this.opfsTensorNames();
  assert.equal(
    names.length,
    distinct.size,
    "the κ-store must hold exactly one blob per distinct tensor content",
  );
});

When("downloading a repository that does not exist", async function () {
  await installCustomEntry(this, {
    id: "does-not-exist",
    hfId: MISSING_REPO,
    displayName: "missing repo",
    description: "",
    modality: "text-chat",
    size: "?",
    approxArchiveMb: 0,
    quantize: "none",
    promptTemplate: null,
    stop: [],
    chatTurnSeparator: null,
  });
  await this.gotoModels();
  const row = this.page.locator(".list-item", { hasText: "missing repo" });
  await row.locator("button", { hasText: "Download" }).click();
  await this.page.getByText(/error:/).first().waitFor({ timeout: 60_000 });
});

Then("the journey fails at the download stage naming the repository", async function () {
  const body = await this.page.locator("body").innerText();
  assert.ok(body.includes(MISSING_REPO), `failure must name ${MISSING_REPO}:\n${body}`);
});

// ── S1 — quota fail-soft (row `memory-guard`) ───────────────────────────────

Given("the origin's storage quota is capped below the model's size", async function () {
  // A REAL quota, enforced by the browser (CDP override) — the measured
  // headroom is a projection; the write is where the environment answers.
  // The cap is far below the fixture's tensor bytes, so caching MUST hit
  // QuotaExceededError mid-stream.
  const cdp = await this.context.newCDPSession(this.page);
  const origin = new URL(this.appUrl).origin;
  // Between the structure size (~100 KB: archive, companions, provenance
  // records — load-bearing) and the tensor bytes (~558 KB): caching hits
  // the wall mid-stream, everything essential still persists.
  await cdp.send("Storage.overrideQuotaForOrigin", { origin, quotaSize: 300 * 1024 });
});

Then("the download reports the quota and continues on recorded provenance", async function () {
  const body = await this.page.locator("body").innerText();
  assert.ok(
    body.includes("quota reached"),
    `the projection must narrate the quota:\n${body.slice(0, 400)}`,
  );
  assert.ok(
    !/\berror:\s/i.test(body),
    `a refused cache write must never error the journey:\n${body.slice(0, 600)}`,
  );
  console.log("[memory-guard] quota hit reported; journey continued on provenance");
});

// ── S1 — transit prior (row `network-skip`) ─────────────────────────────────

/** The committed fixture's shard-body start (8-byte length + JSON header). */
function fixtureBodyStart() {
  const bytes = readFileSync(path.join(ROOT, "oracles/fixture/model.safetensors"));
  const headerLen = Number(new DataView(bytes.buffer, bytes.byteOffset, 8).getBigUint64(0, true));
  return 8 + headerLen;
}

/** Requests for the shard since `mark`, with their Range start (null = full body). */
async function shardRequestsSince(world, mark) {
  const log = await world.fixtureRequests();
  return log
    .slice(mark)
    .filter((entry) => entry.includes("model.safetensors"))
    .map((entry) => {
      const range = /#bytes=(\d+)-/.exec(entry);
      return { entry, rangeStart: range ? Number(range[1]) : null };
    });
}

When("the model directory is removed but the transit prior survives", async function () {
  // The κ-store (tensors/) and the transit prior (provenance/) persist; only
  // the model directory goes — the repeat download must rebuild it from the
  // known set without moving shard bytes.
  await this.page.evaluate(async () => {
    const root = await navigator.storage.getDirectory();
    const models = await root.getDirectoryHandle("models");
    await models.removeEntry("handshake-tiny", { recursive: true });
  });
  // The Models page scans OPFS on load; reload so the row offers Download.
  await this.page.reload({ waitUntil: "networkidle" });
  this.transitMark = (await this.fixtureRequests()).length;
});

When("the fixture model is downloaded again", async function () {
  await this.downloadModel(FIXTURE_DISPLAY);
});

Then("the repeat download transferred no shard body bytes", async function () {
  assert.ok(this.transitMark !== undefined, "the transit mark was set");
  const bodyStart = fixtureBodyStart();
  const shardRequests = await shardRequestsSince(this, this.transitMark);
  assert.ok(shardRequests.length > 0, "the repeat download read the shard header");
  const bodyTransfers = shardRequests.filter(
    (r) => r.rangeStart === null || r.rangeStart >= bodyStart,
  );
  assert.deepEqual(
    bodyTransfers.map((r) => r.entry),
    [],
    "known content must never re-transit — only header ranges may move",
  );
  console.log(
    `[network-skip] repeat download: ${shardRequests.length} header-range request(s), zero body bytes`,
  );
});

When("the shard content pin changes", async function () {
  const res = await fetch(`${this.fixtureBase()}/__etag-salt?value=changed-${Date.now()}`);
  assert.equal(res.status, 200, "the pin toggle responds");
  this.transitMark = (await this.fixtureRequests()).length;
});

Then("the repeat download streamed the shard body", async function () {
  assert.ok(this.transitMark !== undefined, "the transit mark was set");
  const bodyStart = fixtureBodyStart();
  const shardRequests = await shardRequestsSince(this, this.transitMark);
  const bodyTransfers = shardRequests.filter(
    (r) => r.rangeStart === null || r.rangeStart >= bodyStart,
  );
  assert.ok(
    bodyTransfers.length > 0,
    `a changed pin discards the prior — the shard must stream: ${JSON.stringify(shardRequests)}`,
  );
  console.log(`[network-skip] changed pin: ${bodyTransfers.length} body transfer(s)`);
});

// ── S1 — supported-only discovery ───────────────────────────────────────────

When("searching the catalog for {string}", async function (query) {
  await this.gotoModels();
  await this.page.locator('input[type="text"]').fill(query);
  await this.page.locator("button", { hasText: "Search Catalog" }).click();
  // The results panel renders once the (filtered) search resolves.
  await this.page.getByText("Search Results").waitFor({ timeout: 30_000 });
});

Then("the search results include the supported fixture model", async function () {
  const results = this.page.locator(".list-item", { hasText: FIXTURE_REPO });
  await results.first().waitFor({ timeout: 10_000 });
});

Then("the unsupported-family model appears refused with the preflight reason", async function () {
  // The refused repo is NOT hidden: it renders greyed with the preflight's
  // own error verbatim (the GPT-2 fixture cannot supply the generic decoder
  // schema, so the reason names the architecture).
  // The annotation lands when the async probe resolves — wait for the
  // ANNOTATED row, not merely the row (reading too early races the probe).
  const row = this.page.locator(".list-item", {
    hasText: SEARCH_UNSUPPORTED_REPO,
  });
  await row.first().waitFor({ timeout: 10_000 });
  const annotated = row.filter({ hasText: "Not runnable:" });
  await annotated.first().waitFor({ timeout: 10_000 });
  const text = await annotated.first().innerText();
  assert.ok(
    text.includes("GPT2LMHeadModel"),
    `the annotation must carry the preflight reason verbatim (naming the architecture):\n${text}`,
  );
});

// ── S1 — model preflight ────────────────────────────────────────────────────

const BARE_ENTRY = {
  description: "",
  modality: "text-chat",
  size: "?",
  approxArchiveMb: 0,
  quantize: "none",
  promptTemplate: null,
  stop: [],
  chatTurnSeparator: null,
};

async function attemptRejectedDownload(world, id, hfId, displayName) {
  await installCustomEntry(world, { ...BARE_ENTRY, id, hfId, displayName });
  await world.gotoModels();
  const row = world.page.locator(".list-item", { hasText: displayName });
  await row.locator("button", { hasText: "Download" }).click();
  await world.page.getByText(/error:/).first().waitFor({ timeout: 60_000 });
}

When("downloading a model whose config names an unsupported family", async function () {
  await attemptRejectedDownload(this, "unsupported-family", UNSUPPORTED_FAMILY_REPO, "unsupported family model");
  this.rejectedRepo = UNSUPPORTED_FAMILY_REPO;
});

Then("the journey is rejected at preflight naming the family", async function () {
  const body = await this.page.locator("body").innerText();
  assert.ok(
    body.includes("GPT2LMHeadModel"),
    `the rejection must name the unsupported family:\n${body}`,
  );
});

When("downloading a model whose config carries an unimplemented rope-scaling law", async function () {
  await attemptRejectedDownload(this, "exotic-rope", ROPE_EXOTIC_REPO, "exotic rope model");
  this.rejectedRepo = ROPE_EXOTIC_REPO;
});

Then("the journey is rejected at preflight naming the rope law", async function () {
  const body = await this.page.locator("body").innerText();
  assert.ok(
    body.includes("exotic"),
    `the rejection must name the unimplemented rope_scaling type:\n${body}`,
  );
});

When("downloading a model whose config lacks the required keys", async function () {
  await attemptRejectedDownload(this, "bad-config", BAD_CONFIG_REPO, "bad config model");
  this.rejectedRepo = BAD_CONFIG_REPO;
});

Then("the journey is rejected at preflight naming the missing key", async function () {
  const body = await this.page.locator("body").innerText();
  assert.ok(
    /missing required (numeric )?key/.test(body),
    `the rejection must name the missing key:\n${body}`,
  );
});

Then("no shard bytes were transferred for the rejected model", async function () {
  const log = await this.fixtureRequests();
  const shardHits = log.filter(
    (p) => p.includes(this.rejectedRepo.split("/")[1]) && p.endsWith(".safetensors"),
  );
  assert.deepEqual(shardHits, [], "preflight must reject before any shard request");
});

Then("the preflight validated the model before the first shard byte", async function () {
  const body = await this.page.locator("body").innerText();
  assert.ok(
    body.includes("Preflight passed"),
    `the preflight line must precede streaming:\n${body}`,
  );
});

// ── S1 — memory guard ───────────────────────────────────────────────────────

Then("the journey proceeds past the resource guard with figures surfaced", async function () {
  const body = await this.page.locator("body").innerText();
  assert.ok(
    body.includes("Resource projection:") &&
      body.includes("cache coverage") &&
      body.includes("execution window"),
    `the projection figures must be surfaced:\n${body}`,
  );
});

When(
  "downloading a model whose κ-store need exceeds the measured local headroom",
  async function () {
    await installCustomEntry(this, {
      id: "too-large",
      hfId: TOO_LARGE_REPO,
      displayName: "too large model",
      description: "",
      modality: "text-chat",
      size: "800GB",
      approxArchiveMb: 0,
      quantize: "none",
      promptTemplate: null,
      stop: [],
      chatTurnSeparator: null,
    });
    await this.gotoModels();
    const row = this.page.locator(".list-item", { hasText: "too large model" });
    await row.locator("button", { hasText: "Download" }).click();
    await this.page.getByText(/Resource projection:/).first().waitFor({ timeout: 60_000 });
  },
);

Then("the resource projection reports partial cache coverage", async function () {
  const body = await this.page.locator("body").innerText();
  const match = /cache coverage ~(\d+)%/.exec(body);
  assert.ok(match, `the projection must report cache coverage:\n${body}`);
  assert.ok(
    Number(match[1]) < 100,
    `an over-headroom model must report partial coverage, got ${match[1]}%`,
  );
});

Then("the journey is not refused at the guard", async function () {
  const body = await this.page.locator("body").innerText();
  assert.ok(
    !body.includes("Rejecting before transfer") && !body.includes("genuine storage shortfall"),
    `no resource figure may refuse the journey:\n${body}`,
  );
});

// ── S3 — κ-provenance resolution ────────────────────────────────────────────

Given("a zero local cache budget", async function () {
  // Nothing caches locally; every κ must resolve from recorded provenance.
  // Prior scenarios' state is cleared so the assertion below is meaningful.
  await this.page.evaluate(async () => {
    localStorage.setItem("hologram_cache_budget", "0");
    const root = await navigator.storage.getDirectory();
    try {
      const models = await root.getDirectoryHandle("models");
      await models.removeEntry("handshake-tiny", { recursive: true });
    } catch {}
    try {
      const tensors = await root.getDirectoryHandle("tensors");
      const names = [];
      for await (const [name] of tensors.entries()) names.push(name);
      for (const name of names) await tensors.removeEntry(name);
    } catch {}
  });
});

When("one cached fixture tensor is corrupted in the κ-store", async function () {
  // Flip a byte in one cached tensor (same length — the label no longer
  // reproduces). The journey must recover through recorded provenance and
  // evaporate the entry (row `saturation-residency`).
  this.corruptedKappa = await this.retryOpfs(() => this.page.evaluate(async () => {
    const root = await navigator.storage.getDirectory();
    const tensors = await root.getDirectoryHandle("tensors");
    for await (const [name, handle] of tensors.entries()) {
      if (handle.kind !== "file" || !name.endsWith(".bin")) continue;
      const bytes = new Uint8Array(await (await handle.getFile()).arrayBuffer());
      bytes[0] ^= 0xff;
      const writable = await handle.createWritable();
      await writable.write(bytes);
      await writable.close();
      return name.replace(/\.bin$/, "");
    }
    throw new Error("NotFoundError: no cached tensors to corrupt");
  }));
});

Then("the corrupted κ-store entry has evaporated", async function () {
  assert.ok(this.corruptedKappa, "a tensor was corrupted earlier in the scenario");
  const names = await this.opfsTensorNames();
  assert.ok(
    !names.includes(`${this.corruptedKappa}.bin`),
    `the failed entry must leave the cache: ${this.corruptedKappa} still present`,
  );
});

Then("the local κ-store holds no fixture tensors", async function () {
  const names = await this.opfsTensorNames();
  assert.deepEqual(names, [], `a zero budget must cache nothing, found: ${names}`);
});

Then("the model directory records κ provenance for every manifest tensor", async function () {
  const { compute_kappa } = await wasm();
  const raw = await this.opfsModelFile("handshake-tiny", "kappa-sources.json");
  assert.ok(raw, "kappa-sources.json missing from the model directory");
  const sources = JSON.parse(Buffer.from(raw).toString("utf8"));
  for (const [tensor, bytes] of fixtureTensors()) {
    const kappa = compute_kappa(bytes);
    const source = sources[kappa];
    assert.ok(source, `no provenance recorded for tensor ${tensor} (κ ${kappa})`);
    assert.ok(
      source.url && source.end > source.start,
      `malformed provenance for ${tensor}: ${JSON.stringify(source)}`,
    );
  }
});

Then("the completion matches reference turn 1", async function () {
  const reference = referenceTranscript();
  const completions = await this.page.evaluate(
    () => globalThis.__hologram_completions ?? [],
  );
  assert.ok(completions.length >= 1, "turn 1 did not complete");
  assert.equal(
    cleanCompletion(completions[0].text),
    reference.turns[0].completion,
    "provenance-resolved execution must reproduce the committed reference exactly",
  );
});

// ── S1 — companions ─────────────────────────────────────────────────────────

Then(
  "{string} is persisted under the model directory byte-identical to the server copy",
  async function (fileName) {
    const persisted = await this.opfsModelFile("handshake-tiny", fileName);
    assert.ok(persisted, `${fileName} missing from the model directory`);
    const server = readFileSync(path.join(ROOT, "oracles/fixture", fileName));
    assert.ok(
      Buffer.from(persisted).equals(server),
      `${fileName} differs from the server copy`,
    );
  },
);

// ── S4 — journey ────────────────────────────────────────────────────────────

Then(
  "the model directory holds a k-form archive whose κ-map is fully resolvable from OPFS",
  async function () {
    const { kappa_requirements } = await wasm();
    let holo;
    try {
      holo = await this.opfsArchive("handshake-tiny");
    } catch (err) {
      const tree = await this.opfsTree();
      throw new Error(`opfsArchive failed: ${err}\nOPFS tree:\n${tree.join("\n")}`);
    }
    assert.ok(holo, "no .holo archive in the model directory");
    const required = kappa_requirements(new Uint8Array(holo));
    assert.ok(required.length > 0, "the archive must be a k-form (κ-map present)");
    const names = await this.opfsTensorNames();
    for (const kappa of required) {
      assert.ok(names.includes(`${kappa}.bin`), `κ ${kappa} unresolvable from OPFS`);
    }
  },
);

Then("the model is listed as ready to chat", async function () {
  await this.gotoModels();
  const row = this.page.locator(".list-item", { hasText: FIXTURE_DISPLAY });
  await row.locator("button", { hasText: "Ready" }).waitFor({ timeout: 10_000 });
});

When("a single-turn prompt is sent", async function () {
  await this.gotoChat();
  await this.page.locator("input[type=number]").fill("0");
  this.lastCompletion = await this.sendChat(HANDSHAKE[0]);
});

Then("a non-empty completion streams back", async function () {
  // The hook records only *successfully finished* generations — a DOM bubble
  // could be an error message and must not satisfy this step.
  const completions = await this.page.evaluate(
    () => globalThis.__hologram_completions ?? [],
  );
  if (completions.length === 0) {
    const bubbles = await this.page.locator(".bubble.assistant").allInnerTexts();
    assert.fail(`generation did not complete; assistant bubbles: ${JSON.stringify(bubbles)}`);
  }
  assert.ok(
    cleanCompletion(completions[0].text).length > 0,
    `the assistant turn must stream non-empty text (raw: ${JSON.stringify(completions)})`,
  );
});

// ── S1 — quantized rest (row `quantized-rest`) ──────────────────────────────

Given("the quantized tier is forced", async function () {
  // The tier is a per-model catalogue statement; the knob forces it for the
  // hermetic fixture without touching the committed reference entry.
  await this.page.evaluate(() => localStorage.setItem("hologram_quantize", "int8"));
});

Then("the κ-store holds every quantized artifact and no gas-phase wide blob", async function () {
  const state = await this.page.evaluate(async () => {
    const root = await navigator.storage.getDirectory();
    const models = await root.getDirectoryHandle("models");
    const dir = await models.getDirectoryHandle("handshake-tiny");
    const fh = await dir.getFileHandle("stages.json");
    const meta = JSON.parse(await (await fh.getFile()).text());
    const tensors = await root.getDirectoryHandle("tensors");
    const names = [];
    for await (const name of tensors.keys()) names.push(name);
    return { quant: meta.quant ?? [], names };
  });
  assert.ok(
    state.quant.length > 0,
    "the quant-tiered download must record a quantized derived-artifact map",
  );
  for (const entry of state.quant) {
    assert.ok(
      state.names.includes(`${entry.artifact}.bin`),
      `quantized artifact ${entry.artifact} missing from the κ-store`,
    );
    assert.ok(
      !state.names.includes(`${entry.wide}.bin`),
      `wide blob ${entry.wide} must be gas-phase after its derivation crystallized`,
    );
  }
});

Then("the download narrated the quantized tier without erroring", async function () {
  const body = await this.page.locator("body").innerText();
  assert.ok(
    body.includes("Quantized tier"),
    `the download must state its tier under quota pressure:\n${body.slice(-600)}`,
  );
  assert.ok(
    !/\berror:\s/i.test(body),
    `a saturated quota must degrade the tier, never the journey:\n${body.slice(-600)}`,
  );
});

Then("the session narration states the quantized tier", async function () {
  const log = await statusLog(this);
  assert.ok(
    log.some((line) => line.includes("quantized tier")),
    `the pipeline must state its tier, never run it silently: ${JSON.stringify(log)}`,
  );
});

// ── S4 — windowed execution over k ──────────────────────────────────────────

Given("a forced single-layer execution window", async function () {
  // The stage-granularity knob forces finer staging without shrinking the
  // context; a stale monolithic compile of the fixture is removed so the
  // download recompiles staged (the κ-store keeps the tensors — dedup).
  await this.page.evaluate(async () => {
    localStorage.setItem("hologram_stage_window", "200000");
    const root = await navigator.storage.getDirectory();
    try {
      const models = await root.getDirectoryHandle("models");
      await models.removeEntry("handshake-tiny", { recursive: true });
    } catch {
      // No prior compile in this storage partition — nothing to remove.
    }
  });
});

Given("speculative decode is enabled", async function () {
  // The `hologram_speculative` knob = draft width K (row `speculative-decode`).
  // Greedy only, so it engages at temperature 0; the browser's staged decode
  // session drafts from the realized sequence's recurrence and verifies in one
  // M=K pass. Output must be byte-identical to plain decode.
  await this.page.evaluate(() => {
    localStorage.setItem("hologram_speculative", "4");
  });
});

Given("the fixture model is paired with itself as its own draft", async function () {
  // Catalogue pairing (row `speculative-draft-pairing`): set the fixture entry's
  // `draftModel` to its OWN hfId — a self-pairing is a guaranteed
  // vocabulary-compatible draft, so the browser wiring (paired resolution,
  // second-session build, `attach_draft` with its shared residency ledger, the
  // `ModelDrafter`) is witnessed without a second fixture model. The world's own
  // init script re-installs the base entry on every navigation, so this appends
  // AFTER it and reloads to take effect before the download reads the catalogue.
  await this.page.addInitScript((draftHfId) => {
    const cat = JSON.parse(localStorage.getItem("hologram_catalogue_custom") ?? "[]");
    for (const e of cat) if (e.hfId === draftHfId) e.draftModel = draftHfId;
    localStorage.setItem("hologram_catalogue_custom", JSON.stringify(cat));
  }, FIXTURE_REPO);
  await this.page.reload({ waitUntil: "networkidle" });
});

Then("the paired draft model is present in local storage", async function () {
  // The pairing is configured (the catalogue names the draft) AND the paired
  // draft's compiled artifact is resolvable — a self-pairing shares the target's
  // own dir, compiled by the download.
  const paired = await this.page.evaluate((draftHfId) => {
    const cat = JSON.parse(localStorage.getItem("hologram_catalogue_custom") ?? "[]");
    return cat.some((e) => e.draftModel === draftHfId);
  }, FIXTURE_REPO);
  assert.ok(paired, "the catalogue must pair the fixture with its draft model");
  const stages = await this.opfsModelFile("handshake-tiny", "stages.json");
  assert.ok(stages, "the paired draft dir must hold the compiled stages.json");
});

Then("the drafter reports the paired draft model attached", async function () {
  // The worker built the paired draft as a second session and `attach_draft`ed
  // it (vocab guard passed, one shared residency ledger) — the drafter is the
  // paired model, not prompt-lookup.
  const log = await statusLog(this);
  assert.ok(
    log.some((l) => l.includes("draft model attached")),
    `the worker must build and attach the paired draft model:\n${log.join("\n")}`,
  );
});

Given("the fixture is staged at its full context", async function () {
  // A stage-plan budget that FORCES staging (the decode-plan path) yet keeps the
  // fixture at its OWN context (128) — unlike the aggressive knob that shrinks
  // it to 64 (leaving no room to grow). So the decode bucket starts at 64 and
  // can regrow to 128 mid-turn. Remove any stale compile so the download
  // recompiles under this budget (the κ-store keeps the tensors — dedup).
  await this.page.evaluate(async () => {
    localStorage.setItem("hologram_stage_window", "1000000");
    const root = await navigator.storage.getDirectory();
    try {
      const models = await root.getDirectoryHandle("models");
      await models.removeEntry("handshake-tiny", { recursive: true });
    } catch {
      // No prior compile in this storage partition — nothing to remove.
    }
  });
});

Then("the decode bucket regrew to a wider window during the journey", async function () {
  const log = await statusLog(this);
  // The window observer narrates each (re)built window; a window WIDER than the
  // initial 64 proves a geometric growth ran through the real wasm decode path —
  // the exact transition that aborted a large model before the residency handoff.
  // A warm two-turn transcript accumulates past the 64-row bucket (context is the
  // model's own 128 here, not the shrunk-to-64 forced-staging config).
  const grew = log.some((l) => /\b(128|256|512)-token window\b/.test(l));
  assert.ok(
    grew,
    `the decode bucket must have grown past its initial 64 rows:\n${log.join("\n")}`,
  );
});

Given("a small weight-paging budget", async function () {
  // The weight-tier pager (row `lazy-constant-residency`): 1 MB of resident
  // paged-weight bytes per stage, well below the fixture's weight set, so
  // each stage pages its weights from the OPFS κ-store and evicts under the
  // budget instead of pinning them whole.
  await this.page.evaluate(() => {
    localStorage.setItem("hologram_weight_budget", "1");
  });
});

Then("the model directory holds a staged k-form bundle", async function () {
  const meta = await this.opfsModelFile("handshake-tiny", "stages.json");
  assert.ok(meta, "stages.json missing — the download did not compile staged");
  const parsed = JSON.parse(Buffer.from(meta).toString("utf8"));
  assert.ok(parsed.stageCount > 1, `expected a multi-stage bundle, got ${parsed.stageCount}`);
  const stages = await this.page.evaluate(async () => {
    const root = await navigator.storage.getDirectory();
    const models = await root.getDirectoryHandle("models");
    const dir = await models.getDirectoryHandle("handshake-tiny");
    const stagesDir = await dir.getDirectoryHandle("stages");
    const names = [];
    for await (const [name] of stagesDir.entries()) names.push(name);
    return names;
  });
  assert.equal(stages.length, parsed.stageCount, "stage archive count must match stages.json");
});

Then("the staged completion matches reference turn 1", async function () {
  const reference = referenceTranscript();
  const completions = await this.page.evaluate(
    () => globalThis.__hologram_completions ?? [],
  );
  assert.ok(completions.length >= 1, "turn 1 did not complete");
  assert.equal(
    cleanCompletion(completions[0].text),
    reference.turns[0].completion,
    "windowed execution must reproduce the committed reference exactly",
  );
});

// ── S4 — warm turns (row `warm-turn`) ───────────────────────────────────────

/** The generation status log, with the index where each turn began (a
 * "session warm/cold" line opens every staged turn). */
async function statusLog(world) {
  return world.page.evaluate(() => globalThis.__hologram_status ?? []);
}

Then("the second turn reports a warm session", async function () {
  const log = await statusLog(this);
  const sessionLines = log.filter((l) => l.startsWith("session "));
  // The invariant the warm-turn feature guarantees: the session is built EXACTLY
  // ONCE (a single "session cold"), then every turn after the build REUSES it
  // ("session warm"). The build is done by the eager prewarm if it fired (ADR-0018),
  // else by the first turn — so turn 1 may already be warm (prewarm moved the cold
  // build off its TTFT, which is the point). We assert the invariant, not that
  // turn 1 specifically is the cold one.
  const coldCount = sessionLines.filter((l) => l.startsWith("session cold")).length;
  assert.equal(coldCount, 1, `the session is built exactly once (cold):\n${log.join("\n")}`);
  assert.ok(
    sessionLines.length >= 2,
    `the single build plus at least one warm reuse ran:\n${log.join("\n")}`,
  );
  const coldIdx = sessionLines.findIndex((l) => l.startsWith("session cold"));
  assert.ok(
    sessionLines.slice(coldIdx + 1).every((l) => l.startsWith("session warm")),
    `every turn after the single build reuses the warm session:\n${log.join("\n")}`,
  );
  assert.ok(
    sessionLines[sessionLines.length - 1].startsWith("session warm"),
    `the last (second) turn reused the warm session:\n${log.join("\n")}`,
  );
});

Then("the second turn materializes no stages", async function () {
  const log = await statusLog(this);
  // The LAST staged turn (from its opening "session " line onward) is a warm
  // reuse — it must recompile / rematerialize NOTHING (a warm turn pays decode
  // only). Using the LAST turn is robust to the eager prewarm having opened an
  // earlier "session " line.
  let lastTurnStart = -1;
  for (let i = 0; i < log.length; i++) {
    if (log[i].startsWith("session ")) lastTurnStart = i;
  }
  assert.ok(lastTurnStart >= 0, "a staged turn opened in the status log");
  const lastTurn = log.slice(lastTurnStart);
  assert.ok(
    lastTurn[0].startsWith("session warm"),
    `the last turn is a warm reuse:\n${lastTurn.join("\n")}`,
  );
  assert.ok(
    !lastTurn.some((l) => l.includes("materialized") || l.includes("compiling")),
    `a warm turn pays decode only — no recompile, no rematerialization:\n${lastTurn.join("\n")}`,
  );
  console.log(`[warm-turn] last turn status: ${JSON.stringify(lastTurn)}`);
});

// ── S4 — the three-message handshake ────────────────────────────────────────

When("the user sends handshake message {int}", async function (n) {
  if (n === 1) {
    await this.gotoChat();
    await this.page.locator("input[type=number]").fill("0");
  }
  await this.sendChat(HANDSHAKE[n - 1]);
});

Then("assistant turn {int} streams a non-empty completion", async function (n) {
  const completions = await this.page.evaluate(
    () => globalThis.__hologram_completions ?? [],
  );
  if (completions.length < n) {
    const bubbles = await this.page.locator(".bubble.assistant").allInnerTexts();
    assert.fail(`turn ${n} did not complete; assistant bubbles: ${JSON.stringify(bubbles)}`);
  }
  assert.ok(
    cleanCompletion(completions[n - 1].text).length > 0,
    `assistant turn ${n} must be non-empty; raw completions: ${JSON.stringify(completions)}`,
  );
});

Then("the transcript matches the committed reference transcript", async function () {
  const reference = referenceTranscript();
  const completions = await this.page.evaluate(
    () => globalThis.__hologram_completions ?? [],
  );
  assert.equal(completions.length, reference.turns.length, "turn count");
  reference.turns.forEach((turn, i) => {
    // The hook records the pre-template slot; the reference stores the
    // templated prompt (template application is the shared Rust code path).
    assert.equal(
      reference.template.replace("{prompt}", completions[i].prompt),
      turn.prompt,
      `turn ${i + 1} prompt assembly deviates from the committed reference`,
    );
    assert.equal(
      cleanCompletion(completions[i].text),
      turn.completion,
      `assistant turn ${i + 1} deviates from the committed reference`,
    );
  });
});

// ── S4 — the session window ─────────────────────────────────────────────────

const OVERFLOW_MESSAGE = "please keep talking about the weather today ".repeat(2).trim();

When("the user sends a message that overflows the context window", async function () {
  // The fixture tokenizes per character (context 128): the accumulated
  // three-turn history plus this message exceeds the window, forcing the
  // oldest-turn trim; the message alone still fits the model's own context.
  await this.sendChat(OVERFLOW_MESSAGE);
});

Then("the overflow turn completes without error", async function () {
  const completions = await this.page.evaluate(() => globalThis.__hologram_completions ?? []);
  assert.equal(completions.length, 4, "the overflow turn must complete (no dead-end)");
  const bubbles = await this.page.locator(".bubble.assistant").allInnerTexts();
  assert.ok(
    !bubbles.some((b) => b.includes("error:")),
    `no turn may error:\n${JSON.stringify(bubbles)}`,
  );
});

Then("the overflow prompt omits the oldest turn", async function () {
  const completions = await this.page.evaluate(() => globalThis.__hologram_completions ?? []);
  const overflowPrompt = completions[3].prompt;
  assert.ok(overflowPrompt.includes(OVERFLOW_MESSAGE), "the pending message must survive trimming");
  assert.ok(
    !overflowPrompt.includes(HANDSHAKE[0]),
    `the oldest turn must be trimmed from the prompt:\n${overflowPrompt}`,
  );
});

// ── S4 — the live journey (@live) ───────────────────────────────────────────

When("the pinned SmolLM2 model is downloaded", async function () {
  await this.downloadModel("SmolLM2 135M Instruct", { timeoutMs: 1_500_000 });
});

Then("the model reaches the runnable state", async function () {
  const row = this.page.locator(".list-item", { hasText: "SmolLM2 135M Instruct" });
  await row.locator("button", { hasText: "Ready" }).waitFor({ timeout: 10_000 });
});

When("the user completes the three-message handshake", async function () {
  await this.gotoChat();
  await this.page.locator("input[type=number]").fill("0");
  for (const message of HANDSHAKE) {
    await this.sendChat(message, { timeoutMs: 1_500_000 });
  }
});

Then(
  "every assistant turn streams a non-empty completion respecting stop conditions",
  async function () {
    const completions = await this.page.evaluate(
      () => globalThis.__hologram_completions ?? [],
    );
    assert.equal(completions.length, HANDSHAKE.length, "three turns must complete");
    for (const completion of completions) {
      assert.ok(completion.text.trim().length > 0, "every turn must be non-empty");
      assert.ok(!completion.text.includes("<|im_end|>"), "stop token must terminate the turn");
    }
  },
);
