// Step definitions for the browser-executor dictionary rows (S1 + S4).
// Every step drives the real app in real Chromium — no mocks between the UI
// and the substrate. κ verification runs the same wasm binding node-side.
import { Given, When, Then } from "@cucumber/cucumber";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { referenceTranscript } from "./world.mjs";
import { BAD_CONFIG_REPO, FIXTURE_REPO, MISSING_REPO, SEARCH_UNSUPPORTED_REPO, TOO_LARGE_REPO, UNSUPPORTED_FAMILY_REPO } from "./fixture-server.mjs";

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

Then("the search results do not include the unsupported-family model", async function () {
  const body = await this.page.locator("body").innerText();
  assert.ok(
    !body.includes(SEARCH_UNSUPPORTED_REPO),
    `the GPT2-family repo must be filtered out of discovery:\n${body}`,
  );
});

Then("each search result names its architecture family", async function () {
  const metas = await this.page
    .locator(".list-item", { hasText: FIXTURE_REPO })
    .locator(".meta")
    .allInnerTexts();
  assert.ok(
    metas.some((m) => m.includes("Family: LlamaForCausalLM")),
    `the result must carry its family badge: ${JSON.stringify(metas)}`,
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
