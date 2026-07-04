// Step definitions for the browser-executor dictionary rows (S1 + S4).
// Every step drives the real app in real Chromium — no mocks between the UI
// and the substrate. κ verification runs the same wasm binding node-side.
import { Given, When, Then } from "@cucumber/cucumber";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { referenceTranscript } from "./world.mjs";
import { MISSING_REPO, TOO_LARGE_REPO } from "./fixture-server.mjs";

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
  await world.page.evaluate((json) => {
    localStorage.setItem("hologram_catalogue_custom", json);
  }, JSON.stringify([entry]));
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

When("the fixture model is downloaded", async function () {
  await this.downloadModel(FIXTURE_DISPLAY);
});

Given("the fixture model is downloaded", async function () {
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

// ── S1 — memory guard ───────────────────────────────────────────────────────

Then("the journey proceeds past the memory guard", async function () {
  const body = await this.page.locator("body").innerText();
  assert.ok(
    body.includes("Memory guard:") && body.includes("within budget"),
    `the guard line must be visible:\n${body}`,
  );
});

When(
  "downloading a model whose config-derived estimate exceeds the environment budget",
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
    await this.page.getByText(/Memory guard:/).first().waitFor({ timeout: 60_000 });
  },
);

Then("the journey is rejected at the memory guard with the estimate", async function () {
  const body = await this.page.locator("body").innerText();
  assert.ok(
    body.includes("Memory guard:") && body.includes("Rejecting before transfer"),
    `the guard rejection (with figures) must be visible:\n${body}`,
  );
});

Then("no shard bytes were transferred", async function () {
  const log = await this.fixtureRequests();
  const shardHits = log.filter(
    (p) => p.includes(TOO_LARGE_REPO.split("/")[1]) && p.endsWith(".safetensors"),
  );
  assert.deepEqual(shardHits, [], "the guard must reject before any shard request");
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
    const holo = await this.opfsArchive("handshake-tiny");
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
  assert.ok(
    this.lastCompletion && this.lastCompletion.trim().length > 0,
    "the assistant turn must stream non-empty text",
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
  assert.ok(completions.length >= n, `turn ${n} did not complete`);
  assert.ok(
    cleanCompletion(completions[n - 1]).length > 0,
    `assistant turn ${n} must be non-empty`,
  );
});

Then("the transcript matches the committed reference transcript", async function () {
  const reference = referenceTranscript();
  const completions = await this.page.evaluate(
    () => globalThis.__hologram_completions ?? [],
  );
  assert.equal(completions.length, reference.turns.length, "turn count");
  reference.turns.forEach((turn, i) => {
    assert.equal(
      cleanCompletion(completions[i]),
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
      assert.ok(completion.trim().length > 0, "every turn must be non-empty");
      assert.ok(!completion.includes("<|im_end|>"), "stop token must terminate the turn");
    }
  },
);
