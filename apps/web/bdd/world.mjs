// The cucumber World: real Chromium (Playwright) against the built app
// (vite preview) and the hermetic fixture server. One browser per run; a
// fresh context (fresh OPFS/localStorage) per scenario.
import { setWorldConstructor, setDefaultTimeout, BeforeAll, AfterAll, Before, After } from "@cucumber/cucumber";
import { chromium } from "playwright";
import { spawn } from "node:child_process";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { startFixtureServer, FIXTURE_REPO, DEEP_FIXTURE_REPO } from "./fixture-server.mjs";

const WEB_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const ROOT = path.resolve(WEB_DIR, "../..");
const BASE_PATH = "/hologram-ai/";

setDefaultTimeout(180_000);

let browser;
let fixture;
let preview;
let previewPort;

/** The committed deterministic reference (oracle `journey-reference`). */
export function referenceTranscript() {
  return JSON.parse(
    readFileSync(path.join(ROOT, "oracles/fixture/reference-transcript.json"), "utf8"),
  );
}

function waitForOutput(child, pattern) {
  return new Promise((resolve, reject) => {
    let seen = "";
    const timer = setTimeout(
      () => reject(new Error(`preview server did not start; output so far:\n${seen}`)),
      90_000,
    );
    const onData = (chunk) => {
      // Strip ANSI color codes: on CI vite styles the URL, splitting the
      // host:port match with escape sequences.
      // eslint-disable-next-line no-control-regex
      seen += chunk.toString().replace(/\x1b\[[0-9;]*m/g, "");
      const match = seen.match(pattern);
      if (match) {
        clearTimeout(timer);
        resolve(match);
      }
    };
    child.stdout.on("data", onData);
    child.stderr.on("data", onData);
    child.on("error", (err) => reject(new Error(`preview failed to spawn: ${err}`)));
    child.on("exit", (code) => reject(new Error(`preview exited early (${code}); output:\n${seen}`)));
  });
}

BeforeAll(async () => {
  fixture = await startFixtureServer();
  // `pnpm build` must have run first (the `bdd` script does); preview serves
  // the exact bundle the deployment would publish.
  preview = spawn("pnpm", ["exec", "vite", "preview", "--host", "127.0.0.1", "--port", "4173", "--strictPort"], {
    cwd: WEB_DIR,
    stdio: ["ignore", "pipe", "pipe"],
    env: { ...process.env, NO_COLOR: "1", FORCE_COLOR: "0" },
  });
  const match = await waitForOutput(preview, /(?:localhost|127\.0\.0\.1):(\d+)/);
  previewPort = Number(match[1]);
  browser = await chromium.launch();
});

AfterAll(async () => {
  if (browser) await browser.close();
  if (preview) preview.kill();
  if (fixture) await fixture.close();
});

class HologramWorld {
  constructor() {
    this.appUrl = `http://localhost:${previewPort}${BASE_PATH}`;
    this.fixtureRepo = FIXTURE_REPO;
    this.turns = [];
  }

  fixtureBase() {
    return fixture.base;
  }

  async fixtureRequests() {
    const res = await fetch(`${fixture.base}/__log`);
    return res.json();
  }

  /** Open the app in a fresh context, pointed at `hfBase` (the fixture server
   * unless a live run overrides), with the fixture catalogue entry installed
   * from the committed reference (data, not code). */
  /** Install the DEEP hermetic fixture as a catalogue entry (int8 tier,
   * many stages) and point the app at the fixture server — the real-model
   * journey SHAPE with zero network. Must run before navigation. */
  async installDeepFixture() {
    await this.page.addInitScript(
      ([base, entryJson]) => {
        localStorage.setItem("hologram_hf_base", base);
        const cur = JSON.parse(localStorage.getItem("hologram_catalogue_custom") ?? "[]");
        cur.push(JSON.parse(entryJson));
        localStorage.setItem("hologram_catalogue_custom", JSON.stringify(cur));
      },
      [
        fixture.base,
        JSON.stringify({
          id: "deep-tiny",
          hfId: DEEP_FIXTURE_REPO,
          displayName: "deep-tiny (head_dim-128 hermetic fixture)",
          description: "The real-model-shape journey fixture (many stages, int8).",
          modality: "text-chat",
          size: "tiny-deep",
          approxArchiveMb: 28,
          quantize: "int8",
          promptTemplate: "User:\n{prompt}\nAssistant:\n",
          stop: [],
          chatTurnSeparator: "\nAssistant: {response}\nUser: ",
          // Enough tokens to run well past the SECOND decode step (the first
          // resident-K/V-carry step) where the deployed model traps.
          maxTokens: 6,
        }),
      ],
    );
    await this.page.reload({ waitUntil: "networkidle" });
  }

  async openApp({ live = false } = {}) {
    this.context = await browser.newContext();
    this.page = await this.context.newPage();
    this.consoleErrors = [];
    this.page.on("pageerror", (err) => {
      this.lastPageError = String(err);
      this.consoleErrors.push(`pageerror: ${err}`);
    });
    this.page.on("console", (m) => {
      if (m.type() === "error" || m.type() === "warning") {
        this.consoleErrors.push(`console.${m.type()}: ${m.text()}`);
      }
    });
    if (!live) {
      const reference = referenceTranscript();
      const hfBase = fixture.base;
      const entry = {
        id: "handshake-tiny",
        hfId: FIXTURE_REPO,
        displayName: "handshake-tiny (hermetic fixture)",
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
      await this.page.addInitScript(
        ([base, entryJson]) => {
          localStorage.setItem("hologram_hf_base", base);
          localStorage.setItem("hologram_catalogue_custom", entryJson);
        },
        [hfBase, JSON.stringify([entry])],
      );
    } else {
      // The app ships NO default models (catalogue.json is empty — arbitrary
      // models only), so the live journey adds its target as a user would: a
      // custom entry. Its chat template + stops are DERIVED from the model's own
      // tokenizer_config at run time, so the entry carries no per-model boilerplate.
      const entry = {
        id: "smollm2-135m-instruct",
        hfId: "HuggingFaceTB/SmolLM2-135M-Instruct",
        displayName: "SmolLM2 135M Instruct",
        description: "Live-journey target (user-added).",
        modality: "text-chat",
        size: "135M",
        approxArchiveMb: 0,
        quantize: "int8",
        promptTemplate: null,
        stop: [],
        chatTurnSeparator: null,
      };
      await this.page.addInitScript(
        (entryJson) => localStorage.setItem("hologram_catalogue_custom", entryJson),
        JSON.stringify([entry]),
      );
    }
    await this.page.goto(this.appUrl, { waitUntil: "networkidle" });
  }

  async gotoModels() {
    await this.page.goto(`${this.appUrl}#/models`);
    await this.page.waitForSelector("h1:has-text('Models')");
  }

  async gotoChat() {
    await this.page.goto(`${this.appUrl}#/chat`);
    await this.page.waitForSelector("h1:has-text('Chat')");
  }

  /** Click Download on a catalogue row and wait until it reads Ready. */
  async downloadModel(displayName, { timeoutMs = 150_000 } = {}) {
    await this.gotoModels();
    const row = this.page.locator(".list-item", { hasText: displayName });
    await row.locator("button", { hasText: "Download" }).click();
    await row.locator("button", { hasText: "Ready" }).waitFor({ timeout: timeoutMs });
  }

  /** Evaluate inside the page: list OPFS κ-store entries (`tensors/`). */
  async opfsTensorNames() {
    return this.retryOpfs(() => this.page.evaluate(async () => {
      const root = await navigator.storage.getDirectory();
      const dir = await root.getDirectoryHandle("tensors");
      const names = [];
      for await (const [name] of dir.entries()) names.push(name);
      return names;
    }));
  }

  /** Read a file under OPFS `models/<dir>/` (returns bytes as number[]). */
  async opfsModelFile(modelDir, fileName) {
    return this.retryOpfs(() => this.page.evaluate(
      async ([dirName, name]) => {
        const root = await navigator.storage.getDirectory();
        const models = await root.getDirectoryHandle("models");
        const dir = await models.getDirectoryHandle(dirName);
        async function find(handle, target) {
          for await (const [n, h] of handle.entries()) {
            if (h.kind === "file" && n === target) return h;
            if (h.kind === "directory") {
              const found = await find(h, target);
              if (found) return found;
            }
          }
          return null;
        }
        const fh = await find(dir, name);
        if (!fh) return null;
        const file = await fh.getFile();
        return Array.from(new Uint8Array(await file.arrayBuffer()));
      },
      [modelDir, fileName],
    ));
  }

  /** Dump the OPFS tree (diagnostics). */
  async opfsTree() {
    return this.page.evaluate(async () => {
      const root = await navigator.storage.getDirectory();
      async function walk(dir, prefix) {
        const out = [];
        for await (const [name, handle] of dir.entries()) {
          out.push(`${prefix}${name}${handle.kind === "directory" ? "/" : ""}`);
          if (handle.kind === "directory") out.push(...(await walk(handle, `${prefix}${name}/`)));
        }
        return out;
      }
      return walk(root, "");
    });
  }

  /** Retry `fn` briefly: OPFS directory entries written by another realm
   * (worker/page) can lag visibility for a moment after creation. */
  async retryOpfs(fn) {
    let lastErr;
    for (let attempt = 0; attempt < 20; attempt++) {
      try {
        return await fn();
      } catch (err) {
        lastErr = err;
        if (!String(err).includes("NotFoundError")) throw err;
        await new Promise((r) => setTimeout(r, 250));
      }
    }
    throw lastErr;
  }

  /** The OPFS `.holo` under models/<dir>/ (first match), as number[]. */
  async opfsArchive(modelDir) {
    return this.retryOpfs(() => this.page.evaluate(async (dirName) => {
      const root = await navigator.storage.getDirectory();
      const models = await root.getDirectoryHandle("models");
      const dir = await models.getDirectoryHandle(dirName);
      for await (const [name, handle] of dir.entries()) {
        if (handle.kind === "file" && name.endsWith(".holo")) {
          const file = await handle.getFile();
          return Array.from(new Uint8Array(await file.arrayBuffer()));
        }
      }
      return null;
    }, modelDir));
  }

  /** Send a chat message and wait for the assistant turn to finish. */
  async sendChat(text, { timeoutMs = 150_000 } = {}) {
    await this.page.locator(".composer textarea").fill(text);
    await this.page.locator(".composer button", { hasText: "Send" }).click();
    // Generation is done when the composer shows Send again (not Cancel).
    await this.page
      .locator(".composer button", { hasText: "Send" })
      .waitFor({ timeout: timeoutMs });
    const bubbles = this.page.locator(".bubble.assistant .md");
    const count = await bubbles.count();
    const last = await bubbles.nth(count - 1).innerText();
    this.turns.push(last);
    return last;
  }

  async close() {
    if (this.context) await this.context.close();
  }
}

setWorldConstructor(HologramWorld);

Before(async function () {
  this.lastPageError = null;
});

After(async function () {
  await this.close();
});
