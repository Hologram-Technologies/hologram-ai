// The hermetic model server: a HuggingFace-Hub-API-compatible static server
// over the committed fixture (oracles/fixture/). The BDD world points the app
// at it via the `hologram_hf_base` localStorage override, so the journey runs
// the genuine download/compile/materialize/chat path with zero mocks in the
// app itself. Also serves a `too-large` repo (for the memory-guard row) and
// records every request for the "no shard bytes moved" assertion.
import { createServer } from "node:http";
import { readFileSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const FIXTURE_DIR = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../../../oracles/fixture",
);

export const FIXTURE_REPO = "hologram-fixture/handshake-tiny";
export const TOO_LARGE_REPO = "hologram-fixture/too-large";
export const MISSING_REPO = "hologram-fixture/does-not-exist";
export const UNSUPPORTED_FAMILY_REPO = "hologram-fixture/unsupported-family";
export const BAD_CONFIG_REPO = "hologram-fixture/bad-config";

const FIXTURE_FILES = [
  "config.json",
  "model.safetensors",
  "tokenizer.json",
  "tokenizer_config.json",
  "generation_config.json",
];

// The user-reported phi-4 shape: a valid repo whose family the parametric
// registry does not support. Preflight must reject it on config alone.
const UNSUPPORTED_FAMILY_CONFIG = JSON.stringify({
  architectures: ["Phi3ForCausalLM"],
  hidden_size: 5120,
  intermediate_size: 17920,
  num_hidden_layers: 40,
  num_attention_heads: 40,
  num_key_value_heads: 10,
  vocab_size: 100352,
  rms_norm_eps: 1e-5,
  rope_theta: 250000.0,
  max_position_embeddings: 16384,
  tie_word_embeddings: false,
  torch_dtype: "bfloat16",
  model_type: "phi3",
});

// A config that is not a model config at all (the tokenizer_config-overwrite
// failure mode): no architectures, no dimensions.
const BAD_CONFIG = JSON.stringify({
  tokenizer_class: "GPT2Tokenizer",
  model_max_length: 16384,
});

// A config whose parametric estimate exceeds any browser budget. The shard
// size below is what the guard sees via `?blobs=true`.
const TOO_LARGE_CONFIG = JSON.stringify({
  architectures: ["LlamaForCausalLM"],
  hidden_size: 16384,
  intermediate_size: 65536,
  num_hidden_layers: 120,
  num_attention_heads: 128,
  num_key_value_heads: 8,
  vocab_size: 128000,
  rms_norm_eps: 1e-5,
  rope_theta: 500000.0,
  max_position_embeddings: 131072,
  tie_word_embeddings: false,
  torch_dtype: "bfloat16",
  model_type: "llama",
});
const TOO_LARGE_SHARD_BYTES = 800 * 1024 ** 3;

export function startFixtureServer() {
  const requests = [];
  const server = createServer((req, res) => {
    const url = new URL(req.url, "http://localhost");
    requests.push(url.pathname);
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Access-Control-Allow-Headers", "*");
    if (req.method === "OPTIONS") {
      res.writeHead(204);
      res.end();
      return;
    }

    const send = (status, body, type = "application/json") => {
      res.writeHead(status, { "Content-Type": type, "Content-Length": Buffer.byteLength(body) });
      res.end(body);
    };
    // Honor Range for binary payloads (the preflight reads safetensors
    // headers via ranged requests — kilobytes, never the shard body).
    const sendBytes = (body) => {
      const range = req.headers.range;
      const match = range && /^bytes=(\d+)-(\d+)?$/.exec(range);
      if (match) {
        const start = Number(match[1]);
        const end = match[2] !== undefined ? Math.min(Number(match[2]), body.length - 1) : body.length - 1;
        const slice = body.subarray ? body.subarray(start, end + 1) : body.slice(start, end + 1);
        res.writeHead(206, {
          "Content-Type": "application/octet-stream",
          "Content-Length": slice.length,
          "Content-Range": `bytes ${start}-${end}/${body.length}`,
        });
        res.end(slice);
        return;
      }
      send(200, body, "application/octet-stream");
    };

    if (url.pathname === "/__log") {
      send(200, JSON.stringify(requests));
      return;
    }
    if (url.pathname === `/api/models/${FIXTURE_REPO}`) {
      const siblings = FIXTURE_FILES.map((name) => {
        const size = readFileSync(path.join(FIXTURE_DIR, name)).length;
        return { rfilename: name, size };
      });
      send(200, JSON.stringify({ id: FIXTURE_REPO, siblings }));
      return;
    }
    if (url.pathname === `/api/models/${TOO_LARGE_REPO}`) {
      send(
        200,
        JSON.stringify({
          id: TOO_LARGE_REPO,
          siblings: [
            { rfilename: "config.json", size: TOO_LARGE_CONFIG.length },
            { rfilename: "model.safetensors", size: TOO_LARGE_SHARD_BYTES },
          ],
        }),
      );
      return;
    }
    if (url.pathname === `/api/models/${UNSUPPORTED_FAMILY_REPO}`) {
      send(
        200,
        JSON.stringify({
          id: UNSUPPORTED_FAMILY_REPO,
          siblings: [
            { rfilename: "config.json", size: UNSUPPORTED_FAMILY_CONFIG.length },
            { rfilename: "model.safetensors", size: 4 * 1024 ** 2 },
          ],
        }),
      );
      return;
    }
    if (url.pathname === `/api/models/${BAD_CONFIG_REPO}`) {
      send(
        200,
        JSON.stringify({
          id: BAD_CONFIG_REPO,
          siblings: [
            { rfilename: "config.json", size: BAD_CONFIG.length },
            { rfilename: "model.safetensors", size: 4 * 1024 ** 2 },
          ],
        }),
      );
      return;
    }
    if (url.pathname.startsWith("/api/models/")) {
      send(404, JSON.stringify({ error: "Repository not found" }));
      return;
    }

    const fixtureResolve = `/${FIXTURE_REPO}/resolve/main/`;
    if (url.pathname.startsWith(fixtureResolve)) {
      const name = url.pathname.slice(fixtureResolve.length);
      const file = path.join(FIXTURE_DIR, name);
      if (!FIXTURE_FILES.includes(name) || !existsSync(file)) {
        send(404, "not found", "text/plain");
        return;
      }
      sendBytes(readFileSync(file));
      return;
    }
    const unsupportedResolve = `/${UNSUPPORTED_FAMILY_REPO}/resolve/main/`;
    if (url.pathname.startsWith(unsupportedResolve)) {
      const name = url.pathname.slice(unsupportedResolve.length);
      if (name === "config.json") {
        send(200, UNSUPPORTED_FAMILY_CONFIG);
        return;
      }
      // Preflight must reject on config alone: any shard request here is a
      // failure the request log exposes.
      send(500, "shard access must never happen for the unsupported-family repo", "text/plain");
      return;
    }
    const badConfigResolve = `/${BAD_CONFIG_REPO}/resolve/main/`;
    if (url.pathname.startsWith(badConfigResolve)) {
      const name = url.pathname.slice(badConfigResolve.length);
      if (name === "config.json") {
        send(200, BAD_CONFIG);
        return;
      }
      send(500, "shard access must never happen for the bad-config repo", "text/plain");
      return;
    }
    const tooLargeResolve = `/${TOO_LARGE_REPO}/resolve/main/`;
    if (url.pathname.startsWith(tooLargeResolve)) {
      const name = url.pathname.slice(tooLargeResolve.length);
      if (name === "config.json") {
        send(200, TOO_LARGE_CONFIG);
        return;
      }
      // The memory guard must reject BEFORE any shard transfer; a shard
      // request against this repo is itself a failure the log exposes.
      send(500, "shard transfer must never happen for the too-large repo", "text/plain");
      return;
    }
    send(404, "not found", "text/plain");
  });

  return new Promise((resolve) => {
    server.listen(0, "127.0.0.1", () => {
      const { port } = server.address();
      resolve({
        server,
        port,
        base: `http://127.0.0.1:${port}`,
        requests,
        close: () => new Promise((r) => server.close(r)),
      });
    });
  });
}
