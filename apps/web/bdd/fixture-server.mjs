// The hermetic model server: a HuggingFace-Hub-API-compatible static server
// over the committed fixture (oracles/fixture/). The BDD world points the app
// at it via the `hologram_hf_base` localStorage override, so the journey runs
// the genuine download/compile/materialize/chat path with zero mocks in the
// app itself. Also serves a `too-large` repo (for the memory-guard row) and
// records every request for the "no shard bytes moved" assertion.
import { createServer } from "node:http";
import { createHash } from "node:crypto";
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
export const SEARCH_UNSUPPORTED_REPO = "hologram-fixture/gpt2-tiny";
// The SAME committed fixture under a second identity — a genuine model switch
// (two model dirs, two sessions) with zero extra fixture bytes.
export const SECOND_FIXTURE_REPO = "hologram-fixture/handshake-tiny-b";
// The fixture weights under a config that adds an IMPLEMENTED scaled-rope law
// (llama3) — the full journey must download, compile, and stream chat with the
// scaled frequency tables.
export const ROPE_SCALED_REPO = "hologram-fixture/handshake-tiny-llama3rope";
// The fixture config with an UNIMPLEMENTED rope_scaling type — preflight must
// refuse naming the law, before any shard byte.
export const ROPE_EXOTIC_REPO = "hologram-fixture/exotic-rope";
// A DEEP hermetic model at PRODUCTION head_dim (128), SYNTHESIZED at serve time from the same
// deterministic weight law as the committed fixture (zero repo bytes): the
// real-model journey SHAPE — many stages, the int8 tier, growth — that the
// 2-layer fixture cannot exercise (the class of failure that shipped in
// PR #13/#14 hid exactly here).
export const DEEP_FIXTURE_REPO = "hologram-fixture/deep-tiny";

const FIXTURE_FILES = [
  "config.json",
  "model.safetensors",
  "tokenizer.json",
  "tokenizer_config.json",
  "generation_config.json",
];

// A valid repo whose architecture is outside the parametric decoder schema:
// GPT-2 names its dimensions `n_embd` / `n_layer` (learned positions + Conv1D
// attention), not the generic decoder schema the recipe consumes. There is no
// name allowlist — an unknown family is normally derived from its manifest —
// but this config cannot even supply the decoder's dimensions, so preflight
// rejects on config alone, naming the architecture.
const UNSUPPORTED_FAMILY_CONFIG = JSON.stringify({
  architectures: ["GPT2LMHeadModel"],
  n_embd: 768,
  n_layer: 12,
  n_head: 12,
  vocab_size: 50257,
  n_positions: 1024,
  torch_dtype: "float32",
  model_type: "gpt2",
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

// The committed fixture config with extra keys layered on — the rope-law repos
// share the fixture's weights (rope_scaling changes no tensor shape).
const fixtureConfigWith = (extra) =>
  JSON.stringify({
    ...JSON.parse(readFileSync(path.join(FIXTURE_DIR, "config.json"), "utf8")),
    ...extra,
  });
const ROPE_SCALED_CONFIG = fixtureConfigWith({
  rope_scaling: {
    rope_type: "llama3",
    factor: 4.0,
    low_freq_factor: 1.0,
    high_freq_factor: 4.0,
    original_max_position_embeddings: 32,
  },
});
const ROPE_EXOTIC_CONFIG = fixtureConfigWith({
  rope_scaling: { rope_type: "exotic", factor: 2.0 },
});

// ── The synthesized deep fixture ─────────────────────────────────────────────

// PRODUCTION head_dim (128 = DEEP_HIDDEN/DEEP_HEADS = 256/2), GQA (kv_heads 1),
// SwiGLU MLP, 6 layers — the exact attention geometry the v0.9.0 fused decode
// path runs on real Qwen/Llama checkpoints, which the committed fixture's
// head_dim 16 never exercised. Context 512 so a longer prompt grows the bucket
// (the regrow path). Vocab 512 reuses the committed tokenizer.
//
// SMALL hidden by design: it keeps every stage fast to compile so the forced
// STAGING + residency eviction — the deployed real-model shape (stages dropped
// and re-materialized between decode steps, the kv_shadow carry path) — is
// exercised in wasm without a multi-minute compile. head_dim 128 is the
// production value the shallow fixtures never reached; the staged carry across
// an evicted stage is the one browser path the head_dim-16 fixtures cannot
// reach (see `deep_model_journey.feature`, the staged scenario). The dims below
// are the source of truth — keep this comment matched to them.
const DEEP_LAYERS = 6;
const DEEP_HIDDEN = 256;
const DEEP_HEADS = 2;
const DEEP_KV_HEADS = 1;
const DEEP_INTER = 512;
const DEEP_VOCAB = 512;
const DEEP_CONFIG = JSON.stringify({
  architectures: ["LlamaForCausalLM"],
  hidden_size: DEEP_HIDDEN,
  intermediate_size: DEEP_INTER,
  num_hidden_layers: DEEP_LAYERS,
  num_attention_heads: DEEP_HEADS,
  num_key_value_heads: DEEP_KV_HEADS,
  vocab_size: DEEP_VOCAB,
  rms_norm_eps: 1e-6,
  rope_theta: 10000.0,
  max_position_embeddings: 512,
  tie_word_embeddings: false,
  torch_dtype: "float32",
  bos_token_id: 1,
  eos_token_id: 2,
  model_type: "llama",
});

/** The deep manifest: the Llama tensor schema at the deep dims. */
function deepManifest() {
  const kv = DEEP_KV_HEADS * (DEEP_HIDDEN / DEEP_HEADS);
  const m = [["model.embed_tokens.weight", [DEEP_VOCAB, DEEP_HIDDEN]]];
  for (let l = 0; l < DEEP_LAYERS; l++) {
    const p = `model.layers.${l}`;
    m.push([`${p}.input_layernorm.weight`, [DEEP_HIDDEN]]);
    m.push([`${p}.self_attn.q_proj.weight`, [DEEP_HIDDEN, DEEP_HIDDEN]]);
    m.push([`${p}.self_attn.k_proj.weight`, [kv, DEEP_HIDDEN]]);
    m.push([`${p}.self_attn.v_proj.weight`, [kv, DEEP_HIDDEN]]);
    m.push([`${p}.self_attn.o_proj.weight`, [DEEP_HIDDEN, DEEP_HIDDEN]]);
    m.push([`${p}.post_attention_layernorm.weight`, [DEEP_HIDDEN]]);
    m.push([`${p}.mlp.gate_proj.weight`, [DEEP_INTER, DEEP_HIDDEN]]);
    m.push([`${p}.mlp.up_proj.weight`, [DEEP_INTER, DEEP_HIDDEN]]);
    m.push([`${p}.mlp.down_proj.weight`, [DEEP_HIDDEN, DEEP_INTER]]);
  }
  m.push(["model.norm.weight", [DEEP_HIDDEN]]);
  m.push(["lm_head.weight", [DEEP_VOCAB, DEEP_HIDDEN]]);
  return m;
}

/** The committed fixture generator's exact weight law (xtask fixture.rs
 * `bytes_for`): norms are 1.0, everything else cycles ((k % 13) − 6) · 0.01. */
function deepTensorBytes(name, dims) {
  const n = dims.reduce((a, b) => a * b, 1);
  const norm = name.includes("layernorm") || name.endsWith(".norm.weight");
  const out = Buffer.alloc(n * 4);
  for (let k = 0; k < n; k++) {
    out.writeFloatLE(norm ? 1.0 : ((k % 13) - 6) * 0.01, k * 4);
  }
  return out;
}

/** Assemble a valid single-file safetensors image (u64-LE header length,
 * JSON header, raw little-endian tensor data), built once per process. */
let deepSafetensors = null;
function deepSafetensorsBytes() {
  if (deepSafetensors) return deepSafetensors;
  const manifest = deepManifest();
  const header = {};
  let offset = 0;
  const buffers = [];
  for (const [name, dims] of manifest) {
    const bytes = deepTensorBytes(name, dims);
    header[name] = { dtype: "F32", shape: dims, data_offsets: [offset, offset + bytes.length] };
    offset += bytes.length;
    buffers.push(bytes);
  }
  const headerJson = Buffer.from(JSON.stringify(header));
  const len = Buffer.alloc(8);
  len.writeBigUInt64LE(BigInt(headerJson.length));
  deepSafetensors = Buffer.concat([len, headerJson, ...buffers]);
  return deepSafetensors;
}

// The deep repo serves the committed fixture's tokenizer/config companions
// (vocab 512 covers DEEP_VOCAB) with its own config.json + synthesized shard.
const DEEP_COMPANIONS = ["tokenizer.json", "tokenizer_config.json"];
// The deep fixture serves its OWN generation_config with eos moved to an
// unused high token so the deterministic noise weights do NOT emit
// end-of-sequence on the first step — the turn generates multiple tokens and
// thereby exercises the SECOND decode step, the first that binds the resident
// K/V carry (where the deployed real model traps `unreachable`).
const DEEP_GENERATION_CONFIG = JSON.stringify({
  bos_token_id: 1,
  chat_template: "User:\n{prompt}\nAssistant:\n",
  eos_token_id: 500,
});

export function startFixtureServer() {
  const requests = [];
  // The content-pin salt: changing it changes every served ETag, so a
  // recorded transit prior no longer matches (row `network-skip`).
  let etagSalt = "";
  const server = createServer((req, res) => {
    const url = new URL(req.url, "http://localhost");
    requests.push(url.pathname + (req.headers.range ? `#${req.headers.range}` : ""));
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
    // headers via ranged requests — kilobytes, never the shard body). Every
    // binary response carries a strong content-derived ETag — the pin the
    // exact-repeat transit prior keys on, like the HF Hub's blob hash.
    const sendBytes = (body) => {
      const etag = `"${createHash("sha256").update(body).update(etagSalt).digest("hex").slice(0, 32)}"`;
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
          ETag: etag,
          "Access-Control-Expose-Headers": "ETag, Content-Range",
        });
        res.end(slice);
        return;
      }
      res.writeHead(200, {
        "Content-Type": "application/octet-stream",
        "Content-Length": body.length,
        ETag: etag,
        "Access-Control-Expose-Headers": "ETag, Content-Range",
      });
      res.end(body);
    };

    if (url.pathname === "/__log") {
      send(200, JSON.stringify(requests));
      return;
    }
    if (url.pathname === "/__etag-salt") {
      etagSalt = url.searchParams.get("value") ?? "";
      send(200, JSON.stringify({ etagSalt }));
      return;
    }
    if (url.pathname === "/api/models" && url.searchParams.has("search")) {
      send(
        200,
        JSON.stringify([
          { id: FIXTURE_REPO, downloads: 1000, tags: ["text-generation"] },
          { id: SEARCH_UNSUPPORTED_REPO, downloads: 999, tags: ["text-generation"] },
        ]),
      );
      return;
    }
    if (url.pathname === `/api/models/${SEARCH_UNSUPPORTED_REPO}`) {
      send(
        200,
        JSON.stringify({
          id: SEARCH_UNSUPPORTED_REPO,
          config: { architectures: ["GPT2LMHeadModel"] },
          siblings: [{ rfilename: "config.json", size: 200 }],
        }),
      );
      return;
    }
    for (const repo of [FIXTURE_REPO, SECOND_FIXTURE_REPO, ROPE_SCALED_REPO]) {
      if (url.pathname === `/api/models/${repo}`) {
        const siblings = FIXTURE_FILES.map((name) => {
          const size =
            repo === ROPE_SCALED_REPO && name === "config.json"
              ? Buffer.byteLength(ROPE_SCALED_CONFIG)
              : readFileSync(path.join(FIXTURE_DIR, name)).length;
          return { rfilename: name, size };
        });
        send(
          200,
          JSON.stringify({
            id: repo,
            config: { architectures: ["LlamaForCausalLM"] },
            siblings,
          }),
        );
        return;
      }
    }
    if (url.pathname === `/api/models/${DEEP_FIXTURE_REPO}`) {
      const siblings = [
        { rfilename: "config.json", size: Buffer.byteLength(DEEP_CONFIG) },
        { rfilename: "model.safetensors", size: deepSafetensorsBytes().length },
        { rfilename: "generation_config.json", size: Buffer.byteLength(DEEP_GENERATION_CONFIG) },
        ...DEEP_COMPANIONS.map((name) => ({
          rfilename: name,
          size: readFileSync(path.join(FIXTURE_DIR, name)).length,
        })),
      ];
      send(
        200,
        JSON.stringify({
          id: DEEP_FIXTURE_REPO,
          config: { architectures: ["LlamaForCausalLM"] },
          siblings,
        }),
      );
      return;
    }
    if (url.pathname === `/api/models/${ROPE_EXOTIC_REPO}`) {
      send(
        200,
        JSON.stringify({
          id: ROPE_EXOTIC_REPO,
          siblings: [
            { rfilename: "config.json", size: Buffer.byteLength(ROPE_EXOTIC_CONFIG) },
            { rfilename: "model.safetensors", size: 4 * 1024 ** 2 },
          ],
        }),
      );
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

    for (const repo of [FIXTURE_REPO, SECOND_FIXTURE_REPO, ROPE_SCALED_REPO]) {
      const resolve = `/${repo}/resolve/main/`;
      if (url.pathname.startsWith(resolve)) {
        const name = url.pathname.slice(resolve.length);
        if (repo === ROPE_SCALED_REPO && name === "config.json") {
          sendBytes(Buffer.from(ROPE_SCALED_CONFIG));
          return;
        }
        const file = path.join(FIXTURE_DIR, name);
        if (!FIXTURE_FILES.includes(name) || !existsSync(file)) {
          send(404, "not found", "text/plain");
          return;
        }
        sendBytes(readFileSync(file));
        return;
      }
    }
    // The search-derivability probe fetches THIS repo's config at search
    // time (never at download — the repo is refused before that): serve it
    // so the annotation carries the preflight's own reason.
    const searchUnsupportedResolve = `/${SEARCH_UNSUPPORTED_REPO}/resolve/main/`;
    if (url.pathname.startsWith(searchUnsupportedResolve)) {
      const name = url.pathname.slice(searchUnsupportedResolve.length);
      if (name === "config.json") {
        send(200, UNSUPPORTED_FAMILY_CONFIG);
        return;
      }
      send(500, "shard access must never happen for the gpt2 search repo", "text/plain");
      return;
    }
    const deepResolve = `/${DEEP_FIXTURE_REPO}/resolve/main/`;
    if (url.pathname.startsWith(deepResolve)) {
      const name = url.pathname.slice(deepResolve.length);
      if (name === "config.json") {
        send(200, DEEP_CONFIG);
        return;
      }
      if (name === "model.safetensors") {
        sendBytes(deepSafetensorsBytes());
        return;
      }
      if (name === "generation_config.json") {
        send(200, DEEP_GENERATION_CONFIG);
        return;
      }
      if (DEEP_COMPANIONS.includes(name)) {
        sendBytes(readFileSync(path.join(FIXTURE_DIR, name)));
        return;
      }
      send(404, "not found", "text/plain");
      return;
    }
    const exoticResolve = `/${ROPE_EXOTIC_REPO}/resolve/main/`;
    if (url.pathname.startsWith(exoticResolve)) {
      const name = url.pathname.slice(exoticResolve.length);
      if (name === "config.json") {
        send(200, ROPE_EXOTIC_CONFIG);
        return;
      }
      // Preflight must refuse on config alone: any shard request here is a
      // failure the request log exposes.
      send(500, "shard access must never happen for the exotic-rope repo", "text/plain");
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
