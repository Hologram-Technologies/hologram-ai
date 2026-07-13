// Witnesses for the search-time derivability preflight (row
// `supported-search`): the decision logic is a pure function of the injected
// preflight — no architecture names, no wasm needed here. Each test pins one
// honest behavior: derivable → selectable, refused → the preflight's reason
// VERBATIM, missing config named, cache hits cost no fetch.
import { describe, expect, it, vi } from "vitest";
import {
  ProbeCache,
  lacksSafetensorsExport,
  mapBounded,
  probeCandidate,
  probeWithCache,
} from "./searchPreflight";

const CONFIG_URL = "https://hub.example/org/repo/resolve/main/config.json";

function fetchReturning(status: number, body = "{}"): typeof fetch {
  return vi.fn(async () => new Response(body, { status })) as unknown as typeof fetch;
}

describe("lacksSafetensorsExport — the cheap first pass", () => {
  it("keeps a repo whose listing shows a safetensors export", () => {
    expect(
      lacksSafetensorsExport([
        { rfilename: "config.json" },
        { rfilename: "model-00001-of-00002.safetensors" },
      ]),
    ).toBe(false);
  });
  it("drops a repo whose listing proves there is no safetensors export", () => {
    expect(
      lacksSafetensorsExport([{ rfilename: "model.onnx" }, { rfilename: "config.json" }]),
    ).toBe(true);
  });
  it("keeps a repo whose listing carries no file info — the probe decides", () => {
    expect(lacksSafetensorsExport(undefined)).toBe(false);
    expect(lacksSafetensorsExport(null)).toBe(false);
  });
});

describe("probeCandidate — the preflight is the only authority", () => {
  it("a derivable config is selectable", async () => {
    const validate = vi.fn(async () => undefined);
    const fetchImpl = fetchReturning(200, '{"architectures":["AnyNewDecoder"]}');
    const outcome = await probeCandidate(CONFIG_URL, validate, fetchImpl);
    expect(outcome).toEqual({ status: "derivable" });
    // The preflight saw the config text itself, not a re-serialization.
    expect(validate).toHaveBeenCalledWith('{"architectures":["AnyNewDecoder"]}');
  });

  it("an unimplemented knob is refused WITH the preflight's reason verbatim", async () => {
    // The real wasm binding throws a plain string (JsValue::from_str) shaped
    // like parametric.rs's reject_unsupported_knobs message.
    const reason =
      "config.json carries `rope_scaling` — the `Qwen2ForCausalLM` family builder does not " +
      "implement this knob and refuses to silently ignore a semantic config key";
    const validate = () => {
      throw reason; // wasm-bindgen throws the JsValue string, not an Error
    };
    const outcome = await probeCandidate(CONFIG_URL, validate, fetchReturning(200));
    expect(outcome.status).toBe("refused");
    expect(outcome.status === "refused" && outcome.reason).toBe(reason);
  });

  it("an Error-throwing preflight surfaces its message verbatim too", async () => {
    const validate = () => {
      throw new Error(
        "architecture `FooNet` is not a recognized family and its config does not supply the generic decoder schema",
      );
    };
    const outcome = await probeCandidate(CONFIG_URL, validate, fetchReturning(200));
    expect(outcome.status === "refused" && outcome.reason).toMatch(
      /architecture `FooNet` is not a recognized family/,
    );
  });

  it("a missing config.json is refused naming the missing file", async () => {
    const validate = vi.fn();
    const outcome = await probeCandidate(CONFIG_URL, validate, fetchReturning(404, "not found"));
    expect(outcome.status).toBe("refused");
    expect(outcome.status === "refused" && outcome.reason).toMatch(/config\.json missing/);
    expect(outcome.status === "refused" && outcome.reason).toMatch(/404/);
    expect(validate).not.toHaveBeenCalled(); // nothing to validate
  });

  it("a network failure is an honest, TRANSIENT refusal — never a throw", async () => {
    const fetchImpl = vi.fn(async () => {
      throw new TypeError("Failed to fetch");
    }) as unknown as typeof fetch;
    const outcome = await probeCandidate(CONFIG_URL, vi.fn(), fetchImpl);
    expect(outcome.status).toBe("refused");
    expect(outcome.status === "refused" && outcome.transient).toBe(true);
    expect(outcome.status === "refused" && outcome.reason).toMatch(/config\.json unreachable/);
  });
});

function fakeStorage(): Pick<Storage, "getItem" | "setItem"> & { backing: Map<string, string> } {
  const backing = new Map<string, string>();
  return {
    backing,
    getItem: (k: string) => backing.get(k) ?? null,
    setItem: (k: string, v: string) => void backing.set(k, v),
  };
}

describe("ProbeCache + probeWithCache — a verdict is fetched once", () => {
  it("a cache hit costs no second fetch", async () => {
    const cache = new ProbeCache(fakeStorage());
    const fetchImpl = fetchReturning(200);
    const validate = vi.fn(async () => undefined);

    const first = await probeWithCache("org/repo", "abc123", CONFIG_URL, cache, validate, fetchImpl);
    const second = await probeWithCache("org/repo", "abc123", CONFIG_URL, cache, validate, fetchImpl);

    expect(first).toEqual({ status: "derivable" });
    expect(second).toEqual(first);
    expect(fetchImpl).toHaveBeenCalledTimes(1);
    expect(validate).toHaveBeenCalledTimes(1);
  });

  it("refusals are cached too — an honest no is as stable as a yes", async () => {
    const cache = new ProbeCache(fakeStorage());
    const fetchImpl = fetchReturning(404);
    await probeWithCache("org/repo", undefined, CONFIG_URL, cache, vi.fn(), fetchImpl);
    const second = await probeWithCache("org/repo", undefined, CONFIG_URL, cache, vi.fn(), fetchImpl);
    expect(second.status).toBe("refused");
    expect(fetchImpl).toHaveBeenCalledTimes(1);
  });

  it("a TRANSIENT (network) refusal is NOT cached — the next search retries", async () => {
    const cache = new ProbeCache(fakeStorage());
    const fetchImpl = vi.fn(async () => {
      throw new TypeError("Failed to fetch");
    }) as unknown as typeof fetch;
    await probeWithCache("org/repo", undefined, CONFIG_URL, cache, vi.fn(), fetchImpl);
    await probeWithCache("org/repo", undefined, CONFIG_URL, cache, vi.fn(), fetchImpl);
    expect(fetchImpl).toHaveBeenCalledTimes(2);
  });

  it("keys by revision: a new revision re-probes, the old verdict stays", async () => {
    const cache = new ProbeCache(fakeStorage());
    const fetchImpl = fetchReturning(200);
    await probeWithCache("org/repo", "rev1", CONFIG_URL, cache, vi.fn(), fetchImpl);
    await probeWithCache("org/repo", "rev2", CONFIG_URL, cache, vi.fn(), fetchImpl);
    expect(fetchImpl).toHaveBeenCalledTimes(2);
    expect(cache.get("org/repo", "rev1")).toEqual({ status: "derivable" });
    expect(cache.get("org/repo", "rev2")).toEqual({ status: "derivable" });
  });

  it("mirrors verdicts to storage — a fresh cache over the same session re-fetches nothing", async () => {
    const storage = fakeStorage();
    const first = new ProbeCache(storage);
    await probeWithCache("org/repo", "abc", CONFIG_URL, first, vi.fn(), fetchReturning(200));

    const rehydrated = new ProbeCache(storage);
    const fetchImpl = fetchReturning(200);
    const outcome = await probeWithCache("org/repo", "abc", CONFIG_URL, rehydrated, vi.fn(), fetchImpl);
    expect(outcome).toEqual({ status: "derivable" });
    expect(fetchImpl).not.toHaveBeenCalled();
  });

  it("a corrupt storage mirror never blocks probing", async () => {
    const storage = fakeStorage();
    storage.backing.set("hologram_search_preflight_v1", "not json {");
    const cache = new ProbeCache(storage);
    const outcome = await probeWithCache("org/repo", undefined, CONFIG_URL, cache, vi.fn(), fetchReturning(200));
    expect(outcome).toEqual({ status: "derivable" });
  });
});

describe("mapBounded — probes are bounded, complete, and non-throwing by contract", () => {
  it("never exceeds the in-flight bound and processes every item", async () => {
    let inFlight = 0;
    let peak = 0;
    const seen: number[] = [];
    await mapBounded([...Array(11).keys()], 4, async (item) => {
      inFlight++;
      peak = Math.max(peak, inFlight);
      await new Promise((r) => setTimeout(r, 1));
      seen.push(item);
      inFlight--;
    });
    expect(peak).toBeLessThanOrEqual(4);
    expect(seen.sort((a, b) => a - b)).toEqual([...Array(11).keys()]);
  });

  it("handles an empty list and a bound larger than the list", async () => {
    const fn = vi.fn(async () => undefined);
    await mapBounded([], 4, fn);
    expect(fn).not.toHaveBeenCalled();
    await mapBounded([1], 8, fn);
    expect(fn).toHaveBeenCalledTimes(1);
  });
});
