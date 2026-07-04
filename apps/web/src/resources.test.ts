// Unit tests for the parametric resource model (dictionary row
// `memory-guard`): pure functions of the model's own configuration.
import { describe, expect, it } from "vitest";
import {
  chooseContextLength,
  dtypeBytes,
  environmentBudgetBytes,
  estimateResources,
  formatBytes,
} from "./resources";

const TINY = {
  hidden_size: 64,
  num_hidden_layers: 2,
  max_position_embeddings: 128,
  torch_dtype: "float32",
};

const HUGE = {
  hidden_size: 16384,
  num_hidden_layers: 120,
  max_position_embeddings: 131072,
  torch_dtype: "bfloat16",
};

describe("environmentBudgetBytes", () => {
  it("caps at the wasm32 ceiling", () => {
    expect(environmentBudgetBytes(64)).toBe(4 * 1024 ** 3);
  });
  it("halves reported device memory", () => {
    expect(environmentBudgetBytes(4)).toBe(2 * 1024 ** 3);
  });
  it("falls back to the ceiling when unreported", () => {
    expect(environmentBudgetBytes(undefined)).toBe(2 * 1024 ** 3);
  });
});

describe("chooseContextLength", () => {
  it("never exceeds the model's own max_position_embeddings", () => {
    expect(chooseContextLength(TINY, 4 * 1024 ** 3)).toBeLessThanOrEqual(128);
  });
  it("shrinks with the budget — a function of config, not a constant", () => {
    const large = chooseContextLength(HUGE, 4 * 1024 ** 3);
    const small = chooseContextLength(HUGE, 256 * 1024 ** 2);
    expect(small).toBeLessThanOrEqual(large);
  });
});

describe("estimateResources", () => {
  it("inflates BF16 checkpoints to their F32 working set", () => {
    const budget = 4 * 1024 ** 3;
    const shard = 100 * 1024 ** 2;
    const est = estimateResources({ ...TINY, torch_dtype: "bfloat16" }, shard, budget);
    expect(est.runtimeBytes).toBeGreaterThan(shard * 4);
    expect(est.storageBytes).toBe(shard);
  });
  it("rejects-by-numbers: an 800 GiB shard exceeds any browser budget", () => {
    const budget = environmentBudgetBytes(64);
    const est = estimateResources(HUGE, 800 * 1024 ** 3, budget);
    expect(est.runtimeBytes).toBeGreaterThan(budget);
  });
});

describe("dtypeBytes / formatBytes", () => {
  it("maps safetensors dtype tags", () => {
    expect(dtypeBytes("F32")).toBe(4);
    expect(dtypeBytes("BF16")).toBe(2);
    expect(dtypeBytes("bfloat16")).toBe(2);
    expect(dtypeBytes("F64")).toBe(8);
  });
  it("prints human figures", () => {
    expect(formatBytes(2 * 1024 ** 3)).toBe("2.0 GiB");
    expect(formatBytes(350 * 1024 ** 2)).toBe("350.0 MiB");
  });
});
