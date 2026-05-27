#!/usr/bin/env python3
"""EE-3: verify hologram-ai's SmolLM2 prefill logits against ONNX Runtime.

ONNX Runtime is the external authority. hologram-ai runs the with-past decoder
export as an empty-past full-recompute prefill; ORT runs the same .onnx with a
genuinely-empty past (Python ORT accepts 0-length dims, unlike the Rust crate),
so both compute the identical causal prefill and their logits[1, S, vocab] must
agree. Per-position argmax (the next-token prediction) must match exactly.

Usage:
  # 1) hologram-ai logits:
  cargo run -q -p hologram-ai --example dump_logits -- \
      models/smollm2-135m/model.onnx /tmp/holo_logits.f32
  # 2) compare:
  python3 scripts/verify_logits_vs_ort.py \
      models/smollm2-135m/model.onnx /tmp/holo_logits.f32
"""
import sys
import numpy as np
import onnxruntime as ort

# Must match crates/hologram-ai/examples/dump_logits.rs.
TOKENS = [1, 450, 7483, 310, 3444, 338, 263, 8444]


def main(model_path: str, holo_path: str) -> int:
    s = len(TOKENS)
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    # Discover the past_key_values port names + their KV-head / head-dim sizes.
    feeds = {
        "input_ids": np.array([TOKENS], dtype=np.int64),
        "attention_mask": np.ones((1, s), dtype=np.int64),
        "position_ids": np.array([list(range(s))], dtype=np.int64),
    }
    for inp in sess.get_inputs():
        if inp.name.startswith("past_key_values"):
            # [batch, n_kv_heads, past_len=0, head_dim]
            _, n_kv, _, hd = inp.shape
            feeds[inp.name] = np.zeros((1, int(n_kv), 0, int(hd)), dtype=np.float32)

    logits = sess.run(["logits"], feeds)[0].astype(np.float32).reshape(-1)
    holo = np.fromfile(holo_path, dtype=np.float32)

    if holo.shape != logits.shape:
        print(f"FAIL: shape mismatch — hologram-ai {holo.shape} vs ORT {logits.shape}")
        return 1

    vocab = logits.size // s
    ho, re = holo.reshape(s, vocab), logits.reshape(s, vocab)
    max_abs = float(np.max(np.abs(ho - re)))
    max_rel = float(np.max(np.abs(ho - re) / (np.abs(re) + 1e-6)))
    holo_argmax = ho.argmax(axis=1)
    ort_argmax = re.argmax(axis=1)
    mism = int(np.sum(holo_argmax != ort_argmax))

    print(f"positions={s} vocab={vocab}")
    print(f"max|diff|={max_abs:.4e}  max rel={max_rel:.4e}")
    print(f"hologram-ai argmax: {holo_argmax.tolist()}")
    print(f"ORT         argmax: {ort_argmax.tolist()}")
    print(f"argmax mismatches: {mism}/{s}")

    ok = mism == 0 and max_abs < 1.0
    print("RESULT:", "PASS ✓" if ok else "FAIL ✗")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
