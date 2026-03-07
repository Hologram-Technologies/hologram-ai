#!/usr/bin/env python3
"""Generate small ONNX test fixtures for hologram-ai integration tests.

Requirements:
    pip install onnx numpy

Usage:
    python3 scripts/gen-fixtures.py
"""

import struct
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import os

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures", "onnx")
os.makedirs(FIXTURES, exist_ok=True)


def make_tiny_mlp(vocab=32, embd=64, out=32, seq=3):
    """
    Minimal embedding + linear model:
        token_ids (u32 [seq]) → Gather → embd (f32 [seq, embd])
                              → MatMul(W) → logits (f32 [seq, out])

    All weights are zero-initialised for determinism.
    """
    # Embedding table: vocab × embd
    embd_w = numpy_helper.from_array(
        np.zeros((vocab, embd), dtype=np.float32), name="embed_weight"
    )
    # Linear weight: embd × out
    linear_w = numpy_helper.from_array(
        np.zeros((embd, out), dtype=np.float32), name="linear_weight"
    )

    gather = helper.make_node("Gather", inputs=["token_ids", "embed_weight"],
                              outputs=["embedded"], axis=0)
    matmul = helper.make_node("MatMul", inputs=["embedded", "linear_weight"],
                              outputs=["logits"])

    graph = helper.make_graph(
        nodes=[gather, matmul],
        name="tiny_mlp",
        inputs=[
            helper.make_tensor_value_info("token_ids", TensorProto.INT64, [seq]),
        ],
        outputs=[
            helper.make_tensor_value_info("logits", TensorProto.FLOAT, [seq, out]),
        ],
        initializer=[embd_w, linear_w],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)

    path = os.path.join(FIXTURES, "tiny-mlp.onnx")
    onnx.save(model, path)
    print(f"wrote {path}  ({os.path.getsize(path)} bytes)")

    # Write golden shape
    import json
    shape_path = os.path.join(FIXTURES, "tiny-mlp-output-shape.json")
    with open(shape_path, "w") as f:
        json.dump([seq, out], f)
    print(f"wrote {shape_path}")


def make_identity(dim=16):
    """Simplest possible model: Identity op. Used for smoke tests."""
    identity = helper.make_node("Identity", inputs=["x"], outputs=["y"])
    graph = helper.make_graph(
        [identity], "identity",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, dim])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, dim])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)

    path = os.path.join(FIXTURES, "identity.onnx")
    onnx.save(model, path)
    print(f"wrote {path}  ({os.path.getsize(path)} bytes)")


if __name__ == "__main__":
    make_identity()
    make_tiny_mlp()
    print("done.")
