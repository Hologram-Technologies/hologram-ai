#!/usr/bin/env python3
"""Generate ONNX fixture files for hologram-ai conformance tests.

Run from the repo root:
    python3 crates/hologram-ai-conformance/fixtures/generate.py

Each fixture is a minimal ONNX model exercising a specific op pattern.
Tests load these files from disk instead of building models programmatically.
"""

import os
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

FIXTURES_DIR = os.path.dirname(os.path.abspath(__file__))


def save(model, name):
    path = os.path.join(FIXTURES_DIR, f"{name}.onnx")
    onnx.checker.check_model(model)
    onnx.save(model, path)
    size = os.path.getsize(path)
    print(f"  {name}.onnx ({size} bytes)")
    return path


def mk_weight(shape, offset=0.0):
    n = 1
    for d in shape:
        n *= d
    return np.array(
        [(i % 64) * 0.02 - 0.5 + offset for i in range(n)], dtype=np.float32
    ).reshape(shape)


# ── Basic ops ─────────────────────────────────────────────────────────────────


def gen_matmul():
    """MatMul: A[2,4] @ B[4,3] → C[2,3]."""
    m, k, n = 2, 4, 3
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [m, k])
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [k, n])
    C = helper.make_tensor_value_info("C", TensorProto.FLOAT, [m, n])
    node = helper.make_node("MatMul", ["A", "B"], ["C"])
    graph = helper.make_graph([node], "matmul", [A, B], [C])
    return save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]),
        "matmul",
    )


def gen_softmax():
    """Softmax axis=-1 on [2,8]."""
    rows, size = 2, 8
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [rows, size])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [rows, size])
    node = helper.make_node("Softmax", ["X"], ["Y"], axis=-1)
    graph = helper.make_graph([node], "softmax", [X], [Y])
    return save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]),
        "softmax",
    )


def gen_gemm_trans_b():
    """Gemm with trans_b=1: A[3,4] @ B^T[2,4] + bias[2] → C[3,2]."""
    m, k, n = 3, 4, 2
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [m, k])
    C = helper.make_tensor_value_info("C", TensorProto.FLOAT, [m, n])
    B = numpy_helper.from_array(mk_weight([n, k], 0.1), "B")
    bias = numpy_helper.from_array(mk_weight([n], 0.2), "bias")
    node = helper.make_node(
        "Gemm", ["A", "B", "bias"], ["C"], alpha=1.0, beta=1.0, transA=0, transB=1
    )
    graph = helper.make_graph([node], "gemm_trans_b", [A], [C], initializer=[B, bias])
    return save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]),
        "gemm_trans_b",
    )


# ── Composite normalization ───────────────────────────────────────────────────


def gen_rms_norm():
    """RmsNorm composite: x / rms(x) * weight. [2,16]."""
    rows, size = 2, 16
    eps = 1e-6
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [rows, size])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [rows, size])
    weight = numpy_helper.from_array(np.ones(size, dtype=np.float32) * 0.5, "weight")
    eps_t = numpy_helper.from_array(np.array(eps, dtype=np.float32), "eps")

    nodes = [
        helper.make_node("Mul", ["X", "X"], ["x_sq"]),
        helper.make_node("ReduceMean", ["x_sq"], ["mean_sq"], axes=[-1], keepdims=1),
        helper.make_node("Add", ["mean_sq", "eps"], ["mean_sq_eps"]),
        helper.make_node("Sqrt", ["mean_sq_eps"], ["rms"]),
        helper.make_node("Div", ["X", "rms"], ["normed"]),
        helper.make_node("Mul", ["normed", "weight"], ["Y"]),
    ]
    graph = helper.make_graph(
        nodes, "rms_norm", [X], [Y], initializer=[weight, eps_t]
    )
    return save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]),
        "rms_norm",
    )


def gen_layer_norm():
    """LayerNorm composite: (x - mean) / sqrt(var + eps) * w + b. [2,16]."""
    rows, size = 2, 16
    eps = 1e-5
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [rows, size])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [rows, size])
    weight = numpy_helper.from_array(np.ones(size, dtype=np.float32) * 0.5, "weight")
    bias = numpy_helper.from_array(np.zeros(size, dtype=np.float32) + 0.1, "bias")
    eps_t = numpy_helper.from_array(np.array(eps, dtype=np.float32), "eps")

    nodes = [
        helper.make_node("ReduceMean", ["X"], ["mean"], axes=[-1], keepdims=1),
        helper.make_node("Sub", ["X", "mean"], ["x_centered"]),
        helper.make_node("Mul", ["x_centered", "x_centered"], ["x_sq"]),
        helper.make_node("ReduceMean", ["x_sq"], ["var"], axes=[-1], keepdims=1),
        helper.make_node("Add", ["var", "eps"], ["var_eps"]),
        helper.make_node("Sqrt", ["var_eps"], ["std"]),
        helper.make_node("Div", ["x_centered", "std"], ["normed"]),
        helper.make_node("Mul", ["normed", "weight"], ["scaled"]),
        helper.make_node("Add", ["scaled", "bias"], ["Y"]),
    ]
    graph = helper.make_graph(
        nodes, "layer_norm", [X], [Y], initializer=[weight, bias, eps_t]
    )
    return save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]),
        "layer_norm",
    )


# ── 4D tensor / attention ops ────────────────────────────────────────────────


def gen_batched_matmul_4d():
    """Batched 4D MatMul: [1,4,6,8] @ [1,4,8,6] → [1,4,6,6]."""
    batch, heads, seq, hd = 1, 4, 6, 8
    Q = helper.make_tensor_value_info("Q", TensorProto.FLOAT, [batch, heads, seq, hd])
    K = helper.make_tensor_value_info("K", TensorProto.FLOAT, [batch, heads, hd, seq])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [batch, heads, seq, seq])
    node = helper.make_node("MatMul", ["Q", "K"], ["Y"])
    graph = helper.make_graph([node], "batched_matmul_4d", [Q, K], [Y])
    return save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]),
        "batched_matmul_4d",
    )


def gen_concat_4d_last_axis():
    """Concat along axis=3: [1,4,6,8] ++ [1,4,6,8] → [1,4,6,16]."""
    batch, heads, seq, half = 1, 4, 6, 8
    A = helper.make_tensor_value_info(
        "A", TensorProto.FLOAT, [batch, heads, seq, half]
    )
    B = helper.make_tensor_value_info(
        "B", TensorProto.FLOAT, [batch, heads, seq, half]
    )
    Y = helper.make_tensor_value_info(
        "Y", TensorProto.FLOAT, [batch, heads, seq, half * 2]
    )
    node = helper.make_node("Concat", ["A", "B"], ["Y"], axis=3)
    graph = helper.make_graph([node], "concat_4d_last_axis", [A, B], [Y])
    return save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]),
        "concat_4d_last_axis",
    )


def gen_scaled_dot_product_attention():
    """SDPA: Q@K^T → scale → softmax → @V. [1,4,6,8]."""
    batch, heads, seq, hd = 1, 4, 6, 8
    scale = 1.0 / (hd**0.5)
    Q = helper.make_tensor_value_info("Q", TensorProto.FLOAT, [batch, heads, seq, hd])
    K = helper.make_tensor_value_info("K", TensorProto.FLOAT, [batch, heads, seq, hd])
    V = helper.make_tensor_value_info("V", TensorProto.FLOAT, [batch, heads, seq, hd])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [batch, heads, seq, hd])
    scale_t = numpy_helper.from_array(np.array(scale, dtype=np.float32), "scale")

    nodes = [
        helper.make_node("Transpose", ["K"], ["K_T"], perm=[0, 1, 3, 2]),
        helper.make_node("MatMul", ["Q", "K_T"], ["QK"]),
        helper.make_node("Mul", ["QK", "scale"], ["QK_s"]),
        helper.make_node("Softmax", ["QK_s"], ["attn_w"], axis=-1),
        helper.make_node("MatMul", ["attn_w", "V"], ["Y"]),
    ]
    graph = helper.make_graph(
        nodes, "sdpa", [Q, K, V], [Y], initializer=[scale_t]
    )
    return save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]),
        "scaled_dot_product_attention",
    )


def gen_gqa_expand_attention():
    """GQA with Unsqueeze→Expand→Reshape for KV head repetition.
    Static shapes: batch=1, n_heads=8, n_kv_heads=2, seq=6, head_dim=8."""
    batch, n_heads, n_kv, seq, hd = 1, 8, 2, 6, 8
    group = n_heads // n_kv
    scale = 1.0 / (hd**0.5)

    Q = helper.make_tensor_value_info(
        "Q", TensorProto.FLOAT, [batch, n_heads, seq, hd]
    )
    K = helper.make_tensor_value_info(
        "K", TensorProto.FLOAT, [batch, n_kv, seq, hd]
    )
    V = helper.make_tensor_value_info(
        "V", TensorProto.FLOAT, [batch, n_kv, seq, hd]
    )
    Y = helper.make_tensor_value_info(
        "Y", TensorProto.FLOAT, [batch, n_heads, seq, hd]
    )

    unsq_axes = numpy_helper.from_array(np.array([2], dtype=np.int64), "unsq_axes")
    expand_shape_k = numpy_helper.from_array(
        np.array([batch, n_kv, group, seq, hd], dtype=np.int64), "expand_shape_k"
    )
    reshape_shape = numpy_helper.from_array(
        np.array([batch, n_heads, seq, hd], dtype=np.int64), "reshape_shape"
    )
    scale_t = numpy_helper.from_array(np.array(scale, dtype=np.float32), "scale")

    nodes = [
        # K expand: [1,2,6,8] → unsq → [1,2,1,6,8] → expand → [1,2,4,6,8] → reshape → [1,8,6,8]
        helper.make_node("Unsqueeze", ["K", "unsq_axes"], ["K_unsq"]),
        helper.make_node("Expand", ["K_unsq", "expand_shape_k"], ["K_5d"]),
        helper.make_node("Reshape", ["K_5d", "reshape_shape"], ["K_exp"]),
        # V expand
        helper.make_node("Unsqueeze", ["V", "unsq_axes"], ["V_unsq"]),
        helper.make_node("Expand", ["V_unsq", "expand_shape_k"], ["V_5d"]),
        helper.make_node("Reshape", ["V_5d", "reshape_shape"], ["V_exp"]),
        # Attention
        helper.make_node("Transpose", ["K_exp"], ["K_T"], perm=[0, 1, 3, 2]),
        helper.make_node("MatMul", ["Q", "K_T"], ["QK"]),
        helper.make_node("Mul", ["QK", "scale"], ["QK_s"]),
        helper.make_node("Softmax", ["QK_s"], ["attn_w"], axis=-1),
        helper.make_node("MatMul", ["attn_w", "V_exp"], ["Y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "gqa_expand",
        [Q, K, V],
        [Y],
        initializer=[unsq_axes, expand_shape_k, reshape_shape, scale_t],
    )
    return save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]),
        "gqa_expand_attention_static",
    )


# ── Shape ops ─────────────────────────────────────────────────────────────────


def gen_shape_then_cast():
    """Shape(X) → Cast to float. X=[2,6,32]."""
    batch, seq, hidden = 2, 6, 32
    X = helper.make_tensor_value_info(
        "X", TensorProto.FLOAT, [batch, seq, hidden]
    )
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3])
    nodes = [
        helper.make_node("Shape", ["X"], ["shape_i64"]),
        helper.make_node("Cast", ["shape_i64"], ["Y"], to=TensorProto.FLOAT),
    ]
    graph = helper.make_graph(nodes, "shape_cast", [X], [Y])
    return save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]),
        "shape_then_cast",
    )


def gen_expand_dynamic_shape():
    """Expand where shape is built at runtime via Shape→Slice→Concat.
    X=[2,6,32], target shape=[2,6,32] (identity expand via dynamic shape)."""
    batch, seq, hidden = 2, 6, 32
    X = helper.make_tensor_value_info(
        "X", TensorProto.FLOAT, [batch, seq, hidden]
    )
    Y = helper.make_tensor_value_info(
        "Y", TensorProto.FLOAT, [batch, seq, hidden]
    )

    ones = numpy_helper.from_array(np.ones([1], dtype=np.int64), "ones")
    starts_0 = numpy_helper.from_array(np.array([0], dtype=np.int64), "starts_0")
    ends_2 = numpy_helper.from_array(np.array([2], dtype=np.int64), "ends_2")
    axes_0 = numpy_helper.from_array(np.array([0], dtype=np.int64), "axes_0")
    hidden_t = numpy_helper.from_array(
        np.array([hidden], dtype=np.int64), "hidden_const"
    )

    nodes = [
        helper.make_node("Shape", ["X"], ["x_shape"]),
        helper.make_node(
            "Slice", ["x_shape", "starts_0", "ends_2", "axes_0"], ["batch_seq"]
        ),
        helper.make_node("Concat", ["batch_seq", "hidden_const"], ["target"], axis=0),
        helper.make_node("Expand", ["X", "target"], ["Y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "expand_dyn",
        [X],
        [Y],
        initializer=[ones, starts_0, ends_2, axes_0, hidden_t],
    )
    return save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]),
        "expand_dynamic_shape",
    )


def gen_shape_start_end():
    """Shape with start/end attributes. X=[1,2,6,8], Shape(start=0,end=1)→[1]."""
    batch, n_kv, seq, hd = 1, 2, 6, 8
    X = helper.make_tensor_value_info(
        "X", TensorProto.FLOAT, [batch, n_kv, seq, hd]
    )
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])
    nodes = [
        helper.make_node("Shape", ["X"], ["shape_partial"], start=0, end=1),
        helper.make_node("Cast", ["shape_partial"], ["Y"], to=TensorProto.FLOAT),
    ]
    graph = helper.make_graph(nodes, "shape_start_end", [X], [Y])
    return save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]),
        "shape_start_end",
    )


def gen_gqa_k_expand_shape_start_end():
    """GQA K-expand using Shape(start/end) for static shape slicing.
    K=[1,2,6,8] → expand to [1,8,6,8]."""
    batch, n_heads, n_kv, seq, hd = 1, 8, 2, 6, 8
    group = n_heads // n_kv

    K = helper.make_tensor_value_info(
        "K", TensorProto.FLOAT, [batch, n_kv, seq, hd]
    )
    Y = helper.make_tensor_value_info(
        "Y", TensorProto.FLOAT, [batch, n_heads, seq, hd]
    )

    unsq_axes = numpy_helper.from_array(np.array([2], dtype=np.int64), "unsq_axes")
    group_t = numpy_helper.from_array(
        np.array([group], dtype=np.int64), "group_const"
    )
    reshape_t = numpy_helper.from_array(
        np.array([batch, n_heads, seq, hd], dtype=np.int64), "reshape_shape"
    )

    nodes = [
        helper.make_node("Unsqueeze", ["K", "unsq_axes"], ["K_unsq"]),
        # Shape(start=0,end=1) → [batch], Shape(start=1,end=2) → [n_kv]
        helper.make_node("Shape", ["K"], ["K_shape_batch"], start=0, end=1),
        helper.make_node("Shape", ["K"], ["K_shape_nkv"], start=1, end=2),
        helper.make_node("Shape", ["K"], ["K_shape_tail"], start=2, end=4),
        # expand_shape = [batch, n_kv, group, seq, hd]
        helper.make_node(
            "Concat",
            ["K_shape_batch", "K_shape_nkv", "group_const", "K_shape_tail"],
            ["expand_shape"],
            axis=0,
        ),
        helper.make_node("Expand", ["K_unsq", "expand_shape"], ["K_5d"]),
        helper.make_node("Reshape", ["K_5d", "reshape_shape"], ["Y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "gqa_shape_start_end",
        [K],
        [Y],
        initializer=[unsq_axes, group_t, reshape_t],
    )
    return save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]),
        "gqa_k_expand_shape_start_end",
    )


def gen_gqa_k_expand_dynamic_shape():
    """GQA K-expand with runtime Shape for dynamic expansion.
    K=[1,2,6,8] → expand to [1,8,6,8] using Shape→Gather→Concat."""
    batch, n_heads, n_kv, seq, hd = 1, 8, 2, 6, 8
    group = n_heads // n_kv

    K = helper.make_tensor_value_info(
        "K", TensorProto.FLOAT, [batch, n_kv, seq, hd]
    )
    Y = helper.make_tensor_value_info(
        "Y", TensorProto.FLOAT, [batch, n_heads, seq, hd]
    )

    unsq_axes = numpy_helper.from_array(np.array([2], dtype=np.int64), "unsq_axes")
    unsq_axes_0 = numpy_helper.from_array(np.array([0], dtype=np.int64), "unsq_axes_0")
    idx_0 = numpy_helper.from_array(np.array(0, dtype=np.int64), "idx_0")
    idx_1 = numpy_helper.from_array(np.array(1, dtype=np.int64), "idx_1")
    idx_2 = numpy_helper.from_array(np.array(2, dtype=np.int64), "idx_2")
    idx_3 = numpy_helper.from_array(np.array(3, dtype=np.int64), "idx_3")
    group_t = numpy_helper.from_array(
        np.array([group], dtype=np.int64), "group_const"
    )
    reshape_t = numpy_helper.from_array(
        np.array([batch, n_heads, -1, hd], dtype=np.int64), "reshape_shape"
    )

    nodes = [
        helper.make_node("Unsqueeze", ["K", "unsq_axes"], ["K_unsq"]),
        helper.make_node("Shape", ["K"], ["K_shape"]),
        helper.make_node("Gather", ["K_shape", "idx_0"], ["dim_batch"], axis=0),
        helper.make_node("Gather", ["K_shape", "idx_1"], ["dim_nkv"], axis=0),
        helper.make_node("Gather", ["K_shape", "idx_2"], ["dim_seq"], axis=0),
        helper.make_node("Gather", ["K_shape", "idx_3"], ["dim_hd"], axis=0),
        helper.make_node("Unsqueeze", ["dim_batch", "unsq_axes_0"], ["dim_batch_1d"]),
        helper.make_node("Unsqueeze", ["dim_nkv", "unsq_axes_0"], ["dim_nkv_1d"]),
        helper.make_node("Unsqueeze", ["dim_seq", "unsq_axes_0"], ["dim_seq_1d"]),
        helper.make_node("Unsqueeze", ["dim_hd", "unsq_axes_0"], ["dim_hd_1d"]),
        helper.make_node(
            "Concat",
            ["dim_batch_1d", "dim_nkv_1d", "group_const", "dim_seq_1d", "dim_hd_1d"],
            ["expand_shape"],
            axis=0,
        ),
        helper.make_node("Expand", ["K_unsq", "expand_shape"], ["K_5d"]),
        helper.make_node("Reshape", ["K_5d", "reshape_shape"], ["Y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "gqa_dyn_shape",
        [K],
        [Y],
        initializer=[
            unsq_axes,
            unsq_axes_0,
            idx_0,
            idx_1,
            idx_2,
            idx_3,
            group_t,
            reshape_t,
        ],
    )
    return save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]),
        "gqa_k_expand_dynamic_shape",
    )


def gen_shape_start_end_dynamic_seq():
    """Shape(start=0,end=1) with dynamic seq dimension."""
    batch, n_kv, hd = 1, 2, 8
    X = helper.make_tensor_value_info(
        "X", TensorProto.FLOAT, [batch, n_kv, "seq", hd]
    )
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])
    nodes = [
        helper.make_node("Shape", ["X"], ["shape_partial"], start=0, end=1),
        helper.make_node("Cast", ["shape_partial"], ["Y"], to=TensorProto.FLOAT),
    ]
    graph = helper.make_graph(nodes, "shape_dyn_seq", [X], [Y])
    return save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]),
        "shape_start_end_dynamic_seq",
    )


# ── Activation / FFN ops ─────────────────────────────────────────────────────


def gen_swiglu():
    """SwiGLU: silu(gate) * up. X=[4,16]."""
    rows, cols = 4, 16
    half = cols // 2
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [rows, cols])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [rows, half])

    starts_0 = numpy_helper.from_array(np.array([0], dtype=np.int64), "starts_0")
    ends_half = numpy_helper.from_array(np.array([half], dtype=np.int64), "ends_half")
    starts_half = numpy_helper.from_array(
        np.array([half], dtype=np.int64), "starts_half"
    )
    ends_full = numpy_helper.from_array(np.array([cols], dtype=np.int64), "ends_full")
    axes_1 = numpy_helper.from_array(np.array([1], dtype=np.int64), "axes_1")

    nodes = [
        helper.make_node(
            "Slice", ["X", "starts_0", "ends_half", "axes_1"], ["gate"]
        ),
        helper.make_node(
            "Slice", ["X", "starts_half", "ends_full", "axes_1"], ["up"]
        ),
        helper.make_node("Sigmoid", ["gate"], ["gate_s"]),
        helper.make_node("Mul", ["gate", "gate_s"], ["gate_silu"]),
        helper.make_node("Mul", ["gate_silu", "up"], ["Y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "swiglu",
        [X],
        [Y],
        initializer=[starts_0, ends_half, starts_half, ends_full, axes_1],
    )
    return save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]),
        "swiglu",
    )


# ── Edge cases / regressions ─────────────────────────────────────────────────


def gen_range_i64_cast():
    """Range with i64 scalars → Cast to float. Tests i64 input handling."""
    n = 8
    start = helper.make_tensor_value_info("start", TensorProto.INT64, [])
    limit = helper.make_tensor_value_info("limit", TensorProto.INT64, [])
    delta = helper.make_tensor_value_info("delta", TensorProto.INT64, [])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [n])

    nodes = [
        helper.make_node("Range", ["start", "limit", "delta"], ["range_out"]),
        helper.make_node("Cast", ["range_out"], ["Y"], to=TensorProto.FLOAT),
    ]
    graph = helper.make_graph(nodes, "range_i64_cast", [start, limit, delta], [Y])
    return save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]),
        "range_i64_cast",
    )


def gen_causal_mask():
    """Causal mask: LessOrEqual with orthogonal broadcast [seq,1]×[1,seq]→[seq,seq]."""
    seq = 4
    rows = helper.make_tensor_value_info("rows", TensorProto.FLOAT, [seq, 1])
    cols = helper.make_tensor_value_info("cols", TensorProto.FLOAT, [1, seq])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [seq, seq])
    node = helper.make_node("LessOrEqual", ["rows", "cols"], ["Y"])
    graph = helper.make_graph([node], "causal_mask", [rows, cols], [Y])
    return save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]),
        "causal_mask",
    )


def gen_gqa_flat_single_kv():
    """GQA with flat GGUF-style inputs (single KV head, n_q_heads=4).
    Includes causal mask. Q=[1,4,5,8], K=[1,1,5,8], V=[1,1,5,8]."""
    n_q, seq, hd = 4, 5, 8
    scale = 1.0 / (hd**0.5)

    Q = helper.make_tensor_value_info("Q", TensorProto.FLOAT, [1, n_q, seq, hd])
    K = helper.make_tensor_value_info("K", TensorProto.FLOAT, [1, 1, seq, hd])
    V = helper.make_tensor_value_info("V", TensorProto.FLOAT, [1, 1, seq, hd])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, n_q, seq, hd])
    scale_t = numpy_helper.from_array(np.array(scale, dtype=np.float32), "scale")

    nodes = [
        helper.make_node("Transpose", ["K"], ["K_T"], perm=[0, 1, 3, 2]),
        helper.make_node("MatMul", ["Q", "K_T"], ["QK"]),
        helper.make_node("Mul", ["QK", "scale"], ["QK_s"]),
        helper.make_node("Softmax", ["QK_s"], ["attn_w"], axis=-1),
        helper.make_node("MatMul", ["attn_w", "V"], ["Y"]),
    ]
    graph = helper.make_graph(
        nodes, "gqa_flat", [Q, K, V], [Y], initializer=[scale_t]
    )
    return save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]),
        "gqa_flat_single_kv",
    )


# ── Dynamic sequence models ──────────────────────────────────────────────────


def gen_softmax_dyn_seq():
    """Softmax with dynamic seq: [1, seq, 16]."""
    hidden = 16
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, "seq", hidden])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, "seq", hidden])
    node = helper.make_node("Softmax", ["X"], ["Y"], axis=-1)
    graph = helper.make_graph([node], "softmax_dyn", [X], [Y])
    return save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]),
        "softmax_dyn_seq",
    )


def gen_matmul_dyn_seq():
    """MatMul with dynamic seq: X=[1, seq, 8] @ W[8,4] → [1, seq, 4]."""
    k, n = 8, 4
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, "seq", k])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, "seq", n])
    W = numpy_helper.from_array(mk_weight([k, n], 0.0), "W")
    node = helper.make_node("MatMul", ["X", "W"], ["Y"])
    graph = helper.make_graph([node], "matmul_dyn", [X], [Y], initializer=[W])
    return save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]),
        "matmul_dyn_seq",
    )


def gen_reshape_unpack_heads_dyn_seq():
    """Reshape unpacking heads with dynamic seq.
    X=[1, seq, 32] → Shape→Gather→Concat→Reshape → [1, 4, seq, 8]."""
    num_heads, head_dim = 4, 8
    hidden = num_heads * head_dim
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, "seq", hidden])
    Y = helper.make_tensor_value_info(
        "Y", TensorProto.FLOAT, [1, num_heads, "seq", head_dim]
    )

    idx_0 = numpy_helper.from_array(np.array(0, dtype=np.int64), "idx_0")
    idx_1 = numpy_helper.from_array(np.array(1, dtype=np.int64), "idx_1")
    unsq_axes = numpy_helper.from_array(np.array([0], dtype=np.int64), "unsq_axes")
    heads_t = numpy_helper.from_array(
        np.array([num_heads], dtype=np.int64), "heads_const"
    )
    hd_t = numpy_helper.from_array(
        np.array([head_dim], dtype=np.int64), "head_dim_const"
    )

    nodes = [
        helper.make_node("Shape", ["X"], ["x_shape"]),
        helper.make_node("Gather", ["x_shape", "idx_0"], ["dim_batch"], axis=0),
        helper.make_node("Gather", ["x_shape", "idx_1"], ["dim_seq"], axis=0),
        helper.make_node("Unsqueeze", ["dim_batch", "unsq_axes"], ["batch_1d"]),
        helper.make_node("Unsqueeze", ["dim_seq", "unsq_axes"], ["seq_1d"]),
        helper.make_node(
            "Concat",
            ["batch_1d", "heads_const", "seq_1d", "head_dim_const"],
            ["target_shape"],
            axis=0,
        ),
        helper.make_node("Reshape", ["X", "target_shape"], ["Y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "reshape_unpack_heads",
        [X],
        [Y],
        initializer=[idx_0, idx_1, unsq_axes, heads_t, hd_t],
    )
    return save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]),
        "reshape_unpack_heads_dyn_seq",
    )


def gen_mini_transformer():
    """Mini transformer with dynamic seq. hidden=32, heads=2, ffn=64, vocab=32."""
    hidden, ffn, vocab = 32, 64, 32
    scale = 1.0 / (hidden**0.5)

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, ["seq", hidden])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["seq", vocab])

    w_q = numpy_helper.from_array(mk_weight([hidden, hidden], 0.00), "w_q")
    w_k = numpy_helper.from_array(mk_weight([hidden, hidden], 0.10), "w_k")
    w_v = numpy_helper.from_array(mk_weight([hidden, hidden], 0.20), "w_v")
    w_o = numpy_helper.from_array(mk_weight([hidden, hidden], 0.30), "w_o")
    w_gate = numpy_helper.from_array(mk_weight([hidden, ffn], 0.00), "w_gate")
    w_up = numpy_helper.from_array(mk_weight([hidden, ffn], 0.15), "w_up")
    w_down = numpy_helper.from_array(mk_weight([ffn, hidden], 0.05), "w_down")
    w_lm = numpy_helper.from_array(mk_weight([hidden, vocab], 0.00), "w_lm")
    scale_t = numpy_helper.from_array(np.array(scale, dtype=np.float32), "scale_s")

    nodes = [
        # QKV
        helper.make_node("MatMul", ["X", "w_q"], ["Q"]),
        helper.make_node("MatMul", ["X", "w_k"], ["K"]),
        helper.make_node("MatMul", ["X", "w_v"], ["V"]),
        # K^T
        helper.make_node("Transpose", ["K"], ["K_T"], perm=[1, 0]),
        # Attention
        helper.make_node("MatMul", ["Q", "K_T"], ["QK"]),
        helper.make_node("Mul", ["QK", "scale_s"], ["QK_s"]),
        helper.make_node("Softmax", ["QK_s"], ["attn_w"], axis=-1),
        helper.make_node("MatMul", ["attn_w", "V"], ["attn_out"]),
        # Output proj + residual
        helper.make_node("MatMul", ["attn_out", "w_o"], ["o_proj"]),
        helper.make_node("Add", ["X", "o_proj"], ["h2"]),
        # FFN SwiGLU
        helper.make_node("MatMul", ["h2", "w_gate"], ["gate"]),
        helper.make_node("Sigmoid", ["gate"], ["gate_s"]),
        helper.make_node("Mul", ["gate", "gate_s"], ["gate_silu"]),
        helper.make_node("MatMul", ["h2", "w_up"], ["up"]),
        helper.make_node("Mul", ["gate_silu", "up"], ["ffn_h"]),
        helper.make_node("MatMul", ["ffn_h", "w_down"], ["h3"]),
        helper.make_node("Add", ["h2", "h3"], ["Y_pre"]),
        # LM head
        helper.make_node("MatMul", ["Y_pre", "w_lm"], ["Y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "mini_transformer",
        [X],
        [Y],
        initializer=[w_q, w_k, w_v, w_o, w_gate, w_up, w_down, w_lm, scale_t],
    )
    return save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]),
        "mini_transformer",
    )


# ── Fused kernel reference fixtures ──────────────────────────────────────────
# These are decomposed ONNX equivalents of fused AiOps (GroupedQueryAttention,
# FusedSwiGLU). Used as ORT reference for comparing against the hologram
# fused kernel path compiled from AiGraph.


def gen_gqa_fused_reference():
    """Decomposed GQA with flat inputs (GGUF-style): Q [seq, n_q*hd], K/V [seq, n_kv*hd].

    n_q_heads=8, n_kv_heads=2, seq=4, head_dim=8.
    Expands K/V via Unsqueeze→Expand→Reshape, applies causal SDPA.
    This is the ORT reference for testing the fused GroupedQueryAttention kernel."""
    n_q, n_kv, seq, hd = 8, 2, 4, 8
    group = n_q // n_kv
    scale = 1.0 / (hd**0.5)
    q_dim = n_q * hd
    kv_dim = n_kv * hd

    Q = helper.make_tensor_value_info("Q_flat", TensorProto.FLOAT, [seq, q_dim])
    K = helper.make_tensor_value_info("K_flat", TensorProto.FLOAT, [seq, kv_dim])
    V = helper.make_tensor_value_info("V_flat", TensorProto.FLOAT, [seq, kv_dim])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [seq, q_dim])

    # Shape constants
    q_reshape = numpy_helper.from_array(np.array([seq, n_q, hd], dtype=np.int64), "q_reshape")
    kv_reshape = numpy_helper.from_array(np.array([seq, n_kv, hd], dtype=np.int64), "kv_reshape")
    unsq_axes = numpy_helper.from_array(np.array([1], dtype=np.int64), "unsq_axes")
    kv_expand = numpy_helper.from_array(np.array([n_kv, group, seq, hd], dtype=np.int64), "kv_expand")
    kv_final = numpy_helper.from_array(np.array([n_q, seq, hd], dtype=np.int64), "kv_final")
    out_flat = numpy_helper.from_array(np.array([seq, q_dim], dtype=np.int64), "out_flat")
    scale_t = numpy_helper.from_array(np.array(scale, dtype=np.float32), "scale")

    # Causal mask [seq, seq]: 0 on/below diagonal, -inf above
    causal = np.zeros((seq, seq), dtype=np.float32)
    for i in range(seq):
        for j in range(seq):
            if j > i:
                causal[i, j] = -np.inf
    causal_t = numpy_helper.from_array(causal, "causal_mask")

    nodes = [
        # Q: [seq, n_q*hd] → [seq, n_q, hd] → [n_q, seq, hd]
        helper.make_node("Reshape", ["Q_flat", "q_reshape"], ["Q_3d"]),
        helper.make_node("Transpose", ["Q_3d"], ["Q_t"], perm=[1, 0, 2]),

        # K: [seq, n_kv*hd] → [seq, n_kv, hd] → [n_kv, seq, hd] → unsqueeze → expand → reshape
        helper.make_node("Reshape", ["K_flat", "kv_reshape"], ["K_3d"]),
        helper.make_node("Transpose", ["K_3d"], ["K_t"], perm=[1, 0, 2]),
        helper.make_node("Unsqueeze", ["K_t", "unsq_axes"], ["K_4d"]),
        helper.make_node("Expand", ["K_4d", "kv_expand"], ["K_exp4"]),
        helper.make_node("Reshape", ["K_exp4", "kv_final"], ["K_exp"]),

        # V: same as K
        helper.make_node("Reshape", ["V_flat", "kv_reshape"], ["V_3d"]),
        helper.make_node("Transpose", ["V_3d"], ["V_t"], perm=[1, 0, 2]),
        helper.make_node("Unsqueeze", ["V_t", "unsq_axes"], ["V_4d"]),
        helper.make_node("Expand", ["V_4d", "kv_expand"], ["V_exp4"]),
        helper.make_node("Reshape", ["V_exp4", "kv_final"], ["V_exp"]),

        # K^T: [n_q, hd, seq]
        helper.make_node("Transpose", ["K_exp"], ["K_T"], perm=[0, 2, 1]),

        # SDPA: QK^T → scale → mask → softmax → @V
        helper.make_node("MatMul", ["Q_t", "K_T"], ["QK"]),
        helper.make_node("Mul", ["QK", "scale"], ["QK_s"]),
        helper.make_node("Add", ["QK_s", "causal_mask"], ["QK_masked"]),
        helper.make_node("Softmax", ["QK_masked"], ["scores"], axis=-1),
        helper.make_node("MatMul", ["scores", "V_exp"], ["AttnOut_t"]),

        # Output: [n_q, seq, hd] → [seq, n_q, hd] → [seq, n_q*hd]
        helper.make_node("Transpose", ["AttnOut_t"], ["AttnOut_3d"], perm=[1, 0, 2]),
        helper.make_node("Reshape", ["AttnOut_3d", "out_flat"], ["output"]),
    ]
    graph = helper.make_graph(
        nodes, "gqa_flat_multi_kv", [Q, K, V], [Y],
        initializer=[q_reshape, kv_reshape, unsq_axes, kv_expand, kv_final,
                     out_flat, scale_t, causal_t],
    )
    return save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]),
        "gqa_fused_reference",
    )


def gen_swiglu_fused_reference():
    """Decomposed SwiGLU with separate gate/up inputs: silu(gate) * up.

    gate [4, 16], up [4, 16] → output [4, 16].
    This is the ORT reference for testing the fused FusedSwiGLU kernel."""
    rows, cols = 4, 16
    gate = helper.make_tensor_value_info("gate", TensorProto.FLOAT, [rows, cols])
    up = helper.make_tensor_value_info("up", TensorProto.FLOAT, [rows, cols])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [rows, cols])

    nodes = [
        helper.make_node("Sigmoid", ["gate"], ["sig_gate"]),
        helper.make_node("Mul", ["gate", "sig_gate"], ["silu_gate"]),
        helper.make_node("Mul", ["silu_gate", "up"], ["output"]),
    ]
    graph = helper.make_graph(nodes, "swiglu_fused_ref", [gate, up], [Y])
    return save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]),
        "swiglu_fused_reference",
    )

def gen_mlp():
    """MLP: X[4,32] -> Linear(32->64) -> Relu -> Linear(64->32) -> Y[4,32]."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [4, 32])
    W1_init = helper.make_tensor("W1", TensorProto.FLOAT, [32, 64], mk_weight([32, 64]))
    B1_init = helper.make_tensor("B1", TensorProto.FLOAT, [64], mk_weight([64]))
    W2_init = helper.make_tensor("W2", TensorProto.FLOAT, [64, 32], mk_weight([64, 32]))
    B2_init = helper.make_tensor("B2", TensorProto.FLOAT, [32], mk_weight([32]))
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4, 32])
    
    node1 = helper.make_node("MatMul", ["X", "W1"], ["H1"])
    node2 = helper.make_node("Add", ["H1", "B1"], ["H1_b"])
    node3 = helper.make_node("Relu", ["H1_b"], ["H2"])
    node4 = helper.make_node("MatMul", ["H2", "W2"], ["H3"])
    node5 = helper.make_node("Add", ["H3", "B2"], ["Y"])
    
    graph = helper.make_graph(
        [node1, node2, node3, node4, node5], 
        "mlp", 
        [X], 
        [Y], 
        [W1_init, B1_init, W2_init, B2_init]
    )
    return save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), "mlp")

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating ONNX fixtures...")
    gen_matmul()
    gen_softmax()
    gen_gemm_trans_b()
    gen_rms_norm()
    gen_layer_norm()
    gen_batched_matmul_4d()
    gen_concat_4d_last_axis()
    gen_scaled_dot_product_attention()
    gen_gqa_expand_attention()
    gen_shape_then_cast()
    gen_expand_dynamic_shape()
    gen_shape_start_end()
    gen_gqa_k_expand_shape_start_end()
    gen_gqa_k_expand_dynamic_shape()
    gen_shape_start_end_dynamic_seq()
    gen_swiglu()
    gen_range_i64_cast()
    gen_causal_mask()
    gen_gqa_flat_single_kv()
    gen_softmax_dyn_seq()
    gen_matmul_dyn_seq()
    gen_reshape_unpack_heads_dyn_seq()
    gen_mini_transformer()
    gen_gqa_fused_reference()
    gen_swiglu_fused_reference()
    gen_mlp()
    print("Done!")
