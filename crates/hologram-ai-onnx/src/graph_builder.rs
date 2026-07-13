//! Converts a parsed `GraphProto` into an `AiGraph`.

use crate::{
    dtype_map::onnx_dtype,
    onnx_pb::{GraphProto, TensorShapeProto, ValueInfoProto},
    op_map::{map_op, OpContext},
    tensor_map::tensor_to_param,
};
use hologram_ai_common::{
    shape_from_concrete, AiGraph, AiNode, AiOp, AiParam, DType, Dim, DimVarSource, DimVarTable,
    ImportWarning, NodeId, QuantDescriptor, SemanticHint, Shape, TensorId, TensorInfo,
};
use std::collections::HashMap;
use std::path::Path;

/// Build an `AiGraph` from an ONNX `GraphProto`.
///
/// Returns the graph and an oracle map (TensorId → TensorInfo) built from all
/// ONNX `value_info`, `input`, and `output` annotations. The oracle is used by
/// [`ShapeOraclePass`][hologram_ai_common::ShapeOraclePass] to seed any tensor
/// shapes that remain empty after construction.
///
/// `model_dir` is the directory containing the `.onnx` file, used to resolve
/// external data file paths for tensors with `data_location == EXTERNAL`.
pub fn build_ai_graph(
    g: &GraphProto,
    graph_name: &str,
    model_dir: Option<&Path>,
    external_data: Option<&[u8]>,
) -> anyhow::Result<(AiGraph, HashMap<TensorId, TensorInfo>)> {
    let mut next_tid: TensorId = 0;
    // name → TensorId mapping.
    let mut name_to_tid: HashMap<String, TensorId> = HashMap::new();
    let mut tensor_info: HashMap<TensorId, TensorInfo> = HashMap::new();
    let mut params = HashMap::new();
    let mut warnings = Vec::new();
    let mut dim_vars = DimVarTable::default();
    let mut subgraphs: HashMap<String, AiGraph> = HashMap::new();

    let mut alloc_tid = |name: &str, name_to_tid: &mut HashMap<String, TensorId>| -> TensorId {
        if let Some(&tid) = name_to_tid.get(name) {
            return tid;
        }
        let tid = next_tid;
        next_tid += 1;
        name_to_tid.insert(name.to_owned(), tid);
        tid
    };

    // ── Initializers (weights) ────────────────────────────────────────────
    for init in &g.initializer {
        let tid = alloc_tid(&init.name, &mut name_to_tid);
        match tensor_to_param(init, model_dir, external_data) {
            Ok((param, info)) => {
                tensor_info.insert(tid, info);
                params.insert(tid, param);
            }
            Err(e) => {
                warnings.push(ImportWarning {
                    message: format!("skipping initializer '{}': {e}", init.name),
                    node_name: Some(init.name.clone()),
                });
            }
        }
    }

    // ── Graph inputs ──────────────────────────────────────────────────────
    // Collect non-param inputs first (immutable borrow of name_to_tid),
    // then allocate tensor IDs (mutable borrow) in a separate pass.
    let input_vis: Vec<&ValueInfoProto> = g
        .input
        .iter()
        .filter(|vi| !params.contains_key(name_to_tid.get(&vi.name).unwrap_or(&u32::MAX)))
        .collect();
    let graph_inputs_with_names: Vec<(TensorId, String)> = input_vis
        .into_iter()
        .map(|vi| {
            let tid = alloc_tid(&vi.name, &mut name_to_tid);
            // Skip if we can't represent the input's dtype/shape. The
            // tensor_info entry will be left absent — downstream code
            // either fills it in (params, value_info, inference) or
            // surfaces the gap as a propagation precondition failure.
            if let Some(info) = value_info_to_tensor_info(vi, &mut dim_vars) {
                tensor_info.insert(tid, info);
            }
            (tid, vi.name.clone())
        })
        .collect();
    let graph_inputs: Vec<TensorId> = graph_inputs_with_names.iter().map(|(t, _)| *t).collect();
    let input_names: Vec<String> = graph_inputs_with_names
        .into_iter()
        .map(|(_, n)| n)
        .collect();

    // ── Intermediate tensor shapes (value_info) ──────────────────────────
    // ONNX stores type/shape info for intermediate tensors here.
    for vi in &g.value_info {
        if vi.name.is_empty() {
            continue;
        }
        let tid = alloc_tid(&vi.name, &mut name_to_tid);
        // Only insert if not already populated (params/inputs take
        // priority). A ValueInfo we can't represent is skipped — the
        // tensor's dtype/shape will be left to forward inference (or
        // surfaced as a precondition failure).
        if let Some(info) = value_info_to_tensor_info(vi, &mut dim_vars) {
            tensor_info.entry(tid).or_insert(info);
        }
    }

    // ── Nodes ─────────────────────────────────────────────────────────────
    let mut nodes: Vec<AiNode> = Vec::new();
    let mut next_nid: NodeId = 0;

    for n in &g.node {
        // Allocate output TensorIds.
        let output_tids: Vec<TensorId> = n
            .output
            .iter()
            .map(|name| alloc_tid(name, &mut name_to_tid))
            .collect();

        let input_tids: Vec<TensorId> = n
            .input
            .iter()
            .filter(|name| !name.is_empty())
            .map(|name| alloc_tid(name, &mut name_to_tid))
            .collect();

        // Add placeholder TensorInfo for any not-yet-seen outputs.
        for &tid in &output_tids {
            tensor_info.entry(tid).or_insert_with(|| TensorInfo {
                logical_dtype: DType::F32,
                storage_dtype: DType::F32,
                shape: Shape::new(),
                quant: QuantDescriptor::none(),
                known_i64_values: None,
                semantic: SemanticHint::Unknown,
            });
        }
        for &tid in &input_tids {
            tensor_info.entry(tid).or_insert_with(|| TensorInfo {
                logical_dtype: DType::F32,
                storage_dtype: DType::F32,
                shape: Shape::new(),
                quant: QuantDescriptor::none(),
                known_i64_values: None,
                semantic: SemanticHint::Unknown,
            });
        }

        // Decompose ONNX standard `DynamicQuantizeLinear` (opset 11+)
        // into canonical AiOps at import time. The ONNX op packs
        // amax/amin reduction, scale derivation, zero-point quantization,
        // and the x → uint8 quantization into a single node with three
        // outputs; routing to `AiOp::Opaque` leaves all three TensorIds
        // undefined at lowering, which surfaces as `tensor T<n>
        // referenced before definition` once the Qwen2.5 int8 graph (96
        // DQL instances feeding 168 MatMulInteger consumers) hits the
        // pipeline. Spec: https://onnx.ai/onnx/operators/onnx__DynamicQuantizeLinear.html
        //
        //   Inputs:  [x]            (f32)
        //   Outputs: [y, y_scale, y_zero_point]
        //
        //   max_x      = ReduceMax(x)
        //   min_x      = ReduceMin(x)
        //   max_clip   = Max(max_x, 0)
        //   min_clip   = Min(min_x, 0)
        //   range      = max_clip - min_clip
        //   y_scale    = range / 255                       → output[1]
        //   neg_min    = Neg(min_clip)
        //   zp_f32     = neg_min / y_scale
        //   zp_round   = Round(zp_f32)
        //   zp_clip    = Min(Max(zp_round, 0), 255)        // canonical clip
        //   y_zero     = Cast(zp_clip, U8)                 → output[2]
        //   x_scaled   = x / y_scale
        //   x_round    = Round(x_scaled)
        //   x_shift    = x_round + zp_f32
        //   x_clip     = Min(Max(x_shift, 0), 255)         // canonical clip
        //   y          = Cast(x_clip, U8)                  → output[0]
        if n.op_type == "DynamicQuantizeLinear" {
            if input_tids.is_empty() || output_tids.len() < 3 {
                warnings.push(ImportWarning {
                    message: format!(
                        "DynamicQuantizeLinear '{}' has too few inputs/outputs ({}/{}); skipping",
                        n.name,
                        input_tids.len(),
                        output_tids.len()
                    ),
                    node_name: Some(n.name.clone()),
                });
                continue;
            }

            let x_tid = input_tids[0];
            let y_tid = output_tids[0];
            let scale_tid = output_tids[1];
            let zp_tid = output_tids[2];

            // Override the placeholder F32 dtype for the uint8 outputs.
            // `y` keeps `x`'s shape; `y_scale`/`y_zero_point` are scalars.
            if let Some(info) = tensor_info.get_mut(&y_tid) {
                info.logical_dtype = DType::U8;
                info.storage_dtype = DType::U8;
            }
            if let Some(info) = tensor_info.get_mut(&scale_tid) {
                info.logical_dtype = DType::F32;
                info.storage_dtype = DType::F32;
            }
            if let Some(info) = tensor_info.get_mut(&zp_tid) {
                info.logical_dtype = DType::U8;
                info.storage_dtype = DType::U8;
            }

            let mut synth_counter: u32 = 0;
            macro_rules! new_intermediate {
                ($tag:expr, $dtype:expr) => {{
                    let name = format!("__hai_dql_{}_{}_{}", n.name, $tag, synth_counter);
                    synth_counter += 1;
                    let tid = alloc_tid(&name, &mut name_to_tid);
                    tensor_info.insert(
                        tid,
                        TensorInfo {
                            logical_dtype: $dtype,
                            storage_dtype: $dtype,
                            shape: Shape::new(),
                            quant: QuantDescriptor::none(),
                            known_i64_values: None,
                            semantic: SemanticHint::Unknown,
                        },
                    );
                    tid
                }};
            }
            macro_rules! new_f32_const {
                ($tag:expr, $value:expr) => {{
                    let v: f32 = $value;
                    let name = format!("__hai_dql_{}_{}_const_{}", n.name, $tag, synth_counter);
                    synth_counter += 1;
                    let tid = alloc_tid(&name, &mut name_to_tid);
                    let info = TensorInfo {
                        logical_dtype: DType::F32,
                        storage_dtype: DType::F32,
                        shape: Shape::new(),
                        quant: QuantDescriptor::none(),
                        known_i64_values: None,
                        semantic: SemanticHint::Unknown,
                    };
                    tensor_info.insert(tid, info.clone());
                    params.insert(tid, AiParam::inline(v.to_le_bytes().to_vec(), info));
                    tid
                }};
            }
            macro_rules! push_node {
                ($op:expr, $inputs:expr, $outputs:expr) => {{
                    let nid = next_nid;
                    next_nid += 1;
                    nodes.push(AiNode::new(nid, $op, $inputs, $outputs));
                }};
            }

            // 1. max_x = ReduceMax(x, axes=[], keepdims=false)
            let max_x_tid = new_intermediate!("max_x", DType::F32);
            push_node!(
                AiOp::ReduceMax {
                    axes: vec![],
                    keepdims: false,
                },
                vec![x_tid],
                vec![max_x_tid]
            );

            // 2. min_x = ReduceMin(x, axes=[], keepdims=false)
            let min_x_tid = new_intermediate!("min_x", DType::F32);
            push_node!(
                AiOp::ReduceMin {
                    axes: vec![],
                    keepdims: false,
                },
                vec![x_tid],
                vec![min_x_tid]
            );

            // 3. Constants 0.0 and 255.0.
            let c0_tid = new_f32_const!("c0", 0.0);
            let c255_tid = new_f32_const!("c255", 255.0);

            // 4. max_clip = Max(max_x, 0)
            let max_clip_tid = new_intermediate!("max_clip", DType::F32);
            push_node!(AiOp::Max, vec![max_x_tid, c0_tid], vec![max_clip_tid]);

            // 5. min_clip = Min(min_x, 0)
            let min_clip_tid = new_intermediate!("min_clip", DType::F32);
            push_node!(AiOp::Min, vec![min_x_tid, c0_tid], vec![min_clip_tid]);

            // 6. range = max_clip - min_clip
            let range_tid = new_intermediate!("range", DType::F32);
            push_node!(AiOp::Sub, vec![max_clip_tid, min_clip_tid], vec![range_tid]);

            // 7. y_scale = range / 255       → output[1]
            push_node!(AiOp::Div, vec![range_tid, c255_tid], vec![scale_tid]);

            // 8. neg_min = Neg(min_clip)
            let neg_min_tid = new_intermediate!("neg_min", DType::F32);
            push_node!(AiOp::Neg, vec![min_clip_tid], vec![neg_min_tid]);

            // 9. zp_f32 = neg_min / y_scale
            let zp_f32_tid = new_intermediate!("zp_f32", DType::F32);
            push_node!(AiOp::Div, vec![neg_min_tid, scale_tid], vec![zp_f32_tid]);

            // 10. zp_rounded = Round(zp_f32)
            let zp_round_tid = new_intermediate!("zp_round", DType::F32);
            push_node!(AiOp::Round, vec![zp_f32_tid], vec![zp_round_tid]);

            // 11. zp_clip = Min(Max(zp_round, 0), 255)
            //
            // We emit Min∘Max instead of AiOp::Clip because the canonical
            // lowering treats Clip's bounds as trailing operands (read from
            // AiNode.inputs positionally), not from the AiOp::Clip struct
            // fields. With only one input, the backend's Clip kernel fails
            // with UnsupportedOp ("(min, max) bounds not represented in
            // UnaryCall"). Min∘Max routes through binary_w8 directly and
            // has identical semantics for finite bounds.
            let zp_max_tid = new_intermediate!("zp_max", DType::F32);
            push_node!(AiOp::Max, vec![zp_round_tid, c0_tid], vec![zp_max_tid]);
            let zp_clip_tid = new_intermediate!("zp_clip", DType::F32);
            push_node!(AiOp::Min, vec![zp_max_tid, c255_tid], vec![zp_clip_tid]);

            // 12. y_zero = Cast(zp_clip, U8)  → output[2]
            push_node!(
                AiOp::Cast { to: DType::U8 },
                vec![zp_clip_tid],
                vec![zp_tid]
            );

            // 13. x_scaled = x / y_scale
            let x_scaled_tid = new_intermediate!("x_scaled", DType::F32);
            push_node!(AiOp::Div, vec![x_tid, scale_tid], vec![x_scaled_tid]);

            // 14. x_round = Round(x_scaled)
            let x_round_tid = new_intermediate!("x_round", DType::F32);
            push_node!(AiOp::Round, vec![x_scaled_tid], vec![x_round_tid]);

            // 15. x_shift = x_round + zp_f32
            let x_shift_tid = new_intermediate!("x_shift", DType::F32);
            push_node!(AiOp::Add, vec![x_round_tid, zp_f32_tid], vec![x_shift_tid]);

            // 16. x_clip = Min(Max(x_shift, 0), 255)
            //   (see zp_clip above for the Min∘Max vs Clip rationale)
            let x_max_tid = new_intermediate!("x_max", DType::F32);
            push_node!(AiOp::Max, vec![x_shift_tid, c0_tid], vec![x_max_tid]);
            let x_clip_tid = new_intermediate!("x_clip", DType::F32);
            push_node!(AiOp::Min, vec![x_max_tid, c255_tid], vec![x_clip_tid]);

            // 17. y = Cast(x_clip, U8)        → output[0]
            push_node!(AiOp::Cast { to: DType::U8 }, vec![x_clip_tid], vec![y_tid]);

            let _ = synth_counter; // final increment is unused; silence lint.
            continue;
        }

        // Decompose ONNX standard `MatMulInteger` (opset 10+) into
        // canonical AiOps: cast int operands to f32, subtract optional
        // per-row/per-column zero points (also cast), MatMul in f32,
        // then cast the product back to int32. f32 has a 24-bit
        // mantissa, so for K ≤ ~2^24 with u8/i8 inputs the int32
        // accumulator value is exact (Qwen2.5-0.5B hidden_size=896, well
        // under 2^24). Spec: https://onnx.ai/onnx/operators/onnx__MatMulInteger.html
        //
        //   Inputs:  [A, B, a_zero_point?, b_zero_point?]
        //   Output:  [Y]   int32
        //
        //   a_i32 = Cast(A, INT32)
        //   b_i32 = Cast(B, INT32)
        //   a_centered = (a_zp present) ? Sub(a_i32, Cast(a_zp, INT32)) : a_i32
        //   b_centered = (b_zp present) ? Sub(b_i32, Cast(b_zp, INT32)) : b_i32
        //   a_f32 = Cast(a_centered, FLOAT)
        //   b_f32 = Cast(b_centered, FLOAT)
        //   mm    = MatMul(a_f32, b_f32)
        //   Y     = Cast(mm, INT32)
        if n.op_type == "MatMulInteger" {
            if input_tids.len() < 2 || output_tids.is_empty() {
                warnings.push(ImportWarning {
                    message: format!(
                        "MatMulInteger '{}' has too few inputs/outputs ({}/{}); skipping",
                        n.name,
                        input_tids.len(),
                        output_tids.len()
                    ),
                    node_name: Some(n.name.clone()),
                });
                continue;
            }

            let a_tid = input_tids[0];
            let b_tid = input_tids[1];
            let a_zp_tid = input_tids.get(2).copied();
            let b_zp_tid = input_tids.get(3).copied();
            let y_tid = output_tids[0];

            if let Some(info) = tensor_info.get_mut(&y_tid) {
                info.logical_dtype = DType::INT32;
                info.storage_dtype = DType::INT32;
            }

            let mut synth_counter: u32 = 0;
            macro_rules! new_intermediate {
                ($tag:expr, $dtype:expr) => {{
                    let name = format!("__hai_mmi_{}_{}_{}", n.name, $tag, synth_counter);
                    synth_counter += 1;
                    let tid = alloc_tid(&name, &mut name_to_tid);
                    tensor_info.insert(
                        tid,
                        TensorInfo {
                            logical_dtype: $dtype,
                            storage_dtype: $dtype,
                            shape: Shape::new(),
                            quant: QuantDescriptor::none(),
                            known_i64_values: None,
                            semantic: SemanticHint::Unknown,
                        },
                    );
                    tid
                }};
            }
            macro_rules! push_node {
                ($op:expr, $inputs:expr, $outputs:expr) => {{
                    let nid = next_nid;
                    next_nid += 1;
                    nodes.push(AiNode::new(nid, $op, $inputs, $outputs));
                }};
            }

            // Cast A and B to INT32.
            let a_i32_tid = new_intermediate!("a_i32", DType::INT32);
            push_node!(
                AiOp::Cast { to: DType::INT32 },
                vec![a_tid],
                vec![a_i32_tid]
            );
            let b_i32_tid = new_intermediate!("b_i32", DType::INT32);
            push_node!(
                AiOp::Cast { to: DType::INT32 },
                vec![b_tid],
                vec![b_i32_tid]
            );

            // Subtract zero points if present (broadcasted by AiOp::Sub).
            let a_centered_tid = if let Some(zp_tid) = a_zp_tid {
                let a_zp_i32_tid = new_intermediate!("a_zp_i32", DType::INT32);
                push_node!(
                    AiOp::Cast { to: DType::INT32 },
                    vec![zp_tid],
                    vec![a_zp_i32_tid]
                );
                let centered = new_intermediate!("a_centered", DType::INT32);
                push_node!(AiOp::Sub, vec![a_i32_tid, a_zp_i32_tid], vec![centered]);
                centered
            } else {
                a_i32_tid
            };

            let b_centered_tid = if let Some(zp_tid) = b_zp_tid {
                let b_zp_i32_tid = new_intermediate!("b_zp_i32", DType::INT32);
                push_node!(
                    AiOp::Cast { to: DType::INT32 },
                    vec![zp_tid],
                    vec![b_zp_i32_tid]
                );
                let centered = new_intermediate!("b_centered", DType::INT32);
                push_node!(AiOp::Sub, vec![b_i32_tid, b_zp_i32_tid], vec![centered]);
                centered
            } else {
                b_i32_tid
            };

            // Cast to f32 → MatMul → Cast back to INT32 (→ output[0]).
            let a_f32_tid = new_intermediate!("a_f32", DType::F32);
            push_node!(
                AiOp::Cast { to: DType::F32 },
                vec![a_centered_tid],
                vec![a_f32_tid]
            );
            let b_f32_tid = new_intermediate!("b_f32", DType::F32);
            push_node!(
                AiOp::Cast { to: DType::F32 },
                vec![b_centered_tid],
                vec![b_f32_tid]
            );
            let mm_f32_tid = new_intermediate!("mm_f32", DType::F32);
            push_node!(AiOp::MatMul, vec![a_f32_tid, b_f32_tid], vec![mm_f32_tid]);
            push_node!(
                AiOp::Cast { to: DType::INT32 },
                vec![mm_f32_tid],
                vec![y_tid]
            );

            let _ = synth_counter; // final increment is unused; silence lint.
            continue;
        }

        // Decompose ONNX Microsoft-contrib `RotaryEmbedding` into a
        // canonical sequence of shape ops + Gather + arithmetic. The
        // contrib op fuses position lookup and the paired-halves rotation
        // into a single opaque node; routing it to `AiOp::Opaque` leaves
        // every downstream tensor undefined at lowering. We accept the
        // SmolLM2 case (3D input `[B, S, num_heads*head_dim]`, paired
        // halves, `interleaved=0`) and reject all others by skipping —
        // an opaque routing has the same fail-fast effect at lowering
        // but with a structured warning that pinpoints the cause.
        //
        //   Inputs:  [input, position_ids, cos_cache, sin_cache]
        //   Output:  rotated input (same shape)
        //
        //   half = head_dim / 2
        //   cos  = Gather(cos_cache, position_ids, axis=0)   [B, S, half]
        //   sin  = Gather(sin_cache, position_ids, axis=0)
        //   x_4d = Reshape(input, [0, 0, num_heads, head_dim])
        //   x1   = Slice(x_4d, axes=[-1], starts=[0], ends=[half])
        //   x2   = Slice(x_4d, axes=[-1], starts=[half], ends=[head_dim])
        //   cos4 = Unsqueeze(cos, axes=[2])                  [B, S, 1, half]
        //   sin4 = Unsqueeze(sin, axes=[2])
        //   first  = x1*cos4 - x2*sin4
        //   second = x1*sin4 + x2*cos4
        //   y_4d = Concat([first, second], axis=-1)
        //   out  = Reshape(y_4d, [0, 0, -1])
        //
        // `num_heads` and `rotary_embedding_dim` default to 0 in the
        // ORT contrib spec, meaning "derive from cos_cache": head_dim =
        // 2 * cos_cache.last_dim, num_heads = input.last_dim / head_dim.
        // We honor explicit nonzero attribute values when present.
        if n.op_type == "RotaryEmbedding" {
            let interleaved = n
                .attribute
                .iter()
                .find(|a| a.name == "interleaved")
                .map(|a| a.i)
                .unwrap_or(0);
            let num_heads_attr = n
                .attribute
                .iter()
                .find(|a| a.name == "num_heads")
                .map(|a| a.i)
                .unwrap_or(0);
            let rotary_dim_attr = n
                .attribute
                .iter()
                .find(|a| a.name == "rotary_embedding_dim")
                .map(|a| a.i)
                .unwrap_or(0);

            // Skipping would leave this node's output unproduced — every
            // downstream consumer then fails far from the cause (or a shape
            // accident runs un-roped positions). An unrealizable rotary is a
            // REFUSAL of the import, loud, naming the node and the reason.
            anyhow::ensure!(
                interleaved == 0,
                "RotaryEmbedding '{}': interleaved=1 (adjacent-pair rotation) is not \
                 implemented — importing it as rotate-half would be silently wrong",
                n.name
            );
            anyhow::ensure!(
                input_tids.len() >= 4 && !output_tids.is_empty(),
                "RotaryEmbedding '{}' has too few inputs/outputs ({}/{}) to realize",
                n.name,
                input_tids.len(),
                output_tids.len()
            );

            let input_tid = input_tids[0];
            let pos_ids_tid = input_tids[1];
            let cos_cache_tid = input_tids[2];
            let sin_cache_tid = input_tids[3];

            // Derive head_dim. Priority: explicit `rotary_embedding_dim`
            // attr if nonzero, otherwise 2 * cos_cache.last_dim.
            let cos_last_dim = tensor_info
                .get(&cos_cache_tid)
                .and_then(|info| info.shape.last())
                .and_then(|d| d.as_concrete());
            let head_dim_u64: Option<u64> = if rotary_dim_attr > 0 {
                Some(rotary_dim_attr as u64)
            } else {
                cos_last_dim.map(|h| h * 2)
            };

            // Derive num_heads. Priority: explicit `num_heads` attr if
            // nonzero, otherwise input.last_dim / head_dim.
            let input_last_dim = tensor_info
                .get(&input_tid)
                .and_then(|info| info.shape.last())
                .and_then(|d| d.as_concrete());
            let num_heads_u64: Option<u64> = if num_heads_attr > 0 {
                Some(num_heads_attr as u64)
            } else {
                match (input_last_dim, head_dim_u64) {
                    (Some(ld), Some(hd)) if hd > 0 && ld.is_multiple_of(hd) => Some(ld / hd),
                    _ => None,
                }
            };

            let (head_dim, num_heads) = match (head_dim_u64, num_heads_u64) {
                (Some(hd), Some(nh)) if hd > 0 && nh > 0 && hd.is_multiple_of(2) => (hd, nh),
                _ => anyhow::bail!(
                    "RotaryEmbedding '{}': cannot derive head_dim/num_heads \
                     (head_dim={:?}, num_heads={:?}) — refusing to import an \
                     underdetermined rotary",
                    n.name,
                    head_dim_u64,
                    num_heads_u64
                ),
            };
            let half = head_dim / 2;

            // Synthesized names use a per-node counter to stay unique.
            // Macros (not closures) — macros expand inline, so they
            // don't conflict with the outer `alloc_tid` closure that
            // captures `&mut next_tid`. The trailing `synth_counter +=
            // 1` after the last macro invocation produces an unused-
            // assignment warning that's structural to the pattern.
            let mut synth_counter: u32 = 0;
            macro_rules! new_intermediate {
                ($tag:expr) => {{
                    let name = format!("__hai_rope_{}_{}_{}", n.name, $tag, synth_counter);
                    synth_counter += 1;
                    let tid = alloc_tid(&name, &mut name_to_tid);
                    tensor_info.insert(
                        tid,
                        TensorInfo {
                            logical_dtype: DType::F32,
                            storage_dtype: DType::F32,
                            shape: Shape::new(),
                            quant: QuantDescriptor::none(),
                            known_i64_values: None,
                            semantic: SemanticHint::Unknown,
                        },
                    );
                    tid
                }};
            }
            macro_rules! new_i64_const {
                ($tag:expr, $values:expr) => {{
                    let values: &[i64] = $values;
                    let name = format!("__hai_rope_{}_{}_const_{}", n.name, $tag, synth_counter);
                    synth_counter += 1;
                    let tid = alloc_tid(&name, &mut name_to_tid);
                    let info = TensorInfo {
                        logical_dtype: DType::INT64,
                        storage_dtype: DType::INT64,
                        shape: shape_from_concrete(&[values.len() as u64]),
                        quant: QuantDescriptor::none(),
                        known_i64_values: Some(values.iter().map(|v| Some(*v)).collect()),
                        semantic: SemanticHint::Unknown,
                    };
                    let mut bytes = Vec::with_capacity(values.len() * 8);
                    for v in values {
                        bytes.extend_from_slice(&v.to_le_bytes());
                    }
                    tensor_info.insert(tid, info.clone());
                    params.insert(tid, AiParam::inline(bytes, info));
                    tid
                }};
            }
            macro_rules! push_node {
                ($op:expr, $inputs:expr, $outputs:expr) => {{
                    let nid = next_nid;
                    next_nid += 1;
                    nodes.push(AiNode::new(nid, $op, $inputs, $outputs));
                }};
            }

            // 1. cos = Gather(cos_cache, position_ids, axis=0)
            let cos_tid = new_intermediate!("cos");
            push_node!(
                AiOp::Gather { axis: 0 },
                vec![cos_cache_tid, pos_ids_tid],
                vec![cos_tid]
            );

            // 2. sin = Gather(sin_cache, position_ids, axis=0)
            let sin_tid = new_intermediate!("sin");
            push_node!(
                AiOp::Gather { axis: 0 },
                vec![sin_cache_tid, pos_ids_tid],
                vec![sin_tid]
            );

            // 3. input_4d = Reshape(input, [0, 0, num_heads, head_dim])
            let reshape_in_shape_tid =
                new_i64_const!("in_shape", &[0, 0, num_heads as i64, head_dim as i64]);
            let input_4d_tid = new_intermediate!("in_4d");
            push_node!(
                AiOp::Reshape { allow_zero: false },
                vec![input_tid, reshape_in_shape_tid],
                vec![input_4d_tid]
            );

            // 4. x1 = Slice(input_4d, axes=[-1], starts=[0], ends=[half])
            let x1_tid = new_intermediate!("x1");
            push_node!(
                AiOp::Slice {
                    axes: vec![-1],
                    starts: vec![0],
                    ends: vec![half as i64],
                    steps: vec![1],
                },
                vec![input_4d_tid],
                vec![x1_tid]
            );

            // 5. x2 = Slice(input_4d, axes=[-1], starts=[half], ends=[head_dim])
            let x2_tid = new_intermediate!("x2");
            push_node!(
                AiOp::Slice {
                    axes: vec![-1],
                    starts: vec![half as i64],
                    ends: vec![head_dim as i64],
                    steps: vec![1],
                },
                vec![input_4d_tid],
                vec![x2_tid]
            );

            // 6. cos_4d = Unsqueeze(cos, axes=[2])
            let cos_4d_tid = new_intermediate!("cos_4d");
            push_node!(
                AiOp::Unsqueeze { axes: vec![2] },
                vec![cos_tid],
                vec![cos_4d_tid]
            );

            // 7. sin_4d = Unsqueeze(sin, axes=[2])
            let sin_4d_tid = new_intermediate!("sin_4d");
            push_node!(
                AiOp::Unsqueeze { axes: vec![2] },
                vec![sin_tid],
                vec![sin_4d_tid]
            );

            // 8. x1_cos = Mul(x1, cos_4d)
            let x1_cos_tid = new_intermediate!("x1_cos");
            push_node!(AiOp::Mul, vec![x1_tid, cos_4d_tid], vec![x1_cos_tid]);

            // 9. x2_sin = Mul(x2, sin_4d)
            let x2_sin_tid = new_intermediate!("x2_sin");
            push_node!(AiOp::Mul, vec![x2_tid, sin_4d_tid], vec![x2_sin_tid]);

            // 10. first = Sub(x1_cos, x2_sin)
            let first_tid = new_intermediate!("first");
            push_node!(AiOp::Sub, vec![x1_cos_tid, x2_sin_tid], vec![first_tid]);

            // 11. x1_sin = Mul(x1, sin_4d)
            let x1_sin_tid = new_intermediate!("x1_sin");
            push_node!(AiOp::Mul, vec![x1_tid, sin_4d_tid], vec![x1_sin_tid]);

            // 12. x2_cos = Mul(x2, cos_4d)
            let x2_cos_tid = new_intermediate!("x2_cos");
            push_node!(AiOp::Mul, vec![x2_tid, cos_4d_tid], vec![x2_cos_tid]);

            // 13. second = Add(x1_sin, x2_cos)
            let second_tid = new_intermediate!("second");
            push_node!(AiOp::Add, vec![x1_sin_tid, x2_cos_tid], vec![second_tid]);

            // 14. rotated_4d = Concat([first, second], axis=-1)
            let rotated_4d_tid = new_intermediate!("rot_4d");
            push_node!(
                AiOp::Concat { axis: -1 },
                vec![first_tid, second_tid],
                vec![rotated_4d_tid]
            );

            // 15. output = Reshape(rotated_4d, [0, 0, -1])  → bind to output_tids[0]
            let reshape_out_shape_tid = new_i64_const!("out_shape", &[0, 0, -1]);
            push_node!(
                AiOp::Reshape { allow_zero: false },
                vec![rotated_4d_tid, reshape_out_shape_tid],
                vec![output_tids[0]]
            );
            let _ = synth_counter; // Final increment in new_i64_const is unused; silence lint.
            continue;
        }

        // Decompose ONNX Microsoft-contrib `GroupQueryAttention` into
        // canonical Reshape/Transpose ops + `AiOp::GroupedQueryAttention`.
        // The contrib op fuses head-reshape, head-first transpose, SDPA,
        // and the produces-present-K/V semantics; routing to Opaque
        // leaves three downstream tensors undefined at lowering. We
        // accept the SmolLM2 case (do_rotary=0, post-RoPE Q/K/V, no
        // past) and reject `do_rotary=1`.
        //
        //   Inputs:  [Q, K, V, past_k, past_v, seqlens_k, total_seq_len,
        //             cos_cache?, sin_cache?]    (past + seqlens ignored)
        //   Outputs: [output, present_key, present_value]
        //
        //   Q_4d  = Reshape(Q, [0, 0, num_heads, head_dim])
        //   Q_t   = Transpose(Q_4d, perm=[0, 2, 1, 3])
        //   K_4d  = Reshape(K, [0, 0, kv_num_heads, head_dim])
        //   K_t   = Transpose(K_4d, perm=[0, 2, 1, 3])     // → present_key
        //   V_4d  = Reshape(V, [0, 0, kv_num_heads, head_dim])
        //   V_t   = Transpose(V_4d, perm=[0, 2, 1, 3])     // → present_value
        //   out4d = GroupedQueryAttention(Q_t, K_t, V_t)   // [B,H,S,head_dim]
        //   out_t = Transpose(out4d, perm=[0, 2, 1, 3])
        //   out   = Reshape(out_t, [0, 0, -1])             // → output
        if n.op_type == "GroupQueryAttention" {
            let num_heads = n
                .attribute
                .iter()
                .find(|a| a.name == "num_heads")
                .map(|a| a.i)
                .unwrap_or(0);
            let kv_num_heads = n
                .attribute
                .iter()
                .find(|a| a.name == "kv_num_heads")
                .map(|a| a.i)
                .unwrap_or(0);
            let do_rotary = n
                .attribute
                .iter()
                .find(|a| a.name == "do_rotary")
                .map(|a| a.i)
                .unwrap_or(0);

            // Same refusal law as RotaryEmbedding above: this node produces
            // output + present K/V — skipping strands three downstream
            // tensors. Unrealizable forms refuse the import by name.
            anyhow::ensure!(
                do_rotary == 0,
                "GroupQueryAttention '{}': do_rotary=1 (fused in-kernel rotary) is not \
                 implemented — importing it without the rotation would be silently wrong",
                n.name
            );
            anyhow::ensure!(
                input_tids.len() >= 3 && output_tids.len() >= 3,
                "GroupQueryAttention '{}' has too few inputs/outputs ({}/{}) to realize",
                n.name,
                input_tids.len(),
                output_tids.len()
            );
            anyhow::ensure!(
                num_heads > 0 && kv_num_heads > 0,
                "GroupQueryAttention '{}': num_heads={}, kv_num_heads={} (both must be > 0)",
                n.name,
                num_heads,
                kv_num_heads
            );

            let q_tid = input_tids[0];
            let k_tid = input_tids[1];
            let v_tid = input_tids[2];

            // Derive head_dim from Q.last_dim / num_heads.
            let q_last = tensor_info
                .get(&q_tid)
                .and_then(|info| info.shape.last())
                .and_then(|d| d.as_concrete());
            let head_dim_u64 = match q_last {
                Some(ld) if num_heads > 0 && ld % (num_heads as u64) == 0 => {
                    Some(ld / num_heads as u64)
                }
                _ => None,
            };
            let head_dim = match head_dim_u64 {
                Some(hd) if hd > 0 => hd,
                _ => {
                    warnings.push(ImportWarning {
                        message: format!(
                            "GroupQueryAttention '{}': cannot derive head_dim from Q (q_last={:?}, num_heads={}); skipping",
                            n.name, q_last, num_heads
                        ),
                        node_name: Some(n.name.clone()),
                    });
                    continue;
                }
            };

            // Sanity-check K and V's last dim if known. Non-fatal — if
            // the oracle hasn't filled the value_info yet, we skip the
            // check and trust the contrib spec's attribute layout.
            let check_last = |tid: TensorId, expected: u64| -> bool {
                tensor_info
                    .get(&tid)
                    .and_then(|info| info.shape.last())
                    .and_then(|d| d.as_concrete())
                    .is_none_or(|ld| ld == expected)
            };
            let kv_hidden = (kv_num_heads as u64) * head_dim;
            if !check_last(k_tid, kv_hidden) || !check_last(v_tid, kv_hidden) {
                warnings.push(ImportWarning {
                    message: format!(
                        "GroupQueryAttention '{}': K/V last-dim does not match kv_num_heads*head_dim={}; skipping",
                        n.name, kv_hidden
                    ),
                    node_name: Some(n.name.clone()),
                });
                continue;
            }

            // Macro-based helpers (see RotaryEmbedding handler for rationale).
            let mut synth_counter: u32 = 0;
            macro_rules! new_intermediate {
                ($tag:expr) => {{
                    let name = format!("__hai_gqa_{}_{}_{}", n.name, $tag, synth_counter);
                    synth_counter += 1;
                    let tid = alloc_tid(&name, &mut name_to_tid);
                    tensor_info.insert(
                        tid,
                        TensorInfo {
                            logical_dtype: DType::F32,
                            storage_dtype: DType::F32,
                            shape: Shape::new(),
                            quant: QuantDescriptor::none(),
                            known_i64_values: None,
                            semantic: SemanticHint::Unknown,
                        },
                    );
                    tid
                }};
            }
            macro_rules! new_i64_const {
                ($tag:expr, $values:expr) => {{
                    let values: &[i64] = $values;
                    let name = format!("__hai_gqa_{}_{}_const_{}", n.name, $tag, synth_counter);
                    synth_counter += 1;
                    let tid = alloc_tid(&name, &mut name_to_tid);
                    let info = TensorInfo {
                        logical_dtype: DType::INT64,
                        storage_dtype: DType::INT64,
                        shape: shape_from_concrete(&[values.len() as u64]),
                        quant: QuantDescriptor::none(),
                        known_i64_values: Some(values.iter().map(|v| Some(*v)).collect()),
                        semantic: SemanticHint::Unknown,
                    };
                    let mut bytes = Vec::with_capacity(values.len() * 8);
                    for v in values {
                        bytes.extend_from_slice(&v.to_le_bytes());
                    }
                    tensor_info.insert(tid, info.clone());
                    params.insert(tid, AiParam::inline(bytes, info));
                    tid
                }};
            }
            macro_rules! push_node {
                ($op:expr, $inputs:expr, $outputs:expr) => {{
                    let nid = next_nid;
                    next_nid += 1;
                    nodes.push(AiNode::new(nid, $op, $inputs, $outputs));
                }};
            }

            // Q: reshape [0,0,num_heads,head_dim] then transpose [0,2,1,3].
            let q_shape_tid = new_i64_const!("q_shape", &[0, 0, num_heads, head_dim as i64]);
            let q_4d_tid = new_intermediate!("q_4d");
            push_node!(
                AiOp::Reshape { allow_zero: false },
                vec![q_tid, q_shape_tid],
                vec![q_4d_tid]
            );
            let q_t_tid = new_intermediate!("q_t");
            push_node!(
                AiOp::Transpose {
                    perm: vec![0, 2, 1, 3]
                },
                vec![q_4d_tid],
                vec![q_t_tid]
            );

            // K: reshape [0,0,kv_num_heads,head_dim], transpose → present_key (output[1]).
            let kv_shape_tid = new_i64_const!("kv_shape", &[0, 0, kv_num_heads, head_dim as i64]);
            let k_4d_tid = new_intermediate!("k_4d");
            push_node!(
                AiOp::Reshape { allow_zero: false },
                vec![k_tid, kv_shape_tid],
                vec![k_4d_tid]
            );
            push_node!(
                AiOp::Transpose {
                    perm: vec![0, 2, 1, 3]
                },
                vec![k_4d_tid],
                vec![output_tids[1]]
            );
            let k_t_tid = output_tids[1];

            // V: reshape [0,0,kv_num_heads,head_dim], transpose → present_value (output[2]).
            let v_4d_tid = new_intermediate!("v_4d");
            push_node!(
                AiOp::Reshape { allow_zero: false },
                vec![v_tid, kv_shape_tid],
                vec![v_4d_tid]
            );
            push_node!(
                AiOp::Transpose {
                    perm: vec![0, 2, 1, 3]
                },
                vec![v_4d_tid],
                vec![output_tids[2]]
            );
            let v_t_tid = output_tids[2];

            // GroupedQueryAttention(Q_t, K_t, V_t) → out_4d [B, num_heads, S, head_dim].
            let out_4d_tid = new_intermediate!("out_4d");
            push_node!(
                AiOp::GroupedQueryAttention {
                    num_heads: num_heads as u32,
                    num_kv_heads: kv_num_heads as u32,
                    head_dim: head_dim as u32,
                    scale: None,
                    causal: true,
                    heads_first: true,
                    qk_norm: false,
                    rope: None,
                },
                vec![q_t_tid, k_t_tid, v_t_tid],
                vec![out_4d_tid]
            );

            // Transpose back to [B, S, num_heads, head_dim] then reshape [0, 0, -1].
            let out_pre_tid = new_intermediate!("out_pre");
            push_node!(
                AiOp::Transpose {
                    perm: vec![0, 2, 1, 3]
                },
                vec![out_4d_tid],
                vec![out_pre_tid]
            );
            let out_shape_tid = new_i64_const!("out_shape", &[0, 0, -1]);
            push_node!(
                AiOp::Reshape { allow_zero: false },
                vec![out_pre_tid, out_shape_tid],
                vec![output_tids[0]]
            );
            let _ = synth_counter; // Final increment in new_i64_const is unused; silence lint.
            continue;
        }

        // Decompose ONNX Microsoft-contrib `SkipSimplifiedLayerNormalization`
        // into canonical `Add` + `RmsNorm` at import time. This op has
        // residual-fusion semantics that plain `AiOp::RmsNorm` does not
        // express:
        //
        //   Inputs:  [input, skip, gamma (, bias)]
        //   Outputs: [output, mean, inv_std_var, input_skip_sum]
        //   output           = RmsNorm(input + skip) * gamma
        //   input_skip_sum   = input + skip   (consumed by the next layer)
        //
        // Mapping the whole op to a single `AiOp::RmsNorm` discards both
        // the residual fuse AND the input_skip_sum output. Later layers'
        // SkipSimplifiedLayerNormalization nodes take *that orphaned sum*
        // as their `skip` input, so when optimization removes the
        // mis-imported parent the downstream reference becomes
        // "tensor T<...> referenced before definition" at lowering. Fix
        // it at the boundary: emit `Add(input, skip) → output_3` and
        // `RmsNorm(sum, gamma, eps) → output_0`. Output[1] (mean) and
        // output[2] (inv_std) are inference-time unused. The downstream
        // `AddRmsNormFusion` rule then re-fuses Add+RmsNorm into
        // `FusedLayerNormResidual` where the residual sum doesn't leak.
        if n.op_type == "SkipSimplifiedLayerNormalization" {
            // Must have at least input, skip, gamma + at least one
            // output. The bias input (input[3]) and the mean/inv_std
            // outputs (output[1], output[2]) are unused at inference.
            if input_tids.len() < 3 || output_tids.is_empty() {
                warnings.push(ImportWarning {
                    message: format!(
                        "SkipSimplifiedLayerNormalization '{}' has too few inputs/outputs ({}/{}); skipping",
                        n.name,
                        input_tids.len(),
                        output_tids.len()
                    ),
                    node_name: Some(n.name.clone()),
                });
                continue;
            }
            // Determine the residual-sum TID. Where present, it is
            // output[3]; otherwise synthesize one (no downstream
            // consumer will reference an absent output).
            let sum_tid = if output_tids.len() >= 4 {
                output_tids[3]
            } else {
                // Synthesize a fresh TID — internal-only intermediate.
                let synth = format!("__hai_skip_sum_{}", n.name);
                alloc_tid(&synth, &mut name_to_tid)
            };
            tensor_info.entry(sum_tid).or_insert_with(|| TensorInfo {
                logical_dtype: DType::F32,
                storage_dtype: DType::F32,
                shape: Shape::new(),
                quant: QuantDescriptor::none(),
                known_i64_values: None,
                semantic: SemanticHint::Unknown,
            });
            let epsilon = n
                .attribute
                .iter()
                .find(|a| a.name == "epsilon")
                .map(|a| a.f)
                .unwrap_or(1e-5);
            // Emit Add(input, skip) → sum_tid.
            let add_nid = next_nid;
            next_nid += 1;
            nodes.push(AiNode::new(
                add_nid,
                AiOp::Add,
                vec![input_tids[0], input_tids[1]],
                vec![sum_tid],
            ));
            // Emit RmsNorm(sum_tid, gamma, epsilon) → output_0.
            let norm_nid = next_nid;
            next_nid += 1;
            nodes.push(AiNode::new(
                norm_nid,
                AiOp::RmsNorm { epsilon },
                vec![sum_tid, input_tids[2]],
                vec![output_tids[0]],
            ));
            continue;
        }

        // Handle Constant nodes: extract the `value` attribute as a param.
        if n.op_type == "Constant" {
            if let Some(tensor_attr) = n.attribute.iter().find(|a| a.name == "value") {
                if let Some(ref tensor_proto) = tensor_attr.t {
                    if let Some(&out_tid) = output_tids.first() {
                        match tensor_to_param(tensor_proto, model_dir, external_data) {
                            Ok((param, info)) => {
                                tensor_info.insert(out_tid, info);
                                params.insert(out_tid, param);
                            }
                            Err(e) => {
                                warnings.push(ImportWarning {
                                    message: format!(
                                        "error extracting Constant value '{}': {e}",
                                        n.name
                                    ),
                                    node_name: Some(n.name.clone()),
                                });
                            }
                        }
                    }
                }
            }
            continue;
        }

        let ctx = OpContext {
            op_type: &n.op_type,
            domain: &n.domain,
            attrs: &n.attribute,
        };

        match map_op(&ctx) {
            Ok(Some(op)) => {
                if matches!(op, AiOp::Opaque { ref op_type, .. } if !op_type.is_empty()) {
                    warnings.push(ImportWarning {
                        message: format!("unsupported op '{}' mapped to Opaque", n.op_type),
                        node_name: Some(n.name.clone()),
                    });
                }
                let nid = next_nid;
                next_nid += 1;
                nodes.push(AiNode::new(nid, op, input_tids, output_tids));
            }
            Ok(None) => {
                // Intentional no-op (e.g. Dropout at inference).
                tracing::debug!(op_type = %n.op_type, "skipping no-op");
            }
            Err(e) => {
                warnings.push(ImportWarning {
                    message: format!("error mapping op '{}': {e}", n.op_type),
                    node_name: Some(n.name.clone()),
                });
            }
        }

        // ── Recursive subgraph import for control flow ops ─────────────
        if matches!(n.op_type.as_str(), "If" | "Loop" | "Scan") {
            let nid = next_nid.saturating_sub(1);
            import_subgraph_attrs(
                &ctx,
                &n.op_type,
                nid,
                &mut subgraphs,
                &mut warnings,
                model_dir,
                external_data,
            );
            // Rewrite placeholder branch names to actual subgraph keys.
            if let Some(node) = nodes.last_mut() {
                rewrite_subgraph_keys(&mut node.op, nid);
            }
        }
    }

    // Re-resolve graph outputs (tensors may have been allocated during node pass).
    let graph_outputs_with_names: Vec<(TensorId, String)> = g
        .output
        .iter()
        .map(|vi| {
            let tid = alloc_tid(&vi.name, &mut name_to_tid);
            // Populate shape/dtype from output's ValueInfoProto. If the
            // ValueInfo isn't a TensorType with a representable dtype,
            // leave the tensor_info entry alone — forward inference will
            // fill it from the producer's output, and a true gap
            // surfaces downstream.
            if let Some(info) = value_info_to_tensor_info(vi, &mut dim_vars) {
                if !info.shape.is_empty() {
                    tensor_info
                        .entry(tid)
                        .and_modify(|existing| {
                            if existing.shape.is_empty() {
                                *existing = info.clone();
                            }
                        })
                        .or_insert(info);
                }
            }
            (tid, vi.name.clone())
        })
        .collect();
    let graph_outputs: Vec<TensorId> = graph_outputs_with_names.iter().map(|(t, _)| *t).collect();
    let output_names: Vec<String> = graph_outputs_with_names
        .into_iter()
        .map(|(_, n)| n)
        .collect();

    // Post-process: resolve op parameters that are inputs in ONNX opset 10+/13+.
    // Slice, Unsqueeze, Squeeze take their axis/start/end parameters as tensor
    // inputs rather than node attributes. We resolve them from constants here.
    resolve_dynamic_op_params(&mut nodes, &params, &tensor_info, &mut warnings);

    // Build the shape oracle from all ONNX-provided annotations.
    // Covers value_info (intermediate tensors), graph inputs, and graph outputs.
    // Only entries with non-empty shapes are included — empty annotations from
    // untyped outputs would just add noise.
    let mut oracle: HashMap<TensorId, TensorInfo> = HashMap::new();
    let all_vis = g
        .value_info
        .iter()
        .chain(g.input.iter())
        .chain(g.output.iter());
    for vi in all_vis {
        if let Some(&tid) = name_to_tid.get(&vi.name) {
            // Skip params: their shapes come from initializer data.
            if params.contains_key(&tid) {
                continue;
            }
            // Only record oracle entries we can fully represent. A
            // missing/empty TensorType is left absent rather than
            // approximated.
            if let Some(info) = value_info_to_tensor_info(vi, &mut dim_vars) {
                if !info.shape.is_empty() {
                    oracle.insert(tid, info);
                }
            }
        }
    }

    // Resolve head_dim for ONNX MultiHeadAttention nodes that were given
    // head_dim: 0 in op_map (placeholder; real value comes from oracle shapes).
    resolve_attention_head_dims(&mut nodes, &oracle, &tensor_info);

    Ok((
        AiGraph {
            name: graph_name.to_owned(),
            nodes,
            inputs: graph_inputs,
            outputs: graph_outputs,
            input_names,
            output_names,
            params,
            tensor_info,
            metadata: HashMap::new(),
            warnings,
            dim_vars,
            shape_constraints: Default::default(),
            subgraphs,
            tensor_names: name_to_tid
                .into_iter()
                .map(|(name, tid)| (tid, name))
                .collect(),
            topo_cache: Default::default(),
        },
        oracle,
    ))
}

/// Resolve `head_dim: 0` for ONNX MultiHeadAttention / GroupedQueryAttention nodes.
///
/// ONNX MHA nodes are imported with `head_dim: 0` because the value is not
/// available as a node attribute — it must be derived from tensor shapes.
///
/// Priority order:
/// 1. Oracle output shape: if the output is `[_, _, hidden]` with concrete
///    `hidden`, then `head_dim = hidden / num_heads`.
/// 2. Query input shape (input[0]): same formula.
/// 3. Leave as 0 and emit a tracing warning (will produce a zero-dim shape
///    later, caught by the compiler's diagnostic pass).
fn resolve_attention_head_dims(
    nodes: &mut [AiNode],
    oracle: &HashMap<TensorId, TensorInfo>,
    tensor_info: &HashMap<TensorId, TensorInfo>,
) {
    for node in nodes.iter_mut() {
        let (num_heads, head_dim) = match &node.op {
            AiOp::MultiHeadAttention {
                num_heads,
                head_dim,
                ..
            } => (*num_heads, *head_dim),
            AiOp::GroupedQueryAttention {
                num_heads,
                head_dim,
                ..
            } => (*num_heads, *head_dim),
            _ => continue,
        };

        if head_dim != 0 || num_heads == 0 {
            continue;
        }

        // Try to derive head_dim from the output tensor's oracle shape.
        let resolved = node
            .outputs
            .first()
            .and_then(|tid| oracle.get(tid).or_else(|| tensor_info.get(tid)))
            .and_then(|info| info.shape.last())
            .and_then(|d| d.as_concrete())
            .map(|hidden| hidden / num_heads as u64)
            // Fallback: try the query input (input[0]) shape.
            .or_else(|| {
                node.inputs
                    .first()
                    .and_then(|tid| oracle.get(tid).or_else(|| tensor_info.get(tid)))
                    .and_then(|info| info.shape.last())
                    .and_then(|d| d.as_concrete())
                    .map(|hidden| hidden / num_heads as u64)
            });

        match resolved {
            Some(hd) if hd > 0 => {
                match &mut node.op {
                    AiOp::MultiHeadAttention { head_dim, .. } => *head_dim = hd as u32,
                    AiOp::GroupedQueryAttention { head_dim, .. } => *head_dim = hd as u32,
                    _ => {}
                }
                tracing::debug!(
                    node_id = node.id,
                    num_heads,
                    head_dim = hd,
                    "resolved attention head_dim from oracle"
                );
            }
            _ => {
                tracing::warn!(
                    node_id = node.id,
                    num_heads,
                    "could not resolve head_dim for attention node; left as 0"
                );
            }
        }
    }
}

/// Build a `TensorInfo` from an ONNX `ValueInfoProto`.
///
/// UOR-native: returns `None` when the ValueInfo doesn't carry a
/// TensorType (the only kind hologram-ai handles) or when its
/// `elem_type` isn't a representable dtype. Callers either skip the
/// value (when it's truly informational, e.g. an unused intermediate)
/// or propagate the import failure. There is no F32 + empty-shape
/// fallback — fabricating a dtype produces silently-wrong downstream
/// inference (see ADR-0018 + the Qwen2 GQA dtype-prop trail).
fn value_info_to_tensor_info(
    vi: &ValueInfoProto,
    dim_vars: &mut DimVarTable,
) -> Option<TensorInfo> {
    let tp = vi.r#type.as_ref()?;
    let Some(crate::onnx_pb::type_proto::Value::TensorType(t)) = &tp.value else {
        return None;
    };
    let dtype = onnx_dtype(t.elem_type)?;
    let shape = t
        .shape
        .as_ref()
        .map(|s| shape_from_shape_proto(s, dim_vars))
        .unwrap_or_default();
    Some(TensorInfo {
        logical_dtype: dtype,
        storage_dtype: dtype,
        shape,
        quant: QuantDescriptor::none(),
        known_i64_values: None,
        semantic: SemanticHint::Unknown,
    })
}

/// Resolve op parameters that ONNX opset 10+/13+/18+ provides as tensor inputs
/// rather than node attributes.
///
/// # Optional Input Semantics
///
/// ONNX uses empty-string input names (`""`) for optional inputs that are not
/// provided. During graph construction these are filtered out by
/// `filter(|name| !name.is_empty())`, which means positional indexing into
/// `node.inputs` does NOT correspond to the ONNX input position when optional
/// inputs are absent.
///
/// This function resolves the ambiguity for position-sensitive ops by:
/// 1. Extracting constant values from tensor inputs and storing them on the
///    `AiOp` variant (Slice starts/ends, Clip min/max, etc.)
/// 2. Normalizing `node.inputs` to match the arity expected by lowering
///    (Pad → 2 inputs, Resize → 2 inputs, Clip → 1 input, etc.)
///
/// # Resolved ops
///
/// - **Slice** (opset 10+): starts, ends, axes, steps from inputs[1-4]
/// - **Squeeze/Unsqueeze** (opset 13+): axes from input[1]
/// - **ReduceMean/Sum/Max/Min** (opset 18+): axes from input[1]
/// - **Pad** (opset 11+): drop optional constant_value (input[2])
/// - **Resize** (opset 11+): normalize to [X, scales_or_sizes], prefer sizes
/// - **Clip** (opset 11+): min/max from optional inputs[1-2]
fn resolve_dynamic_op_params(
    nodes: &mut [AiNode],
    params: &HashMap<TensorId, hologram_ai_common::AiParam>,
    tensor_info: &HashMap<TensorId, TensorInfo>,
    warnings: &mut Vec<ImportWarning>,
) {
    for node in nodes.iter_mut() {
        match &node.op {
            AiOp::Slice {
                axes, starts, ends, ..
            } if axes.is_empty() && starts.is_empty() && ends.is_empty() => {
                // ONNX opset 10+: Slice(data, starts, ends, [axes], [steps])
                // inputs[0] = data, inputs[1] = starts, inputs[2] = ends,
                // inputs[3] = axes (optional), inputs[4] = steps (optional)
                if node.inputs.len() < 3 {
                    continue;
                }
                let starts_vals = extract_i64_const(node.inputs[1], params, tensor_info);
                let ends_vals = extract_i64_const(node.inputs[2], params, tensor_info);
                if starts_vals.is_none() || ends_vals.is_none() {
                    tracing::debug!(
                        node_id = node.id,
                        n_inputs = node.inputs.len(),
                        starts_tid = node.inputs[1],
                        starts_in_params = params.contains_key(&node.inputs[1]),
                        starts_in_info = tensor_info.contains_key(&node.inputs[1]),
                        ends_tid = node.inputs[2],
                        ends_in_params = params.contains_key(&node.inputs[2]),
                        "Slice debug: starts/ends resolution"
                    );
                }
                let axes_vals = if node.inputs.len() > 3 {
                    extract_i64_const(node.inputs[3], params, tensor_info)
                } else {
                    None
                };
                let steps_vals = if node.inputs.len() > 4 {
                    extract_i64_const(node.inputs[4], params, tensor_info)
                } else {
                    None
                };

                match (starts_vals, ends_vals) {
                    (Some(s), Some(e)) => {
                        let axes = axes_vals.unwrap_or_else(|| (0..s.len() as i64).collect());
                        let steps = steps_vals.unwrap_or_else(|| vec![1; s.len()]);
                        node.op = AiOp::Slice {
                            axes,
                            starts: s,
                            ends: e,
                            steps,
                        };
                        // Keep only the data input; remove the constant inputs.
                        node.inputs.truncate(1);
                    }
                    _ => {
                        // Expected for dynamic/with-past slices (resolved later by
                        // SliceToGather after concretization); recorded structurally
                        // in `warnings`, so keep the per-node log at debug.
                        tracing::debug!(
                            node_id = node.id,
                            starts_tid = node.inputs.get(1).copied().unwrap_or(0),
                            ends_tid = node.inputs.get(2).copied().unwrap_or(0),
                            "Slice: could not resolve starts/ends"
                        );
                        warnings.push(ImportWarning {
                            message: format!(
                                "Slice node {}: could not resolve starts/ends from constant inputs",
                                node.id
                            ),
                            node_name: None,
                        });
                    }
                }
            }

            AiOp::Unsqueeze { axes }
                if axes.is_empty()
                // ONNX opset 13+: Unsqueeze(data, axes)
                && node.inputs.len() >= 2 =>
            {
                if let Some(axes_vals) = extract_i64_const(node.inputs[1], params, tensor_info) {
                    node.op = AiOp::Unsqueeze { axes: axes_vals };
                    node.inputs.truncate(1);
                }
            }

            AiOp::Squeeze { axes }
                if axes.is_empty()
                // ONNX opset 13+: Squeeze(data, [axes])
                && node.inputs.len() >= 2 =>
            {
                if let Some(axes_vals) = extract_i64_const(node.inputs[1], params, tensor_info) {
                    node.op = AiOp::Squeeze { axes: axes_vals };
                    node.inputs.truncate(1);
                }
            }
            // If only 1 input, Squeeze with empty axes = squeeze all size-1 dims (already correct).

            // ONNX opset 18+: ReduceMean/Sum/Max/Min(data, axes) — axes as input tensor.
            AiOp::ReduceMean { axes, keepdims } if axes.is_empty() => {
                let kd = *keepdims;
                if node.inputs.len() >= 2 {
                    if let Some(axes_vals) = extract_i64_const(node.inputs[1], params, tensor_info)
                    {
                        node.op = AiOp::ReduceMean {
                            axes: axes_vals,
                            keepdims: kd,
                        };
                        node.inputs.truncate(1);
                    }
                }
            }
            AiOp::ReduceSum { axes, keepdims } if axes.is_empty() => {
                let kd = *keepdims;
                if node.inputs.len() >= 2 {
                    if let Some(axes_vals) = extract_i64_const(node.inputs[1], params, tensor_info)
                    {
                        node.op = AiOp::ReduceSum {
                            axes: axes_vals,
                            keepdims: kd,
                        };
                        node.inputs.truncate(1);
                    }
                }
            }
            AiOp::ReduceMax { axes, keepdims } if axes.is_empty() => {
                let kd = *keepdims;
                if node.inputs.len() >= 2 {
                    if let Some(axes_vals) = extract_i64_const(node.inputs[1], params, tensor_info)
                    {
                        node.op = AiOp::ReduceMax {
                            axes: axes_vals,
                            keepdims: kd,
                        };
                        node.inputs.truncate(1);
                    }
                }
            }
            AiOp::ReduceMin { axes, keepdims } if axes.is_empty() => {
                let kd = *keepdims;
                if node.inputs.len() >= 2 {
                    if let Some(axes_vals) = extract_i64_const(node.inputs[1], params, tensor_info)
                    {
                        node.op = AiOp::ReduceMin {
                            axes: axes_vals,
                            keepdims: kd,
                        };
                        node.inputs.truncate(1);
                    }
                }
            }

            // ONNX opset 11+: Pad(data, pads, constant_value?)
            // FloatOp::PadOp expects arity 2: [data, pads].
            // Drop optional constant_value input (input[2]).
            AiOp::Pad { .. } if node.inputs.len() > 2 => {
                node.inputs.truncate(2);
            }

            // ONNX opset 11+: Resize(X, roi, scales, sizes)
            // Empty-name inputs are already filtered, so we may have 2-4 inputs.
            // FloatOp::Resize expects arity 2: [data, scales_or_sizes].
            // Prefer sizes (i64) over scales (f32); drop roi.
            AiOp::Resize { .. } if node.inputs.len() > 2 => {
                let data = node.inputs[0];
                // Find the best param: prefer i64 (sizes) over f32 (scales).
                let mut best = node.inputs[node.inputs.len() - 1];
                for &tid in &node.inputs[1..] {
                    if let Some(info) = tensor_info.get(&tid) {
                        if info.logical_dtype == DType::INT64 {
                            best = tid;
                            break;
                        }
                    }
                }
                node.inputs = vec![data, best];
            }

            // ONNX opset 11+: Clip(input, min?, max?)
            // Empty-name inputs are filtered, so we may have 1-3 inputs.
            // Resolve min/max from constant scalar inputs and store on AiOp.
            AiOp::Clip { .. } => {
                let min_val = if node.inputs.len() >= 2 {
                    extract_f32_scalar(node.inputs[1], params, tensor_info)
                } else {
                    None
                };
                let max_val = if node.inputs.len() >= 3 {
                    extract_f32_scalar(node.inputs[2], params, tensor_info)
                } else {
                    None
                };
                node.op = AiOp::Clip {
                    min: min_val.unwrap_or(f32::NEG_INFINITY),
                    max: max_val.unwrap_or(f32::INFINITY),
                };
                // Keep only data input; min/max are now on the op.
                node.inputs.truncate(1);
            }

            _ => {}
        }
    }
}

/// Extract a single f32 scalar from a constant parameter tensor.
fn extract_f32_scalar(
    tid: TensorId,
    params: &HashMap<TensorId, hologram_ai_common::AiParam>,
    tensor_info: &HashMap<TensorId, TensorInfo>,
) -> Option<f32> {
    let param = params.get(&tid)?;
    let info = tensor_info.get(&tid)?;
    let data = match param {
        hologram_ai_common::AiParam::Inline { data, .. } => data.as_slice(),
        _ => return None,
    };

    match info.logical_dtype {
        DType::F32 if data.len() == 4 => Some(f32::from_le_bytes(data.try_into().unwrap())),
        DType::F64 if data.len() == 8 => Some(f64::from_le_bytes(data.try_into().unwrap()) as f32),
        _ => None,
    }
}

/// Extract i64 values from a constant parameter tensor.
fn extract_i64_const(
    tid: TensorId,
    params: &HashMap<TensorId, hologram_ai_common::AiParam>,
    tensor_info: &HashMap<TensorId, TensorInfo>,
) -> Option<Vec<i64>> {
    let param = params.get(&tid)?;
    let info = tensor_info.get(&tid)?;
    tracing::trace!(
        tid,
        dtype = ?info.logical_dtype,
        "extract_i64_const: found param"
    );
    let data: std::borrow::Cow<'_, [u8]> = match param {
        hologram_ai_common::AiParam::Inline { data, .. } => {
            std::borrow::Cow::Borrowed(data.as_slice())
        }
        hologram_ai_common::AiParam::Mmap {
            path, offset, len, ..
        } => {
            // Read small constants from external data files (< 1 KiB).
            // Large tensors (weights) stay mmap'd; only tiny scalars
            // (Slice starts/ends/axes/steps) are loaded eagerly.
            if *len > 1024 {
                return None;
            }
            use std::io::{Read, Seek, SeekFrom};
            let mut file = std::fs::File::open(path).ok()?;
            file.seek(SeekFrom::Start(*offset)).ok()?;
            let mut buf = vec![0u8; *len as usize];
            file.read_exact(&mut buf).ok()?;
            std::borrow::Cow::Owned(buf)
        }
        hologram_ai_common::AiParam::External { .. } => return None,
    };

    match info.logical_dtype {
        DType::INT64 => {
            if !data.len().is_multiple_of(8) {
                return None;
            }
            Some(
                data.chunks_exact(8)
                    .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                    .collect(),
            )
        }
        DType::INT32 => {
            if !data.len().is_multiple_of(4) {
                return None;
            }
            Some(
                data.chunks_exact(4)
                    .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as i64)
                    .collect(),
            )
        }
        _ => None,
    }
}

/// Rewrite placeholder branch names in control flow AiOps to the actual
/// subgraph keys (e.g., "then_branch" → "then_branch_42").
fn rewrite_subgraph_keys(op: &mut AiOp, node_id: u32) {
    match op {
        AiOp::If {
            then_branch,
            else_branch,
        } => {
            *then_branch = format!("then_branch_{node_id}");
            if let Some(eb) = else_branch {
                *eb = format!("else_branch_{node_id}");
            }
        }
        AiOp::Loop { body, .. } | AiOp::Scan { body, .. } => {
            *body = format!("body_{node_id}");
        }
        _ => {}
    }
}

/// Recursively import subgraph attributes from control flow ops (If/Loop/Scan).
///
/// Extracts `GraphProto` attributes from the ONNX node, builds child `AiGraph`s
/// via `build_ai_graph`, and stores them in the parent's `subgraphs` map.
/// The keys match the branch names used by the `AiOp::If`/`Loop`/`Scan` variants.
fn import_subgraph_attrs(
    ctx: &OpContext<'_>,
    op_type: &str,
    node_id: u32,
    subgraphs: &mut HashMap<String, AiGraph>,
    warnings: &mut Vec<ImportWarning>,
    model_dir: Option<&Path>,
    external_data: Option<&[u8]>,
) {
    let branch_attrs: Vec<(&str, &str)> = match op_type {
        "If" => vec![
            ("then_branch", "then_branch"),
            ("else_branch", "else_branch"),
        ],
        "Loop" => vec![("body", "body")],
        "Scan" => vec![("body", "body")],
        _ => return,
    };

    for (attr_name, key_prefix) in branch_attrs {
        if let Some(graph_proto) = ctx.attr_g(attr_name) {
            let subgraph_key = format!("{key_prefix}_{node_id}");
            match build_ai_graph(graph_proto, &subgraph_key, model_dir, external_data) {
                Ok((child_graph, _oracle)) => {
                    tracing::debug!(
                        node_id,
                        op_type,
                        key = %subgraph_key,
                        nodes = child_graph.nodes.len(),
                        "imported subgraph"
                    );
                    subgraphs.insert(subgraph_key, child_graph);
                }
                Err(e) => {
                    warnings.push(ImportWarning {
                        message: format!(
                            "failed to import {attr_name} subgraph for {op_type} node {node_id}: {e}"
                        ),
                        node_name: None,
                    });
                }
            }
        }
    }
}

fn shape_from_shape_proto(s: &TensorShapeProto, dim_vars: &mut DimVarTable) -> Shape {
    s.dim
        .iter()
        .map(|d| {
            match &d.value {
                Some(crate::onnx_pb::tensor_shape_proto::dimension::Value::DimValue(v)) => {
                    Dim::Concrete(*v as u64)
                }
                Some(crate::onnx_pb::tensor_shape_proto::dimension::Value::DimParam(name)) => {
                    // Intern the ONNX dim_param as a named DimVar.
                    let var_id = dim_vars.intern_with_bounds(
                        name,
                        Some(1),
                        None, // upper bound unknown from ONNX alone
                        DimVarSource::Import,
                    );
                    Dim::Var(var_id)
                }
                None => Dim::Dynamic,
            }
        })
        .collect()
}
