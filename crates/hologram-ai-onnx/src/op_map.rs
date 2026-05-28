//! Map ONNX `op_type` strings to `AiOp`.

use crate::onnx_pb::AttributeProto;
use hologram_ai_common::AiOp;

/// Context passed to op converters (attributes, domain, etc.)
pub struct OpContext<'a> {
    pub op_type: &'a str,
    #[allow(dead_code)]
    pub domain: &'a str,
    pub attrs: &'a [AttributeProto],
}

impl<'a> OpContext<'a> {
    pub fn attr_f(&self, name: &str) -> Option<f32> {
        self.attrs.iter().find(|a| a.name == name).map(|a| a.f)
    }

    pub fn attr_i(&self, name: &str) -> Option<i64> {
        self.attrs.iter().find(|a| a.name == name).map(|a| a.i)
    }

    pub fn attr_ints(&self, name: &str) -> Option<&[i64]> {
        self.attrs
            .iter()
            .find(|a| a.name == name)
            .map(|a| a.ints.as_slice())
    }

    #[allow(dead_code)]
    pub fn attr_floats(&self, name: &str) -> Option<&[f32]> {
        self.attrs
            .iter()
            .find(|a| a.name == name)
            .map(|a| a.floats.as_slice())
    }

    pub fn attr_s(&self, name: &str) -> Option<String> {
        self.attrs
            .iter()
            .find(|a| a.name == name)
            .filter(|a| !a.s.is_empty())
            .map(|a| String::from_utf8_lossy(&a.s).into_owned())
    }

    /// Get a float attribute with a default value.
    pub fn attr_f_or(&self, name: &str, default: f32) -> f32 {
        self.attr_f(name).unwrap_or(default)
    }

    /// Get an integer attribute with a default value.
    pub fn attr_i_or(&self, name: &str, default: i64) -> i64 {
        self.attr_i(name).unwrap_or(default)
    }

    /// Get an integer list attribute as `Vec<u64>`.
    pub fn attr_ints_as_u64(&self, name: &str) -> Option<Vec<u64>> {
        self.attr_ints(name)
            .map(|v| v.iter().map(|&i| i as u64).collect())
    }

    /// Get a graph-valued attribute (used for control flow: If/Loop/Scan subgraphs).
    pub fn attr_g(&self, name: &str) -> Option<&crate::onnx_pb::GraphProto> {
        self.attrs
            .iter()
            .find(|a| a.name == name)
            .and_then(|a| a.g.as_ref())
    }
}

/// Convert an ONNX node op to `AiOp`.
///
/// Returns `Ok(None)` for ops that should be silently dropped (e.g. `Dropout` at inference).
/// Returns `Err` for hard failures; `Ok(Some(AiOp::Opaque))` for unrecognised ops.
pub fn map_op(ctx: &OpContext<'_>) -> anyhow::Result<Option<AiOp>> {
    use AiOp::*;

    let op = match ctx.op_type {
        // ── Linear algebra ────────────────────────────────────────────────
        "MatMul" => MatMul,
        "BatchMatMul" => BatchMatMul,
        "Gemm" => Gemm {
            alpha: ctx.attr_f_or("alpha", 1.0),
            beta: ctx.attr_f_or("beta", 1.0),
            trans_a: ctx.attr_i_or("transA", 0) != 0,
            trans_b: ctx.attr_i_or("transB", 0) != 0,
        },
        "Einsum" => {
            let eq = ctx
                .attrs
                .iter()
                .find(|a| a.name == "equation")
                .map(|a| String::from_utf8_lossy(&a.s).into_owned())
                .unwrap_or_default();
            Einsum { equation: eq }
        }

        // ── Activations ───────────────────────────────────────────────────
        "Relu" => Relu,
        "Gelu" => Gelu,
        "Silu" => Silu,
        "Tanh" => Tanh,
        "Sigmoid" => Sigmoid,
        "Softmax" => Softmax {
            axis: ctx.attr_i_or("axis", -1),
        },
        "LogSoftmax" => LogSoftmax {
            axis: ctx.attr_i_or("axis", -1),
        },
        // onnxruntime contrib approx gelu
        "FastGelu" | "BiasGelu" => GeluApprox,

        // ── Normalization ─────────────────────────────────────────────────
        "LayerNormalization" => LayerNorm {
            axis: ctx.attr_i_or("axis", -1),
            epsilon: ctx.attr_f_or("epsilon", 1e-5),
        },
        "GroupNormalization" => GroupNorm {
            num_groups: ctx.attr_i_or("num_groups", 1) as u32,
            epsilon: ctx.attr_f_or("epsilon", 1e-5),
        },
        "BatchNormalization" => BatchNorm {
            epsilon: ctx.attr_f_or("epsilon", 1e-5),
            momentum: ctx.attr_f_or("momentum", 0.9),
            training: ctx.attr_i_or("training_mode", 0) != 0,
        },
        "SimplifiedLayerNormalization" | "RMSNorm" | "SkipSimplifiedLayerNormalization" => {
            RmsNorm {
                epsilon: ctx.attr_f_or("epsilon", 1e-6),
            }
        }

        // ── Shape manipulation ────────────────────────────────────────────
        "Reshape" => Reshape {
            allow_zero: ctx.attr_i_or("allowzero", 0) != 0,
        },
        "Transpose" => {
            let perm = ctx
                .attr_ints("perm")
                .map(|v| v.iter().map(|&i| i as u32).collect())
                .unwrap_or_default();
            Transpose { perm }
        }
        "Concat" => Concat {
            axis: ctx.attr_i_or("axis", 0),
        },
        "Split" => {
            let axis = ctx.attr_i_or("axis", 0);
            let sizes = ctx.attr_ints_as_u64("split").unwrap_or_default();
            Split { axis, sizes }
        }
        "Slice" => {
            // ONNX opset 10+: axes/starts/ends/steps are *inputs*, not attrs.
            // We emit a placeholder; the builder resolves from constant inputs.
            Slice {
                axes: vec![],
                starts: vec![],
                ends: vec![],
                steps: vec![],
            }
        }
        "Gather" | "GatherElements" => {
            let axis = ctx.attr_i_or("axis", 0);
            if ctx.op_type == "GatherElements" {
                GatherElements { axis }
            } else {
                Gather { axis }
            }
        }
        "Unsqueeze" => {
            let axes = ctx
                .attr_ints("axes")
                .map(|v| v.to_vec())
                .unwrap_or_default();
            Unsqueeze { axes }
        }
        "Squeeze" => {
            let axes = ctx
                .attr_ints("axes")
                .map(|v| v.to_vec())
                .unwrap_or_default();
            Squeeze { axes }
        }
        "Expand" => Expand,
        "Tile" => Tile { repeats: vec![] }, // repeats resolved from constant input
        "GatherND" => GatherND {
            batch_dims: ctx.attr_i_or("batch_dims", 0),
        },
        "Shape" => Shape {
            start: ctx.attr_i("start"),
            end: ctx.attr_i("end"),
        },
        "Where" => Where,
        "Range" => Range,
        "Flatten" => Flatten {
            axis: ctx.attr_i_or("axis", 1),
        },

        // ── Elementwise binary ────────────────────────────────────────────
        "Add" => Add,
        "Sub" => Sub,
        "Mul" => Mul,
        "Div" => Div,
        "Pow" => Pow,
        "Mod" => Mod,
        "Min" => Min,
        "Max" => Max,
        "And" => And,
        "Or" => Or,
        "Xor" => Xor,
        "Equal" => Equal,
        "Less" => Less,
        "LessOrEqual" => LessOrEqual,
        "Greater" => Greater,
        "GreaterOrEqual" => GreaterOrEqual,

        // ── Elementwise unary ─────────────────────────────────────────────
        "Abs" => Abs,
        "Neg" => Neg,
        "Sqrt" => Sqrt,
        "Exp" => Exp,
        "Log" => Log,
        "Sign" => Sign,
        "Floor" => Floor,
        "Ceil" => Ceil,
        "Round" => Round,
        "Clip" => Clip {
            min: f32::NEG_INFINITY,
            max: f32::INFINITY,
        },
        "Erf" => Erf,
        "Reciprocal" => Reciprocal,
        "Cos" => Cos,
        "Sin" => Sin,
        "IsNaN" => IsNaN,
        "Not" => Not,

        // ── Reductions ────────────────────────────────────────────────────
        "ReduceSum" => ReduceSum {
            axes: reduce_axes(ctx),
            keepdims: keepdims(ctx),
        },
        "ReduceMean" => ReduceMean {
            axes: reduce_axes(ctx),
            keepdims: keepdims(ctx),
        },
        "ReduceMax" => ReduceMax {
            axes: reduce_axes(ctx),
            keepdims: keepdims(ctx),
        },
        "ReduceMin" => ReduceMin {
            axes: reduce_axes(ctx),
            keepdims: keepdims(ctx),
        },
        "ArgMax" => ArgMax {
            axis: ctx.attr_i_or("axis", 0),
            keepdims: keepdims(ctx),
        },
        "ArgMin" => ArgMin {
            axis: ctx.attr_i_or("axis", 0),
            keepdims: keepdims(ctx),
        },

        // ── Type / cast ───────────────────────────────────────────────────
        "Cast" => {
            // `to` is required on ONNX Cast (no default in the spec). The
            // canonical realization needs the exact target dtype — silently
            // substituting F32 for an unrecognized `to` would produce
            // wrong-typed values. Refuse instead of approximate.
            let to = ctx
                .attr_i("to")
                .ok_or_else(|| anyhow::anyhow!("Cast: missing required `to` attribute"))?;
            let dtype = crate::dtype_map::onnx_dtype(to as i32).ok_or_else(|| {
                anyhow::anyhow!(
                    "Cast: unsupported `to` dtype {to} ({})",
                    crate::dtype_map::onnx_dtype_name(to as i32)
                )
            })?;
            Cast { to: dtype }
        }
        "Identity" => Identity,
        "ConstantOfShape" => {
            let fill_value: f32 = ctx
                .attrs
                .iter()
                .find(|a| a.name == "value")
                .and_then(|a| a.t.as_ref())
                .and_then(|t| {
                    if !t.float_data.is_empty() {
                        Some(t.float_data[0])
                    } else if !t.raw_data.is_empty() && t.raw_data.len() >= 4 {
                        Some(f32::from_le_bytes([
                            t.raw_data[0],
                            t.raw_data[1],
                            t.raw_data[2],
                            t.raw_data[3],
                        ]))
                    } else {
                        None
                    }
                })
                .unwrap_or(0.0);
            ConstantOfShape {
                fill_value: fill_value.to_bits(),
            }
        }
        "Constant" => {
            // Handled separately in the graph builder via initializer or attr.
            return Ok(None);
        }

        // ── Embedding ─────────────────────────────────────────────────────
        "Embedding" => Embed,

        // ── Dropout (inference no-op) ─────────────────────────────────────
        "Dropout" | "DropoutGrad" => return Ok(None),

        // ── Attention (contrib ops) ───────────────────────────────────────
        "MultiHeadAttention" | "Attention" => MultiHeadAttention {
            num_heads: ctx.attr_i_or("num_heads", 1) as u32,
            head_dim: 0, // resolved during shape propagation
            scale: ctx.attr_f("scale"),
            causal: ctx.attr_i_or("unidirectional", 0) != 0,
        },

        // ── Convolution / pooling ────────────────────────────────────────
        "Conv" => Conv {
            kernel_shape: ctx.attr_ints_as_u64("kernel_shape").unwrap_or_default(),
            strides: ctx.attr_ints_as_u64("strides").unwrap_or_default(),
            pads: ctx.attr_ints_as_u64("pads").unwrap_or_default(),
            dilations: ctx.attr_ints_as_u64("dilations").unwrap_or_default(),
            group: ctx.attr_i_or("group", 1) as u64,
            auto_pad: ctx.attr_s("auto_pad").unwrap_or_default(),
        },
        "ConvTranspose" => ConvTranspose {
            kernel_shape: ctx.attr_ints_as_u64("kernel_shape").unwrap_or_default(),
            strides: ctx.attr_ints_as_u64("strides").unwrap_or_default(),
            pads: ctx.attr_ints_as_u64("pads").unwrap_or_default(),
            output_padding: ctx.attr_ints_as_u64("output_padding").unwrap_or_default(),
            dilations: ctx.attr_ints_as_u64("dilations").unwrap_or_default(),
            group: ctx.attr_i_or("group", 1) as u64,
            auto_pad: ctx.attr_s("auto_pad").unwrap_or_default(),
        },
        "MaxPool" => MaxPool {
            kernel_shape: ctx.attr_ints_as_u64("kernel_shape").unwrap_or_default(),
            strides: ctx.attr_ints_as_u64("strides").unwrap_or_default(),
            pads: ctx.attr_ints_as_u64("pads").unwrap_or_default(),
            dilations: ctx.attr_ints_as_u64("dilations").unwrap_or_default(),
            auto_pad: ctx.attr_s("auto_pad").unwrap_or_default(),
            ceil_mode: ctx.attr_i_or("ceil_mode", 0) != 0,
        },
        "AveragePool" => AveragePool {
            kernel_shape: ctx.attr_ints_as_u64("kernel_shape").unwrap_or_default(),
            strides: ctx.attr_ints_as_u64("strides").unwrap_or_default(),
            pads: ctx.attr_ints_as_u64("pads").unwrap_or_default(),
            count_include_pad: ctx.attr_i_or("count_include_pad", 0) != 0,
            auto_pad: ctx.attr_s("auto_pad").unwrap_or_default(),
            ceil_mode: ctx.attr_i_or("ceil_mode", 0) != 0,
        },
        "GlobalAveragePool" => GlobalAveragePool,
        "Resize" | "Upsample" => Resize {
            mode: ctx.attr_s("mode").unwrap_or_else(|| "nearest".into()),
            coordinate_transform_mode: ctx
                .attr_s("coordinate_transformation_mode")
                .unwrap_or_else(|| "half_pixel".into()),
            nearest_mode: ctx
                .attr_s("nearest_mode")
                .unwrap_or_else(|| "round_prefer_floor".into()),
        },
        "Pad" => Pad {
            mode: ctx.attr_s("mode").unwrap_or_else(|| "constant".into()),
        },
        "InstanceNormalization" => InstanceNorm {
            epsilon: ctx.attr_f_or("epsilon", 1e-5),
        },
        "LRN" => LRN {
            alpha: ctx.attr_f_or("alpha", 1e-4),
            beta: ctx.attr_f_or("beta", 0.75),
            bias: ctx.attr_f_or("bias", 1.0),
            size: ctx.attr_i_or("size", 5) as u64,
        },

        // ── Additional reductions ───────────────────────────────────────
        "ReduceProd" => ReduceProd {
            axes: reduce_axes(ctx),
            keepdims: keepdims(ctx),
        },
        "ReduceL1" => ReduceL1 {
            axes: reduce_axes(ctx),
            keepdims: keepdims(ctx),
        },
        "ReduceL2" => ReduceL2 {
            axes: reduce_axes(ctx),
            keepdims: keepdims(ctx),
        },

        // ── Data selection / manipulation ────────────────────────────────
        "TopK" => TopK {
            axis: ctx.attr_i_or("axis", -1),
            largest: ctx.attr_i_or("largest", 1) != 0,
            sorted: ctx.attr_i_or("sorted", 1) != 0,
        },
        "ScatterND" => ScatterND {
            reduce: match ctx.attr_s("reduction").as_deref() {
                Some("add") => hologram_ai_common::ScatterReduce::Add,
                Some("mul") => hologram_ai_common::ScatterReduce::Mul,
                Some("min") => hologram_ai_common::ScatterReduce::Min,
                Some("max") => hologram_ai_common::ScatterReduce::Max,
                _ => hologram_ai_common::ScatterReduce::None,
            },
        },
        "CumSum" => CumSum {
            exclusive: ctx.attr_i_or("exclusive", 0) != 0,
            reverse: ctx.attr_i_or("reverse", 0) != 0,
        },
        "NonZero" => NonZero,
        "OneHot" => OneHot {
            axis: ctx.attr_i_or("axis", -1),
        },
        "DepthToSpace" => DepthToSpace {
            blocksize: ctx.attr_i_or("blocksize", 1) as u64,
            mode: ctx.attr_s("mode").unwrap_or_else(|| "DCR".into()),
        },
        "SpaceToDepth" => SpaceToDepth {
            blocksize: ctx.attr_i_or("blocksize", 1) as u64,
        },
        "Compress" => Compress {
            axis: ctx.attr_i("axis"),
        },
        "ReverseSequence" => ReverseSequence {
            batch_axis: ctx.attr_i_or("batch_axis", 1),
            time_axis: ctx.attr_i_or("time_axis", 0),
        },

        // ── Quantization ────────────────────────────────────────────────
        "QuantizeLinear" => Quantize {
            scheme: hologram_ai_quant::QuantScheme::Q8_0,
        },
        "DequantizeLinear" => Dequantize {
            axis: ctx.attr_i_or("axis", 1),
        },

        // ── Control flow (subgraphs imported by graph_builder) ────────
        // Branch names are placeholders; graph_builder rewrites them to
        // "{prefix}_{node_id}" after importing the subgraphs.
        "If" => If {
            then_branch: "then_branch".into(),
            else_branch: if ctx.attr_g("else_branch").is_some() {
                Some("else_branch".into())
            } else {
                None
            },
        },
        "Loop" => Loop {
            body: "body".into(),
            max_trip_count: None,
        },
        "Scan" => Scan {
            body: "body".into(),
            num_scan_inputs: ctx.attr_i_or("num_scan_inputs", 0) as u32,
        },

        // ── Explicitly known but unsupported ops (Phase 5: long-tail) ────
        // These produce Opaque with the op name so lowering gives clear errors.

        // RNG ops — require runtime random state, not compilable.
        "RandomNormal" | "RandomNormalLike" | "RandomUniform" | "RandomUniformLike"
        | "Bernoulli" | "Multinomial" => Opaque {
            op_type: ctx.op_type.to_string(),
            raw_attrs: vec![],
        },

        // ONNX-ML ops — not part of standard DNN inference.
        "StringNormalizer"
        | "TfIdfVectorizer"
        | "LinearClassifier"
        | "LinearRegressor"
        | "SVMClassifier"
        | "SVMRegressor"
        | "TreeEnsembleClassifier"
        | "TreeEnsembleRegressor"
        | "Normalizer"
        | "Binarizer"
        | "LabelEncoder" => Opaque {
            op_type: ctx.op_type.to_string(),
            raw_attrs: vec![],
        },

        // Triangular matrix extraction (causal mask generation).
        "Trilu" => {
            let upper = ctx.attr_i_or("upper", 1) != 0;
            Trilu { upper }
        }

        // Linear algebra — rare in inference graphs.
        "Det" | "Inverse" | "EyeLike" => Opaque {
            op_type: ctx.op_type.to_string(),
            raw_attrs: vec![],
        },

        // String/sequence ops — no tensor representation.
        "SequenceConstruct" | "SequenceAt" | "SequenceLength" | "SequenceInsert"
        | "SequenceErase" | "SequenceEmpty" | "ConcatFromSequence" | "SplitToSequence" => Opaque {
            op_type: ctx.op_type.to_string(),
            raw_attrs: vec![],
        },

        // Optional type ops.
        "Optional" | "OptionalGetElement" | "OptionalHasElement" => Opaque {
            op_type: ctx.op_type.to_string(),
            raw_attrs: vec![],
        },

        // ── Fallback ──────────────────────────────────────────────────────
        _ => Opaque {
            op_type: ctx.op_type.to_string(),
            raw_attrs: vec![],
        },
    };

    Ok(Some(op))
}

fn reduce_axes(ctx: &OpContext<'_>) -> Vec<i64> {
    ctx.attr_ints("axes")
        .map(|v| v.to_vec())
        .unwrap_or_default()
}

fn keepdims(ctx: &OpContext<'_>) -> bool {
    ctx.attr_i_or("keepdims", 1) != 0
}
