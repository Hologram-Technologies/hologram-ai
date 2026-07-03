use crate::builder::GraphBuilder;
use anyhow::{anyhow, Result};
use hologram_ai_common::ir::{dtype::DType, graph::AiGraph};
use safetensors::{Dtype as SafeDtype, SafeTensors};
use serde_json::Value;

#[allow(dead_code)]
fn map_dtype(d: SafeDtype) -> Result<DType> {
    match d {
        SafeDtype::F32 => Ok(DType::F32),
        SafeDtype::F16 => Ok(DType::F16),
        SafeDtype::I64 => Ok(DType::INT64),
        SafeDtype::I32 => Ok(DType::INT32),
        _ => Err(anyhow!("Unsupported safetensors dtype: {:?}", d)),
    }
}

pub fn build_parametric_graph(config: &Value, safetensors_shards: &[&[u8]]) -> Result<AiGraph> {
    let mut st_instances = Vec::new();
    for shard in safetensors_shards {
        let st = SafeTensors::deserialize(shard)?;
        st_instances.push(st);
    }

    let mut keys = Vec::new();
    for st in &st_instances {
        for (k, _) in st.tensors() {
            keys.push(k.clone());
        }
    }

    let mut graph = build_parametric_graph_from_keys(config, &keys)?;

    // Inject the actual safetensors weights into the graph's params.
    let mut name_to_id = std::collections::HashMap::new();
    for (id, name) in &graph.tensor_names {
        name_to_id.insert(name.clone(), *id);
    }

    let mut next_id = graph.tensor_names.keys().max().copied().unwrap_or(0) + 1;
    for st in &st_instances {
        for (k, tensor_view) in st.tensors() {
            let id = if let Some(existing_id) = name_to_id.get(&k) {
                *existing_id
            } else {
                let new_id = next_id;
                next_id += 1;
                graph.tensor_names.insert(new_id, k.clone());
                new_id
            };

            let dtype = map_dtype(tensor_view.dtype())?;
            let shape = hologram_ai_common::shape_from_concrete(
                &tensor_view
                    .shape()
                    .iter()
                    .map(|&x| x as u64)
                    .collect::<Vec<_>>(),
            );
            let info = hologram_ai_common::TensorInfo::new(dtype, shape);
            graph.tensor_info.insert(id, info.clone());

            let data = tensor_view.data().to_vec();
            graph.params.insert(
                id,
                hologram_ai_common::ir::param::AiParam::inline(data, info),
            );
        }
    }

    Ok(graph)
}


fn add_linear_layer(
    builder: &mut GraphBuilder,
    input: hologram_ai_common::ir::node::TensorId,
    weight_name: &str,
    in_features: hologram_ai_common::ir::shape::DimExpr,
    out_features: hologram_ai_common::ir::shape::DimExpr,
    output_name: &str,
    output_shape: Vec<hologram_ai_common::ir::shape::DimExpr>,
) -> hologram_ai_common::ir::node::TensorId {
    let weight = builder.add_tensor(
        weight_name,
        hologram_ai_common::ir::dtype::DType::F32,
        vec![out_features.clone(), in_features.clone()],
    );
    let transposed_weight = builder.add_tensor(
        &format!("{}_transposed", weight_name),
        hologram_ai_common::ir::dtype::DType::F32,
        vec![in_features, out_features],
    );
    builder.add_node(
        hologram_ai_common::ir::op::AiOp::Transpose { perm: vec![1, 0] },
        vec![weight],
        vec![transposed_weight],
    );
    let output = builder.add_tensor(
        output_name,
        hologram_ai_common::ir::dtype::DType::F32,
        output_shape,
    );
    builder.add_node(
        hologram_ai_common::ir::op::AiOp::MatMul,
        vec![input, transposed_weight],
        vec![output],
    );
    output
}

struct LinearLayerParams<'a> {
    weight_name: &'a str,
    in_features: hologram_ai_common::ir::shape::DimExpr,
    out_features: hologram_ai_common::ir::shape::DimExpr,
    output_name: &'a str,
    output_shape: Vec<hologram_ai_common::ir::shape::DimExpr>,
}

fn add_linear_layer_from_tensor(
    builder: &mut GraphBuilder,
    input: hologram_ai_common::ir::node::TensorId,
    weight: hologram_ai_common::ir::node::TensorId,
    params: LinearLayerParams<'_>,
) -> hologram_ai_common::ir::node::TensorId {
    let transposed_weight = builder.add_tensor(
        &format!("{}_transposed", params.weight_name),
        hologram_ai_common::ir::dtype::DType::F32,
        vec![params.in_features, params.out_features],
    );
    builder.add_node(
        hologram_ai_common::ir::op::AiOp::Transpose { perm: vec![1, 0] },
        vec![weight],
        vec![transposed_weight],
    );
    let output = builder.add_tensor(
        params.output_name,
        hologram_ai_common::ir::dtype::DType::F32,
        params.output_shape,
    );
    builder.add_node(
        hologram_ai_common::ir::op::AiOp::MatMul,
        vec![input, transposed_weight],
        vec![output],
    );
    output
}

pub fn build_parametric_graph_from_keys(config: &Value, keys: &[String]) -> Result<AiGraph> {
    let mut builder = GraphBuilder::new("parametric_model".to_string());

    let mut num_layers = 0;
    for key in keys {
        if let Some(idx) = extract_layer_idx(key) {
            if idx >= num_layers {
                num_layers = idx + 1;
            }
        }
    }

    if num_layers == 0 {
        return Err(anyhow!(
            "Could not infer any layers from tensor keys. Is this a transformer?"
        ));
    }

    // Config defaults
    let hidden_size = config
        .get("hidden_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(4096) as u32;
    let num_heads = config
        .get("num_attention_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or(32) as u32;
    let _num_kv_heads = config
        .get("num_key_value_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or(num_heads as u64) as u32;
    let _head_dim = hidden_size / num_heads;

    // Inputs
    let batch = builder.register_var("batch");
    let seq = builder.register_var("seq");
    let input_ids = builder.add_input("input_ids", DType::INT64, vec![batch.clone(), seq.clone()]);

    let vocab_size = config
        .get("vocab_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(32000) as u32;

    let vocab = hologram_ai_common::ir::shape::DimExpr::Concrete(vocab_size as u64);
    let hidden = hologram_ai_common::ir::shape::DimExpr::Concrete(hidden_size as u64);
    let n_heads_expr = hologram_ai_common::ir::shape::DimExpr::Concrete(num_heads as u64);
    let n_kv_heads_expr = hologram_ai_common::ir::shape::DimExpr::Concrete(_num_kv_heads as u64);
    let head_dim_expr = hologram_ai_common::ir::shape::DimExpr::Concrete(_head_dim as u64);

    // 1. Embedding
    let embed_weight = builder.add_tensor(
        "model.embed_tokens.weight",
        DType::F32,
        vec![vocab.clone(), hidden.clone()],
    );
    let mut current = builder.add_tensor(
        "hidden_states",
        DType::F32,
        vec![batch.clone(), seq.clone(), hidden.clone()],
    );
    builder.add_node(
        hologram_ai_common::ir::op::AiOp::Gather { axis: 0 },
        vec![embed_weight, input_ids],
        vec![current],
    );

    // 2. Transformer blocks
    for l in 0..num_layers {
        // Attention Norm
        let attn_norm_weight = builder.add_tensor(
            &format!("model.layers.{l}.input_layernorm.weight"),
            DType::F32,
            vec![hidden.clone()],
        );
        let attn_norm_out = builder.add_tensor(
            &format!("attn_norm_{l}"),
            DType::F32,
            vec![batch.clone(), seq.clone(), hidden.clone()],
        );
        builder.add_node(
            hologram_ai_common::ir::op::AiOp::RmsNorm { epsilon: 1e-5 },
            vec![current, attn_norm_weight],
            vec![attn_norm_out],
        );

        // QKV Projection
        let q_out_var =
            hologram_ai_common::ir::shape::DimExpr::Concrete((num_heads * _head_dim) as u64);
        let k_out_var =
            hologram_ai_common::ir::shape::DimExpr::Concrete((_num_kv_heads * _head_dim) as u64);
        let v_out_var =
            hologram_ai_common::ir::shape::DimExpr::Concrete((_num_kv_heads * _head_dim) as u64);

        let q_flat;
        let k_flat;
        let v_flat;

        if keys.contains(&format!("model.layers.{}.self_attn.qkv_proj.weight", l)) {
            let total_out = hologram_ai_common::ir::shape::DimExpr::Concrete(
                (num_heads * _head_dim + 2 * _num_kv_heads * _head_dim) as u64,
            );
            let qkv_weight = builder.add_tensor(
                &format!("model.layers.{}.self_attn.qkv_proj.weight", l),
                DType::F32,
                vec![total_out.clone(), hidden.clone()],
            );

            let q_weight = builder.add_tensor(&format!("q_weight_{l}"), DType::F32, vec![q_out_var.clone(), hidden.clone()]);
            let k_weight = builder.add_tensor(&format!("k_weight_{l}"), DType::F32, vec![k_out_var.clone(), hidden.clone()]);
            let v_weight = builder.add_tensor(&format!("v_weight_{l}"), DType::F32, vec![v_out_var.clone(), hidden.clone()]);

            let q_dim = (num_heads * _head_dim) as u64;
            let kv_dim = (_num_kv_heads * _head_dim) as u64;

            builder.add_node(
                hologram_ai_common::ir::op::AiOp::Slice { axes: vec![0], starts: vec![0], ends: vec![q_dim as i64], steps: vec![1] },
                vec![qkv_weight],
                vec![q_weight],
            );
            builder.add_node(
                hologram_ai_common::ir::op::AiOp::Slice { axes: vec![0], starts: vec![q_dim as i64], ends: vec![(q_dim + kv_dim) as i64], steps: vec![1] },
                vec![qkv_weight],
                vec![k_weight],
            );
            builder.add_node(
                hologram_ai_common::ir::op::AiOp::Slice { axes: vec![0], starts: vec![(q_dim + kv_dim) as i64], ends: vec![(q_dim + 2 * kv_dim) as i64], steps: vec![1] },
                vec![qkv_weight],
                vec![v_weight],
            );

            q_flat = add_linear_layer_from_tensor(
                &mut builder,
                attn_norm_out,
                q_weight,
                LinearLayerParams {
                    weight_name: &format!("q_proj_{l}"),
                    in_features: hidden.clone(),
                    out_features: q_out_var.clone(),
                    output_name: &format!("q_flat_{l}"),
                    output_shape: vec![batch.clone(), seq.clone(), q_out_var.clone()],
                },
            );
            k_flat = add_linear_layer_from_tensor(
                &mut builder,
                attn_norm_out,
                k_weight,
                LinearLayerParams {
                    weight_name: &format!("k_proj_{l}"),
                    in_features: hidden.clone(),
                    out_features: k_out_var.clone(),
                    output_name: &format!("k_flat_{l}"),
                    output_shape: vec![batch.clone(), seq.clone(), k_out_var.clone()],
                },
            );
            v_flat = add_linear_layer_from_tensor(
                &mut builder,
                attn_norm_out,
                v_weight,
                LinearLayerParams {
                    weight_name: &format!("v_proj_{l}"),
                    in_features: hidden.clone(),
                    out_features: v_out_var.clone(),
                    output_name: &format!("v_flat_{l}"),
                    output_shape: vec![batch.clone(), seq.clone(), v_out_var.clone()],
                },
            );
        } else {
            q_flat = add_linear_layer(
                &mut builder,
                attn_norm_out,
                &format!("model.layers.{}.self_attn.q_proj.weight", l),
                hidden.clone(),
                q_out_var.clone(),
                &format!("q_flat_{}", l),
                vec![batch.clone(), seq.clone(), q_out_var.clone()],
            );
            k_flat = add_linear_layer(
                &mut builder,
                attn_norm_out,
                &format!("model.layers.{}.self_attn.k_proj.weight", l),
                hidden.clone(),
                k_out_var.clone(),
                &format!("k_flat_{}", l),
                vec![batch.clone(), seq.clone(), k_out_var.clone()],
            );
            v_flat = add_linear_layer(
                &mut builder,
                attn_norm_out,
                &format!("model.layers.{}.self_attn.v_proj.weight", l),
                hidden.clone(),
                v_out_var.clone(),
                &format!("v_flat_{}", l),
                vec![batch.clone(), seq.clone(), v_out_var.clone()],
            );
        }

        let q_out = builder.add_tensor(&format!("q_{}", l), DType::F32, vec![batch.clone(), seq.clone(), n_heads_expr.clone(), head_dim_expr.clone()]);
        let k_out = builder.add_tensor(&format!("k_{}", l), DType::F32, vec![batch.clone(), seq.clone(), n_kv_heads_expr.clone(), head_dim_expr.clone()]);
        let v_out = builder.add_tensor(&format!("v_{}", l), DType::F32, vec![batch.clone(), seq.clone(), n_kv_heads_expr.clone(), head_dim_expr.clone()]);

        // Reshape flat QKV to 4D for GQA
        builder.add_node(hologram_ai_common::ir::op::AiOp::Reshape { allow_zero: false }, vec![q_flat], vec![q_out]);
        builder.add_node(hologram_ai_common::ir::op::AiOp::Reshape { allow_zero: false }, vec![k_flat], vec![k_out]);
        builder.add_node(hologram_ai_common::ir::op::AiOp::Reshape { allow_zero: false }, vec![v_flat], vec![v_out]);

        // GQA
        let attn_out = builder.add_tensor(
            &format!("attn_out_{l}"),
            DType::F32,
            vec![batch.clone(), seq.clone(), n_heads_expr.clone(), head_dim_expr.clone()],
        );
        builder.add_node(
            hologram_ai_common::ir::op::AiOp::GroupedQueryAttention {
                num_heads,
                num_kv_heads: _num_kv_heads,
                head_dim: _head_dim,
                scale: None,
                causal: true,
                heads_first: false,
                qk_norm: false,
                rope: true,
                rope_base: 10000.0,
            },
            vec![q_out, k_out, v_out],
            vec![attn_out],
        );

        let attn_out_flat = builder.add_tensor(
            &format!("attn_out_flat_{l}"),
            DType::F32,
            vec![batch.clone(), seq.clone(), hidden.clone()],
        );
        builder.add_node(
            hologram_ai_common::ir::op::AiOp::Reshape { allow_zero: false },
            vec![attn_out],
            vec![attn_out_flat],
        );

        // O Projection
        let o_out = add_linear_layer(
            &mut builder,
            attn_out_flat,
            &format!("model.layers.{}.self_attn.o_proj.weight", l),
            q_out_var.clone(),
            hidden.clone(),
            &format!("o_out_{}", l),
            vec![batch.clone(), seq.clone(), hidden.clone()],
        );

        // Add (residual 1)
        let res1_out = builder.add_tensor(
            &format!("res1_{l}"),
            DType::F32,
            vec![batch.clone(), seq.clone(), hidden.clone()],
        );
        builder.add_node(
            hologram_ai_common::ir::op::AiOp::Add,
            vec![current, o_out],
            vec![res1_out],
        );

        // MLP Norm
        let mlp_norm_weight = builder.add_tensor(
            &format!("model.layers.{l}.post_attention_layernorm.weight"),
            DType::F32,
            vec![hidden.clone()],
        );
        let mlp_norm_out = builder.add_tensor(
            &format!("mlp_norm_{l}"),
            DType::F32,
            vec![batch.clone(), seq.clone(), hidden.clone()],
        );
        builder.add_node(
            hologram_ai_common::ir::op::AiOp::RmsNorm { epsilon: 1e-5 },
            vec![res1_out, mlp_norm_weight],
            vec![mlp_norm_out],
        );

        // MLP Gate + Up
        let intermediate_size = config
            .get("intermediate_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(hidden_size as u64 * 4);
        let ffn_hidden = hologram_ai_common::ir::shape::DimExpr::Concrete(intermediate_size);
        
        let gate_out;
        let up_out;
        
        if keys.contains(&format!("model.layers.{}.mlp.gate_up_proj.weight", l)) {
            let total_ffn = hologram_ai_common::ir::shape::DimExpr::Concrete(intermediate_size * 2);
            let gate_up_weight = builder.add_tensor(
                &format!("model.layers.{}.mlp.gate_up_proj.weight", l),
                DType::F32,
                vec![total_ffn.clone(), hidden.clone()],
            );

            let gate_weight = builder.add_tensor(&format!("gate_weight_{l}"), DType::F32, vec![ffn_hidden.clone(), hidden.clone()]);
            let up_weight = builder.add_tensor(&format!("up_weight_{l}"), DType::F32, vec![ffn_hidden.clone(), hidden.clone()]);

            builder.add_node(
                hologram_ai_common::ir::op::AiOp::Slice { axes: vec![0], starts: vec![0], ends: vec![intermediate_size as i64], steps: vec![1] },
                vec![gate_up_weight],
                vec![gate_weight],
            );
            builder.add_node(
                hologram_ai_common::ir::op::AiOp::Slice { axes: vec![0], starts: vec![intermediate_size as i64], ends: vec![(intermediate_size * 2) as i64], steps: vec![1] },
                vec![gate_up_weight],
                vec![up_weight],
            );

            gate_out = add_linear_layer_from_tensor(
                &mut builder,
                mlp_norm_out,
                gate_weight,
                LinearLayerParams {
                    weight_name: &format!("gate_proj_{l}"),
                    in_features: hidden.clone(),
                    out_features: ffn_hidden.clone(),
                    output_name: &format!("gate_out_{l}"),
                    output_shape: vec![batch.clone(), seq.clone(), ffn_hidden.clone()],
                },
            );
            up_out = add_linear_layer_from_tensor(
                &mut builder,
                mlp_norm_out,
                up_weight,
                LinearLayerParams {
                    weight_name: &format!("up_proj_{l}"),
                    in_features: hidden.clone(),
                    out_features: ffn_hidden.clone(),
                    output_name: &format!("up_out_{l}"),
                    output_shape: vec![batch.clone(), seq.clone(), ffn_hidden.clone()],
                },
            );
        } else {
            gate_out = add_linear_layer(
                &mut builder,
                mlp_norm_out,
                &format!("model.layers.{}.mlp.gate_proj.weight", l),
                hidden.clone(),
                ffn_hidden.clone(),
                &format!("gate_out_{}", l),
                vec![batch.clone(), seq.clone(), ffn_hidden.clone()],
            );
            up_out = add_linear_layer(
                &mut builder,
                mlp_norm_out,
                &format!("model.layers.{}.mlp.up_proj.weight", l),
                hidden.clone(),
                ffn_hidden.clone(),
                &format!("up_out_{}", l),
                vec![batch.clone(), seq.clone(), ffn_hidden.clone()],
            );
        }

        let silu_out = builder.add_tensor(
            &format!("silu_out_{}", l),
            DType::F32,
            vec![batch.clone(), seq.clone(), ffn_hidden.clone()],
        );
        builder.add_node(
            hologram_ai_common::ir::op::AiOp::Silu,
            vec![gate_out],
            vec![silu_out],
        );

        let mul_out = builder.add_tensor(
            &format!("mul_out_{}", l),
            DType::F32,
            vec![batch.clone(), seq.clone(), ffn_hidden.clone()],
        );
        builder.add_node(
            hologram_ai_common::ir::op::AiOp::Mul,
            vec![silu_out, up_out],
            vec![mul_out],
        );

        // MLP Down
        let down_out = add_linear_layer(
            &mut builder,
            mul_out,
            &format!("model.layers.{}.mlp.down_proj.weight", l),
            ffn_hidden.clone(),
            hidden.clone(),
            &format!("down_out_{}", l),
            vec![batch.clone(), seq.clone(), hidden.clone()],
        );

        // Add (residual 2)
        let res2_out = builder.add_tensor(
            &format!("res2_{l}"),
            DType::F32,
            vec![batch.clone(), seq.clone(), hidden.clone()],
        );
        builder.add_node(
            hologram_ai_common::ir::op::AiOp::Add,
            vec![res1_out, down_out],
            vec![res2_out],
        );

        current = res2_out;
    }

    // Final Norm
    let norm_weight = builder.add_tensor("model.norm.weight", DType::F32, vec![hidden.clone()]);
    let norm_out = builder.add_tensor(
        "norm_out",
        DType::F32,
        vec![batch.clone(), seq.clone(), hidden.clone()],
    );
    builder.add_node(
        hologram_ai_common::ir::op::AiOp::RmsNorm { epsilon: 1e-5 },
        vec![current, norm_weight],
        vec![norm_out],
    );

    // LM Head
    let _logits = add_linear_layer(
        &mut builder,
        norm_out,
        "lm_head.weight",
        hidden.clone(),
        vocab.clone(),
        "logits",
        vec![batch.clone(), seq.clone(), vocab.clone()],
    );

    // Output
    builder.add_output(_logits, "logits");

    let mut graph = builder.build();

    graph.metadata.insert(
        "vocab_size".to_string(),
        hologram_ai_common::MetaValue::Int(vocab_size as i64),
    );
    graph.metadata.insert(
        "arch".to_string(),
        hologram_ai_common::MetaValue::Str("parametric_transformer".to_string()),
    );
    graph.metadata.insert(
        "n_layers".to_string(),
        hologram_ai_common::MetaValue::Int(num_layers as i64),
    );

    Ok(graph)
}

fn extract_layer_idx(key: &str) -> Option<usize> {
    let parts: Vec<&str> = key.split('.').collect();
    for (i, part) in parts.iter().enumerate() {
        if (*part == "layers" || *part == "h" || *part == "blocks") && i + 1 < parts.len() {
            if let Ok(idx) = parts[i + 1].parse::<usize>() {
                return Some(idx);
            }
        }
    }
    None
}
