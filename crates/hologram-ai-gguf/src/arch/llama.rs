//! LLaMA architecture graph construction from GGUF tensors.
//!
//! Builds an `AiGraph` representing the LLaMA transformer:
//!   embed → (norm → attn → residual → norm → ffn → residual) × N → final_norm → lm_head

use std::collections::HashMap;
use std::path::Path;
use hologram_ai_common::{
    AiGraph, AiNode, AiOp, AiParam, DType, TensorId, TensorInfo,
    shape_from_concrete, DimVarTable, ConstraintStore, QuantDescriptor,
};
use crate::metadata::ArchParams;
use crate::parser::{GgufFile, GgmlType};
use anyhow::{Context, Result};

/// Build an `AiGraph` for a LLaMA-family model from parsed GGUF data.
pub fn build_llama_graph(
    gguf: &GgufFile,
    params: &ArchParams,
    model_path: &Path,
) -> Result<AiGraph> {
    let mut b = GraphAssembler::new(params, model_path);

    // ── Token embedding ────────────────────────────────────────────────
    let input_ids = b.add_input("input_ids", DType::INT64, &[1, 0]); // [batch, seq_len]
    let embed_weight = b.add_tensor("token_embd.weight", gguf, params.vocab_size as u64, params.embedding_length as u64)?;
    let embedded = b.add_node(AiOp::Embed, vec![input_ids, embed_weight], DType::F32, &[1, 0, params.embedding_length as u64]);

    let mut hidden = embedded;

    // ── Transformer blocks ─────────────────────────────────────────────
    for layer in 0..params.block_count {
        let prefix = format!("blk.{layer}");

        // Attention norm (RMSNorm)
        let attn_norm_w = b.add_tensor(&format!("{prefix}.attn_norm.weight"), gguf, params.embedding_length as u64, 1)?;
        let normed = b.add_node(
            AiOp::RmsNorm { epsilon: params.layer_norm_rms_epsilon },
            vec![hidden, attn_norm_w],
            DType::F32,
            &[1, 0, params.embedding_length as u64],
        );

        // Q/K/V projections
        let head_dim = params.embedding_length / params.head_count;
        let q_w = b.add_tensor(&format!("{prefix}.attn_q.weight"), gguf, params.embedding_length as u64, params.embedding_length as u64)?;
        let k_w = b.add_tensor(&format!("{prefix}.attn_k.weight"), gguf, params.head_count_kv as u64 * head_dim as u64, params.embedding_length as u64)?;
        let v_w = b.add_tensor(&format!("{prefix}.attn_v.weight"), gguf, params.head_count_kv as u64 * head_dim as u64, params.embedding_length as u64)?;

        let q = b.add_node(AiOp::MatMul, vec![normed, q_w], DType::F32, &[1, 0, params.embedding_length as u64]);
        let k = b.add_node(AiOp::MatMul, vec![normed, k_w], DType::F32, &[1, 0, params.head_count_kv as u64 * head_dim as u64]);
        let v = b.add_node(AiOp::MatMul, vec![normed, v_w], DType::F32, &[1, 0, params.head_count_kv as u64 * head_dim as u64]);

        // RoPE
        let q_rope = b.add_node(
            AiOp::RotaryEmbedding { base: params.rope_freq_base, dim: params.rope_dimension_count },
            vec![q],
            DType::F32,
            &[1, 0, params.embedding_length as u64],
        );
        let k_rope = b.add_node(
            AiOp::RotaryEmbedding { base: params.rope_freq_base, dim: params.rope_dimension_count },
            vec![k],
            DType::F32,
            &[1, 0, params.head_count_kv as u64 * head_dim as u64],
        );

        // Grouped-query attention
        let attn_out = b.add_node(
            AiOp::GroupedQueryAttention {
                num_heads: params.head_count,
                num_kv_heads: params.head_count_kv,
                head_dim,
                scale: None,
                causal: true,
            },
            vec![q_rope, k_rope, v],
            DType::F32,
            &[1, 0, params.embedding_length as u64],
        );

        // Output projection
        let o_w = b.add_tensor(&format!("{prefix}.attn_output.weight"), gguf, params.embedding_length as u64, params.embedding_length as u64)?;
        let attn_proj = b.add_node(AiOp::MatMul, vec![attn_out, o_w], DType::F32, &[1, 0, params.embedding_length as u64]);

        // Residual connection
        let residual1 = b.add_node(AiOp::Add, vec![hidden, attn_proj], DType::F32, &[1, 0, params.embedding_length as u64]);

        // FFN norm (RMSNorm)
        let ffn_norm_w = b.add_tensor(&format!("{prefix}.ffn_norm.weight"), gguf, params.embedding_length as u64, 1)?;
        let ffn_normed = b.add_node(
            AiOp::RmsNorm { epsilon: params.layer_norm_rms_epsilon },
            vec![residual1, ffn_norm_w],
            DType::F32,
            &[1, 0, params.embedding_length as u64],
        );

        // FFN: gate + up projections → SwiGLU → down projection
        let gate_w = b.add_tensor(&format!("{prefix}.ffn_gate.weight"), gguf, params.feed_forward_length as u64, params.embedding_length as u64)?;
        let up_w = b.add_tensor(&format!("{prefix}.ffn_up.weight"), gguf, params.feed_forward_length as u64, params.embedding_length as u64)?;
        let down_w = b.add_tensor(&format!("{prefix}.ffn_down.weight"), gguf, params.feed_forward_length as u64, params.embedding_length as u64)?;

        let gate = b.add_node(AiOp::MatMul, vec![ffn_normed, gate_w], DType::F32, &[1, 0, params.feed_forward_length as u64]);
        let up = b.add_node(AiOp::MatMul, vec![ffn_normed, up_w], DType::F32, &[1, 0, params.feed_forward_length as u64]);
        let swiglu = b.add_node(AiOp::FusedSwiGLU, vec![gate, up], DType::F32, &[1, 0, params.feed_forward_length as u64]);
        let down = b.add_node(AiOp::MatMul, vec![swiglu, down_w], DType::F32, &[1, 0, params.embedding_length as u64]);

        // Residual connection
        hidden = b.add_node(AiOp::Add, vec![residual1, down], DType::F32, &[1, 0, params.embedding_length as u64]);
    }

    // ── Final norm + LM head ───────────────────────────────────────────
    let final_norm_w = b.add_tensor("output_norm.weight", gguf, params.embedding_length as u64, 1)?;
    let final_normed = b.add_node(
        AiOp::RmsNorm { epsilon: params.layer_norm_rms_epsilon },
        vec![hidden, final_norm_w],
        DType::F32,
        &[1, 0, params.embedding_length as u64],
    );

    // LM head — may share weights with token embedding (tied embeddings).
    let lm_head_w = if b.has_tensor("output.weight", gguf) {
        b.add_tensor("output.weight", gguf, params.vocab_size as u64, params.embedding_length as u64)?
    } else {
        embed_weight // tied embeddings
    };
    let logits = b.add_node(AiOp::MatMul, vec![final_normed, lm_head_w], DType::F32, &[1, 0, params.vocab_size as u64]);

    b.finish(vec![input_ids], vec![logits])
}

// ── Graph assembler helper ─────────────────────────────────────────────

struct GraphAssembler<'a> {
    params: &'a ArchParams,
    model_path: &'a Path,
    next_tid: TensorId,
    next_nid: u32,
    nodes: Vec<AiNode>,
    tensor_info: HashMap<TensorId, TensorInfo>,
    ai_params: HashMap<TensorId, AiParam>,
    tensor_name_to_tid: HashMap<String, TensorId>,
    warnings: Vec<hologram_ai_common::ImportWarning>,
}

impl<'a> GraphAssembler<'a> {
    fn new(params: &'a ArchParams, model_path: &'a Path) -> Self {
        Self {
            params,
            model_path,
            next_tid: 0,
            next_nid: 0,
            nodes: Vec::new(),
            tensor_info: HashMap::new(),
            ai_params: HashMap::new(),
            tensor_name_to_tid: HashMap::new(),
            warnings: Vec::new(),
        }
    }

    fn alloc_tid(&mut self) -> TensorId {
        let tid = self.next_tid;
        self.next_tid += 1;
        tid
    }

    fn alloc_nid(&mut self) -> u32 {
        let nid = self.next_nid;
        self.next_nid += 1;
        nid
    }

    fn add_input(&mut self, _name: &str, dtype: DType, shape: &[u64]) -> TensorId {
        let tid = self.alloc_tid();
        self.tensor_info.insert(tid, TensorInfo::new(dtype, shape_from_concrete(shape)));
        tid
    }

    fn has_tensor(&self, name: &str, gguf: &GgufFile) -> bool {
        gguf.tensors.iter().any(|t| t.name == name)
    }

    fn add_tensor(
        &mut self,
        name: &str,
        gguf: &GgufFile,
        rows: u64,
        cols: u64,
    ) -> Result<TensorId> {
        // Check if we've already registered this tensor (e.g., tied embeddings).
        if let Some(&tid) = self.tensor_name_to_tid.get(name) {
            return Ok(tid);
        }

        let desc = gguf.tensors.iter()
            .find(|t| t.name == name)
            .with_context(|| format!("missing GGUF tensor: {name}"))?;

        let tid = self.alloc_tid();
        let (storage_dtype, logical_dtype) = ggml_type_to_dtypes(desc.ggml_type);

        let shape = if desc.dims.is_empty() {
            shape_from_concrete(&[rows, cols])
        } else {
            shape_from_concrete(&desc.dims)
        };

        let quant = match desc.ggml_type {
            GgmlType::Q4_0 => QuantDescriptor::q4_0(),
            GgmlType::Q8_0 => QuantDescriptor::q8_0(),
            _ => QuantDescriptor::none(),
        };

        let info = TensorInfo {
            logical_dtype,
            storage_dtype,
            shape,
            quant,
        };

        let byte_offset = gguf.data_offset + desc.offset;
        let byte_len = desc.byte_size();

        self.tensor_info.insert(tid, info.clone());
        self.ai_params.insert(
            tid,
            AiParam::mmap(self.model_path.to_path_buf(), byte_offset, byte_len, info),
        );
        self.tensor_name_to_tid.insert(name.to_string(), tid);

        Ok(tid)
    }

    fn add_node(
        &mut self,
        op: AiOp,
        inputs: Vec<TensorId>,
        output_dtype: DType,
        output_shape: &[u64],
    ) -> TensorId {
        let output_tid = self.alloc_tid();
        self.tensor_info.insert(
            output_tid,
            TensorInfo::new(output_dtype, shape_from_concrete(output_shape)),
        );

        let nid = self.alloc_nid();
        self.nodes.push(AiNode::new(nid, op, inputs, vec![output_tid]));

        output_tid
    }

    fn finish(self, inputs: Vec<TensorId>, outputs: Vec<TensorId>) -> Result<AiGraph> {
        let mut metadata = HashMap::new();
        metadata.insert(
            "arch".to_string(),
            hologram_ai_common::MetaValue::Str(self.params.arch.clone()),
        );
        metadata.insert(
            "vocab_size".to_string(),
            hologram_ai_common::MetaValue::Int(self.params.vocab_size as i64),
        );
        metadata.insert(
            "context_length".to_string(),
            hologram_ai_common::MetaValue::Int(self.params.context_length as i64),
        );
        metadata.insert(
            "n_layers".to_string(),
            hologram_ai_common::MetaValue::Int(self.params.block_count as i64),
        );
        metadata.insert(
            "n_embd".to_string(),
            hologram_ai_common::MetaValue::Int(self.params.embedding_length as i64),
        );

        Ok(AiGraph {
            name: format!("{}-llama", self.params.arch),
            nodes: self.nodes,
            inputs,
            outputs,
            params: self.ai_params,
            tensor_info: self.tensor_info,
            metadata,
            warnings: self.warnings,
            dim_vars: DimVarTable::default(),
            shape_constraints: ConstraintStore::default(),
        })
    }
}

fn ggml_type_to_dtypes(gt: GgmlType) -> (DType, DType) {
    match gt {
        GgmlType::F32 => (DType::F32, DType::F32),
        GgmlType::F16 => (DType::F16, DType::F16),
        GgmlType::BF16 => (DType::BF16, DType::BF16),
        GgmlType::Q4_0 | GgmlType::Q4_1 | GgmlType::Q4K => (DType::U8, DType::F32),
        GgmlType::Q5_0 | GgmlType::Q5_1 | GgmlType::Q5K => (DType::U8, DType::F32),
        GgmlType::Q8_0 | GgmlType::Q8_1 | GgmlType::Q8K => (DType::U8, DType::F32),
        GgmlType::Q2K | GgmlType::Q3K | GgmlType::Q6K => (DType::U8, DType::F32),
        GgmlType::I8 => (DType::INT8, DType::INT8),
        GgmlType::I16 | GgmlType::I32 => (DType::INT32, DType::INT32),
        GgmlType::I64 => (DType::INT64, DType::INT64),
        _ => (DType::U8, DType::F32),
    }
}
