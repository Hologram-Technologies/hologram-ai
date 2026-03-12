//! LLaMA architecture graph construction from GGUF tensors.
//!
//! Builds an `AiGraph` representing the LLaMA transformer:
//!   embed → (norm → attn → residual → norm → ffn → residual) × N → final_norm → lm_head

use crate::metadata::ArchParams;
use crate::parser::{GgmlType, GgufFile};
use anyhow::{Context, Result};
use hologram_ai_common::{
    canonical_vars, shape_from_concrete, AiGraph, AiNode, AiOp, AiParam, ConstraintStore, DType,
    DimExpr, DimVarSource, DimVarTable, QuantDescriptor, Shape, TensorId, TensorInfo,
};
use std::collections::HashMap;
use std::path::Path;

/// Build an `AiGraph` for a LLaMA-family model from parsed GGUF data.
pub fn build_llama_graph(
    gguf: &GgufFile,
    params: &ArchParams,
    model_path: &Path,
) -> Result<AiGraph> {
    let mut b = GraphAssembler::new(params, model_path);

    // ── Token embedding ────────────────────────────────────────────────
    let input_shape = Shape::from_vec(vec![b.batch_dim.clone(), b.seq_len_dim.clone()]);
    let input_ids = b.add_input("input_ids", DType::INT64, input_shape);
    let embed_weight = b.add_tensor(
        "token_embd.weight",
        gguf,
        params.vocab_size as u64,
        params.embedding_length as u64,
    )?;
    let emb_dim = params.embedding_length as u64;
    let embedded = b.add_node(
        AiOp::Embed,
        vec![input_ids, embed_weight],
        DType::F32,
        b.bsv(emb_dim),
    );

    let mut hidden = embedded;
    let ffn_dim = params.feed_forward_length as u64;
    let kv_dim = params.head_count_kv as u64 * (params.embedding_length / params.head_count) as u64;
    let vocab = params.vocab_size as u64;

    // ── Transformer blocks ─────────────────────────────────────────────
    for layer in 0..params.block_count {
        let prefix = format!("blk.{layer}");

        // Attention norm (RMSNorm)
        let attn_norm_w = b.add_tensor(&format!("{prefix}.attn_norm.weight"), gguf, emb_dim, 1)?;
        let normed = b.add_node(
            AiOp::RmsNorm {
                epsilon: params.layer_norm_rms_epsilon,
            },
            vec![hidden, attn_norm_w],
            DType::F32,
            b.bsv(emb_dim),
        );

        // Q/K/V projections
        let head_dim = params.embedding_length / params.head_count;
        let q_w = b.add_tensor(&format!("{prefix}.attn_q.weight"), gguf, emb_dim, emb_dim)?;
        let k_w = b.add_tensor(&format!("{prefix}.attn_k.weight"), gguf, kv_dim, emb_dim)?;
        let v_w = b.add_tensor(&format!("{prefix}.attn_v.weight"), gguf, kv_dim, emb_dim)?;

        let gemm = |b: &mut GraphAssembler, x, w, out_dim| {
            b.add_node(
                AiOp::Gemm { alpha: 1.0, beta: 0.0, trans_a: false, trans_b: true },
                vec![x, w],
                DType::F32,
                b.bsv(out_dim),
            )
        };
        let q = gemm(&mut b, normed, q_w, emb_dim);
        let k = gemm(&mut b, normed, k_w, kv_dim);
        let v = gemm(&mut b, normed, v_w, kv_dim);

        // RoPE
        let q_rope = b.add_node(
            AiOp::RotaryEmbedding {
                base: params.rope_freq_base,
                dim: params.rope_dimension_count,
            },
            vec![q],
            DType::F32,
            b.bsv(emb_dim),
        );
        let k_rope = b.add_node(
            AiOp::RotaryEmbedding {
                base: params.rope_freq_base,
                dim: params.rope_dimension_count,
            },
            vec![k],
            DType::F32,
            b.bsv(kv_dim),
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
            b.bsv(emb_dim),
        );

        // Output projection
        let o_w = b.add_tensor(
            &format!("{prefix}.attn_output.weight"),
            gguf,
            emb_dim,
            emb_dim,
        )?;
        let attn_proj = gemm(&mut b, attn_out, o_w, emb_dim);

        // Residual connection
        let residual1 = b.add_node(
            AiOp::Add,
            vec![hidden, attn_proj],
            DType::F32,
            b.bsv(emb_dim),
        );

        // FFN norm (RMSNorm)
        let ffn_norm_w = b.add_tensor(&format!("{prefix}.ffn_norm.weight"), gguf, emb_dim, 1)?;
        let ffn_normed = b.add_node(
            AiOp::RmsNorm {
                epsilon: params.layer_norm_rms_epsilon,
            },
            vec![residual1, ffn_norm_w],
            DType::F32,
            b.bsv(emb_dim),
        );

        // FFN: gate + up projections → SwiGLU → down projection
        let gate_w = b.add_tensor(&format!("{prefix}.ffn_gate.weight"), gguf, ffn_dim, emb_dim)?;
        let up_w = b.add_tensor(&format!("{prefix}.ffn_up.weight"), gguf, ffn_dim, emb_dim)?;
        let down_w = b.add_tensor(&format!("{prefix}.ffn_down.weight"), gguf, ffn_dim, emb_dim)?;

        let gate = gemm(&mut b, ffn_normed, gate_w, ffn_dim);
        let up = gemm(&mut b, ffn_normed, up_w, ffn_dim);
        let swiglu = b.add_node(
            AiOp::FusedSwiGLU,
            vec![gate, up],
            DType::F32,
            b.bsv(ffn_dim),
        );
        let down = gemm(&mut b, swiglu, down_w, emb_dim);

        // Residual connection
        hidden = b.add_node(AiOp::Add, vec![residual1, down], DType::F32, b.bsv(emb_dim));
    }

    // ── Final norm + LM head ───────────────────────────────────────────
    let final_norm_w = b.add_tensor("output_norm.weight", gguf, emb_dim, 1)?;
    let final_normed = b.add_node(
        AiOp::RmsNorm {
            epsilon: params.layer_norm_rms_epsilon,
        },
        vec![hidden, final_norm_w],
        DType::F32,
        b.bsv(emb_dim),
    );

    // LM head — may share weights with token embedding (tied embeddings).
    let lm_head_w = if b.has_tensor("output.weight", gguf) {
        b.add_tensor("output.weight", gguf, vocab, emb_dim)?
    } else {
        embed_weight // tied embeddings
    };
    let logits = b.add_node(
        AiOp::Gemm { alpha: 1.0, beta: 0.0, trans_a: false, trans_b: true },
        vec![final_normed, lm_head_w],
        DType::F32,
        b.bsv(vocab),
    );

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
    dim_vars: DimVarTable,
    /// Symbolic batch dimension expression.
    batch_dim: DimExpr,
    /// Symbolic seq_len dimension expression.
    seq_len_dim: DimExpr,
}

impl<'a> GraphAssembler<'a> {
    fn new(params: &'a ArchParams, model_path: &'a Path) -> Self {
        let mut dim_vars = DimVarTable::default();

        // Register canonical symbolic dimensions with bounds.
        let batch_id = dim_vars.intern_with_bounds(
            canonical_vars::BATCH,
            Some(1),
            Some(1), // MVP: batch=1 for now
            DimVarSource::Import,
        );
        let seq_len_id = dim_vars.intern_with_bounds(
            canonical_vars::SEQ_LEN,
            Some(1),
            Some(params.context_length as u64),
            DimVarSource::Import,
        );

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
            dim_vars,
            batch_dim: DimExpr::Var(batch_id),
            seq_len_dim: DimExpr::Var(seq_len_id),
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

    fn add_input(&mut self, _name: &str, dtype: DType, shape: Shape) -> TensorId {
        let tid = self.alloc_tid();
        self.tensor_info.insert(tid, TensorInfo::new(dtype, shape));
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

        let desc = gguf
            .tensors
            .iter()
            .find(|t| t.name == name)
            .with_context(|| format!("missing GGUF tensor: {name}"))?;

        let tid = self.alloc_tid();
        let (storage_dtype, logical_dtype) = ggml_type_to_dtypes(desc.ggml_type);

        // GGUF stores dims in ggml order [ne[0], ne[1]] where ne[0] is stride-1
        // (contiguous/inner dimension). Reverse to ML convention [outer, inner]
        // so that e.g. embedding weights become [vocab, dim] not [dim, vocab].
        // Note: all linear weights use Gemm { trans_b: true } since ggml
        // computes y = x @ W^T, so the physical data is transposed vs y = x @ W.
        let shape = if desc.dims.is_empty() {
            shape_from_concrete(&[rows, cols])
        } else {
            let mut reversed = desc.dims.clone();
            reversed.reverse();
            shape_from_concrete(&reversed)
        };

        let quant = match desc.ggml_type {
            GgmlType::Q4_0 => QuantDescriptor::q4_0(),
            GgmlType::Q8_0 => QuantDescriptor::q8_0(),
            GgmlType::Q6K => QuantDescriptor::q6_k(),
            _ => QuantDescriptor::none(),
        };

        let info = TensorInfo {
            logical_dtype,
            storage_dtype,
            shape,
            quant,
            known_i64_values: None,
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
        output_shape: Shape,
    ) -> TensorId {
        let output_tid = self.alloc_tid();
        self.tensor_info
            .insert(output_tid, TensorInfo::new(output_dtype, output_shape));

        let nid = self.alloc_nid();
        self.nodes
            .push(AiNode::new(nid, op, inputs, vec![output_tid]));

        output_tid
    }

    /// Build a symbolic shape: [batch, seq_len, concrete_last_dim].
    fn bsv(&self, last_dim: u64) -> Shape {
        Shape::from_vec(vec![
            self.batch_dim.clone(),
            self.seq_len_dim.clone(),
            DimExpr::Concrete(last_dim),
        ])
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
            input_names: vec!["input_ids".into()],
            output_names: vec!["logits".into()],
            params: self.ai_params,
            tensor_info: self.tensor_info,
            metadata,
            warnings: self.warnings,
            dim_vars: self.dim_vars,
            shape_constraints: ConstraintStore::default(),
            subgraphs: HashMap::new(),
            tensor_names: HashMap::new(),
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
