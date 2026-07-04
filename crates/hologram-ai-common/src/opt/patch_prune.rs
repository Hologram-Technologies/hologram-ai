//! ViT patch pruning injection (PixelPrune).
//!
//! Rewrites a Vision Transformer graph to accept a reduced token sequence
//! selected by a runtime patch pruning kernel. Instead of processing all
//! `grid_h × grid_w` patches from the image, the compiled graph accepts
//! `max_kept = ceil(grid_h × grid_w × budget_ratio)` patches with
//! explicitly-provided position indices.
//!
//! # How it works
//!
//! A ViT typically has this structure:
//!
//! ```text
//! pixel_input [1, C, H, W]
//!     │
//!     ▼
//! Conv2d (patch_embed, kernel=stride=patch_size)
//!     │
//!     ▼
//! Reshape → [1, N_patches, embed_dim]
//!     │
//!     ▼
//! Add(patches, pos_embed)       ← learned position embedding
//!     │
//!     ▼
//! TransformerEncoder            ← N layers of MHA + FFN
//! ```
//!
//! After this pass:
//!
//! ```text
//! pixel_input [1, C, H, W]
//!     │
//!     ▼
//! Conv2d (patch_embed)
//!     │
//!     ▼
//! Reshape → [1, N_patches, embed_dim]
//!     │
//!     ▼
//! Gather(axis=1, kept_indices)  ← new: select budget patches
//!     │   → [1, max_kept, embed_dim]
//!     │
//!     ▼                            pos_embed [1, N_patches, embed_dim]
//!     │                                │
//!     │                                ▼
//!     │                  Gather(axis=1, kept_indices)
//!     │                                │
//!     │                    → [1, max_kept, embed_dim]
//!     │                                │
//!     ▼                                ▼
//! Add(pruned_patches, pruned_pos_embed)
//!     │   → [1, max_kept, embed_dim]
//!     ▼
//! TransformerEncoder   ← all shapes: seq = max_kept (static)
//! ```
//!
//! The runtime PatchPrune kernel (in hologram base) runs before the compiled
//! graph, producing `kept_indices` via Pred-2D predictive coding.
//!
//! # References
//!
//! - PixelPrune paper: arXiv 2604.00886
//! 

use super::pipeline::Pass;
use crate::ir::node::AiNode;
use crate::ir::shape::shape_from_concrete;
use crate::ir::{AiGraph, AiOp, DType, SemanticHint, TensorInfo};
use hologram_ai_quant::QuantDescriptor;

/// Inject patch-pruning Gather nodes into a ViT graph.
///
/// Budget ratio controls what fraction of the total patch grid is retained.
/// Range: `(0.0, 1.0]`. At 1.0, the pass is a no-op (all patches kept).
pub struct PatchPruneInjection {
    pub budget_ratio: f32,
}

impl Pass for PatchPruneInjection {
    fn name(&self) -> &str {
        "PatchPruneInjection"
    }

    fn run(&self, mut graph: AiGraph) -> anyhow::Result<AiGraph> {
        if self.budget_ratio >= 1.0 {
            return Ok(graph);
        }

        // Step 1: Find the patch embedding Conv2d.
        // It must be a Conv2d where kernel_shape == strides (non-overlapping
        // patch extraction) and the input is a 4D image tensor from graph inputs.
        let patch_conv_idx = match find_patch_embed_conv(&graph) {
            Some(idx) => idx,
            None => {
                tracing::debug!("PatchPruneInjection: no patch embedding Conv2d found, skipping");
                return Ok(graph);
            }
        };

        // Step 2: Find the Reshape that flattens spatial dims to a sequence.
        // Pattern: Conv2d output → Reshape → [1, N_patches, embed_dim]
        let (reshape_idx, n_patches, embed_dim) =
            match find_post_conv_reshape(&graph, patch_conv_idx) {
                Some(v) => v,
                None => {
                    tracing::debug!(
                        "PatchPruneInjection: no suitable Reshape after patch Conv2d, skipping"
                    );
                    return Ok(graph);
                }
            };

        // Step 3: Find the position embedding Add.
        // Pattern: Add(reshape_output, pos_embed_param) where pos_embed is a constant.
        let reshape_output = graph.nodes[reshape_idx].outputs[0];
        let (add_idx, pos_embed_tid) = match find_pos_embed_add(&graph, reshape_output) {
            Some(v) => v,
            None => {
                tracing::debug!("PatchPruneInjection: no position embedding Add found, skipping");
                return Ok(graph);
            }
        };

        // Step 4: Compute the budget.
        let max_kept = ((n_patches as f64 * self.budget_ratio as f64).ceil() as u64).max(1);

        tracing::info!(
            n_patches,
            max_kept,
            budget_ratio = self.budget_ratio,
            embed_dim,
            "PatchPruneInjection: rewriting ViT graph for patch pruning"
        );

        // Step 5: Allocate new TensorIds and NodeIds.
        let max_tid = graph.tensor_info.keys().copied().max().unwrap_or(0);
        let max_nid = graph.nodes.iter().map(|n| n.id).max().unwrap_or(0);

        let kept_indices_tid = max_tid + 1;
        let pruned_patches_tid = max_tid + 2;
        let pruned_pos_embed_tid = max_tid + 3;

        let gather_patches_nid = max_nid + 1;
        let gather_pos_embed_nid = max_nid + 2;

        // Step 6: Register kept_indices as a new graph input.
        // Shape: [max_kept] — 1D index tensor. The runtime fills this with
        // flattened indices into the [N_patches] sequence dimension.
        graph.tensor_info.insert(
            kept_indices_tid,
            TensorInfo {
                logical_dtype: DType::INT64,
                storage_dtype: DType::INT64,
                shape: shape_from_concrete(&[max_kept]),
                quant: QuantDescriptor::none(),
                known_i64_values: None,
                semantic: SemanticHint::Unknown,
            },
        );
        graph.inputs.push(kept_indices_tid);
        graph.input_names.push("kept_indices".into());
        graph
            .tensor_names
            .insert(kept_indices_tid, "kept_indices".into());

        // Step 7: Register output tensors for the two Gather nodes.
        // pruned_patches: [1, max_kept, embed_dim]
        graph.tensor_info.insert(
            pruned_patches_tid,
            TensorInfo {
                logical_dtype: DType::F32,
                storage_dtype: DType::F32,
                shape: shape_from_concrete(&[1, max_kept, embed_dim]),
                quant: QuantDescriptor::none(),
                known_i64_values: None,
                semantic: SemanticHint::Embedding,
            },
        );
        // pruned_pos_embed: [1, max_kept, embed_dim]
        graph.tensor_info.insert(
            pruned_pos_embed_tid,
            TensorInfo {
                logical_dtype: DType::F32,
                storage_dtype: DType::F32,
                shape: shape_from_concrete(&[1, max_kept, embed_dim]),
                quant: QuantDescriptor::none(),
                known_i64_values: None,
                semantic: SemanticHint::Position,
            },
        );

        // Step 8: Create Gather nodes.
        // Gather(axis=1) selects along the sequence dimension.
        let gather_patches_node = AiNode::new(
            gather_patches_nid,
            AiOp::Gather { axis: 1 },
            vec![reshape_output, kept_indices_tid],
            vec![pruned_patches_tid],
        );
        let gather_pos_embed_node = AiNode::new(
            gather_pos_embed_nid,
            AiOp::Gather { axis: 1 },
            vec![pos_embed_tid, kept_indices_tid],
            vec![pruned_pos_embed_tid],
        );

        // Step 9: Rewire the Add node to consume pruned tensors.
        let add_node = &graph.nodes[add_idx];
        let add_input_0 = add_node.inputs[0];
        let add_input_1 = add_node.inputs[1];
        let new_input_0 = if add_input_0 == reshape_output {
            pruned_patches_tid
        } else {
            add_input_0
        };
        let new_input_1 = if add_input_1 == pos_embed_tid {
            pruned_pos_embed_tid
        } else if add_input_1 == reshape_output {
            pruned_patches_tid
        } else {
            add_input_1
        };
        // Also handle the case where pos_embed is input[0] and patches is input[1].
        let final_input_0 = if add_input_0 == pos_embed_tid {
            pruned_pos_embed_tid
        } else {
            new_input_0
        };
        graph.nodes[add_idx].inputs = vec![final_input_0, new_input_1];

        // Update the Add output shape to [1, max_kept, embed_dim].
        let add_output = graph.nodes[add_idx].outputs[0];
        if let Some(info) = graph.tensor_info.get_mut(&add_output) {
            info.shape = shape_from_concrete(&[1, max_kept, embed_dim]);
        }

        // Step 10: Insert Gather nodes into the graph before the Add node.
        // They must appear after the Reshape and before the Add in topological order.
        graph.nodes.insert(add_idx, gather_pos_embed_node);
        graph.nodes.insert(add_idx, gather_patches_node);
        // add_idx shifted by 2 after the insertions — no need to update since
        // we already rewired inputs above.

        // Step 11: Store pruning metadata for the runtime.
        use crate::ir::MetaValue;
        graph
            .metadata
            .insert("patch_prune_budget".into(), MetaValue::Int(max_kept as i64));
        graph.metadata.insert(
            "patch_prune_grid_patches".into(),
            MetaValue::Int(n_patches as i64),
        );
        graph.metadata.insert(
            "patch_prune_embed_dim".into(),
            MetaValue::Int(embed_dim as i64),
        );

        graph.invalidate_topo_cache();

        tracing::info!(
            max_kept,
            n_patches,
            embed_dim,
            "PatchPruneInjection: injected Gather nodes, seq_len {n_patches} → {max_kept}"
        );

        Ok(graph)
    }
}

/// Find the patch embedding Conv2d node.
///
/// Criteria:
/// - Op is Conv2d with kernel_shape == strides (non-overlapping patch extraction)
/// - Input is a graph input with 4D shape [N, C, H, W] where C ∈ {1, 3, 4}
fn find_patch_embed_conv(graph: &AiGraph) -> Option<usize> {
    let graph_input_set: std::collections::HashSet<u32> = graph.inputs.iter().copied().collect();

    graph.nodes.iter().enumerate().find_map(|(idx, node)| {
        let (kernel_shape, strides) = match &node.op {
            AiOp::Conv {
                kernel_shape,
                strides,
                ..
            } => (kernel_shape, strides),
            _ => return None,
        };

        // Non-overlapping: kernel == stride
        if kernel_shape != strides {
            return None;
        }

        // Input must be a graph input (the image tensor).
        let input_tid = *node.inputs.first()?;
        if !graph_input_set.contains(&input_tid) {
            return None;
        }

        // Input must be 4D with C ∈ {1, 3, 4}.
        let info = graph.tensor_info.get(&input_tid)?;
        if info.shape.len() != 4 {
            return None;
        }
        let channels = info.shape[1].as_concrete()?;
        if channels != 1 && channels != 3 && channels != 4 {
            return None;
        }

        Some(idx)
    })
}

/// Find the Reshape node that flattens Conv2d output to [1, N_patches, embed_dim].
///
/// Returns (reshape_idx, n_patches, embed_dim).
fn find_post_conv_reshape(graph: &AiGraph, conv_idx: usize) -> Option<(usize, u64, u64)> {
    let conv_output = *graph.nodes[conv_idx].outputs.first()?;

    for (idx, node) in graph.nodes.iter().enumerate() {
        if !matches!(node.op, AiOp::Reshape { .. }) {
            continue;
        }
        if !node.inputs.contains(&conv_output) {
            continue;
        }
        // Output must be 3D: [batch, seq, embed_dim]
        let out_tid = *node.outputs.first()?;
        let info = graph.tensor_info.get(&out_tid)?;
        if info.shape.len() != 3 {
            continue;
        }
        let n_patches = info.shape[1].as_concrete()?;
        let embed_dim = info.shape[2].as_concrete()?;
        if n_patches > 1 && embed_dim > 1 {
            return Some((idx, n_patches, embed_dim));
        }
    }
    None
}

/// Find the position embedding Add node.
///
/// Pattern: Add where one input is `reshape_output` and the other is a
/// constant parameter (the learned position embedding table).
///
/// Returns (add_idx, pos_embed_tid).
fn find_pos_embed_add(graph: &AiGraph, reshape_output: u32) -> Option<(usize, u32)> {
    for (idx, node) in graph.nodes.iter().enumerate() {
        if !matches!(node.op, AiOp::Add) {
            continue;
        }
        if node.inputs.len() != 2 {
            continue;
        }
        let (a, b) = (node.inputs[0], node.inputs[1]);
        if a == reshape_output {
            // b should be the position embedding (a constant param).
            if graph.params.contains_key(&b) || is_constant_output(graph, b) {
                return Some((idx, b));
            }
        } else if b == reshape_output {
            // a should be the position embedding.
            if graph.params.contains_key(&a) || is_constant_output(graph, a) {
                return Some((idx, a));
            }
        }
    }
    None
}

/// Check if a TensorId is produced by a Constant op.
fn is_constant_output(graph: &AiGraph, tid: u32) -> bool {
    graph
        .nodes
        .iter()
        .any(|n| matches!(n.op, AiOp::Constant { .. }) && n.outputs.contains(&tid))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::node::AiNode;
    use crate::ir::param::AiParam;
    use crate::ir::shape::{shape_from_concrete, ConstraintStore, DimVarTable};
    use std::collections::HashMap;

    /// Build a minimal ViT-shaped graph:
    /// input [1, 3, 224, 224] → Conv2d(16×16) → Reshape [1, 196, 768] → Add(pos_embed) → output
    fn make_vit_graph() -> AiGraph {
        let mut ti = HashMap::new();
        let mut params = HashMap::new();

        // TID 0: image input [1, 3, 224, 224]
        ti.insert(
            0u32,
            TensorInfo::new(DType::F32, shape_from_concrete(&[1, 3, 224, 224])),
        );

        // TID 1: Conv2d weight [768, 3, 16, 16]
        ti.insert(
            1u32,
            TensorInfo::new(DType::F32, shape_from_concrete(&[768, 3, 16, 16])),
        );
        params.insert(
            1u32,
            AiParam::Inline {
                data: vec![0u8; 4].into(), // dummy
                info: TensorInfo::new(DType::F32, shape_from_concrete(&[768, 3, 16, 16])),
            },
        );

        // TID 2: Conv2d output [1, 768, 14, 14]
        ti.insert(
            2u32,
            TensorInfo::new(DType::F32, shape_from_concrete(&[1, 768, 14, 14])),
        );

        // TID 3: Reshape output [1, 196, 768]
        ti.insert(
            3u32,
            TensorInfo::new(DType::F32, shape_from_concrete(&[1, 196, 768])),
        );

        // TID 4: position embedding [1, 196, 768] (constant param)
        ti.insert(
            4u32,
            TensorInfo::new(DType::F32, shape_from_concrete(&[1, 196, 768])),
        );
        params.insert(
            4u32,
            AiParam::Inline {
                data: vec![0u8; 4].into(), // dummy
                info: TensorInfo::new(DType::F32, shape_from_concrete(&[1, 196, 768])),
            },
        );

        // TID 5: Add output [1, 196, 768]
        ti.insert(
            5u32,
            TensorInfo::new(DType::F32, shape_from_concrete(&[1, 196, 768])),
        );

        let nodes = vec![
            AiNode::new(
                0,
                AiOp::Conv {
                    kernel_shape: vec![16, 16],
                    strides: vec![16, 16],
                    pads: vec![0, 0, 0, 0],
                    dilations: vec![1, 1],
                    group: 1,
                    auto_pad: String::new(),
                },
                vec![0, 1],
                vec![2],
            ),
            AiNode::new(1, AiOp::Reshape { allow_zero: false }, vec![2], vec![3]),
            AiNode::new(2, AiOp::Add, vec![3, 4], vec![5]),
        ];

        AiGraph {
            name: "vit_test".into(),
            nodes,
            inputs: vec![0],
            outputs: vec![5],
            input_names: vec!["pixel_values".into()],
            output_names: vec!["last_hidden_state".into()],
            params,
            tensor_info: ti,
            metadata: HashMap::new(),
            warnings: vec![],
            dim_vars: DimVarTable::default(),
            shape_constraints: ConstraintStore::default(),
            subgraphs: HashMap::new(),
            tensor_names: HashMap::new(),
            topo_cache: Default::default(),
        }
    }

    #[test]
    fn patch_prune_rewrites_vit_graph() {
        let graph = make_vit_graph();
        let pass = PatchPruneInjection { budget_ratio: 0.5 };
        let result = pass.run(graph).expect("pass should succeed");

        // max_kept = ceil(196 * 0.5) = 98
        let max_kept = 98u64;

        // Should have kept_indices as a new graph input.
        assert!(
            result.input_names.contains(&"kept_indices".into()),
            "kept_indices must be a graph input"
        );

        // Should have 5 nodes: Conv2d, Reshape, Gather(patches), Gather(pos_embed), Add.
        assert_eq!(
            result.nodes.len(),
            5,
            "expected 5 nodes (Conv2d + Reshape + 2 Gathers + Add), got {}",
            result.nodes.len()
        );

        // The two Gather nodes should have axis=1.
        let gathers: Vec<_> = result
            .nodes
            .iter()
            .filter(|n| matches!(n.op, AiOp::Gather { axis: 1 }))
            .collect();
        assert_eq!(gathers.len(), 2, "expected 2 Gather(axis=1) nodes");

        // The Add output should have shape [1, max_kept, 768].
        let add_node = result
            .nodes
            .iter()
            .find(|n| matches!(n.op, AiOp::Add))
            .expect("Add node");
        let add_out_info = result
            .tensor_info
            .get(&add_node.outputs[0])
            .expect("Add output info");
        let add_shape: Vec<u64> = add_out_info
            .shape
            .iter()
            .filter_map(|d| d.as_concrete())
            .collect();
        assert_eq!(add_shape, vec![1, max_kept, 768]);

        // Metadata should record the budget.
        assert!(result.metadata.contains_key("patch_prune_budget"));

        // Validation should pass.
        let errs = result.validate();
        assert!(errs.is_empty(), "validation errors: {errs:?}");
    }

    #[test]
    fn patch_prune_noop_at_ratio_one() {
        let graph = make_vit_graph();
        let original_node_count = graph.nodes.len();
        let pass = PatchPruneInjection { budget_ratio: 1.0 };
        let result = pass.run(graph).expect("pass should succeed");
        assert_eq!(
            result.nodes.len(),
            original_node_count,
            "ratio=1.0 should be a no-op"
        );
    }

    #[test]
    fn patch_prune_skips_non_vit() {
        // A graph without a patch-embed Conv2d should be untouched.
        let mut ti = HashMap::new();
        ti.insert(
            0u32,
            TensorInfo::new(DType::F32, shape_from_concrete(&[1, 64])),
        );
        ti.insert(
            1u32,
            TensorInfo::new(DType::F32, shape_from_concrete(&[1, 64])),
        );

        let graph = AiGraph {
            name: "not_vit".into(),
            nodes: vec![AiNode::new(0, AiOp::Relu, vec![0], vec![1])],
            inputs: vec![0],
            outputs: vec![1],
            input_names: vec!["input".into()],
            output_names: vec!["output".into()],
            params: HashMap::new(),
            tensor_info: ti,
            metadata: HashMap::new(),
            warnings: vec![],
            dim_vars: DimVarTable::default(),
            shape_constraints: ConstraintStore::default(),
            subgraphs: HashMap::new(),
            tensor_names: HashMap::new(),
            topo_cache: Default::default(),
        };

        let pass = PatchPruneInjection { budget_ratio: 0.5 };
        let result = pass.run(graph).expect("pass should succeed");
        assert_eq!(result.nodes.len(), 1, "non-ViT graph should be unchanged");
    }

    #[test]
    fn patch_prune_minimum_one_patch() {
        let graph = make_vit_graph();
        // Extreme budget: 0.001 on 196 patches → max(ceil(0.196), 1) = 1
        let pass = PatchPruneInjection {
            budget_ratio: 0.001,
        };
        let result = pass.run(graph).expect("pass should succeed");

        let add_node = result
            .nodes
            .iter()
            .find(|n| matches!(n.op, AiOp::Add))
            .expect("Add node");
        let add_out_info = result.tensor_info.get(&add_node.outputs[0]).expect("info");
        let seq_dim = add_out_info.shape[1]
            .as_concrete()
            .expect("concrete seq dim");
        assert!(seq_dim >= 1, "must keep at least 1 patch");
    }
}
