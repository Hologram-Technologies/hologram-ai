//! Convert Slice ops to Gather ops.
//!
//! The hologram runtime doesn't have a native Slice operation. Slice is
//! currently dispatched as Identity (pass-through), which is wrong for any
//! non-trivial slice. This pass converts Slice nodes to Gather nodes with
//! constant index tensors, which the runtime handles correctly.
//!
//! Only converts simple cases: single-axis slices with step=1 where the
//! slice dimension is concrete. Complex slices (multi-axis, non-unit steps)
//! are left as-is (the lowering will handle them or error).

use super::pipeline::Pass;
use crate::ir::{shape_from_concrete, AiGraph, AiOp, AiParam, DType, TensorInfo};
use hologram_ai_quant::QuantDescriptor;

/// Convert Slice nodes to Gather nodes with constant index tensors.
pub struct SliceToGather;

impl Pass for SliceToGather {
    fn name(&self) -> &str {
        "SliceToGather"
    }

    fn run(&self, mut graph: AiGraph) -> anyhow::Result<AiGraph> {
        let mut next_tid = graph
            .tensor_info
            .keys()
            .copied()
            .max()
            .unwrap_or(0)
            + 1;

        // Collect nodes to convert (can't mutate while iterating).
        let conversions: Vec<(usize, i64, Vec<i64>)> = graph
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(idx, node)| {
                if let AiOp::Slice {
                    axes,
                    starts,
                    ends,
                    steps,
                } = &node.op
                {
                    // Only convert single-axis slices with step=1.
                    if axes.len() != 1 || starts.len() != 1 || ends.len() != 1 {
                        return None;
                    }
                    let step = steps.first().copied().unwrap_or(1);
                    if step != 1 {
                        return None;
                    }

                    let axis = axes[0];
                    let start = starts[0];
                    let end = ends[0];

                    // Resolve the actual start/end against the input dim.
                    let data_tid = *node.inputs.first()?;
                    let info = graph.tensor_info.get(&data_tid)?;
                    let ndim = info.shape.len();
                    let norm_axis = if axis < 0 {
                        (ndim as i64 + axis).max(0) as usize
                    } else {
                        axis as usize
                    };
                    let dim_val = info.shape.get(norm_axis)?.as_concrete()? as i64;

                    let s = normalize_bound(start, dim_val);
                    let e = normalize_bound(end, dim_val);
                    if s >= e {
                        return None;
                    }

                    // If selecting ALL elements (start=0, end=dim), this is
                    // a no-op slice. Convert to Identity instead of Gather.
                    if s == 0 && e == dim_val {
                        return None;
                    }

                    // Only convert to Gather when hologram's row-based Gather
                    // can handle it: all dimensions BEFORE the axis must be 1
                    // (i.e., no batching). hologram's Gather treats data as
                    // [total/dim, dim] which doesn't support batched axis gathers.
                    let pre_axis_product: u64 = info.shape[..norm_axis]
                        .iter()
                        .filter_map(|d| d.as_concrete())
                        .product::<u64>()
                        .max(1);
                    if pre_axis_product != 1 {
                        tracing::debug!(axis, ?info.shape, pre_axis_product, "slice-to-gather: skipping non-trivial pre-axis");
                        return None;
                    }

                    let indices: Vec<i64> = (s..e).collect();
                    Some((idx, axis, indices))
                } else {
                    None
                }
            })
            .collect();

        // Apply conversions.
        for (idx, axis, indices) in conversions {
            let num_indices = indices.len();

            // Create constant index tensor.
            let indices_tid = next_tid;
            next_tid += 1;

            let index_bytes: Vec<u8> = indices.iter().flat_map(|&v| v.to_le_bytes()).collect();
            let index_shape = shape_from_concrete(&[num_indices as u64]);
            let index_info = TensorInfo {
                logical_dtype: DType::INT64,
                storage_dtype: DType::INT64,
                shape: index_shape,
                quant: QuantDescriptor::none(),
                known_i64_values: Some(indices.iter().map(|&v| Some(v)).collect()),
            };

            graph
                .tensor_info
                .insert(indices_tid, index_info.clone());
            graph
                .params
                .insert(indices_tid, AiParam::inline(index_bytes, index_info));

            // Convert Slice to Gather.
            let node = &mut graph.nodes[idx];
            node.op = AiOp::Gather { axis };
            // Gather inputs: (data, indices).
            let data_tid = node.inputs[0];
            node.inputs = vec![data_tid, indices_tid];

            // Update output shape: replace the sliced dim with num_indices.
            if let Some(&out_tid) = node.outputs.first() {
                if let Some(info) = graph.tensor_info.get_mut(&out_tid) {
                    let ndim = info.shape.len();
                    let norm_axis = if axis < 0 {
                        (ndim as i64 + axis).max(0) as usize
                    } else {
                        axis as usize
                    };
                    if norm_axis < info.shape.len() {
                        info.shape[norm_axis] =
                            crate::ir::shape::DimExpr::Concrete(num_indices as u64);
                    }
                }
            }
        }

        Ok(graph)
    }
}

fn normalize_bound(val: i64, dim_size: i64) -> i64 {
    let v = if val < 0 { dim_size + val } else { val };
    v.clamp(0, dim_size)
}
