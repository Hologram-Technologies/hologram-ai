use super::{
    graph_utils::{build_producer_map, next_node_id},
    pipeline::Pass,
    shape_helpers::pool_output_dim,
};
use crate::ir::{
    shape_from_concrete, AiGraph, AiNode, AiOp, AiParam, DType, Dim, TensorId, TensorInfo,
};

/// Normalize padded ONNX Conv into explicit zero-Pad + pad-free Conv.
///
/// The pinned hologram runtime commit executes convolution as valid-only
/// im2col/GEMM and ignores `pad_h/pad_w` at the kernel boundary. Reifying the
/// ONNX pads as a first-class `Pad` op preserves semantics inside hologram-ai
/// without requiring downstream runtime changes.
pub struct ConvPadNormalization;
pub struct NonNegativeMaxPoolPadNormalization;

#[derive(Clone)]
struct MaxPoolSpec {
    kernel_shape: Vec<u64>,
    strides: Vec<u64>,
    pads: Vec<u64>,
    ceil_mode: bool,
}

struct RewriteState<'a> {
    graph: &'a mut AiGraph,
    rewritten: &'a mut Vec<AiNode>,
    next_tid: &'a mut TensorId,
    next_nid: &'a mut u32,
}

struct PadAxisRequest<'a> {
    input_tid: TensorId,
    input_shape: &'a [Dim],
    axis: usize,
    begin: u64,
    end: u64,
}

struct Axis0PadRequest<'a> {
    input_tid: TensorId,
    input_shape: &'a [Dim],
    begin: u64,
    end: u64,
}

struct TransposeInsert<'a> {
    input_tid: TensorId,
    output_shape: &'a [Dim],
    perm: &'a [u32],
    suffix: &'a str,
}

struct SpatialCropPlan<'a> {
    input_tid: TensorId,
    final_output_tid: TensorId,
    input_shape: &'a [Dim],
    final_output_shape: &'a [Dim],
    spatial_offset: usize,
    spatial_rank: usize,
}

struct PaddedMaxPoolGatherPlan<'a> {
    padded_input_tid: TensorId,
    padded_input_shape: &'a [Dim],
    output_shape: &'a [Dim],
    spatial_offset: usize,
    kernel_shape: &'a [u64],
    strides: &'a [u64],
    output_tid: TensorId,
}

struct GatherInsert {
    data_tid: TensorId,
    indices_tid: TensorId,
    axis: usize,
    output_shape: smallvec::SmallVec<[Dim; 4]>,
    output_name: String,
}

struct GatherOutputInsert<'a> {
    data_tid: TensorId,
    indices_tid: TensorId,
    output_tid: TensorId,
    axis: usize,
    output_shape: &'a [Dim],
    output_name: String,
}

impl Pass for ConvPadNormalization {
    fn name(&self) -> &str {
        "ConvPadNormalization"
    }

    fn should_run(&self, graph: &AiGraph) -> bool {
        graph.nodes.iter().any(|node| match &node.op {
            AiOp::Conv { pads, .. } => pads.iter().any(|&pad| pad != 0),
            _ => false,
        })
    }

    fn run(&self, mut graph: AiGraph) -> anyhow::Result<AiGraph> {
        let mut next_tid = next_tensor_id(&graph);
        let mut next_nid = next_node_id(&graph);
        let original = std::mem::take(&mut graph.nodes);
        let mut rewritten = Vec::with_capacity(original.len());

        for mut node in original {
            let Some(conv_pads) = conv_pads(&node.op) else {
                rewritten.push(node);
                continue;
            };
            if conv_pads.iter().all(|&pad| pad == 0) {
                rewritten.push(node);
                continue;
            }

            let input_tid = node.inputs[0];
            let mut current_tid = input_tid;
            let spatial_rank = conv_pads.len() / 2;
            let mut current_shape = graph
                .tensor_info
                .get(&input_tid)
                .map(|info| info.shape.clone())
                .unwrap_or_default();
            if current_shape.is_empty() {
                current_shape = std::iter::repeat_n(Dim::Dynamic, spatial_rank + 2).collect();
            }
            let spatial_offset = current_shape.len().saturating_sub(spatial_rank);
            for spatial_axis in 0..spatial_rank {
                let begin = conv_pads[spatial_axis];
                let end = conv_pads[spatial_rank + spatial_axis];
                if begin == 0 && end == 0 {
                    continue;
                }
                let axis = spatial_offset + spatial_axis;
                current_tid =
                    RewriteState::new(&mut graph, &mut rewritten, &mut next_tid, &mut next_nid)
                        .pad_axis_via_front(PadAxisRequest {
                            input_tid: current_tid,
                            input_shape: &current_shape,
                            axis,
                            begin,
                            end,
                        });
                current_shape = pad_shape_axis(&current_shape, axis, begin, end);
            }

            node.inputs[0] = current_tid;
            if let AiOp::Conv { pads, .. } = &mut node.op {
                pads.fill(0);
            }
            rewritten.push(node);
        }

        graph.nodes = rewritten;
        Ok(graph)
    }
}

impl Pass for NonNegativeMaxPoolPadNormalization {
    fn name(&self) -> &str {
        "NonNegativeMaxPoolPadNormalization"
    }

    fn should_run(&self, graph: &AiGraph) -> bool {
        graph
            .nodes
            .iter()
            .any(|node| matches!(node.op, AiOp::MaxPool { .. }))
    }

    fn run(&self, mut graph: AiGraph) -> anyhow::Result<AiGraph> {
        let source_nodes = graph.nodes.clone();
        let producers = build_producer_map(&graph);
        let mut next_tid = next_tensor_id(&graph);
        let mut next_nid = next_node_id(&graph);
        let original = std::mem::take(&mut graph.nodes);
        let mut rewritten = Vec::with_capacity(original.len());

        for mut node in original {
            let Some(spec) = max_pool_spec(&node.op) else {
                rewritten.push(node);
                continue;
            };
            let input_tid = node.inputs[0];
            if !definitely_non_negative(&source_nodes, &producers, input_tid) {
                rewritten.push(node);
                continue;
            }

            let mut current_tid = input_tid;
            let spatial_rank = spec.pads.len() / 2;
            let mut current_shape = graph
                .tensor_info
                .get(&input_tid)
                .map(|info| info.shape.clone())
                .unwrap_or_default();
            if current_shape.is_empty() {
                current_shape = std::iter::repeat_n(Dim::Dynamic, spatial_rank + 2).collect();
            }
            let spatial_offset = current_shape.len().saturating_sub(spatial_rank);
            let output_shape = graph
                .tensor_info
                .get(&node.outputs[0])
                .map(|info| info.shape.clone())
                .unwrap_or_default();
            let has_explicit_pad = spec.pads.iter().any(|&pad| pad != 0);
            let mut changed = false;
            for spatial_axis in 0..spatial_rank {
                let begin = spec.pads[spatial_axis];
                let end = spec.pads[spatial_rank + spatial_axis];
                let axis = spatial_offset + spatial_axis;
                if begin != 0 || end != 0 {
                    current_tid =
                        RewriteState::new(&mut graph, &mut rewritten, &mut next_tid, &mut next_nid)
                            .pad_axis_via_front(PadAxisRequest {
                                input_tid: current_tid,
                                input_shape: &current_shape,
                                axis,
                                begin,
                                end,
                            });
                    current_shape = pad_shape_axis(&current_shape, axis, begin, end);
                    changed = true;
                }
            }

            if has_explicit_pad
                && dilations_are_unity(&node.op)
                && !spec.ceil_mode
                && RewriteState::new(&mut graph, &mut rewritten, &mut next_tid, &mut next_nid)
                    .rewrite_padded_max_pool_with_gathers(PaddedMaxPoolGatherPlan {
                        padded_input_tid: current_tid,
                        padded_input_shape: &current_shape,
                        output_shape: &output_shape,
                        spatial_offset,
                        kernel_shape: &spec.kernel_shape,
                        strides: &spec.strides,
                        output_tid: node.outputs[0],
                    })
            {
                continue;
            }

            for spatial_axis in 0..spatial_rank {
                let axis = spatial_offset + spatial_axis;
                let Some(current_dim) = current_shape.get(axis).and_then(|dim| dim.as_concrete())
                else {
                    continue;
                };
                let Some(output_dim) = output_shape.get(axis).and_then(|dim| dim.as_concrete())
                else {
                    continue;
                };
                let Some(&kernel) = spec.kernel_shape.get(spatial_axis) else {
                    continue;
                };
                let target_dim = output_dim.saturating_mul(kernel);
                if current_dim >= target_dim {
                    continue;
                }
                current_tid =
                    RewriteState::new(&mut graph, &mut rewritten, &mut next_tid, &mut next_nid)
                        .pad_axis_via_front(PadAxisRequest {
                            input_tid: current_tid,
                            input_shape: &current_shape,
                            axis,
                            begin: 0,
                            end: target_dim - current_dim,
                        });
                current_shape = pad_shape_axis(&current_shape, axis, 0, target_dim - current_dim);
                changed = true;
            }

            if !changed {
                rewritten.push(node);
                continue;
            }

            if let Some(raw_shape) = max_pool_output_shape(
                &current_shape,
                spatial_offset,
                &spec.kernel_shape,
                &spec.strides,
                spec.ceil_mode,
            ) {
                if raw_shape != output_shape {
                    let original_output_tid = node.outputs[0];
                    let raw_output_tid = next_tid;
                    next_tid += 1;
                    if let Some(mut info) = graph.tensor_info.get(&original_output_tid).cloned() {
                        info.shape = raw_shape.clone();
                        graph.tensor_info.insert(raw_output_tid, info);
                    }
                    graph
                        .tensor_names
                        .insert(raw_output_tid, format!("tensor_{original_output_tid}.raw"));
                    node.inputs[0] = current_tid;
                    if let AiOp::MaxPool { pads, .. } = &mut node.op {
                        pads.fill(0);
                    }
                    node.outputs[0] = raw_output_tid;
                    rewritten.push(node.clone());
                    RewriteState::new(&mut graph, &mut rewritten, &mut next_tid, &mut next_nid)
                        .insert_spatial_crop_slices(SpatialCropPlan {
                            input_tid: raw_output_tid,
                            final_output_tid: original_output_tid,
                            input_shape: &raw_shape,
                            final_output_shape: &output_shape,
                            spatial_offset,
                            spatial_rank,
                        });
                    continue;
                }
            }

            if let Some(info) = graph.tensor_info.get_mut(&node.outputs[0]) {
                if !output_shape.is_empty() {
                    info.shape = output_shape.clone();
                }
            }
            node.inputs[0] = current_tid;
            if let AiOp::MaxPool { pads, .. } = &mut node.op {
                pads.fill(0);
            }
            rewritten.push(node);
        }

        graph.nodes = rewritten;
        Ok(graph)
    }
}

fn next_tensor_id(graph: &AiGraph) -> TensorId {
    let mut next_tid = graph
        .nodes
        .iter()
        .flat_map(|node| node.inputs.iter().chain(node.outputs.iter()))
        .copied()
        .max()
        .unwrap_or(0)
        + 1;
    if let Some(&max_param) = graph.params.keys().max() {
        next_tid = next_tid.max(max_param + 1);
    }
    if let Some(&max_input) = graph.inputs.iter().max() {
        next_tid = next_tid.max(max_input + 1);
    }
    if let Some(&max_output) = graph.outputs.iter().max() {
        next_tid = next_tid.max(max_output + 1);
    }
    next_tid
}

fn conv_pads(op: &AiOp) -> Option<Vec<u64>> {
    match op {
        AiOp::Conv { pads, .. } => Some(pads.clone()),
        _ => None,
    }
}

fn max_pool_spec(op: &AiOp) -> Option<MaxPoolSpec> {
    match op {
        AiOp::MaxPool {
            kernel_shape,
            strides,
            pads,
            ceil_mode,
            ..
        } => Some(MaxPoolSpec {
            kernel_shape: kernel_shape.clone(),
            strides: strides.clone(),
            pads: pads.clone(),
            ceil_mode: *ceil_mode,
        }),
        _ => None,
    }
}

fn dilations_are_unity(op: &AiOp) -> bool {
    match op {
        AiOp::MaxPool { dilations, .. } => dilations.iter().all(|&d| d == 1),
        _ => false,
    }
}

fn max_pool_output_shape(
    input_shape: &[Dim],
    spatial_offset: usize,
    kernel_shape: &[u64],
    strides: &[u64],
    ceil_mode: bool,
) -> Option<smallvec::SmallVec<[Dim; 4]>> {
    let mut shape: smallvec::SmallVec<[Dim; 4]> = input_shape.iter().cloned().collect();
    for spatial_axis in 0..kernel_shape.len() {
        let axis = spatial_offset + spatial_axis;
        let in_dim = shape.get(axis)?.as_concrete()?;
        let kernel = *kernel_shape.get(spatial_axis)?;
        let stride = strides.get(spatial_axis).copied().unwrap_or(1).max(1);
        shape[axis] = Dim::Concrete(pool_output_dim(in_dim, kernel, stride, 0, 0, ceil_mode));
    }
    Some(shape)
}

fn definitely_non_negative(
    nodes: &[AiNode],
    producers: &std::collections::HashMap<TensorId, usize>,
    tid: TensorId,
) -> bool {
    let Some(&producer_idx) = producers.get(&tid) else {
        return false;
    };
    matches!(
        nodes[producer_idx].op,
        AiOp::Relu | AiOp::Sigmoid | AiOp::Softmax { .. } | AiOp::Abs | AiOp::Sqrt | AiOp::Exp
    )
}

impl<'a> RewriteState<'a> {
    fn new(
        graph: &'a mut AiGraph,
        rewritten: &'a mut Vec<AiNode>,
        next_tid: &'a mut TensorId,
        next_nid: &'a mut u32,
    ) -> Self {
        Self {
            graph,
            rewritten,
            next_tid,
            next_nid,
        }
    }

    fn pad_axis_via_front(&mut self, req: PadAxisRequest<'_>) -> TensorId {
        if req.axis == 0 {
            return self.insert_axis0_pad(Axis0PadRequest {
                input_tid: req.input_tid,
                input_shape: req.input_shape,
                begin: req.begin,
                end: req.end,
            });
        }

        let perm = move_axis_to_front(req.input_shape.len(), req.axis);
        let inverse = inverse_perm(&perm);
        let transposed_shape = permute_shape(req.input_shape, &perm);
        let transposed_tid = self.insert_transpose(TransposeInsert {
            input_tid: req.input_tid,
            output_shape: &transposed_shape,
            perm: &perm,
            suffix: "pad_front",
        });
        let padded_front_tid = self.insert_axis0_pad(Axis0PadRequest {
            input_tid: transposed_tid,
            input_shape: &transposed_shape,
            begin: req.begin,
            end: req.end,
        });
        let restored_shape = permute_shape(
            &pad_shape_axis(&transposed_shape, 0, req.begin, req.end),
            &inverse,
        );
        self.insert_transpose(TransposeInsert {
            input_tid: padded_front_tid,
            output_shape: &restored_shape,
            perm: &inverse,
            suffix: "pad_restore",
        })
    }

    fn insert_axis0_pad(&mut self, req: Axis0PadRequest<'_>) -> TensorId {
        let rank = req.input_shape.len();
        let pad_values = axis0_pad_values(rank, req.begin, req.end);
        let pad_tid = *self.next_tid;
        *self.next_tid += 1;
        let output_tid = *self.next_tid;
        *self.next_tid += 1;

        let mut pad_info = TensorInfo::new(
            DType::INT64,
            shape_from_concrete(&[pad_values.len() as u64]),
        );
        pad_info.known_i64_values = Some(pad_values.iter().copied().map(Some).collect());
        let pad_bytes: Vec<u8> = pad_values.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.graph
            .params
            .insert(pad_tid, AiParam::inline(pad_bytes, pad_info.clone()));
        self.graph.tensor_info.insert(pad_tid, pad_info);
        self.graph
            .tensor_names
            .insert(pad_tid, format!("axis0_pad_{pad_tid}"));

        if let Some(mut input_info) = self.graph.tensor_info.get(&req.input_tid).cloned() {
            input_info.shape = pad_shape_axis(&input_info.shape, 0, req.begin, req.end);
            input_info.known_i64_values = None;
            self.graph.tensor_info.insert(output_tid, input_info);
        }
        self.graph
            .tensor_names
            .insert(output_tid, format!("tensor_{}_axis0_padded", req.input_tid));

        self.rewritten.push(AiNode::new(
            *self.next_nid,
            AiOp::Pad {
                mode: "constant".into(),
            },
            vec![req.input_tid, pad_tid],
            vec![output_tid],
        ));
        *self.next_nid += 1;
        output_tid
    }

    fn insert_transpose(&mut self, req: TransposeInsert<'_>) -> TensorId {
        let output_tid = *self.next_tid;
        *self.next_tid += 1;
        if let Some(mut info) = self.graph.tensor_info.get(&req.input_tid).cloned() {
            info.shape = req.output_shape.iter().cloned().collect();
            info.known_i64_values = None;
            self.graph.tensor_info.insert(output_tid, info);
        }
        let input_name = self
            .graph
            .tensor_names
            .get(&req.input_tid)
            .cloned()
            .unwrap_or_else(|| format!("tensor_{}", req.input_tid));
        self.graph
            .tensor_names
            .insert(output_tid, format!("{input_name}.{}", req.suffix));
        self.rewritten.push(AiNode::new(
            *self.next_nid,
            AiOp::Transpose {
                perm: req.perm.to_vec(),
            },
            vec![req.input_tid],
            vec![output_tid],
        ));
        *self.next_nid += 1;
        output_tid
    }

    fn insert_spatial_crop_slices(&mut self, plan: SpatialCropPlan<'_>) {
        let mut current_tid = plan.input_tid;
        let mut current_shape: smallvec::SmallVec<[Dim; 4]> =
            plan.input_shape.iter().cloned().collect();

        for spatial_axis in 0..plan.spatial_rank {
            let axis = plan.spatial_offset + spatial_axis;
            let end = plan
                .final_output_shape
                .get(axis)
                .and_then(|dim| dim.as_concrete())
                .unwrap_or(0) as i64;
            let is_last = spatial_axis + 1 == plan.spatial_rank;
            let output_tid = if is_last {
                plan.final_output_tid
            } else {
                *self.next_tid
            };
            if !is_last {
                *self.next_tid += 1;
            }

            let mut output_shape = current_shape.clone();
            if let Some(dim) = plan.final_output_shape.get(axis) {
                output_shape[axis] = dim.clone();
            }

            if let Some(mut info) = self.graph.tensor_info.get(&current_tid).cloned() {
                info.shape = output_shape.clone();
                info.known_i64_values = None;
                self.graph.tensor_info.insert(output_tid, info);
            }
            let output_name = if is_last {
                self.graph
                    .tensor_names
                    .get(&plan.final_output_tid)
                    .cloned()
                    .unwrap_or_else(|| format!("tensor_{}", plan.final_output_tid))
            } else {
                format!("tensor_{}.crop_axis_{axis}", plan.input_tid)
            };
            self.graph.tensor_names.insert(output_tid, output_name);

            self.rewritten.push(AiNode::new(
                *self.next_nid,
                AiOp::Slice {
                    axes: vec![axis as i64],
                    starts: vec![0],
                    ends: vec![end],
                    steps: vec![1],
                },
                vec![current_tid],
                vec![output_tid],
            ));
            *self.next_nid += 1;
            current_tid = output_tid;
            current_shape = output_shape;
        }
    }

    fn rewrite_padded_max_pool_with_gathers(&mut self, plan: PaddedMaxPoolGatherPlan<'_>) -> bool {
        if plan.kernel_shape.len() != 2 || plan.strides.len() != 2 {
            return false;
        }
        let Some(height_out) = plan
            .output_shape
            .get(plan.spatial_offset)
            .and_then(|dim| dim.as_concrete())
        else {
            return false;
        };
        let Some(width_out) = plan
            .output_shape
            .get(plan.spatial_offset + 1)
            .and_then(|dim| dim.as_concrete())
        else {
            return false;
        };

        let mut window_tids =
            Vec::with_capacity((plan.kernel_shape[0] * plan.kernel_shape[1]) as usize);
        for kh in 0..plan.kernel_shape[0] {
            let h_indices: Vec<i64> = (0..height_out)
                .map(|oh| (kh + oh * plan.strides[0]) as i64)
                .collect();
            let h_indices_tid = insert_i64_param(
                self.graph,
                self.next_tid,
                &format!("maxpool_h_indices_{}_{}", plan.output_tid, kh),
                &h_indices,
            );
            let h_gather_tid = self.insert_gather(GatherInsert {
                data_tid: plan.padded_input_tid,
                indices_tid: h_indices_tid,
                axis: plan.spatial_offset,
                output_shape: shape_with_axis(
                    plan.padded_input_shape,
                    plan.spatial_offset,
                    Dim::Concrete(height_out),
                ),
                output_name: format!("tensor_{}.gather_h_{kh}", plan.output_tid),
            });

            for kw in 0..plan.kernel_shape[1] {
                let w_indices: Vec<i64> = (0..width_out)
                    .map(|ow| (kw + ow * plan.strides[1]) as i64)
                    .collect();
                let w_indices_tid = insert_i64_param(
                    self.graph,
                    self.next_tid,
                    &format!("maxpool_w_indices_{}_{}_{}", plan.output_tid, kh, kw),
                    &w_indices,
                );
                let window_tid = *self.next_tid;
                *self.next_tid += 1;
                let window_shape = shape_with_axis(
                    &shape_with_axis(
                        plan.padded_input_shape,
                        plan.spatial_offset,
                        Dim::Concrete(height_out),
                    ),
                    plan.spatial_offset + 1,
                    Dim::Concrete(width_out),
                );
                self.insert_gather_with_output(GatherOutputInsert {
                    data_tid: h_gather_tid,
                    indices_tid: w_indices_tid,
                    output_tid: window_tid,
                    axis: plan.spatial_offset + 1,
                    output_shape: &window_shape,
                    output_name: format!("tensor_{}.window_{kh}_{kw}", plan.output_tid),
                });
                window_tids.push(window_tid);
            }
        }

        let mut accum_tid = match window_tids.first().copied() {
            Some(tid) => tid,
            None => return false,
        };
        let max_shape = plan
            .output_shape
            .iter()
            .cloned()
            .collect::<smallvec::SmallVec<[Dim; 4]>>();
        if window_tids.len() == 1 {
            self.rewritten.push(AiNode::new(
                *self.next_nid,
                AiOp::Identity,
                vec![accum_tid],
                vec![plan.output_tid],
            ));
            *self.next_nid += 1;
            return true;
        }
        for (idx, window_tid) in window_tids.into_iter().enumerate().skip(1) {
            let is_last = idx + 1 == (plan.kernel_shape[0] * plan.kernel_shape[1]) as usize;
            let max_tid = if is_last {
                plan.output_tid
            } else {
                *self.next_tid
            };
            if !is_last {
                *self.next_tid += 1;
            }
            if max_tid != plan.output_tid {
                if let Some(mut info) = self.graph.tensor_info.get(&accum_tid).cloned() {
                    info.shape = max_shape.clone();
                    info.known_i64_values = None;
                    self.graph.tensor_info.insert(max_tid, info);
                }
                self.graph
                    .tensor_names
                    .insert(max_tid, format!("tensor_{}.max_{idx}", plan.output_tid));
            }
            self.rewritten.push(AiNode::new(
                *self.next_nid,
                AiOp::Max,
                vec![accum_tid, window_tid],
                vec![max_tid],
            ));
            *self.next_nid += 1;
            accum_tid = max_tid;
        }

        true
    }

    fn insert_gather(&mut self, req: GatherInsert) -> TensorId {
        let output_tid = *self.next_tid;
        *self.next_tid += 1;
        self.insert_gather_with_output(GatherOutputInsert {
            data_tid: req.data_tid,
            indices_tid: req.indices_tid,
            output_tid,
            axis: req.axis,
            output_shape: &req.output_shape,
            output_name: req.output_name,
        });
        output_tid
    }

    fn insert_gather_with_output(&mut self, req: GatherOutputInsert<'_>) {
        if let Some(mut info) = self.graph.tensor_info.get(&req.data_tid).cloned() {
            info.shape = req.output_shape.iter().cloned().collect();
            info.known_i64_values = None;
            self.graph.tensor_info.insert(req.output_tid, info);
        }
        self.graph
            .tensor_names
            .insert(req.output_tid, req.output_name);
        self.rewritten.push(AiNode::new(
            *self.next_nid,
            AiOp::Gather {
                axis: req.axis as i64,
            },
            vec![req.data_tid, req.indices_tid],
            vec![req.output_tid],
        ));
        *self.next_nid += 1;
    }
}

fn insert_i64_param(
    graph: &mut AiGraph,
    next_tid: &mut TensorId,
    name: &str,
    values: &[i64],
) -> TensorId {
    let tid = *next_tid;
    *next_tid += 1;
    let mut info = TensorInfo::new(DType::INT64, shape_from_concrete(&[values.len() as u64]));
    info.known_i64_values = Some(values.iter().copied().map(Some).collect());
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    graph
        .params
        .insert(tid, AiParam::inline(bytes, info.clone()));
    graph.tensor_info.insert(tid, info);
    graph.tensor_names.insert(tid, name.into());
    tid
}

fn shape_with_axis(
    input_shape: &[Dim],
    axis: usize,
    replacement: Dim,
) -> smallvec::SmallVec<[Dim; 4]> {
    input_shape
        .iter()
        .enumerate()
        .map(|(idx, dim)| {
            if idx == axis {
                replacement.clone()
            } else {
                dim.clone()
            }
        })
        .collect()
}

fn axis0_pad_values(rank: usize, begin: u64, end: u64) -> Vec<i64> {
    let mut out = vec![0i64; rank * 2];
    out[0] = begin as i64;
    out[rank] = end as i64;
    out
}

fn move_axis_to_front(rank: usize, axis: usize) -> Vec<u32> {
    let mut perm = Vec::with_capacity(rank);
    perm.push(axis as u32);
    for idx in 0..rank {
        if idx != axis {
            perm.push(idx as u32);
        }
    }
    perm
}

fn inverse_perm(perm: &[u32]) -> Vec<u32> {
    let mut inverse = vec![0u32; perm.len()];
    for (idx, &axis) in perm.iter().enumerate() {
        inverse[axis as usize] = idx as u32;
    }
    inverse
}

fn permute_shape(input_shape: &[Dim], perm: &[u32]) -> smallvec::SmallVec<[Dim; 4]> {
    perm.iter()
        .map(|&axis| input_shape[axis as usize].clone())
        .collect()
}

fn pad_shape_axis(
    input_shape: &[Dim],
    target_axis: usize,
    begin: u64,
    end: u64,
) -> smallvec::SmallVec<[Dim; 4]> {
    input_shape
        .iter()
        .enumerate()
        .map(|(axis, dim)| {
            if axis != target_axis {
                return dim.clone();
            }
            match dim.as_concrete() {
                Some(size) => Dim::Concrete(size + begin + end),
                None => Dim::Dynamic,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{ConvPadNormalization, NonNegativeMaxPoolPadNormalization};
    use crate::ir::{
        shape_from_concrete, AiGraph, AiNode, AiOp, ConstraintStore, DType, DimVarTable, TensorInfo,
    };
    use crate::Pass;
    use std::collections::HashMap;

    #[test]
    fn rewrites_padded_conv_to_explicit_pad() {
        let mut graph = AiGraph {
            name: "conv_pad_norm".into(),
            nodes: Vec::new(),
            inputs: vec![0, 1],
            outputs: vec![2],
            input_names: Vec::new(),
            output_names: Vec::new(),
            params: HashMap::new(),
            tensor_info: HashMap::new(),
            metadata: HashMap::new(),
            warnings: Vec::new(),
            dim_vars: DimVarTable::default(),
            shape_constraints: ConstraintStore::default(),
            subgraphs: HashMap::new(),
            tensor_names: HashMap::new(),
            topo_cache: Default::default(),
        };
        graph.tensor_info.insert(
            0,
            TensorInfo::new(DType::F32, shape_from_concrete(&[1, 3, 7, 7])),
        );
        graph.tensor_info.insert(
            1,
            TensorInfo::new(DType::F32, shape_from_concrete(&[4, 3, 3, 3])),
        );
        graph.tensor_info.insert(
            2,
            TensorInfo::new(DType::F32, shape_from_concrete(&[1, 4, 4, 4])),
        );
        graph.nodes = vec![AiNode::new(
            0,
            AiOp::Conv {
                kernel_shape: vec![3, 3],
                strides: vec![2, 2],
                pads: vec![1, 1, 1, 1],
                dilations: vec![1, 1],
                group: 1,
                auto_pad: String::new(),
            },
            vec![0, 1],
            vec![2],
        )];

        let out = ConvPadNormalization.run(graph).expect("pass succeeds");
        assert_eq!(out.nodes.len(), 7);
        assert!(out
            .nodes
            .iter()
            .any(|node| matches!(node.op, AiOp::Pad { .. })));
        assert!(out
            .nodes
            .iter()
            .any(|node| matches!(node.op, AiOp::Transpose { .. })));
        match &out.nodes[6].op {
            AiOp::Conv { pads, .. } => assert_eq!(pads, &vec![0, 0, 0, 0]),
            other => panic!("expected Conv, got {other:?}"),
        }
        assert_eq!(out.nodes[6].inputs.len(), 2);
        assert_ne!(
            out.nodes[6].inputs[0], 0,
            "conv input should be rewritten to padded tensor"
        );
    }

    #[test]
    fn rewrites_relu_maxpool_to_explicit_pad_and_kernel_fixup() {
        let mut graph = AiGraph {
            name: "max_pool_norm".into(),
            nodes: Vec::new(),
            inputs: vec![0],
            outputs: vec![2],
            input_names: Vec::new(),
            output_names: Vec::new(),
            params: HashMap::new(),
            tensor_info: HashMap::new(),
            metadata: HashMap::new(),
            warnings: Vec::new(),
            dim_vars: DimVarTable::default(),
            shape_constraints: ConstraintStore::default(),
            subgraphs: HashMap::new(),
            tensor_names: HashMap::new(),
            topo_cache: Default::default(),
        };
        graph.tensor_info.insert(
            0,
            TensorInfo::new(DType::F32, shape_from_concrete(&[1, 2, 5, 5])),
        );
        graph.tensor_info.insert(
            1,
            TensorInfo::new(DType::F32, shape_from_concrete(&[1, 2, 5, 5])),
        );
        graph.tensor_info.insert(
            2,
            TensorInfo::new(DType::F32, shape_from_concrete(&[1, 2, 2, 2])),
        );
        graph.nodes = vec![
            AiNode::new(0, AiOp::Relu, vec![0], vec![1]),
            AiNode::new(
                1,
                AiOp::MaxPool {
                    kernel_shape: vec![3, 3],
                    strides: vec![2, 2],
                    pads: vec![0, 0, 0, 0],
                    dilations: vec![1, 1],
                    auto_pad: String::new(),
                    ceil_mode: false,
                },
                vec![1],
                vec![2],
            ),
        ];

        let out = NonNegativeMaxPoolPadNormalization
            .run(graph)
            .expect("pass succeeds");
        assert!(out.nodes.len() > 2, "expected helper nodes before MaxPool");
        match &out.nodes.last().expect("pool remains").op {
            AiOp::MaxPool { pads, .. } => assert_eq!(pads, &vec![0, 0, 0, 0]),
            other => panic!("expected MaxPool, got {other:?}"),
        }
        assert_ne!(
            out.nodes.last().expect("pool remains").inputs[0],
            1,
            "pool input should be rewritten to padded tensor"
        );
    }
}
