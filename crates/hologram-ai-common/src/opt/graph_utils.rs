//! Shared graph-traversal and mutation helpers for optimization passes.

use crate::ir::{AiGraph, AiNode, TensorId};
use std::collections::{HashMap, HashSet};

/// Map each output tensor to the index of the node that produces it.
pub fn build_producer_map(graph: &AiGraph) -> HashMap<TensorId, usize> {
    graph
        .nodes
        .iter()
        .enumerate()
        .flat_map(|(i, n)| n.outputs.iter().map(move |&tid| (tid, i)))
        .collect()
}

/// Map each tensor to all `(consumer_node_index, input_position)` pairs.
pub fn build_consumer_map(graph: &AiGraph) -> HashMap<TensorId, Vec<(usize, usize)>> {
    let mut consumers: HashMap<TensorId, Vec<(usize, usize)>> = HashMap::new();
    for (i, n) in graph.nodes.iter().enumerate() {
        for (pos, &tid) in n.inputs.iter().enumerate() {
            consumers.entry(tid).or_default().push((i, pos));
        }
    }
    consumers
}

/// True when the tensor has exactly one consumer.
pub fn has_single_consumer(
    tid: TensorId,
    consumers: &HashMap<TensorId, Vec<(usize, usize)>>,
) -> bool {
    consumers.get(&tid).is_some_and(|c| c.len() == 1)
}

/// Next available node ID (one past the current max).
pub fn next_node_id(graph: &AiGraph) -> u32 {
    graph.nodes.iter().map(|n| n.id).max().unwrap_or(0) + 1
}

/// Apply removals and replacements to `graph.nodes`, then invalidate the
/// topological sort cache.
///
/// - Nodes at indices in `to_remove` are dropped.
/// - Nodes at indices in `replacements` are swapped with the replacement.
/// - All other nodes are kept as-is.
///
/// Returns the number of mutations applied (removals + replacements).
pub fn apply_node_mutations(
    graph: &mut AiGraph,
    to_remove: &HashSet<usize>,
    replacements: &mut HashMap<usize, AiNode>,
) -> usize {
    if to_remove.is_empty() && replacements.is_empty() {
        return 0;
    }
    let count = to_remove.len() + replacements.len();
    let new_nodes: Vec<AiNode> = graph
        .nodes
        .drain(..)
        .enumerate()
        .filter_map(|(idx, node)| {
            if let Some(replacement) = replacements.remove(&idx) {
                Some(replacement)
            } else if to_remove.contains(&idx) {
                None
            } else {
                Some(node)
            }
        })
        .collect();
    graph.nodes = new_nodes;
    graph.invalidate_topo_cache();
    count
}

/// Remove nodes at the given indices. Shorthand for `apply_node_mutations`
/// with no replacements.
pub fn remove_nodes(graph: &mut AiGraph, to_remove: &HashSet<usize>) -> usize {
    let mut empty = HashMap::new();
    apply_node_mutations(graph, to_remove, &mut empty)
}
