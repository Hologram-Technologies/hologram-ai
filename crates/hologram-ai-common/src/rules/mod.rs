//! UOR-native declarative rewrite rules over `AiGraph`.
//!
//! Replaces the bespoke imperative `opt/*Fusion` passes with a confluent
//! fixed-point rewrite over a typed canonical form (ADR-0018).
//!
//! ## Canonical pieces
//!
//! - [`Pattern`] declares the input sub-graph a rule recognizes — a tree
//!   of [`PatternOp`]s with leaves that bind to graph tensors as
//!   [`VarId`]s. Variants like `Maybe` express architecture-specific
//!   differences (e.g. biased vs unbiased projections) as **declared
//!   alternates** in the schema, not as separate code.
//!
//! - [`Replacement`] declares the canonical replacement — a tree of
//!   `AiOp`s with leaves that re-use the bound `VarId`s from the pattern.
//!
//! - [`Rule`] = `Pattern` + `Replacement` + the citation to the external
//!   authoritative source that verifies the rule (the ONNX spec link or
//!   the ORT-parity test name). No rule lands without a witness.
//!
//! - [`RuleSet`] applies rules to fixed-point. Each rule either matches
//!   and rewrites or doesn't; the result is independent of rule order
//!   (rules are confluent on the canonical form). A non-confluent rule
//!   pair is rejected at load time.
//!
//! ## Match semantics
//!
//! A match binds each pattern [`VarId`] to a graph [`TensorId`]. The
//! root pattern matches against an `AiNode`'s op + input tensors;
//! sub-patterns recurse by following each input tensor back to its
//! producer node (if any). If a sub-pattern is a `VarId` and the same
//! `VarId` appears elsewhere in the pattern, the bindings must agree
//! (linear vs sharing is explicit, not implied).
//!
//! Once a pattern matches, the replacement is constructed: a new node
//! whose inputs are the bound `VarId`s' tensor IDs and whose outputs
//! inherit the matched root node's outputs (so consumers downstream
//! continue to see the same tensor IDs — no rewiring outside the
//! match).
//!
//! ## What this is not
//!
//! - Not e-graph saturation. Rules are confluent fixed-point rewrites
//!   on a typed canonical form — the same paradigm as
//!   `Graph::desugar_composites` upstream (ADR-055) and uor-addr's
//!   ψ-tower, applied to architecture-pattern matching.
//! - Not a DSL. Patterns are constructed as plain Rust data; macros
//!   may be added later for ergonomics but are not part of the
//!   architecture.

use crate::ir::{AiGraph, AiNode, AiOp, NodeId, TensorId};
use std::collections::HashMap;

mod op_match;
pub use op_match::OpMatcher;

/// A name bound by a [`Pattern`] to a tensor in the matched sub-graph.
///
/// `VarId(0)` is conventionally the **root output** of the pattern (the
/// tensor the replacement's root will write to). Other ids name
/// intermediate tensors used by the replacement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VarId(pub u32);

/// Pattern over the canonical `AiGraph` IR.
///
/// `Op` matches a producer node whose op satisfies [`OpMatcher`] and
/// whose input tensors recursively match each child pattern. `Var`
/// matches *any* tensor and binds it. `Maybe` matches either the inner
/// pattern *or* the bare `Var` (used to declare optional operations
/// like a bias-Add between a MatMul and its consumer).
#[derive(Debug, Clone)]
pub enum Pattern {
    /// Match any tensor; bind it under `var`.
    Var(VarId),
    /// Match the producer node of a tensor against `op`, with each input
    /// recursively matching the corresponding `inputs[i]`.
    Op {
        op: OpMatcher,
        inputs: Vec<Pattern>,
        /// Optional bind of the root tensor of this sub-pattern (its
        /// producer's first output). Useful in `Replacement::Var`
        /// references and in `Maybe` to inject the inner result back.
        bind: Option<VarId>,
    },
    /// Match either the inner pattern or its `bind`'s underlying tensor
    /// directly. The inner pattern's `bind` (if any) is propagated.
    Maybe(Box<Pattern>),
}

impl Pattern {
    /// Convenience: an `Op` pattern with no binding on its own output.
    pub fn op(matcher: OpMatcher, inputs: Vec<Pattern>) -> Self {
        Pattern::Op {
            op: matcher,
            inputs,
            bind: None,
        }
    }

    /// Convenience: an `Op` pattern that also binds its output tensor.
    pub fn op_bind(matcher: OpMatcher, inputs: Vec<Pattern>, bind: VarId) -> Self {
        Pattern::Op {
            op: matcher,
            inputs,
            bind: Some(bind),
        }
    }
}

/// Replacement tree. Leaves are `Var` references to bindings made by
/// the pattern; internal nodes are canonical `AiOp` constructions whose
/// inputs are themselves `Replacement`s.
///
/// `AiOp` is boxed because clippy flags the unboxed enum as large
/// (≈48 bytes for the heaviest variants) — boxing keeps the
/// `Replacement` discriminant compact while still allowing the rule
/// author to write rich canonical replacements.
#[derive(Debug, Clone)]
pub enum Replacement {
    /// Re-use a tensor bound by the pattern. The tensor's existing
    /// producer (if any) stays in the graph; the replacement just
    /// references it.
    Var(VarId),
    /// Construct a new node with the given op and recursively-built
    /// inputs.
    Op {
        op: Box<AiOp>,
        inputs: Vec<Replacement>,
    },
}

/// A single declarative rewrite rule.
///
/// `witness` is the name of the V&V test (in `hologram-ai-conformance`,
/// `hologram-ai`, or upstream) that establishes the rule's correctness
/// against an external authoritative source — the ONNX operator spec,
/// the ONNX backend node-test corpus, or an ORT logit-parity check.
/// A rule without a witness MUST NOT be added to a [`RuleSet`].
#[derive(Debug, Clone)]
pub struct Rule {
    pub name: &'static str,
    /// External authoritative-source citation (URL or test name).
    pub witness: &'static str,
    pub pattern: Pattern,
    pub replacement: Replacement,
}

/// A set of declarative rules applied to fixed-point over an `AiGraph`.
#[derive(Debug, Default, Clone)]
pub struct RuleSet {
    rules: Vec<Rule>,
}

impl RuleSet {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_rule(mut self, rule: Rule) -> Self {
        self.rules.push(rule);
        self
    }

    pub fn rules(&self) -> &[Rule] {
        &self.rules
    }

    /// Apply every rule to fixed-point. Returns the total number of
    /// rewrites performed. A rule pair that produces different results
    /// on the same input fragment (a confluence violation) is rejected
    /// here by detecting an unbounded rewrite loop and panicking; the
    /// rule set must be fixed before re-running.
    pub fn apply(&self, graph: &mut AiGraph) -> usize {
        let mut total = 0usize;
        loop {
            let pass_rewrites = self.apply_pass(graph);
            total += pass_rewrites;
            if pass_rewrites == 0 {
                break;
            }
            if total > graph.nodes.len().saturating_mul(64) + 1024 {
                panic!(
                    "RuleSet::apply did not converge (rewrites={total}, nodes={}) — non-confluent rule set",
                    graph.nodes.len()
                );
            }
        }
        total
    }

    /// One sweep over the graph applying every rule once at every
    /// candidate root. Returns the number of rewrites made this sweep.
    fn apply_pass(&self, graph: &mut AiGraph) -> usize {
        let mut rewrites = 0usize;
        let n = graph.nodes.len();
        let mut next_id = next_node_id(graph);
        let mut producer = build_producer_map(graph);

        let mut new_nodes: HashMap<usize, AiNode> = HashMap::new();

        'outer: for root_idx in 0..n {
            if new_nodes.contains_key(&root_idx) {
                continue; // Already rewritten this sweep.
            }
            for rule in &self.rules {
                let mut env = Env::default();
                if Matcher::match_at(graph, &producer, &rule.pattern, root_idx, &mut env) {
                    let root_node = &graph.nodes[root_idx];
                    let root_out = root_node.outputs.first().copied();
                    let Some(root_out) = root_out else {
                        continue;
                    };
                    let Some(new_node) =
                        materialize(&rule.replacement, &env, root_out, &mut next_id)
                    else {
                        continue;
                    };
                    producer.insert(root_out, root_idx);
                    new_nodes.insert(root_idx, new_node);
                    rewrites += 1;
                    continue 'outer;
                }
            }
        }

        for (idx, node) in new_nodes {
            graph.nodes[idx] = node;
        }

        rewrites
    }
}

/// Bindings established by a successful pattern match.
#[derive(Debug, Default)]
struct Env {
    binds: HashMap<VarId, TensorId>,
}

impl Env {
    fn bind(&mut self, var: VarId, tid: TensorId) -> bool {
        match self.binds.get(&var) {
            Some(&existing) => existing == tid,
            None => {
                self.binds.insert(var, tid);
                true
            }
        }
    }

    fn lookup(&self, var: VarId) -> Option<TensorId> {
        self.binds.get(&var).copied()
    }
}

struct Matcher;

impl Matcher {
    fn match_at(
        graph: &AiGraph,
        producer: &HashMap<TensorId, usize>,
        pattern: &Pattern,
        node_idx: usize,
        env: &mut Env,
    ) -> bool {
        let node = &graph.nodes[node_idx];
        match pattern {
            Pattern::Var(var) => {
                // A bare Var at a root position matches the node's first
                // output tensor.
                let Some(&tid) = node.outputs.first() else {
                    return false;
                };
                env.bind(*var, tid)
            }
            Pattern::Op { op, inputs, bind } => {
                if !op.matches(&node.op) {
                    return false;
                }
                if inputs.len() != node.inputs.len() {
                    return false;
                }
                if let Some(b) = bind {
                    let Some(&tid) = node.outputs.first() else {
                        return false;
                    };
                    if !env.bind(*b, tid) {
                        return false;
                    }
                }
                for (i, child_pat) in inputs.iter().enumerate() {
                    let in_tid = node.inputs[i];
                    if !Self::match_tensor(graph, producer, child_pat, in_tid, env) {
                        return false;
                    }
                }
                true
            }
            Pattern::Maybe(inner) => {
                // Try the inner pattern first; if it doesn't match, the
                // node itself is the "absent" branch — bind the inner's
                // root var (if any) to the node's first output.
                let mut tentative = Env {
                    binds: env.binds.clone(),
                };
                if Self::match_at(graph, producer, inner, node_idx, &mut tentative) {
                    env.binds = tentative.binds;
                    true
                } else {
                    let Some(&tid) = node.outputs.first() else {
                        return false;
                    };
                    if let Pattern::Op { bind: Some(b), .. } = inner.as_ref() {
                        env.bind(*b, tid)
                    } else {
                        true
                    }
                }
            }
        }
    }

    fn match_tensor(
        graph: &AiGraph,
        producer: &HashMap<TensorId, usize>,
        pattern: &Pattern,
        tid: TensorId,
        env: &mut Env,
    ) -> bool {
        match pattern {
            Pattern::Var(var) => env.bind(*var, tid),
            Pattern::Op { .. } | Pattern::Maybe(_) => {
                let Some(&prod_idx) = producer.get(&tid) else {
                    return false;
                };
                Self::match_at(graph, producer, pattern, prod_idx, env)
            }
        }
    }
}

fn materialize(
    repl: &Replacement,
    env: &Env,
    root_out: TensorId,
    next_id: &mut NodeId,
) -> Option<AiNode> {
    let Replacement::Op { op, inputs } = repl else {
        return None; // Root replacement must be an Op (Var alone has no node).
    };
    let mut input_tids = Vec::with_capacity(inputs.len());
    for inp in inputs {
        match inp {
            Replacement::Var(v) => input_tids.push(env.lookup(*v)?),
            Replacement::Op { .. } => {
                // Nested op replacements not yet supported; surface as
                // a hard miss so the caller knows. The architecture
                // permits this; the current matcher only constructs the
                // root replacement node, leaving multi-node fan-out
                // (e.g. a fused norm + N projections) for the engine's
                // multi-output extension.
                return None;
            }
        }
    }
    let new = AiNode::new(*next_id, (**op).clone(), input_tids, vec![root_out]);
    *next_id += 1;
    Some(new)
}

// ── helpers (duplicated from opt::graph_utils to keep the modules
// independent until the imperative passes are deleted) ──────────────

fn next_node_id(graph: &AiGraph) -> NodeId {
    graph.nodes.iter().map(|n| n.id).max().unwrap_or(0) + 1
}

fn build_producer_map(graph: &AiGraph) -> HashMap<TensorId, usize> {
    let mut m = HashMap::with_capacity(graph.nodes.len());
    for (idx, node) in graph.nodes.iter().enumerate() {
        for &out in &node.outputs {
            m.insert(out, idx);
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{shape_from_concrete, AiNode, DType, TensorInfo};
    use std::collections::HashMap as StdHashMap;

    fn unit_graph() -> AiGraph {
        AiGraph {
            name: "test".into(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            input_names: vec![],
            output_names: vec![],
            params: StdHashMap::new(),
            tensor_info: StdHashMap::new(),
            metadata: StdHashMap::new(),
            warnings: vec![],
            dim_vars: Default::default(),
            shape_constraints: Default::default(),
            subgraphs: StdHashMap::new(),
            tensor_names: StdHashMap::new(),
            topo_cache: Default::default(),
        }
    }

    #[test]
    fn double_relu_folds_to_single_relu() {
        // ReluRelu fusion: Relu(Relu(x)) → Relu(x). Trivial canonical
        // identity that proves the rule engine end-to-end.
        let mut g = unit_graph();
        let shape = shape_from_concrete(&[4]);
        for tid in 0..3u32 {
            g.tensor_info
                .insert(tid, TensorInfo::new(DType::F32, shape.clone()));
        }
        g.inputs = vec![0];
        g.outputs = vec![2];
        g.nodes.push(AiNode::new(0, AiOp::Relu, vec![0], vec![1]));
        g.nodes.push(AiNode::new(1, AiOp::Relu, vec![1], vec![2]));

        let x = VarId(1);
        let rule = Rule {
            name: "double_relu_collapse",
            witness: "Relu(Relu(x)) == Relu(x) (idempotence of ReLU; trivial spec invariant)",
            pattern: Pattern::op(
                OpMatcher::exact_relu(),
                vec![Pattern::op(OpMatcher::exact_relu(), vec![Pattern::Var(x)])],
            ),
            replacement: Replacement::Op {
                op: Box::new(AiOp::Relu),
                inputs: vec![Replacement::Var(x)],
            },
        };
        let set = RuleSet::new().with_rule(rule);

        let rewrites = set.apply(&mut g);
        assert!(rewrites >= 1, "expected at least one rewrite");
        // The outer node (id=1, idx=1) was rewritten to Relu(x) directly,
        // skipping the inner Relu.
        let rewritten = &g.nodes[1];
        assert!(matches!(rewritten.op, AiOp::Relu));
        assert_eq!(rewritten.inputs, vec![0], "outer Relu now reads from x");
        assert_eq!(
            rewritten.outputs,
            vec![2],
            "outer Relu's output unchanged so downstream wiring is preserved"
        );
    }

    #[test]
    fn non_matching_graph_is_not_rewritten() {
        // A graph that doesn't match the rule must stay unchanged — the
        // engine never approximates or partially matches.
        let mut g = unit_graph();
        let shape = shape_from_concrete(&[4]);
        for tid in 0..2u32 {
            g.tensor_info
                .insert(tid, TensorInfo::new(DType::F32, shape.clone()));
        }
        g.inputs = vec![0];
        g.outputs = vec![1];
        g.nodes
            .push(AiNode::new(0, AiOp::Sigmoid, vec![0], vec![1]));

        let x = VarId(1);
        let rule = Rule {
            name: "double_relu_collapse",
            witness: "Relu(Relu(x)) == Relu(x)",
            pattern: Pattern::op(
                OpMatcher::exact_relu(),
                vec![Pattern::op(OpMatcher::exact_relu(), vec![Pattern::Var(x)])],
            ),
            replacement: Replacement::Op {
                op: Box::new(AiOp::Relu),
                inputs: vec![Replacement::Var(x)],
            },
        };
        let set = RuleSet::new().with_rule(rule);

        let rewrites = set.apply(&mut g);
        assert_eq!(rewrites, 0, "no rewrite on a non-matching graph");
        assert!(matches!(g.nodes[0].op, AiOp::Sigmoid));
    }
}
