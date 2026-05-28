//! UOR-native declarative rewrite rules over `AiGraph`.
//!
//! Replaces the bespoke imperative `opt/*Fusion` passes with a confluent
//! fixed-point rewrite over a typed canonical form (ADR-0018).
//!
//! ## Canonical pieces
//!
//! - [`Pattern`] declares the input sub-graph a rule recognizes — a tree
//!   of `Op`s with leaves that bind to graph tensors as [`VarId`]s.
//!   `Maybe` expresses architecture-specific differences (e.g. biased
//!   vs unbiased projections) as **declared alternates**, not as
//!   separate code.
//!
//! - [`Replacement`] declares the canonical replacement — a single
//!   `AiOp` node whose inputs are bound `VarId`s from the pattern. The
//!   replacement reuses the matched root node's first output tensor
//!   id, so downstream consumers see the same tensor id and require no
//!   rewiring.
//!
//! - [`Rule`] = `Pattern` + `Replacement` + the citation to the external
//!   authoritative source (ONNX spec, ORT logit parity, ONNX backend
//!   node-test corpus). No rule lands without a witness.
//!
//! - [`RuleSet`] applies rules to fixed-point. Each rule either matches
//!   and rewrites or doesn't; the result is independent of rule order
//!   (rules are confluent on the canonical form). A non-confluent rule
//!   pair is caught at apply time by non-convergence.
//!
//! ## Match semantics
//!
//! A match binds each pattern [`VarId`] to a graph [`TensorId`]. The
//! root pattern matches against an `AiNode`'s op + input tensors;
//! sub-patterns recurse by following each input tensor back to its
//! producer node. When a `VarId` appears more than once in the pattern,
//! the bindings must agree (sharing is explicit, not implied).
//!
//! Op patterns at *interior* positions (non-root) require their matched
//! node to have **exactly one consumer**. Removing a multi-consumer
//! interior node would break downstream paths, so the matcher refuses
//! to match it in the first place.
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

use crate::ir::{AiGraph, AiNode, AiOp, AiParam, NodeId, TensorId};
use std::collections::HashMap;

mod op_match;
pub mod pattern_rules;
pub use op_match::{AiOpDiscriminant, OpMatcher};

/// Adapter that wraps a [`RuleSet`] as an `opt::Pass`, so the rule
/// engine plugs into the existing `OptPipeline` while the imperative
/// passes are being replaced.
///
/// `should_run` always returns `true` — the matcher itself is the
/// cheap predicate; a rule that doesn't match does no graph mutation.
pub struct RulePass {
    pub name: &'static str,
    pub set: RuleSet,
}

impl RulePass {
    pub fn new(name: &'static str, set: RuleSet) -> Self {
        Self { name, set }
    }
}

impl crate::opt::Pass for RulePass {
    fn name(&self) -> &str {
        self.name
    }

    fn run(&self, mut graph: AiGraph) -> anyhow::Result<AiGraph> {
        let rewrites = self.set.apply(&mut graph);
        if rewrites > 0 {
            tracing::info!(
                pass = self.name,
                rewrites,
                "RulePass: applied {} declarative rewrite(s)",
                rewrites
            );
        }
        Ok(graph)
    }
}

/// A name bound by a [`Pattern`] to a tensor in the matched sub-graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VarId(pub u32);

/// Pattern over the canonical `AiGraph` IR.
#[derive(Clone)]
pub enum Pattern {
    /// Match any tensor; bind it under `var`.
    Var(VarId),
    /// Match a constant tensor (an `AiParam` in `graph.params`); bind
    /// it under `var`. Use [`MatchView::scalar_f32`] / [`MatchView::i64_vec`]
    /// to read the value in a `Replacement::from_match` builder.
    Const(VarId),
    /// Match the producer node of a tensor against `op`, with each input
    /// recursively matching the corresponding `inputs[i]`.
    Op {
        op: OpMatcher,
        inputs: Vec<Pattern>,
        /// Optional bind of the root tensor of this sub-pattern (its
        /// producer's first output).
        bind: Option<VarId>,
        /// If true, try both `[A, B]` and `[B, A]` orderings when
        /// matching the inputs (binary commutative ops only — must have
        /// `inputs.len() == 2`).
        commutative: bool,
        /// Optional predicate on the matched node's `AiOp` — used for
        /// attribute-level constraints (e.g. a `Transpose` whose
        /// `perm` swaps the last two axes, a `Concat` on a specific
        /// `axis`). A `fn` pointer keeps the predicate purely data
        /// (no captured state); the engine refuses to match if the
        /// predicate returns false.
        predicate: Option<fn(&AiOp) -> bool>,
    },
    /// Match either the inner pattern or its `bind`'s underlying tensor
    /// directly.
    Maybe(Box<Pattern>),
}

impl std::fmt::Debug for Pattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Pattern::Var(v) => f.debug_tuple("Var").field(v).finish(),
            Pattern::Const(v) => f.debug_tuple("Const").field(v).finish(),
            Pattern::Op {
                op,
                inputs,
                bind,
                commutative,
                predicate,
            } => f
                .debug_struct("Op")
                .field("op", op)
                .field("inputs", inputs)
                .field("bind", bind)
                .field("commutative", commutative)
                .field("predicate", &predicate.map(|_| "<fn>"))
                .finish(),
            Pattern::Maybe(inner) => f.debug_tuple("Maybe").field(inner).finish(),
        }
    }
}

impl Pattern {
    pub fn op(matcher: OpMatcher, inputs: Vec<Pattern>) -> Self {
        Pattern::Op {
            op: matcher,
            inputs,
            bind: None,
            commutative: false,
            predicate: None,
        }
    }

    pub fn op_bind(matcher: OpMatcher, inputs: Vec<Pattern>, bind: VarId) -> Self {
        Pattern::Op {
            op: matcher,
            inputs,
            bind: Some(bind),
            commutative: false,
            predicate: None,
        }
    }

    /// Commutative binary Op pattern. The matcher tries both input
    /// orderings (`[A, B]` and `[B, A]`); the first one whose sub-
    /// patterns all match wins.
    pub fn op_comm(matcher: OpMatcher, a: Pattern, b: Pattern) -> Self {
        Pattern::Op {
            op: matcher,
            inputs: vec![a, b],
            bind: None,
            commutative: true,
            predicate: None,
        }
    }

    /// Attach a predicate to this `Pattern::Op` — the matched node's
    /// `AiOp` must satisfy the predicate. No-op on `Var` and `Maybe`.
    pub fn with_predicate(self, pred: fn(&AiOp) -> bool) -> Self {
        match self {
            Pattern::Op {
                op,
                inputs,
                bind,
                commutative,
                predicate: _,
            } => Pattern::Op {
                op,
                inputs,
                bind,
                commutative,
                predicate: Some(pred),
            },
            other => other,
        }
    }
}

/// Replacement: a single canonical `AiOp` node whose inputs are bound
/// `VarId`s from the pattern.
///
/// Two construction modes:
/// * `Replacement::new(op, inputs)` — static op (e.g. `FusedSwiGLU`).
/// * `Replacement::from_root(builder, inputs)` — the replacement op is
///   computed from the matched root's `AiOp` value via a pure `fn`
///   pointer. This handles patterns whose canonical replacement carries
///   an attribute from the matched root (e.g. `Add(x, r) → RmsNorm` →
///   `FusedLayerNormResidual { epsilon }` carries the `RmsNorm`'s
///   epsilon). The builder is a plain `fn` (not a closure capturing
///   state), so its identity is its address — content-addressable.
#[derive(Clone)]
pub struct Replacement {
    op: ReplacementOp,
    inputs: Vec<VarId>,
}

#[derive(Clone)]
enum ReplacementOp {
    /// Static op — the same value for every successful match.
    Static(Box<AiOp>),
    /// Computed from the matched root's `AiOp` (e.g. carry epsilon).
    FromRoot(fn(&AiOp) -> Option<AiOp>),
    /// Computed from the matched root op + a typed view of the bound
    /// variables (their params, shapes, and scalar values). Used by
    /// fusions whose canonical replacement carries an attribute
    /// extracted from a bound constant or from a tensor's shape —
    /// e.g. an `epsilon` scalar param on `RmsNormFusion`, a `perm`
    /// i64 array on `TransposeMatMulFusion`, or `num_heads` /
    /// `head_dim` derived from the K tensor's shape on
    /// `AttentionFusion`. Returning `None` aborts the rewrite — the
    /// engine refuses to approximate.
    FromMatch(fn(&AiOp, &MatchView) -> Option<AiOp>),
    /// Graph-mutating rewrite. The closure takes the graph mutably,
    /// the matched bindings, and the matched root's node index; it
    /// applies whatever multi-node / new-param / new-input mutation
    /// the rule needs and returns the **new root node** to be put in
    /// place of the matched root (or `None` to abort with no
    /// approximation). The engine then deletes the consumed interior
    /// nodes as for other replacements.
    ///
    /// Used by rewrites that don't fit the single-node-replace model:
    /// `SliceToGather` (emits a new `i64[]` indices param + a Gather
    /// node), `NormProjectionFusion` (multi-output projection), the
    /// injection passes (`KvSlotInjection`, `PositionIdsInjection`).
    Custom(fn(&mut AiGraph, &HashMap<VarId, TensorId>, usize) -> Option<AiNode>),
}

impl std::fmt::Debug for Replacement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kind = match &self.op {
            ReplacementOp::Static(_) => "Static",
            ReplacementOp::FromRoot(_) => "FromRoot",
            ReplacementOp::FromMatch(_) => "FromMatch",
            ReplacementOp::Custom(_) => "Custom",
        };
        f.debug_struct("Replacement")
            .field("kind", &kind)
            .field("inputs", &self.inputs)
            .finish()
    }
}

impl Replacement {
    /// Static replacement — the canonical op is the same for every match.
    pub fn new(op: AiOp, inputs: Vec<VarId>) -> Self {
        Self {
            op: ReplacementOp::Static(Box::new(op)),
            inputs,
        }
    }

    /// Dynamic replacement — the canonical op is computed from the
    /// matched root's `AiOp` value. Returns `None` to abort the rewrite
    /// (e.g. an attribute that should have been resolved at the matched
    /// root is unexpectedly missing — the engine refuses to approximate).
    pub fn from_root(builder: fn(&AiOp) -> Option<AiOp>, inputs: Vec<VarId>) -> Self {
        Self {
            op: ReplacementOp::FromRoot(builder),
            inputs,
        }
    }

    /// Dynamic replacement with access to the bound variables. The
    /// builder receives the matched root's `AiOp` plus a typed view of
    /// the bound vars (their `AiParam`s if they are constants, their
    /// declared tensor shapes, and helpers for the common scalar /
    /// 1-D vector extractions). Returning `None` aborts the rewrite —
    /// the engine refuses to approximate.
    pub fn from_match(builder: fn(&AiOp, &MatchView) -> Option<AiOp>, inputs: Vec<VarId>) -> Self {
        Self {
            op: ReplacementOp::FromMatch(builder),
            inputs,
        }
    }

    /// Graph-mutating rewrite. The closure mutates the graph (params,
    /// inputs, tensor_info, nodes) and returns the new root node to be
    /// put in place of the matched root, or `None` to abort. The
    /// engine deletes consumed interior nodes as usual. The closure
    /// is a `fn` pointer (no captured state), keeping its identity
    /// content-addressable. The `inputs` list is unused for `Custom`
    /// rewrites — the rewrite reads bindings directly from the
    /// supplied `&HashMap<VarId, TensorId>`.
    pub fn custom(
        rewrite: fn(&mut AiGraph, &HashMap<VarId, TensorId>, usize) -> Option<AiNode>,
    ) -> Self {
        Self {
            op: ReplacementOp::Custom(rewrite),
            inputs: Vec::new(),
        }
    }

    pub fn inputs(&self) -> &[VarId] {
        &self.inputs
    }

    fn build(&self, root_op: &AiOp, view: &MatchView) -> Option<AiOp> {
        match &self.op {
            ReplacementOp::Static(op) => Some((**op).clone()),
            ReplacementOp::FromRoot(builder) => builder(root_op),
            ReplacementOp::FromMatch(builder) => builder(root_op, view),
            ReplacementOp::Custom(_) => None, // dispatched separately in apply_pass
        }
    }

    fn is_custom(&self) -> bool {
        matches!(self.op, ReplacementOp::Custom(_))
    }

    fn run_custom(
        &self,
        graph: &mut AiGraph,
        binds: &HashMap<VarId, TensorId>,
        root_idx: usize,
    ) -> Option<AiNode> {
        if let ReplacementOp::Custom(rewrite) = self.op {
            rewrite(graph, binds, root_idx)
        } else {
            None
        }
    }
}

/// A typed view of a successful match: the bound variables and the
/// graph context they live in. Available to `Replacement::from_match`
/// builders.
pub struct MatchView<'a> {
    binds: &'a HashMap<VarId, TensorId>,
    graph: &'a AiGraph,
}

impl<'a> MatchView<'a> {
    pub fn tensor(&self, var: VarId) -> Option<TensorId> {
        self.binds.get(&var).copied()
    }

    /// The bound var's `AiParam`, if the tensor is a constant. Useful
    /// for extracting scalar epsilons, integer arrays, etc.
    pub fn param(&self, var: VarId) -> Option<&AiParam> {
        let tid = self.tensor(var)?;
        self.graph.params.get(&tid)
    }

    /// Read the bound var as a scalar f32 (e.g. epsilon, scale).
    /// Returns `None` if the param is missing, not a constant, or not
    /// a singleton f32 value.
    pub fn scalar_f32(&self, var: VarId) -> Option<f32> {
        let param = self.param(var)?;
        if param.info().logical_dtype != crate::ir::DType::F32 {
            return None;
        }
        let s = param.as_f32_slice()?;
        if s.len() != 1 {
            return None;
        }
        Some(s[0])
    }

    /// The bound var's declared shape, if it's a tensor with known
    /// `TensorInfo` (a graph input, param, or a node output the
    /// shape-propagation passes have inferred). Used by rules that
    /// derive replacement attributes from tensor dimensions —
    /// `AttentionFusion`'s `num_heads`/`num_kv_heads`/`head_dim` come
    /// from Q and K's shapes.
    pub fn shape(&self, var: VarId) -> Option<&crate::ir::Shape> {
        let tid = self.tensor(var)?;
        self.graph.tensor_info.get(&tid).map(|info| &info.shape)
    }

    /// Concrete dim value of the bound var at `axis`, or `None` if the
    /// shape isn't known concretely at that axis. Negative axes index
    /// from the end of the shape (`-1` = last axis).
    pub fn dim(&self, var: VarId, axis: i64) -> Option<u64> {
        let s = self.shape(var)?;
        let r = s.len();
        let idx = if axis < 0 {
            ((r as i64) + axis).max(0) as usize
        } else {
            axis as usize
        };
        s.get(idx)?.as_concrete()
    }

    /// True if the bound var is a graph input port (a runtime-bound
    /// tensor, not a constant). Used by rules whose `VarId` should
    /// match only a graph input (e.g. a model weight passed at
    /// runtime).
    pub fn is_graph_input(&self, var: VarId) -> bool {
        let Some(tid) = self.tensor(var) else {
            return false;
        };
        self.graph.inputs.contains(&tid)
    }

    /// Read the bound var as an i64 array (e.g. a `Transpose.perm`).
    pub fn i64_vec(&self, var: VarId) -> Option<Vec<i64>> {
        let param = self.param(var)?;
        if param.info().logical_dtype != crate::ir::DType::INT64 {
            return None;
        }
        let bytes: &[u8] = match param {
            AiParam::Inline { data, .. } => data.as_ref(),
            AiParam::Mmap { .. } => return None,
        };
        if !bytes.len().is_multiple_of(8) {
            return None;
        }
        Some(
            bytes
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().expect("chunk_exact")))
                .collect(),
        )
    }
}

/// A single declarative rewrite rule.
///
/// `witness` is the V&V test name (in `hologram-ai-conformance`,
/// `hologram-ai`, or upstream) that verifies the rule's correctness
/// against an external authoritative source. A rule without a witness
/// MUST NOT be added to a [`RuleSet`].
#[derive(Debug, Clone)]
pub struct Rule {
    pub name: &'static str,
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

    /// Apply every rule to fixed-point. Returns the number of rewrites.
    /// A non-confluent rule set is detected by non-convergence (an
    /// unbounded loop) and panics — the engine refuses rather than
    /// approximates.
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

    /// One sweep: at each candidate root, try each rule. A successful
    /// match (a) replaces the root node's op + inputs with the rule's
    /// replacement, and (b) marks every *interior* matched node for
    /// removal. After the sweep, removed nodes are spliced out and any
    /// stale producer-map entries are rebuilt on the next pass.
    fn apply_pass(&self, graph: &mut AiGraph) -> usize {
        let mut rewrites = 0usize;
        let n = graph.nodes.len();
        let mut next_id = next_node_id(graph);
        let producer = build_producer_map(graph);
        let consumer_counts = build_consumer_counts(graph);
        let mut rewritten: HashMap<usize, AiNode> = HashMap::new();
        let mut to_remove: Vec<bool> = vec![false; n];

        'outer: for root_idx in 0..n {
            if rewritten.contains_key(&root_idx) || to_remove[root_idx] {
                continue;
            }
            for rule in &self.rules {
                let mut m = Match::default();
                if !Matcher::match_at(
                    graph,
                    &producer,
                    &consumer_counts,
                    &rule.pattern,
                    root_idx,
                    true, // root position
                    &mut m,
                ) {
                    continue;
                }

                let root_op = graph.nodes[root_idx].op.clone();
                let Some(root_out) = graph.nodes[root_idx].outputs.first().copied() else {
                    continue;
                };
                let new_node = if rule.replacement.is_custom() {
                    // Custom rewrite: mutates graph (params, tensor_info,
                    // possibly new nodes) and returns the new root node.
                    // The engine puts the returned node at `root_idx` and
                    // deletes consumed interior nodes as usual.
                    match rule.replacement.run_custom(graph, &m.binds, root_idx) {
                        Some(n) => n,
                        None => continue,
                    }
                } else {
                    let view = MatchView {
                        binds: &m.binds,
                        graph,
                    };
                    let Some(n) = materialize(
                        &rule.replacement,
                        &m,
                        &root_op,
                        &view,
                        root_out,
                        &mut next_id,
                    ) else {
                        continue;
                    };
                    n
                };
                rewritten.insert(root_idx, new_node);
                for &idx in &m.consumed {
                    if idx != root_idx {
                        to_remove[idx] = true;
                    }
                }
                rewrites += 1;
                continue 'outer;
            }
        }

        if rewrites == 0 {
            return 0;
        }

        // Apply rewrites + removals in one pass over `nodes`.
        let mut new_nodes = Vec::with_capacity(graph.nodes.len());
        for (idx, node) in graph.nodes.iter().enumerate() {
            if to_remove[idx] {
                continue;
            }
            if let Some(replacement) = rewritten.remove(&idx) {
                new_nodes.push(replacement);
            } else {
                new_nodes.push(node.clone());
            }
        }
        graph.nodes = new_nodes;

        rewrites
    }
}

/// Bindings + matched-node indices established by a successful match.
#[derive(Debug, Default)]
struct Match {
    binds: HashMap<VarId, TensorId>,
    consumed: Vec<usize>,
}

impl Match {
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
        consumer_counts: &HashMap<TensorId, usize>,
        pattern: &Pattern,
        node_idx: usize,
        is_root: bool,
        m: &mut Match,
    ) -> bool {
        let node = &graph.nodes[node_idx];

        // Interior matched Op nodes must have single-consumer outputs;
        // otherwise removing them would break a downstream path.
        if !is_root {
            if let Pattern::Op { .. } = pattern {
                let Some(&out) = node.outputs.first() else {
                    return false;
                };
                if consumer_counts.get(&out).copied().unwrap_or(0) != 1 {
                    return false;
                }
            }
        }

        match pattern {
            Pattern::Var(var) => {
                let Some(&tid) = node.outputs.first() else {
                    return false;
                };
                m.bind(*var, tid)
            }
            Pattern::Const(_) => {
                // Const can only match a leaf tensor (a graph constant),
                // not the output of an arbitrary node. At a node root
                // position this is a structural error in the pattern.
                false
            }
            Pattern::Op {
                op,
                inputs,
                bind,
                commutative,
                predicate,
            } => {
                if !op.matches(&node.op) {
                    return false;
                }
                if inputs.len() != node.inputs.len() {
                    return false;
                }
                if let Some(pred) = predicate {
                    if !pred(&node.op) {
                        return false;
                    }
                }
                if let Some(b) = bind {
                    let Some(&tid) = node.outputs.first() else {
                        return false;
                    };
                    if !m.bind(*b, tid) {
                        return false;
                    }
                }

                // Try the natural input order; if commutative and that
                // order fails, try the swapped order.
                let orderings: &[[usize; 2]] = if *commutative && inputs.len() == 2 {
                    &[[0, 1], [1, 0]]
                } else {
                    &[[0, 1]] // ignored when inputs.len() != 2
                };

                let try_order = |order: &[usize], saved: &mut Match| -> bool {
                    for (pos, &perm_idx) in order.iter().enumerate() {
                        let child_pat = &inputs[perm_idx];
                        let in_tid = node.inputs[pos];
                        if !Matcher::match_tensor(
                            graph,
                            producer,
                            consumer_counts,
                            child_pat,
                            in_tid,
                            saved,
                        ) {
                            return false;
                        }
                    }
                    true
                };

                if *commutative && inputs.len() == 2 {
                    let snapshot_binds = m.binds.clone();
                    let snapshot_consumed = m.consumed.clone();
                    for order in orderings {
                        if try_order(order, m) {
                            m.consumed.push(node_idx);
                            return true;
                        }
                        m.binds = snapshot_binds.clone();
                        m.consumed = snapshot_consumed.clone();
                    }
                    false
                } else {
                    let natural: Vec<usize> = (0..inputs.len()).collect();
                    if try_order(&natural, m) {
                        m.consumed.push(node_idx);
                        true
                    } else {
                        false
                    }
                }
            }
            Pattern::Maybe(inner) => {
                let snapshot = (m.binds.clone(), m.consumed.clone());
                if Self::match_at(
                    graph,
                    producer,
                    consumer_counts,
                    inner,
                    node_idx,
                    is_root,
                    m,
                ) {
                    return true;
                }
                m.binds = snapshot.0;
                m.consumed = snapshot.1;
                // Absent branch: bind the inner's root var (if any) to
                // this node's first output.
                let Some(&tid) = node.outputs.first() else {
                    return false;
                };
                if let Pattern::Op { bind: Some(b), .. } = inner.as_ref() {
                    m.bind(*b, tid)
                } else {
                    true
                }
            }
        }
    }

    fn match_tensor(
        graph: &AiGraph,
        producer: &HashMap<TensorId, usize>,
        consumer_counts: &HashMap<TensorId, usize>,
        pattern: &Pattern,
        tid: TensorId,
        m: &mut Match,
    ) -> bool {
        match pattern {
            Pattern::Var(var) => m.bind(*var, tid),
            Pattern::Const(var) => {
                if !graph.params.contains_key(&tid) {
                    return false;
                }
                m.bind(*var, tid)
            }
            Pattern::Op { .. } | Pattern::Maybe(_) => {
                let Some(&prod_idx) = producer.get(&tid) else {
                    return false;
                };
                Self::match_at(
                    graph,
                    producer,
                    consumer_counts,
                    pattern,
                    prod_idx,
                    false,
                    m,
                )
            }
        }
    }
}

fn materialize(
    repl: &Replacement,
    m: &Match,
    root_op: &AiOp,
    view: &MatchView,
    root_out: TensorId,
    next_id: &mut NodeId,
) -> Option<AiNode> {
    let mut input_tids = Vec::with_capacity(repl.inputs.len());
    for v in &repl.inputs {
        input_tids.push(m.lookup(*v)?);
    }
    let op = repl.build(root_op, view)?;
    let new = AiNode::new(*next_id, op, input_tids, vec![root_out]);
    *next_id += 1;
    Some(new)
}

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

fn build_consumer_counts(graph: &AiGraph) -> HashMap<TensorId, usize> {
    let mut m: HashMap<TensorId, usize> = HashMap::new();
    for node in &graph.nodes {
        for &in_tid in &node.inputs {
            *m.entry(in_tid).or_insert(0) += 1;
        }
    }
    // A tensor that's a graph output is also "consumed" (by the world).
    for &out_tid in &graph.outputs {
        *m.entry(out_tid).or_insert(0) += 1;
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

    fn add_node(g: &mut AiGraph, id: NodeId, op: AiOp, inputs: Vec<TensorId>, output: TensorId) {
        let shape = shape_from_concrete(&[4]);
        g.tensor_info
            .insert(output, TensorInfo::new(DType::F32, shape));
        g.nodes.push(AiNode::new(id, op, inputs, vec![output]));
    }

    #[test]
    fn double_relu_folds_to_single_relu() {
        // Relu(Relu(x)) → Relu(x). Two matched nodes: the outer Relu
        // becomes the rewrite; the inner Relu is interior, single-
        // consumer, removed.
        let mut g = unit_graph();
        let shape = shape_from_concrete(&[4]);
        g.tensor_info.insert(0, TensorInfo::new(DType::F32, shape));
        g.inputs = vec![0];
        g.outputs = vec![2];
        add_node(&mut g, 0, AiOp::Relu, vec![0], 1);
        add_node(&mut g, 1, AiOp::Relu, vec![1], 2);

        let x = VarId(1);
        let rule = Rule {
            name: "double_relu_collapse",
            witness: "Relu(Relu(x)) == Relu(x) (idempotence; spec invariant)",
            pattern: Pattern::op(
                OpMatcher::exact_relu(),
                vec![Pattern::op(OpMatcher::exact_relu(), vec![Pattern::Var(x)])],
            ),
            replacement: Replacement::new(AiOp::Relu, vec![x]),
        };
        let set = RuleSet::new().with_rule(rule);

        let rewrites = set.apply(&mut g);
        assert!(rewrites >= 1, "expected at least one rewrite");
        // After the rewrite, only one Relu node remains.
        assert_eq!(g.nodes.len(), 1);
        assert!(matches!(g.nodes[0].op, AiOp::Relu));
        assert_eq!(g.nodes[0].inputs, vec![0], "outer Relu now reads from x");
        assert_eq!(
            g.nodes[0].outputs,
            vec![2],
            "outer Relu retains the root output tensor id"
        );
    }

    #[test]
    fn multi_consumer_interior_is_not_rewritten() {
        // If the interior Relu has more than one consumer, removing it
        // would break the second consumer's input — the matcher must
        // refuse to match.
        let mut g = unit_graph();
        let shape = shape_from_concrete(&[4]);
        for tid in 0..4u32 {
            g.tensor_info
                .insert(tid, TensorInfo::new(DType::F32, shape.clone()));
        }
        g.inputs = vec![0];
        g.outputs = vec![2, 3];
        add_node(&mut g, 0, AiOp::Relu, vec![0], 1);
        add_node(&mut g, 1, AiOp::Relu, vec![1], 2);
        add_node(&mut g, 2, AiOp::Sigmoid, vec![1], 3); // second consumer of inner Relu

        let x = VarId(1);
        let rule = Rule {
            name: "double_relu_collapse",
            witness: "Relu(Relu(x)) == Relu(x)",
            pattern: Pattern::op(
                OpMatcher::exact_relu(),
                vec![Pattern::op(OpMatcher::exact_relu(), vec![Pattern::Var(x)])],
            ),
            replacement: Replacement::new(AiOp::Relu, vec![x]),
        };
        let set = RuleSet::new().with_rule(rule);

        let rewrites = set.apply(&mut g);
        assert_eq!(
            rewrites, 0,
            "multi-consumer interior must not be rewritten (no approximation)"
        );
        assert_eq!(g.nodes.len(), 3);
    }

    #[test]
    fn commutative_match_tries_both_orderings() {
        // SwiGLU pattern: Mul(Silu(gate), up) → FusedSwiGLU(gate, up).
        // Test both Mul orderings: Mul(Silu(g), u) and Mul(u, Silu(g)).
        for swap in [false, true] {
            let mut g = unit_graph();
            let shape = shape_from_concrete(&[4]);
            for tid in 0..4u32 {
                g.tensor_info
                    .insert(tid, TensorInfo::new(DType::F32, shape.clone()));
            }
            g.inputs = vec![0, 1];
            g.outputs = vec![3];
            add_node(&mut g, 0, AiOp::Silu, vec![0], 2);
            let mul_inputs = if swap { vec![1, 2] } else { vec![2, 1] };
            add_node(&mut g, 1, AiOp::Mul, mul_inputs, 3);

            let gate = VarId(1);
            let up = VarId(2);
            let rule = Rule {
                name: "swiglu_fusion_direct",
                witness: "real_model_generation::smollm2_paris (EE-3 ORT parity)",
                pattern: Pattern::op_comm(
                    OpMatcher::exact_mul(),
                    Pattern::op(OpMatcher::exact_silu(), vec![Pattern::Var(gate)]),
                    Pattern::Var(up),
                ),
                replacement: Replacement::new(AiOp::FusedSwiGLU, vec![gate, up]),
            };
            let set = RuleSet::new().with_rule(rule);

            let rewrites = set.apply(&mut g);
            assert!(rewrites >= 1, "swap={swap}: expected a rewrite");
            assert_eq!(g.nodes.len(), 1, "swap={swap}: Silu removed");
            assert!(matches!(g.nodes[0].op, AiOp::FusedSwiGLU));
            assert_eq!(
                g.nodes[0].inputs,
                vec![0, 1],
                "swap={swap}: gate=tid 0, up=tid 1"
            );
        }
    }
}
