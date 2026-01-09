//! Translator registry for ONNX operations.
//!
//! The registry manages all registered translators and provides
//! the central dispatch mechanism for translating ONNX nodes.

use std::collections::HashMap;
use std::sync::Arc;
use hologram::ir::{GraphBuilder, NodeIndex};
use crate::proto::NodeProto;
use super::{OnnxTranslator, TranslationError};

/// Registry of ONNX translators.
///
/// The registry holds all registered translators and provides the
/// `translate()` method to dispatch ONNX nodes to the appropriate translator.
///
/// # Example
///
/// ```ignore
/// let registry = TranslatorRegistry::new();
/// let outputs = registry.translate(&node, &inputs, &mut builder)?;
/// ```
pub struct TranslatorRegistry {
    translators: HashMap<&'static str, Arc<dyn OnnxTranslator>>,
}

impl Default for TranslatorRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl TranslatorRegistry {
    /// Create a new registry with all built-in translators registered.
    pub fn new() -> Self {
        let mut registry = Self {
            translators: HashMap::with_capacity(64),
        };

        // Register all translators by category
        registry.register_activation();
        registry.register_binary();
        registry.register_unary();
        registry.register_matmul();
        registry.register_constant();
        registry.register_reduce();
        registry.register_shape();
        registry.register_indexing();
        registry.register_norm();
        registry.register_conv();
        registry.register_pool();
        registry.register_logical();
        registry.register_advanced();
        registry.register_resize();
        registry.register_pad();

        registry
    }

    /// Register a single translator.
    ///
    /// The translator's `onnx_op_type()` is used as the key.
    /// If a translator for the same op type already exists, it is replaced.
    pub fn register(&mut self, translator: Arc<dyn OnnxTranslator>) {
        self.translators
            .insert(translator.onnx_op_type(), translator);
    }

    /// Get a translator by operation type.
    pub fn get(&self, op_type: &str) -> Option<&Arc<dyn OnnxTranslator>> {
        self.translators.get(op_type)
    }

    /// Check if an operation type is supported.
    pub fn is_supported(&self, op_type: &str) -> bool {
        self.translators.contains_key(op_type)
    }

    /// Get the number of registered translators.
    pub fn len(&self) -> usize {
        self.translators.len()
    }

    /// Check if the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.translators.is_empty()
    }

    /// Get an iterator over all supported operation types.
    pub fn supported_ops(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.translators.keys().copied()
    }

    /// Translate an ONNX node to IR.
    ///
    /// This is the main entry point for translation. It:
    /// 1. Looks up the translator for the node's operation type
    /// 2. Validates the input count
    /// 3. Calls the translator's `translate()` method
    ///
    /// # Arguments
    ///
    /// * `node` - The ONNX node to translate
    /// * `inputs` - IR node indices for the input tensors
    /// * `builder` - The IR graph builder
    ///
    /// # Returns
    ///
    /// A vector of output IR node indices.
    pub fn translate(
        &self,
        node: &NodeProto,
        inputs: &[NodeIndex],
        builder: &mut GraphBuilder,
    ) -> Result<Vec<NodeIndex>, TranslationError> {
        let translator = self.translators.get(node.op_type.as_str()).ok_or_else(|| {
            TranslationError::unsupported_op(&node.op_type, 13)
        })?;

        // Validate input count
        translator
            .input_requirement()
            .validate(inputs.len(), &node.op_type)?;

        // Perform translation
        translator.translate(node, inputs, builder)
    }

    // ===== Registration methods by category =====

    fn register_activation(&mut self) {
        use super::activation::*;
        self.register(Arc::new(ReluTranslator));
        self.register(Arc::new(SigmoidTranslator));
        self.register(Arc::new(TanhTranslator));
        self.register(Arc::new(GeluTranslator));
        self.register(Arc::new(SoftmaxTranslator));
        self.register(Arc::new(ClipTranslator));
        self.register(Arc::new(LeakyReluTranslator));
        self.register(Arc::new(EluTranslator));
        self.register(Arc::new(SeluTranslator));
        self.register(Arc::new(PReluTranslator));
        self.register(Arc::new(SwishTranslator));
        self.register(Arc::new(ErfTranslator));
    }

    fn register_binary(&mut self) {
        use super::binary::*;
        self.register(Arc::new(AddTranslator));
        self.register(Arc::new(SubTranslator));
        self.register(Arc::new(MulTranslator));
        self.register(Arc::new(DivTranslator));
        self.register(Arc::new(PowTranslator));
        self.register(Arc::new(MinTranslator));
        self.register(Arc::new(MaxTranslator));
    }

    fn register_unary(&mut self) {
        use super::unary::*;
        self.register(Arc::new(SqrtTranslator));
        self.register(Arc::new(ExpTranslator));
        self.register(Arc::new(LogTranslator));
        self.register(Arc::new(AbsTranslator));
        self.register(Arc::new(NegTranslator));
        self.register(Arc::new(ReciprocalTranslator));
        self.register(Arc::new(SinTranslator));
        self.register(Arc::new(CosTranslator));
        self.register(Arc::new(TanTranslator));
    }

    fn register_matmul(&mut self) {
        use super::matmul::*;
        self.register(Arc::new(MatMulTranslator));
        self.register(Arc::new(GemmTranslator));
    }

    fn register_constant(&mut self) {
        use super::constant::*;
        self.register(Arc::new(ConstantTranslator));
        self.register(Arc::new(ConstantOfShapeTranslator));
        self.register(Arc::new(ShapeOpTranslator));
        self.register(Arc::new(IdentityTranslator));
    }

    fn register_reduce(&mut self) {
        use super::reduce::*;
        self.register(Arc::new(ReduceSumTranslator));
        self.register(Arc::new(ReduceMeanTranslator));
        self.register(Arc::new(ReduceMaxTranslator));
        self.register(Arc::new(ReduceMinTranslator));
        self.register(Arc::new(ReduceProdTranslator));
    }

    fn register_shape(&mut self) {
        use super::shape::*;
        self.register(Arc::new(ReshapeTranslator));
        self.register(Arc::new(TransposeTranslator));
        self.register(Arc::new(ConcatTranslator));
        self.register(Arc::new(SqueezeTranslator));
        self.register(Arc::new(UnsqueezeTranslator));
        self.register(Arc::new(FlattenTranslator));
        self.register(Arc::new(ExpandTranslator));
        self.register(Arc::new(SplitTranslator));
        self.register(Arc::new(TileTranslator));
    }

    fn register_indexing(&mut self) {
        use super::indexing::*;
        self.register(Arc::new(GatherTranslator));
        self.register(Arc::new(SliceTranslator));
        self.register(Arc::new(GatherElementsTranslator));
        self.register(Arc::new(ScatterNDTranslator));
    }

    fn register_norm(&mut self) {
        use super::norm::*;
        self.register(Arc::new(LayerNormTranslator));
        self.register(Arc::new(BatchNormTranslator));
        self.register(Arc::new(GroupNormTranslator));
        self.register(Arc::new(InstanceNormTranslator));
    }

    fn register_conv(&mut self) {
        use super::conv::*;
        self.register(Arc::new(ConvTranslator));
        self.register(Arc::new(ConvTransposeTranslator));
    }

    fn register_pool(&mut self) {
        use super::pool::*;
        self.register(Arc::new(MaxPoolTranslator));
        self.register(Arc::new(AveragePoolTranslator));
        self.register(Arc::new(GlobalAveragePoolTranslator));
        self.register(Arc::new(GlobalMaxPoolTranslator));
    }

    fn register_logical(&mut self) {
        use super::logical::*;
        self.register(Arc::new(EqualTranslator));
        self.register(Arc::new(GreaterTranslator));
        self.register(Arc::new(LessTranslator));
        self.register(Arc::new(GreaterOrEqualTranslator));
        self.register(Arc::new(LessOrEqualTranslator));
        self.register(Arc::new(AndTranslator));
        self.register(Arc::new(OrTranslator));
        self.register(Arc::new(NotTranslator));
        self.register(Arc::new(WhereTranslator));
    }

    fn register_advanced(&mut self) {
        use super::advanced::*;
        self.register(Arc::new(CastTranslator));
        self.register(Arc::new(RangeTranslator));
        self.register(Arc::new(TriluTranslator));
        self.register(Arc::new(DropoutTranslator));
    }

    fn register_resize(&mut self) {
        use super::resize::*;
        self.register(Arc::new(ResizeTranslator));
        self.register(Arc::new(UpsampleTranslator));
        self.register(Arc::new(DepthToSpaceTranslator));
        self.register(Arc::new(SpaceToDepthTranslator));
    }

    fn register_pad(&mut self) {
        use super::pad::*;
        self.register(Arc::new(PadTranslator));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_creation() {
        let registry = TranslatorRegistry::new();
        assert!(!registry.is_empty());
        // Activation(12) + Binary(7) + Unary(9) + Matmul(2) + Constant(4) + Reduce(5) + Shape(9)
        // + Indexing(4) + Norm(4) + Conv(2) + Pool(4) + Logical(9) + Advanced(4) + Resize(4) + Pad(1)
        // = 80+
        assert!(registry.len() >= 70);
    }

    #[test]
    fn test_is_supported() {
        let registry = TranslatorRegistry::new();

        // Common operations should be supported
        assert!(registry.is_supported("Relu"));
        assert!(registry.is_supported("Add"));
        assert!(registry.is_supported("MatMul"));
        assert!(registry.is_supported("Softmax"));
        assert!(registry.is_supported("Constant"));
        assert!(registry.is_supported("ConstantOfShape"));
        assert!(registry.is_supported("Shape"));
        assert!(registry.is_supported("Identity"));

        // Unknown operations should not be supported
        assert!(!registry.is_supported("CustomOp"));
        assert!(!registry.is_supported(""));
    }

    #[test]
    fn test_supported_ops_iteration() {
        let registry = TranslatorRegistry::new();
        let ops: Vec<_> = registry.supported_ops().collect();

        assert!(ops.contains(&"Relu"));
        assert!(ops.contains(&"Add"));
        assert!(ops.contains(&"MatMul"));
        assert!(ops.contains(&"Constant"));
        assert!(ops.contains(&"Identity"));
    }

    #[test]
    fn test_get_translator() {
        let registry = TranslatorRegistry::new();

        let relu = registry.get("Relu");
        assert!(relu.is_some());
        assert_eq!(relu.unwrap().onnx_op_type(), "Relu");

        let constant = registry.get("Constant");
        assert!(constant.is_some());
        assert_eq!(constant.unwrap().onnx_op_type(), "Constant");

        let missing = registry.get("NonExistent");
        assert!(missing.is_none());
    }

    #[test]
    fn test_constant_translators_registered() {
        let registry = TranslatorRegistry::new();

        // Verify all constant translators are registered
        assert!(registry.is_supported("Constant"));
        assert!(registry.is_supported("ConstantOfShape"));
        assert!(registry.is_supported("Shape"));
        assert!(registry.is_supported("Identity"));
    }

    #[test]
    fn test_matmul_translators_registered() {
        let registry = TranslatorRegistry::new();

        // Verify matmul translators are registered
        assert!(registry.is_supported("MatMul"));
        assert!(registry.is_supported("Gemm"));

        // Verify correct op types
        let matmul = registry.get("MatMul");
        assert!(matmul.is_some());
        assert_eq!(matmul.unwrap().onnx_op_type(), "MatMul");

        let gemm = registry.get("Gemm");
        assert!(gemm.is_some());
        assert_eq!(gemm.unwrap().onnx_op_type(), "Gemm");
    }

    #[test]
    fn test_reduce_translators_registered() {
        let registry = TranslatorRegistry::new();

        // Verify all reduce translators are registered
        assert!(registry.is_supported("ReduceSum"));
        assert!(registry.is_supported("ReduceMean"));
        assert!(registry.is_supported("ReduceMax"));
        assert!(registry.is_supported("ReduceMin"));
        assert!(registry.is_supported("ReduceProd"));

        // Verify correct op types
        let reduce_sum = registry.get("ReduceSum");
        assert!(reduce_sum.is_some());
        assert_eq!(reduce_sum.unwrap().onnx_op_type(), "ReduceSum");

        let reduce_mean = registry.get("ReduceMean");
        assert!(reduce_mean.is_some());
        assert_eq!(reduce_mean.unwrap().onnx_op_type(), "ReduceMean");

        let reduce_max = registry.get("ReduceMax");
        assert!(reduce_max.is_some());
        assert_eq!(reduce_max.unwrap().onnx_op_type(), "ReduceMax");

        let reduce_min = registry.get("ReduceMin");
        assert!(reduce_min.is_some());
        assert_eq!(reduce_min.unwrap().onnx_op_type(), "ReduceMin");

        let reduce_prod = registry.get("ReduceProd");
        assert!(reduce_prod.is_some());
        assert_eq!(reduce_prod.unwrap().onnx_op_type(), "ReduceProd");
    }

    #[test]
    fn test_shape_translators_registered() {
        let registry = TranslatorRegistry::new();

        // Verify all shape translators are registered
        assert!(registry.is_supported("Reshape"));
        assert!(registry.is_supported("Transpose"));
        assert!(registry.is_supported("Concat"));
        assert!(registry.is_supported("Squeeze"));
        assert!(registry.is_supported("Unsqueeze"));
        assert!(registry.is_supported("Flatten"));
        assert!(registry.is_supported("Expand"));
        assert!(registry.is_supported("Split"));
        assert!(registry.is_supported("Tile"));

        // Verify correct op types
        let reshape = registry.get("Reshape");
        assert!(reshape.is_some());
        assert_eq!(reshape.unwrap().onnx_op_type(), "Reshape");

        let transpose = registry.get("Transpose");
        assert!(transpose.is_some());
        assert_eq!(transpose.unwrap().onnx_op_type(), "Transpose");

        let concat = registry.get("Concat");
        assert!(concat.is_some());
        assert_eq!(concat.unwrap().onnx_op_type(), "Concat");

        let squeeze = registry.get("Squeeze");
        assert!(squeeze.is_some());
        assert_eq!(squeeze.unwrap().onnx_op_type(), "Squeeze");

        let unsqueeze = registry.get("Unsqueeze");
        assert!(unsqueeze.is_some());
        assert_eq!(unsqueeze.unwrap().onnx_op_type(), "Unsqueeze");

        let flatten = registry.get("Flatten");
        assert!(flatten.is_some());
        assert_eq!(flatten.unwrap().onnx_op_type(), "Flatten");

        let expand = registry.get("Expand");
        assert!(expand.is_some());
        assert_eq!(expand.unwrap().onnx_op_type(), "Expand");

        let split = registry.get("Split");
        assert!(split.is_some());
        assert_eq!(split.unwrap().onnx_op_type(), "Split");

        let tile = registry.get("Tile");
        assert!(tile.is_some());
        assert_eq!(tile.unwrap().onnx_op_type(), "Tile");
    }

    #[test]
    fn test_indexing_translators_registered() {
        let registry = TranslatorRegistry::new();

        assert!(registry.is_supported("Gather"));
        assert!(registry.is_supported("Slice"));
        assert!(registry.is_supported("GatherElements"));
        assert!(registry.is_supported("ScatterND"));

        let gather = registry.get("Gather");
        assert!(gather.is_some());
        assert_eq!(gather.unwrap().onnx_op_type(), "Gather");
    }

    #[test]
    fn test_norm_translators_registered() {
        let registry = TranslatorRegistry::new();

        assert!(registry.is_supported("LayerNormalization"));
        assert!(registry.is_supported("BatchNormalization"));
        assert!(registry.is_supported("GroupNormalization"));
        assert!(registry.is_supported("InstanceNormalization"));

        let layer_norm = registry.get("LayerNormalization");
        assert!(layer_norm.is_some());
        assert_eq!(layer_norm.unwrap().onnx_op_type(), "LayerNormalization");
    }

    #[test]
    fn test_conv_translators_registered() {
        let registry = TranslatorRegistry::new();

        assert!(registry.is_supported("Conv"));
        assert!(registry.is_supported("ConvTranspose"));

        let conv = registry.get("Conv");
        assert!(conv.is_some());
        assert_eq!(conv.unwrap().onnx_op_type(), "Conv");
    }

    #[test]
    fn test_pool_translators_registered() {
        let registry = TranslatorRegistry::new();

        assert!(registry.is_supported("MaxPool"));
        assert!(registry.is_supported("AveragePool"));
        assert!(registry.is_supported("GlobalAveragePool"));
        assert!(registry.is_supported("GlobalMaxPool"));

        let max_pool = registry.get("MaxPool");
        assert!(max_pool.is_some());
        assert_eq!(max_pool.unwrap().onnx_op_type(), "MaxPool");
    }

    #[test]
    fn test_logical_translators_registered() {
        let registry = TranslatorRegistry::new();

        assert!(registry.is_supported("Equal"));
        assert!(registry.is_supported("Greater"));
        assert!(registry.is_supported("Less"));
        assert!(registry.is_supported("GreaterOrEqual"));
        assert!(registry.is_supported("LessOrEqual"));
        assert!(registry.is_supported("And"));
        assert!(registry.is_supported("Or"));
        assert!(registry.is_supported("Not"));
        assert!(registry.is_supported("Where"));

        let equal = registry.get("Equal");
        assert!(equal.is_some());
        assert_eq!(equal.unwrap().onnx_op_type(), "Equal");
    }

    #[test]
    fn test_advanced_translators_registered() {
        let registry = TranslatorRegistry::new();

        assert!(registry.is_supported("Cast"));
        assert!(registry.is_supported("Range"));
        assert!(registry.is_supported("Trilu"));
        assert!(registry.is_supported("Dropout"));

        let cast = registry.get("Cast");
        assert!(cast.is_some());
        assert_eq!(cast.unwrap().onnx_op_type(), "Cast");
    }

    #[test]
    fn test_resize_translators_registered() {
        let registry = TranslatorRegistry::new();

        assert!(registry.is_supported("Resize"));
        assert!(registry.is_supported("Upsample"));
        assert!(registry.is_supported("DepthToSpace"));
        assert!(registry.is_supported("SpaceToDepth"));

        let resize = registry.get("Resize");
        assert!(resize.is_some());
        assert_eq!(resize.unwrap().onnx_op_type(), "Resize");
    }

    #[test]
    fn test_pad_translators_registered() {
        let registry = TranslatorRegistry::new();

        assert!(registry.is_supported("Pad"));

        let pad = registry.get("Pad");
        assert!(pad.is_some());
        assert_eq!(pad.unwrap().onnx_op_type(), "Pad");
    }
}
