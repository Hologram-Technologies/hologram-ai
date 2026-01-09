> **Status:** Not Started | **Verified:** 2026-01-09
>
> Refactoring proposal. Not implemented.

# Plan: ONNX Translator Traits for hologram-onnx

## Problem Statement

The `hologram-onnx` crate translates ONNX models to hologram IR. Currently, this translation logic is scattered across multiple files with large match statements:

### Current Files with ONNX Translation Logic
- `src/ops/translator.rs` - Main ONNX node to NodeOp translation (50+ match arms)
- `src/ops/activation.rs` - Activation operation handlers
- `src/ops/binary.rs` - Binary operation handlers
- `src/ops/reduction.rs` - Reduction operation handlers
- `src/ops/shape.rs` - Shape operation handlers
- `src/ops/indexing.rs` - Gather, Slice, etc.
- `src/ops/norm.rs` - Normalization operations
- `src/ops/conv.rs` - Convolution operations
- `src/ops/advanced.rs` - Cast, Range, etc.
- `src/core/translator.rs` - ONNX graph-level translation

This leads to:
1. Large match statements that are hard to maintain
2. Inconsistent handling of attributes across operations
3. Difficult to test individual operation translators in isolation
4. New ONNX operations require changes to multiple files

---

## Core Traits

### Primary Translation Trait

```rust
use hologram_ir::{NodeOp, IrBuilder, Shape};
use crate::onnx::{NodeProto, AttributeProto};

/// Trait for translating an ONNX operation to hologram IR
pub trait OnnxTranslator: std::fmt::Debug + Send + Sync {
    /// ONNX operation type name (e.g., "MatMul", "Relu", "Conv")
    fn onnx_op_type(&self) -> &'static str;

    /// Translate an ONNX node to hologram IR NodeOp
    ///
    /// # Arguments
    /// * `node` - The ONNX node to translate
    /// * `input_shapes` - Shapes of the input tensors
    /// * `builder` - IR builder for creating nodes
    ///
    /// # Returns
    /// The translated NodeOp or an error
    fn translate(
        &self,
        node: &NodeProto,
        input_shapes: &[&Shape],
        builder: &mut IrBuilder,
    ) -> Result<NodeOp, TranslationError>;

    /// Number of required inputs for this operation
    fn required_inputs(&self) -> InputRequirement;

    /// Number of outputs this operation produces
    fn num_outputs(&self) -> usize { 1 }

    /// Whether this operation supports constant folding at translation time
    fn supports_constant_folding(&self) -> bool { false }

    /// Attempt to constant-fold if all inputs are constants
    ///
    /// Returns None if constant folding is not possible or not supported.
    fn constant_fold(
        &self,
        _node: &NodeProto,
        _constant_inputs: &[&[u8]],
        _input_shapes: &[&Shape],
    ) -> Option<Vec<u8>> {
        None
    }
}

/// Input requirement specification
#[derive(Debug, Clone, Copy)]
pub enum InputRequirement {
    /// Exact number of inputs required
    Exact(usize),
    /// Range of valid input counts (min, max)
    Range(usize, usize),
    /// At least N inputs required
    AtLeast(usize),
    /// Any number of inputs
    Variadic,
}
```

### Attribute Extraction Trait

```rust
/// Trait for extracting ONNX attributes with type safety
pub trait OnnxAttributes {
    /// Get an integer attribute
    fn get_int(&self, name: &str) -> Option<i64>;

    /// Get an integer attribute with default
    fn get_int_or(&self, name: &str, default: i64) -> i64 {
        self.get_int(name).unwrap_or(default)
    }

    /// Get a float attribute
    fn get_float(&self, name: &str) -> Option<f32>;

    /// Get a float attribute with default
    fn get_float_or(&self, name: &str, default: f32) -> f32 {
        self.get_float(name).unwrap_or(default)
    }

    /// Get a string attribute
    fn get_string(&self, name: &str) -> Option<&str>;

    /// Get an integer array attribute
    fn get_ints(&self, name: &str) -> Option<&[i64]>;

    /// Get an integer array with default
    fn get_ints_or<'a>(&'a self, name: &str, default: &'a [i64]) -> &'a [i64];

    /// Get a float array attribute
    fn get_floats(&self, name: &str) -> Option<&[f32]>;

    /// Get a tensor attribute
    fn get_tensor(&self, name: &str) -> Option<&TensorProto>;
}

impl OnnxAttributes for NodeProto {
    fn get_int(&self, name: &str) -> Option<i64> {
        self.attribute.iter()
            .find(|a| a.name == name)
            .and_then(|a| if a.r#type == AttributeType::Int as i32 {
                Some(a.i)
            } else {
                None
            })
    }

    fn get_float(&self, name: &str) -> Option<f32> {
        self.attribute.iter()
            .find(|a| a.name == name)
            .and_then(|a| if a.r#type == AttributeType::Float as i32 {
                Some(a.f)
            } else {
                None
            })
    }

    fn get_string(&self, name: &str) -> Option<&str> {
        self.attribute.iter()
            .find(|a| a.name == name)
            .and_then(|a| if a.r#type == AttributeType::String as i32 {
                std::str::from_utf8(&a.s).ok()
            } else {
                None
            })
    }

    fn get_ints(&self, name: &str) -> Option<&[i64]> {
        self.attribute.iter()
            .find(|a| a.name == name)
            .and_then(|a| if a.r#type == AttributeType::Ints as i32 {
                Some(a.ints.as_slice())
            } else {
                None
            })
    }

    fn get_ints_or<'a>(&'a self, name: &str, default: &'a [i64]) -> &'a [i64] {
        self.get_ints(name).unwrap_or(default)
    }

    fn get_floats(&self, name: &str) -> Option<&[f32]> {
        self.attribute.iter()
            .find(|a| a.name == name)
            .and_then(|a| if a.r#type == AttributeType::Floats as i32 {
                Some(a.floats.as_slice())
            } else {
                None
            })
    }

    fn get_tensor(&self, name: &str) -> Option<&TensorProto> {
        self.attribute.iter()
            .find(|a| a.name == name)
            .and_then(|a| a.t.as_ref())
    }
}
```

### Shape Inference Trait

```rust
/// Trait for ONNX operation shape inference
pub trait OnnxShapeInference {
    /// Infer output shape from input shapes and attributes
    fn infer_shape(
        &self,
        node: &NodeProto,
        input_shapes: &[&Shape],
    ) -> Result<Vec<Shape>, ShapeError>;
}
```

---

## Example Translator Implementations

### Simple Unary Operation (Relu)

```rust
/// Translator for ONNX Relu operation
#[derive(Debug, Default)]
pub struct ReluTranslator;

impl OnnxTranslator for ReluTranslator {
    fn onnx_op_type(&self) -> &'static str { "Relu" }

    fn required_inputs(&self) -> InputRequirement { InputRequirement::Exact(1) }

    fn translate(
        &self,
        _node: &NodeProto,
        _input_shapes: &[&Shape],
        _builder: &mut IrBuilder,
    ) -> Result<NodeOp, TranslationError> {
        Ok(NodeOp::Relu)
    }

    fn supports_constant_folding(&self) -> bool { true }

    fn constant_fold(
        &self,
        _node: &NodeProto,
        constant_inputs: &[&[u8]],
        _input_shapes: &[&Shape],
    ) -> Option<Vec<u8>> {
        let input = constant_inputs.first()?;
        // Interpret as f32 slice
        let floats: &[f32] = bytemuck::cast_slice(input);
        let result: Vec<f32> = floats.iter().map(|x| x.max(0.0)).collect();
        Some(bytemuck::cast_slice(&result).to_vec())
    }
}

impl OnnxShapeInference for ReluTranslator {
    fn infer_shape(
        &self,
        _node: &NodeProto,
        input_shapes: &[&Shape],
    ) -> Result<Vec<Shape>, ShapeError> {
        // Relu preserves input shape
        Ok(vec![input_shapes[0].clone()])
    }
}
```

### Binary Operation (Add)

```rust
/// Translator for ONNX Add operation
#[derive(Debug, Default)]
pub struct AddTranslator;

impl OnnxTranslator for AddTranslator {
    fn onnx_op_type(&self) -> &'static str { "Add" }

    fn required_inputs(&self) -> InputRequirement { InputRequirement::Exact(2) }

    fn translate(
        &self,
        _node: &NodeProto,
        _input_shapes: &[&Shape],
        _builder: &mut IrBuilder,
    ) -> Result<NodeOp, TranslationError> {
        Ok(NodeOp::Add)
    }

    fn supports_constant_folding(&self) -> bool { true }

    fn constant_fold(
        &self,
        _node: &NodeProto,
        constant_inputs: &[&[u8]],
        _input_shapes: &[&Shape],
    ) -> Option<Vec<u8>> {
        let a: &[f32] = bytemuck::cast_slice(constant_inputs.get(0)?);
        let b: &[f32] = bytemuck::cast_slice(constant_inputs.get(1)?);
        if a.len() != b.len() { return None; }
        let result: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
        Some(bytemuck::cast_slice(&result).to_vec())
    }
}

impl OnnxShapeInference for AddTranslator {
    fn infer_shape(
        &self,
        _node: &NodeProto,
        input_shapes: &[&Shape],
    ) -> Result<Vec<Shape>, ShapeError> {
        // Broadcasting rules
        let a = input_shapes.get(0).ok_or(ShapeError::MissingInput)?;
        let b = input_shapes.get(1).ok_or(ShapeError::MissingInput)?;
        Ok(vec![Shape::broadcast(a, b)?])
    }
}
```

### Complex Operation with Attributes (Conv)

```rust
/// Translator for ONNX Conv operation
#[derive(Debug, Default)]
pub struct ConvTranslator;

impl OnnxTranslator for ConvTranslator {
    fn onnx_op_type(&self) -> &'static str { "Conv" }

    fn required_inputs(&self) -> InputRequirement { InputRequirement::Range(2, 3) }

    fn translate(
        &self,
        node: &NodeProto,
        input_shapes: &[&Shape],
        _builder: &mut IrBuilder,
    ) -> Result<NodeOp, TranslationError> {
        let input_shape = input_shapes.get(0)
            .ok_or(TranslationError::MissingInput)?;
        let weight_shape = input_shapes.get(1)
            .ok_or(TranslationError::MissingInput)?;

        // Extract attributes with defaults
        let kernel_shape = node.get_ints("kernel_shape")
            .ok_or(TranslationError::MissingAttribute("kernel_shape"))?;
        let strides = node.get_ints_or("strides", &[1, 1]);
        let pads = node.get_ints_or("pads", &[0, 0, 0, 0]);
        let dilations = node.get_ints_or("dilations", &[1, 1]);
        let group = node.get_int_or("group", 1) as usize;

        // Extract dimensions
        let batch = input_shape.dims().get(0)
            .and_then(|d| d.static_value())
            .unwrap_or(1);
        let channels = input_shape.dims().get(1)
            .and_then(|d| d.static_value())
            .unwrap_or(1);
        let height = input_shape.dims().get(2)
            .and_then(|d| d.static_value())
            .unwrap_or(1);
        let width = input_shape.dims().get(3)
            .and_then(|d| d.static_value())
            .unwrap_or(1);
        let filters = weight_shape.dims().get(0)
            .and_then(|d| d.static_value())
            .unwrap_or(1);

        Ok(NodeOp::Conv2D {
            input_h: height,
            input_w: width,
            kernel_h: kernel_shape[0] as usize,
            kernel_w: kernel_shape[1] as usize,
            channels,
            filters,
            stride: strides[0] as usize,
            padding: pads[0] as usize,
            groups: group,
        })
    }
}

impl OnnxShapeInference for ConvTranslator {
    fn infer_shape(
        &self,
        node: &NodeProto,
        input_shapes: &[&Shape],
    ) -> Result<Vec<Shape>, ShapeError> {
        let input = input_shapes.get(0).ok_or(ShapeError::MissingInput)?;
        let weight = input_shapes.get(1).ok_or(ShapeError::MissingInput)?;

        let kernel_shape = node.get_ints("kernel_shape")
            .ok_or(ShapeError::MissingAttribute)?;
        let strides = node.get_ints_or("strides", &[1, 1]);
        let pads = node.get_ints_or("pads", &[0, 0, 0, 0]);
        let dilations = node.get_ints_or("dilations", &[1, 1]);

        // NCHW format
        let n = input.dims().get(0).cloned().unwrap_or(Dim::fixed(1));
        let c_out = weight.dims().get(0).cloned().unwrap_or(Dim::fixed(1));

        // Calculate output spatial dimensions
        let h_in = input.dims().get(2).and_then(|d| d.static_value()).unwrap_or(1) as i64;
        let w_in = input.dims().get(3).and_then(|d| d.static_value()).unwrap_or(1) as i64;

        let h_out = (h_in + pads[0] + pads[2] - dilations[0] * (kernel_shape[0] - 1) - 1) / strides[0] + 1;
        let w_out = (w_in + pads[1] + pads[3] - dilations[1] * (kernel_shape[1] - 1) - 1) / strides[1] + 1;

        Ok(vec![Shape::from_dims(vec![n, c_out, Dim::fixed(h_out as usize), Dim::fixed(w_out as usize)])])
    }
}
```

### Operation with Constant Folding (Shape)

```rust
/// Translator for ONNX Shape operation
#[derive(Debug, Default)]
pub struct ShapeTranslator;

impl OnnxTranslator for ShapeTranslator {
    fn onnx_op_type(&self) -> &'static str { "Shape" }

    fn required_inputs(&self) -> InputRequirement { InputRequirement::Exact(1) }

    fn translate(
        &self,
        node: &NodeProto,
        input_shapes: &[&Shape],
        builder: &mut IrBuilder,
    ) -> Result<NodeOp, TranslationError> {
        let input_shape = input_shapes.get(0)
            .ok_or(TranslationError::MissingInput)?;

        // Shape op always constant-folds at translation time
        let start = node.get_int_or("start", 0) as usize;
        let end = node.get_int("end")
            .map(|e| e as usize)
            .unwrap_or(input_shape.dims().len());

        let shape_values: Vec<i64> = input_shape.dims()
            .iter()
            .skip(start)
            .take(end - start)
            .map(|d| d.static_value().unwrap_or(1) as i64)
            .collect();

        // Return as constant
        Ok(NodeOp::Constant {
            data: bytemuck::cast_slice(&shape_values).to_vec(),
        })
    }

    fn supports_constant_folding(&self) -> bool { true }
}

impl OnnxShapeInference for ShapeTranslator {
    fn infer_shape(
        &self,
        node: &NodeProto,
        input_shapes: &[&Shape],
    ) -> Result<Vec<Shape>, ShapeError> {
        let input_shape = input_shapes.get(0).ok_or(ShapeError::MissingInput)?;
        let start = node.get_int_or("start", 0) as usize;
        let end = node.get_int("end")
            .map(|e| e as usize)
            .unwrap_or(input_shape.dims().len());

        let output_len = end.saturating_sub(start);
        Ok(vec![Shape::from_dims(vec![Dim::fixed(output_len)])])
    }
}
```

---

## Translator Registry

```rust
use std::collections::HashMap;
use std::sync::Arc;

/// Registry of ONNX translators
pub struct TranslatorRegistry {
    translators: HashMap<&'static str, Arc<dyn OnnxTranslator>>,
}

impl TranslatorRegistry {
    /// Create a new registry with all built-in translators
    pub fn new() -> Self {
        let mut registry = Self {
            translators: HashMap::new(),
        };

        // Register all built-in translators
        registry.register(Arc::new(ReluTranslator));
        registry.register(Arc::new(SigmoidTranslator));
        registry.register(Arc::new(TanhTranslator));
        registry.register(Arc::new(GeluTranslator));
        registry.register(Arc::new(SiluTranslator));

        registry.register(Arc::new(AddTranslator));
        registry.register(Arc::new(SubTranslator));
        registry.register(Arc::new(MulTranslator));
        registry.register(Arc::new(DivTranslator));
        registry.register(Arc::new(PowTranslator));

        registry.register(Arc::new(MatMulTranslator));
        registry.register(Arc::new(GemmTranslator));
        registry.register(Arc::new(ConvTranslator));

        registry.register(Arc::new(ReduceSumTranslator));
        registry.register(Arc::new(ReduceMeanTranslator));
        registry.register(Arc::new(ReduceMaxTranslator));
        registry.register(Arc::new(ReduceMinTranslator));

        registry.register(Arc::new(ReshapeTranslator));
        registry.register(Arc::new(TransposeTranslator));
        registry.register(Arc::new(SqueezeTranslator));
        registry.register(Arc::new(UnsqueezeTranslator));

        registry.register(Arc::new(GatherTranslator));
        registry.register(Arc::new(SliceTranslator));
        registry.register(Arc::new(ConcatTranslator));
        registry.register(Arc::new(SplitTranslator));

        registry.register(Arc::new(ShapeTranslator));
        registry.register(Arc::new(CastTranslator));
        registry.register(Arc::new(ConstantOfShapeTranslator));
        registry.register(Arc::new(RangeTranslator));

        registry.register(Arc::new(SoftmaxTranslator));
        registry.register(Arc::new(LayerNormTranslator));
        registry.register(Arc::new(BatchNormTranslator));

        // ... register all others

        registry
    }

    /// Register a translator
    pub fn register(&mut self, translator: Arc<dyn OnnxTranslator>) {
        self.translators.insert(translator.onnx_op_type(), translator);
    }

    /// Get a translator for an operation type
    pub fn get(&self, op_type: &str) -> Option<&Arc<dyn OnnxTranslator>> {
        self.translators.get(op_type)
    }

    /// Translate an ONNX node
    pub fn translate(
        &self,
        node: &NodeProto,
        input_shapes: &[&Shape],
        builder: &mut IrBuilder,
    ) -> Result<NodeOp, TranslationError> {
        let translator = self.get(&node.op_type)
            .ok_or_else(|| TranslationError::UnsupportedOp(node.op_type.clone()))?;

        // Validate input count
        let input_count = input_shapes.len();
        match translator.required_inputs() {
            InputRequirement::Exact(n) if input_count != n => {
                return Err(TranslationError::WrongInputCount {
                    op: node.op_type.clone(),
                    expected: n,
                    got: input_count,
                });
            }
            InputRequirement::Range(min, max) if input_count < min || input_count > max => {
                return Err(TranslationError::WrongInputCount {
                    op: node.op_type.clone(),
                    expected: min,
                    got: input_count,
                });
            }
            InputRequirement::AtLeast(min) if input_count < min => {
                return Err(TranslationError::WrongInputCount {
                    op: node.op_type.clone(),
                    expected: min,
                    got: input_count,
                });
            }
            _ => {}
        }

        translator.translate(node, input_shapes, builder)
    }
}
```

---

## File Structure After Refactor

```
src/
├── translators/
│   ├── mod.rs           # TranslatorRegistry, re-exports
│   ├── traits.rs        # OnnxTranslator, OnnxAttributes, OnnxShapeInference
│   ├── error.rs         # TranslationError, ShapeError
│   ├── activation.rs    # ReluTranslator, SigmoidTranslator, etc.
│   ├── binary.rs        # AddTranslator, SubTranslator, MulTranslator, etc.
│   ├── matmul.rs        # MatMulTranslator, GemmTranslator
│   ├── conv.rs          # ConvTranslator, ConvTransposeTranslator
│   ├── reduce.rs        # ReduceSumTranslator, ReduceMeanTranslator, etc.
│   ├── view.rs          # ReshapeTranslator, TransposeTranslator, etc.
│   ├── gather.rs        # GatherTranslator, GatherNDTranslator
│   ├── slice.rs         # SliceTranslator, SliceDynamicTranslator
│   ├── concat.rs        # ConcatTranslator, SplitTranslator
│   ├── shape_ops.rs     # ShapeTranslator, CastTranslator, RangeTranslator
│   ├── norm.rs          # LayerNormTranslator, BatchNormTranslator
│   ├── attention.rs     # AttentionTranslator (for fused attention)
│   └── recurrent.rs     # LSTMTranslator, GRUTranslator
├── core/
│   └── translator.rs    # Simplified: uses TranslatorRegistry
└── ops/                 # Deprecated - will be removed
```

---

## Migration Strategy

### Phase 1: Create Trait Infrastructure
1. Define `OnnxTranslator`, `OnnxAttributes`, `OnnxShapeInference` in `src/translators/traits.rs`
2. Create `TranslatorRegistry` in `src/translators/mod.rs`
3. Create `TranslationError` types in `src/translators/error.rs`

### Phase 2: Migrate Translators by Category
Priority order (simplest to most complex):
1. **Activation ops**: Relu, Sigmoid, Tanh, Gelu, Silu, LeakyRelu
2. **Binary ops**: Add, Sub, Mul, Div, Pow, Max, Min
3. **Unary math**: Abs, Neg, Sqrt, Exp, Log, Sin, Cos, Tan
4. **Shape ops**: Shape, Cast, Constant, ConstantOfShape, Range
5. **View ops**: Reshape, Transpose, Squeeze, Unsqueeze, Flatten
6. **Indexing ops**: Gather, GatherND, Slice, Split, Concat
7. **Reduction ops**: ReduceSum, ReduceMean, ReduceMax, ReduceMin, ReduceProd
8. **Linear ops**: MatMul, Gemm
9. **Conv ops**: Conv, ConvTranspose, MaxPool, AveragePool
10. **Normalization**: LayerNorm, BatchNorm, InstanceNorm, GroupNorm
11. **Attention**: Attention (fused), MultiHeadAttention
12. **Recurrent**: LSTM, GRU

### Phase 3: Update Core Translator
1. Replace match statements in `src/core/translator.rs` with registry lookups
2. Remove old `src/ops/` directory

### Phase 4: Add Comprehensive Tests
1. Unit test each translator individually
2. Test constant folding where applicable
3. Test shape inference
4. Integration tests with real ONNX models

---

## Benefits

1. **Single Source of Truth**: Each ONNX operation's translation logic is in one file
2. **Easy to Add Operations**: Create a new translator struct, implement traits, register
3. **Testability**: Each translator can be unit tested independently
4. **Consistent Attribute Handling**: `OnnxAttributes` trait ensures type-safe attribute extraction
5. **Clear Constant Folding**: Each op explicitly declares if it supports constant folding
6. **Better Error Messages**: Structured errors with context

---

## Testing Strategy

Each translator should have:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu_translate() {
        let translator = ReluTranslator;
        let node = create_test_node("Relu", &[], &[]);
        let input_shapes = vec![&Shape::from_dims(vec![1, 128])];

        let mut builder = IrBuilder::new();
        let result = translator.translate(&node, &input_shapes, &mut builder);

        assert!(matches!(result, Ok(NodeOp::Relu)));
    }

    #[test]
    fn test_relu_constant_fold() {
        let translator = ReluTranslator;
        let input: Vec<f32> = vec![-1.0, 0.0, 1.0, 2.0];
        let input_bytes = bytemuck::cast_slice(&input);

        let result = translator.constant_fold(
            &create_test_node("Relu", &[], &[]),
            &[input_bytes],
            &[&Shape::from_dims(vec![4])],
        );

        let expected: Vec<f32> = vec![0.0, 0.0, 1.0, 2.0];
        assert_eq!(result, Some(bytemuck::cast_slice(&expected).to_vec()));
    }

    #[test]
    fn test_relu_shape_inference() {
        let translator = ReluTranslator;
        let input_shape = Shape::from_dims(vec![1, 64, 128]);

        let result = translator.infer_shape(
            &create_test_node("Relu", &[], &[]),
            &[&input_shape],
        );

        assert_eq!(result.unwrap()[0], input_shape);
    }
}
```

---

## Success Criteria

1. All existing ONNX models compile correctly
2. No match statements on op types in translation code
3. Each translator has comprehensive unit tests
4. Constant folding tests for all ops that support it
5. Shape inference tests for all ops
6. Adding a new ONNX operation requires only one new file
7. Clear documentation for each translator

---

## Related Documents

- **Hologram Trait Refactor**: See `/hologram/specs/plans/22-opnode-trait-refactor.md` for the compiler-side traits (OpNode, NodeOp)
