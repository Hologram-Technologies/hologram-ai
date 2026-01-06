Please make sure that the intent of this library crate implements all the features as described in @docs/plans/onnx-integration.md 

The high-level goal is to be able to take an onnx graph and compile it to a `.holo` file (`OperationGraph`). We want to be able to run text-to-text and text-to-image onnx graphs (arbitrary onnx graphs) utilizing the `hologram` architecture.

We want to be able to accept arbitrary data size and shape (symbolic shapes). 

We want to be able to serialize the weights and still keep all the compilation using lower memory and being able to be referenced at runtime efficiently.

Since we're refactoring this project from the ground-up, I want you to make sure we're cleaned up and simplified as much as possible.

I also want you to ensure we have as close to 100% of test coverage as well as integration tests.

Implement ALL stubbed code, don't leave any TODOS or placeholders. Rewrite all the ignored tests or remove them completely if they don't matter anymore, are outdated, or don't test anything relevant anymore.

---

Please add benchmarks that show how fast and how much we can run successfully.

---

I want you to look at the operations you just implemented here and look for corresponding operations that have now been implemented in `hologram` so that you don't have to re-implement them here

---

Should I implement the updates to the `hologram` library using a prompt for the `hologram` library instead of you directory adding them?

I don't actually think we need those wrappers in the `hologram-backend` crate in the first place. Shouldn't those wrappers be implemented here, not in the dependency?

---

# Hologram-ONNX

Convert ONNX models to Hologram IR format.

## Implementation Requirements

1. **ONNX Parsing**: Use `prost` or `onnx-rs` to parse ONNX protobuf
2. **Operator Mapping**: Convert ONNX ops to `hologram_ir::NodeOp` variants:
   - Conv → Conv2D
   - Gemm/MatMul → MatMul
   - Relu, Sigmoid, Tanh → corresponding activations
   - BatchNormalization → fused ops
3. **Graph Building**: Use `hologram_ir::GraphBuilder` to construct `OperationGraph`
4. **Output Options**:
   - `convert`: Save as `.ir` file (serialized OperationGraph)
   - `compile`: Directly compile to `.holo` file using `hologram_compiler::compile_ir()`

## CLI Commands

```bash
hologram-onnx convert model.onnx -o model.ir
hologram-onnx compile model.onnx -o model.holo --backend cpu
