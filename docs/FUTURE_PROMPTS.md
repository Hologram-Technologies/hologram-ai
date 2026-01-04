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

Notes like this need to be finished. I need you to actually implement the code, not just leave notes about simplified versions. I don't want simplified versions, I want full implementations. If they are no longer necessary, just remove the functions.

Any notes like so indicate failure. 

//! These operations are not yet implemented in the simplified version and will
//! return errors when encountered during translation.

There should be no simplified versions, placeholders, todos, etc.

/// Translate ONNX DepthToSpace operation.
///
/// DepthToSpace rearranges data from the depth dimension into spatial dimensions,
/// commonly used in super-resolution networks.
///
/// # Note
/// Not implemented in simplified version.