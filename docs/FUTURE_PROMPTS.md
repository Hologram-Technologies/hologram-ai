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

Now models themselves have weights, which can be really large. We need to be able to take weights and serialize them using memory maps so we don't have to load all the weights into memory when running a model. For big models, like Stable Diffusion we have Gb of weight files. We need to make sure this works for large models.

---

We need organization in our `docs/` folder.

---

One thing that I want to include here is that the `.holo` format is an archive, so we can compile all of those required models and weights and tokenizer into a single `.holo` file. Can you help update the compilation and execution to be able to support this?

---

Shouldn't this run faster than it actually is? Everything in `hologram` is a O(1) lookup with zero-copy and has minimal runtime overhead, but yet this `cargo run --release -- run --config configs/t5-generate.toml` command takes a long time (I'm timing it with the `time` command)

Shouldn't it execute a LOT faster than seconds/minutes?

The purpose of this is to build compiled `.holo` files and have it run on `hologram` which obeys the O(1), constant-time lookup of data, and `hologram` benchmarks say these operations take much less time (so far my command is still running... and it's been 5 minutes and it's still running). It looks like it's never ending

We also want to take advantage of the archive pipeline of `.holo` files so that when we compile with or without a config file, it builds 1 single `.holo` file which contains all of the onnx model and weights that are memory-mapped.

---

Can you continue debugging, compiling, and running the T5 model that gives us the response in english from T5 execution when I send in a prompt asking for it to generate a joke

---

If this doesn't work in the next few attempts, can you pause so we can restructure this crate entirely. But I'd prefer to get T5 working before a big restructure, but 

---

I would prefer that any model can be built using the `hologram-ai` tools without having to generate a custom builder for each one. So I don't want a `Qwen2Builder`, `LlamaBuilder`, etc. I want to be able to read the formats and build the `OperationGraph` through the tooling provided in this crate already.