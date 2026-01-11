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

You should be able to download `bert` with the `cli` -- can yo u

---

Why is it so hard to get the onnx model working? All the hard work is done in `hologram`. The flow should go like this: 

ONNX graph model is compiled by `hologram-ai` (this crate) which just generates a graph of an `OperationGraph` which is turned into a `CompileGraph`. That `CompileGraph` should carry the operations and the weights serialized. When it's executed, `hologram-ai` or this library (`hologram-ai`) should execute the model for each node of the graph where it "starts."

Right now we have to reimplement all of the operations constantly to find runtime bugs. I think one issue is having dynamic dimensions (symbolic shapes), but it doesn't make sense to me that we have to constantly reimplement those functions that are defined by `hologram` and the `hologram_backends` (exposed by `hologram`).

Help me figure out how we fix this or if my understanding of what we are doing is incorrect.

---

I'd like a cli command to extract a `.holo` bundle into multiple `.holo` files and everything that's embedded in 1. For instance, if our `.holo` archive contains 2 `.holo` files and the weights, this `extract-bundle` command should unarchive it into 2 `.holo` files and a weights file

---

Then I want to try to download (with the cli), compile (bundle), and execute the GPT-2 model and ask it to tell me a joke and receive the response in English.

---

BERT End-to-End Success
Stage	Result
Compilation	ONNX → .holo bundle (522 MB with embedded weights)
Loading	HOLB format auto-detected, mmap'd weights
Execution	828 operations executed on CPU
Output	[1, 1, 512, 28996] - vocab logits per position
Runtime stats:

Workspace allocated: 926 MB
Operations: 828 (199 single-input, 580 dual-input, 49 multi-input)
Execution time: ~48 seconds (CPU, unoptimized build)
Output elements: 14.8M (25% non-zero)
Sample output values: [4.77, 4.08, 4.00, 4.62, 4.30, ...]

The full ONNX → compile → execute pipeline works. The model output shape [1, 1, 512, 28996] represents vocabulary logits (28,996 vocab size) for each of the 512 positions - consistent with BERT's masked language model head.


48 second runtime is FAR too slow.

---

I want to try to download (with the cli), compile (bundle), and execute the GPT-2 model and ask it to tell me a joke and receive the response in English.

---

time cargo test -p hologram-ai-onnx test_bert_mask_prediction -- --nocapture 2>&1

---

Can we define the sections with a trait so if there are other things we need to embed they can just implement the trait?