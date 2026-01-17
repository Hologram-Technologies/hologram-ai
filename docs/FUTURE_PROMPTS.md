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

Can we define the sections with a trait so if there are other things we need to embed they can just implement the trait

---

In `hologram` we support networking (investigate `hologram-network`). One of the features we're trying to realize is that we can have distributed compute and distributed storage. We want to be able to support subgraphs as well, so that we can have intelligent compilation and distributed computing. Subgraphs enable "docker-like" layer support (layers being groups of computation/storage that other inputs can depend upon). 

When we compile we want to have groups where dependencies are resolved at compile time and at runtime we should be able to execute groups of computation in a multi-thread way so that we have faster execution overall. Will our new graph engine be able to support this?

Remember, O(1) (with `hologram`), zero-copy (all data is located in a specific plane), and as minimal runtime performance interaction as much as possible.

A hard requirement is that our graphs can be composed of multiple subgraphs. Subgraphs are basically other graphs that have already been compiled (either previously in another `.holo` file or simultaneously compiled at the same time). We must support subgraphs.

---

We need to make sure we still have traits that define what can go into a `.holo` file. We implemented this in a previous section, but I believe it's in this crate where I actually think we might want to move that functionality into `hologram` and `hologram-compiler` rather than in here.

---

And once the T5 compilations and execution works, I want to run it with the `--profile` so we can see where we can optimize.

---

Can you help identify places for optimization using instrumentation. Give me a report of where the most time was spent in all the functions we're running so we can target finding and squeezing out more performance gains.

We have instrumentation implemented here.

---

We're working on the `hologram-ai-onnx` and `hologram-ai-gguf` crate here. For this session you have write read and write access to `/hologram` (the locaton of the `hologram` crate for now)

We don't really want to do much custom work inside this crate. All the work we should be doing in this crate is map to operations in the `hologram` dependency. If we need to create fused operations in the `hologram` crate, you should. The rationale here is that `hologram` should be the crate that contains all the low-level operatons supported by the ultra fast backends. 

The whole point of this `hologram-ai-onnx` and `hologram-ai-gguf` crates are to just handle the mapping. *IF* there is a case where operations are just for ai work, that's what should be contained either in `hologram-ai`/`hologram-ai-common` (or a new crate, where it makes sense `hologram-ai-operations`), then it needs to be contained in this library crate. Any operations described here have to be _fused_ operations that sit inside the `OperationGraph` that `hologram` depends upon.

This library is just an implementation of the compiler here (our `hologram-ai`) that takes advantage of the IR Graph (`OperationGraph`) that's exposed to `hologram` and runs through the entire chain (as described in `/hologram/crates/compiler/README.md`). 

Can you examine this crate and all the code we have here and tell me how far we've drifted off this idea, if we have and create a plan that shrinks that gap?

---

HOLOGRAM_TRACE_OPS=1 RUST_BACKTRACE=1 cargo run -p hologram-ai --release -- run-pipeline models/t5-small/t5-pipeline-new.holo --prompt "tell me a joke" --max-tokens 1 > /workspace/tmp-run-manual.log 2>&1

---

I want you to delete all the legacy `.holo` files. 

The goal of all of this is to use the latest `hologram` pipeline with layers and all the updated workflow. Can you please try to update this crate to make sure we can run models with memory-mapped weights using our `EmbeddableSection` as well as the tokenizer, etc. The `LayerHeader` for running the model is in the entrypoint of our model (either onnx or gguf).

We want to take advantage of the performance runtime of `hologram` and run onnx models atop the computational runtime.

Can you explore this approach with the latest updated `hologram` library crate.

---

I want you to keep integrating on quality. 

Add beam search with length penalty and no‑repeat‑ngram to stabilize outputs.
Align SentencePiece normalization with tokenizer.json normalizer sequence (implement NFKC/StripAccents support).
Add vocab/logits filtering to exclude <unk>/special tokens during sampling and force EOS only after an “end‑of‑sentence” probability threshold.

---

Does `hologram` have traits? In the ideal world, we would have consumers of that `hologram` as a dependency be able to take advantage of the `hologram` compiler and all the optimizations in there, but have external crates (like this one) be able to define their own individual options. `hologram-ai` for onnx/gguf/(others?) and hologram-python, hologram-typescript are other examples that shouldn't touch the `hologram` compiler

---

We're not loading from a `.weights` file though, we're embedding those weights in the `.holo` archive. Can you confirm this? That's what the `EmbeddableSection` trait should be doing.

I don't want you to default a sequence length. Why would we do that? We want to preserve symbolic dimensions if there are none in the compiled.

---

How can we speed up tose computational costs though? The compiler generates a graph which should enable us to process subgraphs in parallel... what can we do to speed-up those computational costs with this parallel nature of our compled graph?

---

I want you to make the real fix... I prefer propagating symbolic dimensions through shape inference, but if we need to have a dynamic workspace allocation at runtime that seems like a sensible fix

---

Since we have operation and symbolic shape support in `hologram`,  my expectation for these graphs is that we could just use those operations to satisfy the graph naturally. Why are we having such a difficult time getting these graphs to run. Symbolic shapes and dynamic dimensions should allow us to map these graph operations naturally to those that are supported in `hologram`, right?

---

Next Steps
Investigate why execution is slow - there might be workspace allocation or kernel execution issues
Continue with Phase 2 - fix Split, Reshape, Transpose translators to handle symbolic dims gracefully
Consider architecture improvements - extend DimExpr to kernel parameters for true runtime resolution
