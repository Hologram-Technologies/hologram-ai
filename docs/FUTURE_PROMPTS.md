The `hologram` and it's `hologram-backend` dependency expose an ISA, are we using that with the onnx graph? We have a lot of optimizations there that we want to take advantage of.

We also need to make sure we compile not just the graph, but the weights as well that can be used at runtime. We need to make sure we don't use too much memory in this process also so we don't hit an OOM error.

We also want to provide an output trait as different onnx graphs have different outputs. For instance there are ai models that are text->text, text->image, text->audio, etc. etc. We want to support all of those output shapes.

Ideally we can configure all the runtime information in a config file. In a previous iteration, we had this working. I've added a doc that we used in the previous iteration of this project (don't need to create the same files, etc.) so use it to inform what we're doing with the configuration file. (@docs/config-output.md). I also added the @docs/output-handlers.md documenting the output handlers.

For loops, we have done some serious work in the `hologram` crate to take those loops from O^5-O^8 to O(1) (nested loops translated into a single operation/set of operations). 

We have fusion implemented in `hologram`, can we take advantage of that work in this crate?

We need to support symbolic shapes as we want to make sure we support any size and shape of input. This is a CRITICAL design. We fail if this doesn't work.

We have a @docs/graph-partitioning.md doc that describes how we can nest large graphs into multiple nodes.

For the `hologram-onnx-cli` can we have a download model as well?