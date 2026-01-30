# Hologram-AI Integration Specification

**Version:** 1.0
**Date:** 2026-01-27
**Audience:** hologram-ai project team
**Prerequisites:** Read [integration-guide.md](integration-guide.md) first

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [AI/ML-Specific Patterns](#2-aiml-specific-patterns)
3. [Weight Management for ML Models](#3-weight-management-for-ml-models)
4. [Common ML Operations](#4-common-ml-operations)
5. [Integration with Autograd](#5-integration-with-autograd)
6. [Deployment Patterns](#6-deployment-patterns)
7. [Example: End-to-End Transformer](#7-example-end-to-end-transformer)
8. [Benchmarking](#8-benchmarking)

---

## 1. Introduction

### 1.1 Overview

This specification builds on the [general integration guide](integration-guide.md) with patterns specific to the **hologram-ai** project. While the general guide covers basic Hologram usage, this document focuses on:

- Training vs inference workflows
- Gradient computation on top of Hologram operations
- Common ML architectures (Transformers, CNNs, RNNs)
- ML-specific weight management (checkpoints, quantization)
- Production deployment patterns
- Performance benchmarking against PyTorch/TensorFlow

### 1.2 Architecture Philosophy

**hologram-ai** should use Hologram as a **high-performance compute backend** for forward passes, while managing:

- **Graph construction** (model architecture definition)
- **Gradient computation** (autograd/backpropagation)
- **Weight updates** (optimizer logic)
- **Training loops** (batching, learning rate scheduling, etc.)

```
┌───────────────────────────────────────────────────┐
│             hologram-ai Framework                  │
│                                                    │
│  ┌──────────────────────────────────────────┐    │
│  │  Model Definition (nn.Module style)      │    │
│  └──────────────────┬───────────────────────┘    │
│                     │                             │
│  ┌──────────────────▼───────────────────────┐    │
│  │  Autograd Engine (gradient computation)  │    │
│  └──────────────────┬───────────────────────┘    │
│                     │                             │
│  ┌──────────────────▼───────────────────────┐    │
│  │  Hologram IR Builder (GraphBuilder)      │    │
│  └──────────────────┬───────────────────────┘    │
│                     │                             │
└─────────────────────┼─────────────────────────────┘
                      │
                      ▼
         ┌────────────────────────────┐
         │   Hologram Compiler         │
         └────────────┬───────────────┘
                      │
                      ▼
         ┌────────────────────────────┐
         │   Hologram Executor         │
         │   (Forward Pass Only)       │
         └────────────────────────────┘
```

**Key insight:** Hologram handles forward pass execution with optimizations (Winograd, epilogue fusion, SIMD), while hologram-ai handles gradient computation and weight updates.

---

## 2. AI/ML-Specific Patterns

### 2.1 Training vs Inference

#### Inference Workflow (Recommended)

Use pre-compiled `.holp` files for fast inference:

```rust
// Load pre-compiled model
let plan = BackendPlan::from_bytes(&std::fs::read("model.holp")?)?;
let executor = PlanExecutor::with_external_constants(
    plan,
    &*backend,
    Path::new("model.weights"),
)?;

// Run inference loop
for batch in data_loader {
    executor.execute(&[batch.as_bytes()], &mut [output_buffer])?;
    // Post-process outputs
}
```

**Performance characteristics:**
- Cold start: ~1-2ms (load plan)
- Per-batch: ~100μs-10ms (depends on model size)
- Memory: Lazy-loaded weights (low RSS)

#### Training Workflow

Build IR dynamically for each forward pass (to track gradients):

```rust
// Forward pass (build IR)
let mut builder = GraphBuilder::new();
let x = builder.add_input("x", vec![batch_size, input_dim]);

// Add operations (track for backward pass)
let linear1 = add_linear_layer(&mut builder, x, w1, b1);
let relu = builder.add_op(OpNode::FusedActivation(FusedActivation::relu()), vec![linear1]);
let linear2 = add_linear_layer(&mut builder, relu, w2, b2);
let loss = builder.add_op(OpNode::MSELoss(...), vec![linear2, targets]);

builder.add_output("loss", loss);

// Compile and execute forward pass
let graph = builder.build()?;
let plan = graph.compile_to_plan(&backend)?;
let mut executor = PlanExecutor::new(plan, &*backend)?;
executor.execute(&[input_data, target_data], &mut [loss_output])?;

// Backward pass (hologram-ai autograd)
let gradients = autograd.backward(loss)?;

// Weight update (hologram-ai optimizer)
optimizer.step(&gradients)?;
```

**Performance characteristics:**
- Per-batch: ~10-100ms (compilation + execution + backward)
- Memory: Higher (IR graph + gradients)

**Optimization:** Cache compiled plans per input shape to avoid recompilation:

```rust
let cache_key = (batch_size, seq_len);
let plan = plan_cache.entry(cache_key).or_insert_with(|| {
    graph.compile_to_plan(&backend).unwrap()
});
```

### 2.2 Batch Processing Patterns

#### Fixed Batch Size

```rust
// Compile once for fixed batch
let plan = graph.compile_to_plan(&backend)?;
let executor = PlanExecutor::new(plan, &*backend)?;

// Reuse for all batches
for batch in data_loader {
    executor.execute(&[batch.as_bytes()], &mut [output_buffer])?;
}
```

#### Dynamic Batch Size

```rust
// Compile with dynamic dimension
let mut builder = GraphBuilder::new();
let x = builder.add_input("x", vec![0, input_dim]);  // batch=0 (dynamic)
// ... build graph

let plan = graph.compile_to_plan(&backend)?;
let mut executor = PlanExecutor::new(plan, &*backend)?;

// Register shape for each batch
for batch in data_loader {
    let batch_size = batch.len();
    executor.register_input_shape(0, &[batch_size, input_dim])?;
    executor.execute(&[batch.as_bytes()], &mut [output_buffer])?;
}
```

**Recommendation:** Use fixed batch size for training (better performance), dynamic batch size for inference (flexibility).

### 2.3 Model Serving Architectures

#### Singleton Executor (Low Latency)

```rust
// Single executor, single thread
lazy_static! {
    static ref EXECUTOR: Mutex<PlanExecutor> = {
        let plan = BackendPlan::from_bytes(&load_plan()).unwrap();
        let backend = create_best_backend().unwrap();
        Mutex::new(PlanExecutor::new(plan, &*backend).unwrap())
    };
}

fn predict(input: &[f32]) -> Vec<f32> {
    let mut executor = EXECUTOR.lock().unwrap();
    let mut output = vec![0.0; output_size];
    executor.execute(&[input.as_bytes()], &mut [output.as_bytes_mut()])?;
    output
}
```

#### Thread Pool (High Throughput)

```rust
// Multiple executors, one per thread
lazy_static! {
    static ref EXECUTOR_POOL: Arc<Mutex<Vec<PlanExecutor>>> = {
        let plan = BackendPlan::from_bytes(&load_plan()).unwrap();
        let backend = create_best_backend().unwrap();
        let num_threads = num_cpus::get();
        let pool = (0..num_threads)
            .map(|_| PlanExecutor::new(plan.clone(), &*backend).unwrap())
            .collect();
        Arc::new(Mutex::new(pool))
    };
}

fn predict_batch(inputs: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    inputs.par_iter().map(|input| {
        let mut pool = EXECUTOR_POOL.lock().unwrap();
        let mut executor = pool.pop().unwrap();
        drop(pool);  // Release lock

        let mut output = vec![0.0; output_size];
        executor.execute(&[input.as_bytes()], &mut [output.as_bytes_mut()])?;

        EXECUTOR_POOL.lock().unwrap().push(executor);
        output
    }).collect()
}
```

---

## 3. Weight Management for ML Models

### 3.1 Checkpoint Loading Strategies

#### Load from PyTorch Checkpoint

```rust
use safetensors::SafeTensors;

fn load_from_pytorch(checkpoint_path: &str) -> Result<HashMap<String, Vec<u8>>, Error> {
    let bytes = std::fs::read(checkpoint_path)?;
    let safetensors = SafeTensors::deserialize(&bytes)?;

    let mut weights = HashMap::new();
    for (name, tensor) in safetensors.tensors() {
        weights.insert(name.to_string(), tensor.data().to_vec());
    }

    Ok(weights)
}

// Build Hologram graph with loaded weights
let weights = load_from_pytorch("model.safetensors")?;
let mut builder = GraphBuilder::new();

let w1 = builder.add_constant("W1", vec![784, 512], weights["layer1.weight"].clone());
let b1 = builder.add_constant("b1", vec![512], weights["layer1.bias"].clone());
// ... rest of model
```

#### Load from HuggingFace Hub

```rust
use hf_hub::api::sync::Api;

fn load_from_hf(repo_id: &str, filename: &str) -> Result<Vec<u8>, Error> {
    let api = Api::new()?;
    let repo = api.model(repo_id.to_string());
    let path = repo.get(filename)?;
    std::fs::read(path)
}

let safetensors_bytes = load_from_hf("bert-base-uncased", "model.safetensors")?;
let weights = SafeTensors::deserialize(&safetensors_bytes)?;
```

### 3.2 Model Quantization Integration

#### Post-Training Quantization

```rust
// Load full-precision weights
let fp32_weights = load_weights("model.safetensors")?;

// Quantize to INT8
let quantizer = Quantizer::new(QuantizationScheme::SymmetricInt8);
let quantized_weights: HashMap<String, (Vec<i8>, f32, f32)> = fp32_weights
    .into_iter()
    .map(|(name, data)| {
        let (quantized, scale, zero_point) = quantizer.quantize(&data);
        (name, (quantized, scale, zero_point))
    })
    .collect();

// Build Hologram graph with quantized weights
let mut builder = GraphBuilder::new();
for (name, (data, scale, zero_point)) in quantized_weights {
    let const_id = builder.add_constant(&name, shape, data);
    // Store scale and zero_point as metadata
    builder.set_quantization_params(const_id, scale, zero_point);
}
```

#### Quantization-Aware Training (QAT)

```rust
// During training, simulate quantization
fn quantize_simulate(x: &Tensor, scale: f32, zero_point: f32) -> Tensor {
    let quantized = ((x / scale) + zero_point).round().clamp(-128.0, 127.0);
    (quantized - zero_point) * scale  // Dequantize for gradient flow
}

// In forward pass
let w_quantized = quantize_simulate(&weights, scale, zero_point);
let output = matmul(input, w_quantized);
```

### 3.3 Multi-GPU Weight Distribution

#### Model Parallelism

```rust
// Split model across GPUs
let devices = vec![
    Device::Cuda(0),
    Device::Cuda(1),
    Device::Cuda(2),
    Device::Cuda(3),
];

// Layer 1-10 on GPU 0
let plan1 = graph_layers_1_10.compile_to_plan(&backend_gpu0)?;

// Layer 11-20 on GPU 1
let plan2 = graph_layers_11_20.compile_to_plan(&backend_gpu1)?;

// Execute sequentially with transfers
let mut output1 = execute_on_device(&plan1, input, &device[0])?;
let output1_on_device1 = transfer_to_device(output1, &device[0], &device[1])?;
let mut output2 = execute_on_device(&plan2, output1_on_device1, &device[1])?;
```

#### Data Parallelism

```rust
// Replicate model on each GPU, split batch
let plan = graph.compile_to_plan(&backend)?;
let executors: Vec<_> = devices
    .iter()
    .map(|device| {
        let backend = create_backend_for_device(device)?;
        PlanExecutor::new(plan.clone(), &*backend)
    })
    .collect::<Result<_, _>>()?;

// Split batch and execute in parallel
let mini_batches = split_batch(input, devices.len());
let outputs: Vec<_> = mini_batches
    .into_par_iter()
    .zip(executors.par_iter_mut())
    .map(|(mini_batch, executor)| {
        executor.execute(&[mini_batch], &mut [output_buffer])
    })
    .collect::<Result<_, _>>()?;

let combined_output = concat_outputs(outputs);
```

---

## 4. Common ML Operations

### 4.1 Transformer Blocks

#### Self-Attention

```rust
fn build_self_attention(
    builder: &mut GraphBuilder,
    x: NodeId,              // [batch, seq_len, d_model]
    wq: NodeId,             // [d_model, d_k]
    wk: NodeId,             // [d_model, d_k]
    wv: NodeId,             // [d_model, d_v]
    wo: NodeId,             // [d_v, d_model]
    batch: usize,
    seq_len: usize,
    d_model: usize,
    d_k: usize,
) -> NodeId {
    // Q = x @ Wq
    let q = builder.add_op(
        OpNode::MatMul(MatMul::new(batch * seq_len, d_model, d_k)),
        vec![x, wq],
    );

    // K = x @ Wk
    let k = builder.add_op(
        OpNode::MatMul(MatMul::new(batch * seq_len, d_model, d_k)),
        vec![x, wk],
    );

    // V = x @ Wv
    let v = builder.add_op(
        OpNode::MatMul(MatMul::new(batch * seq_len, d_model, d_k)),
        vec![x, wv],
    );

    // Attention scores: Q @ K^T / sqrt(d_k)
    let k_t = builder.add_op(
        OpNode::Transpose(Transpose::new(vec![0, 2, 1])),
        vec![k],
    );

    let scores = builder.add_op(
        OpNode::MatMul(MatMul::new(batch, seq_len, seq_len)),
        vec![q, k_t],
    );

    let scale = (d_k as f32).sqrt();
    let scale_const = builder.add_constant("scale", vec![1], vec![1.0 / scale]);
    let scaled_scores = builder.add_op(
        OpNode::Mul(Mul::new(batch * seq_len * seq_len)),
        vec![scores, scale_const],
    );

    // Softmax
    let attn_weights = builder.add_op(
        OpNode::Softmax(Softmax::new(-1)),
        vec![scaled_scores],
    );

    // attn_weights @ V
    let attn_output = builder.add_op(
        OpNode::MatMul(MatMul::new(batch, seq_len, d_k)),
        vec![attn_weights, v],
    );

    // Output projection: @ Wo
    let output = builder.add_op(
        OpNode::MatMul(MatMul::new(batch * seq_len, d_k, d_model)),
        vec![attn_output, wo],
    );

    output
}
```

#### Feed-Forward Network (FFN)

```rust
fn build_ffn(
    builder: &mut GraphBuilder,
    x: NodeId,              // [batch, seq_len, d_model]
    w1: NodeId,             // [d_model, d_ff]
    b1: NodeId,             // [d_ff]
    w2: NodeId,             // [d_ff, d_model]
    b2: NodeId,             // [d_model]
    batch: usize,
    seq_len: usize,
    d_model: usize,
    d_ff: usize,
) -> NodeId {
    // Linear 1: x @ W1 + b1
    let linear1 = builder.add_op(
        OpNode::MatMul(MatMul::new(batch * seq_len, d_model, d_ff)),
        vec![x, w1],
    );
    let add1 = builder.add_op(
        OpNode::Add(Add::new(d_ff)),
        vec![linear1, b1],
    );

    // Activation (GELU is common in Transformers)
    let gelu = builder.add_op(
        OpNode::FusedActivation(FusedActivation::gelu()),
        vec![add1],
    );

    // Linear 2: @ W2 + b2
    let linear2 = builder.add_op(
        OpNode::MatMul(MatMul::new(batch * seq_len, d_ff, d_model)),
        vec![gelu, w2],
    );
    let add2 = builder.add_op(
        OpNode::Add(Add::new(d_model)),
        vec![linear2, b2],
    );

    add2
}
```

**Note:** Hologram's epilogue fusion will automatically fuse `MatMul + Add + Activation` into single kernels.

### 4.2 CNN Architectures

#### ResNet Block

```rust
fn build_resnet_block(
    builder: &mut GraphBuilder,
    x: NodeId,              // [batch, h, w, c]
    conv1_w: NodeId,        // [3, 3, c, c]
    conv1_b: NodeId,        // [c]
    conv2_w: NodeId,        // [3, 3, c, c]
    conv2_b: NodeId,        // [c]
    batch: usize,
    h: usize,
    w: usize,
    c: usize,
) -> NodeId {
    // Conv1: 3x3, stride=1
    let conv1 = builder.add_op(
        OpNode::Conv2D(Conv2D::new(h, w, 3, 3, c, c, 1)),
        vec![x, conv1_w],
    );
    let add1 = builder.add_op(OpNode::Add(Add::new(c)), vec![conv1, conv1_b]);
    let relu1 = builder.add_op(OpNode::FusedActivation(FusedActivation::relu()), vec![add1]);

    // Conv2: 3x3, stride=1
    let conv2 = builder.add_op(
        OpNode::Conv2D(Conv2D::new(h, w, 3, 3, c, c, 1)),
        vec![relu1, conv2_w],
    );
    let add2 = builder.add_op(OpNode::Add(Add::new(c)), vec![conv2, conv2_b]);

    // Skip connection: x + conv2_output
    let residual = builder.add_op(OpNode::Add(Add::new(batch * h * w * c)), vec![x, add2]);
    let relu2 = builder.add_op(OpNode::FusedActivation(FusedActivation::relu()), vec![residual]);

    relu2
}
```

**Optimization:** Winograd F(2,3) is automatically applied for 3x3 convolutions.

### 4.3 RNN/LSTM Patterns

#### Simple RNN Cell

```rust
fn build_rnn_cell(
    builder: &mut GraphBuilder,
    x_t: NodeId,            // [batch, input_dim]
    h_prev: NodeId,         // [batch, hidden_dim]
    w_ih: NodeId,           // [input_dim, hidden_dim]
    w_hh: NodeId,           // [hidden_dim, hidden_dim]
    b: NodeId,              // [hidden_dim]
    batch: usize,
    input_dim: usize,
    hidden_dim: usize,
) -> NodeId {
    // h_t = tanh(x_t @ W_ih + h_prev @ W_hh + b)
    let x_contrib = builder.add_op(
        OpNode::MatMul(MatMul::new(batch, input_dim, hidden_dim)),
        vec![x_t, w_ih],
    );

    let h_contrib = builder.add_op(
        OpNode::MatMul(MatMul::new(batch, hidden_dim, hidden_dim)),
        vec![h_prev, w_hh],
    );

    let add1 = builder.add_op(
        OpNode::Add(Add::new(batch * hidden_dim)),
        vec![x_contrib, h_contrib],
    );

    let add2 = builder.add_op(
        OpNode::Add(Add::new(hidden_dim)),
        vec![add1, b],
    );

    let h_t = builder.add_op(
        OpNode::FusedActivation(FusedActivation::tanh()),
        vec![add2],
    );

    h_t
}
```

#### LSTM Cell

```rust
fn build_lstm_cell(
    builder: &mut GraphBuilder,
    x_t: NodeId,
    h_prev: NodeId,
    c_prev: NodeId,
    w_ih: NodeId,           // [input_dim, 4 * hidden_dim]
    w_hh: NodeId,           // [hidden_dim, 4 * hidden_dim]
    b: NodeId,              // [4 * hidden_dim]
    batch: usize,
    input_dim: usize,
    hidden_dim: usize,
) -> (NodeId, NodeId) {     // (h_t, c_t)
    // Compute gates: [i, f, g, o] = x @ W_ih + h @ W_hh + b
    let x_contrib = builder.add_op(
        OpNode::MatMul(MatMul::new(batch, input_dim, 4 * hidden_dim)),
        vec![x_t, w_ih],
    );

    let h_contrib = builder.add_op(
        OpNode::MatMul(MatMul::new(batch, hidden_dim, 4 * hidden_dim)),
        vec![h_prev, w_hh],
    );

    let gates = builder.add_op(
        OpNode::Add(Add::new(batch * 4 * hidden_dim)),
        vec![x_contrib, h_contrib],
    );

    let gates_biased = builder.add_op(
        OpNode::Add(Add::new(4 * hidden_dim)),
        vec![gates, b],
    );

    // Split into [i, f, g, o]
    let chunks = builder.add_op(
        OpNode::Split(Split::new(1, 4)),
        vec![gates_biased],
    );

    // i_t = sigmoid(i), f_t = sigmoid(f), g_t = tanh(g), o_t = sigmoid(o)
    // c_t = f_t * c_prev + i_t * g_t
    // h_t = o_t * tanh(c_t)

    // ... (implementation details)

    (h_t, c_t)
}
```

### 4.4 Embedding Layers

```rust
fn build_embedding(
    builder: &mut GraphBuilder,
    input_ids: NodeId,      // [batch, seq_len] (integer indices)
    embedding_table: NodeId, // [vocab_size, embedding_dim]
    batch: usize,
    seq_len: usize,
    vocab_size: usize,
    embedding_dim: usize,
) -> NodeId {
    // Use Gather operation
    builder.add_op(
        OpNode::Gather(Gather::new(0)),  // Gather along axis 0
        vec![embedding_table, input_ids],
    )
}
```

---

## 5. Integration with Autograd

### 5.1 Forward Pass with Hologram

```rust
// hologram-ai wraps Hologram executor
struct HologramTensor {
    data: Vec<f32>,
    grad: Option<Vec<f32>>,
    grad_fn: Option<Box<dyn GradFn>>,
}

impl HologramTensor {
    fn forward_hologram(
        &self,
        op: OpNode,
        inputs: Vec<&HologramTensor>,
    ) -> Result<HologramTensor, Error> {
        // Build IR
        let mut builder = GraphBuilder::new();
        let input_ids: Vec<_> = inputs
            .iter()
            .enumerate()
            .map(|(i, t)| builder.add_input(&format!("input_{}", i), t.shape()))
            .collect();

        let op_id = builder.add_op(op, input_ids);
        builder.add_output("output", op_id);

        // Compile and execute via Hologram
        let graph = builder.build()?;
        let plan = graph.compile_to_plan(&backend)?;
        let mut executor = PlanExecutor::new(plan, &*backend)?;

        let input_bytes: Vec<_> = inputs.iter().map(|t| t.data.as_bytes()).collect();
        let mut output_data = vec![0.0f32; output_size];
        executor.execute(&input_bytes, &mut [output_data.as_bytes_mut()])?;

        // Wrap with gradient function
        Ok(HologramTensor {
            data: output_data,
            grad: None,
            grad_fn: Some(Box::new(HologramGradFn { op, inputs })),
        })
    }
}
```

### 5.2 Backward Pass Computation

```rust
trait GradFn {
    fn backward(&self, grad_output: &[f32]) -> Vec<Vec<f32>>;
}

struct MatMulGradFn {
    x: HologramTensor,
    w: HologramTensor,
}

impl GradFn for MatMulGradFn {
    fn backward(&self, grad_output: &[f32]) -> Vec<Vec<f32>> {
        // dL/dx = grad_output @ W^T
        let grad_x = matmul(grad_output, &self.w.data.transpose());

        // dL/dW = x^T @ grad_output
        let grad_w = matmul(&self.x.data.transpose(), grad_output);

        vec![grad_x, grad_w]
    }
}
```

### 5.3 Weight Update Strategies

#### SGD

```rust
struct SGD {
    learning_rate: f32,
}

impl SGD {
    fn step(&self, params: &mut [HologramTensor], grads: &[Vec<f32>]) {
        for (param, grad) in params.iter_mut().zip(grads) {
            for (p, g) in param.data.iter_mut().zip(grad) {
                *p -= self.learning_rate * g;
            }
        }
    }
}
```

#### Adam

```rust
struct Adam {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    m: Vec<Vec<f32>>,  // First moment
    v: Vec<Vec<f32>>,  // Second moment
    t: usize,          // Time step
}

impl Adam {
    fn step(&mut self, params: &mut [HologramTensor], grads: &[Vec<f32>]) {
        self.t += 1;
        let lr_t = self.learning_rate * (1.0 - self.beta2.powi(self.t as i32)).sqrt()
            / (1.0 - self.beta1.powi(self.t as i32));

        for (i, (param, grad)) in params.iter_mut().zip(grads).enumerate() {
            for (j, (p, g)) in param.data.iter_mut().zip(grad).enumerate() {
                // Update biased first moment estimate
                self.m[i][j] = self.beta1 * self.m[i][j] + (1.0 - self.beta1) * g;

                // Update biased second raw moment estimate
                self.v[i][j] = self.beta2 * self.v[i][j] + (1.0 - self.beta2) * g * g;

                // Update parameter
                *p -= lr_t * self.m[i][j] / (self.v[i][j].sqrt() + self.epsilon);
            }
        }
    }
}
```

---

## 6. Deployment Patterns

### 6.1 Model Packaging

#### Create Deployable Archive

```rust
use hologram::compiler::ArchiveWriter;

fn package_model(
    graph: CompileGraph,
    weights: HashMap<String, Vec<u8>>,
    output_path: &str,
) -> Result<(), Error> {
    // Create manifest
    let mut manifest = Manifest::new("my-model");
    manifest.metadata.description = Some("Production model v1.0".to_string());
    manifest.metadata.author = Some("hologram-ai team".to_string());
    manifest.add_metadata("model_type", "transformer");
    manifest.add_metadata("num_parameters", "110M");

    // Add layer
    let mut layer = Layer::new("model");
    layer.holo_path = "layers/model.holo".to_string();
    layer.weights_path = Some("weights/model.weights".to_string());
    manifest.add_layer(layer);

    // Write archive
    let mut writer = ArchiveWriter::with_compression(manifest, 6);
    let graph_bytes = graph.to_bytes()?;
    writer.add_layer_bytes("model", &graph_bytes);

    let weight_bytes: Vec<u8> = weights.values().flat_map(|v| v.clone()).collect();
    writer.add_weights("model", &weight_bytes);

    let file = File::create(output_path)?;
    writer.write_to(file)?;

    println!("Model packaged to {}", output_path);
    Ok(())
}
```

#### Verify Package Integrity

```rust
fn verify_package(path: &str) -> Result<(), Error> {
    let reader = ArchiveReader::from_reader(File::open(path)?)?;

    // Check manifest
    let manifest = reader.manifest();
    println!("Model: {}", manifest.metadata.name);
    println!("Version: {}", manifest.version);

    // Verify checksums
    if !reader.all_checksums_valid() {
        eprintln!("Checksum validation failed!");
        for result in reader.invalid_checksums() {
            eprintln!("  Layer '{}': expected {}, got {}",
                result.layer_name, result.expected, result.actual);
        }
        return Err("Integrity check failed");
    }

    println!("Package integrity verified ✓");
    Ok(())
}
```

### 6.2 Versioning and Reproducibility

#### Model Versioning

```rust
// Embed version info in manifest
let mut manifest = Manifest::new("my-model");
manifest.version = "1.2.3";
manifest.add_metadata("git_commit", "abc123def456");
manifest.add_metadata("training_dataset", "v2.0");
manifest.add_metadata("trained_at", "2026-01-27T12:00:00Z");

// Content-addressable layer IDs ensure reproducibility
// Same weights → same SHA-256 → same layer ID
```

#### Loading Specific Version

```rust
fn load_model_version(repo_path: &str, version: &str) -> Result<PlanExecutor, Error> {
    let model_path = format!("{}/model-{}.holo", repo_path, version);

    let reader = ArchiveReader::from_reader(File::open(model_path)?)?;
    verify_checksum(&reader)?;

    let graph = reader.get_layer("model")?;
    let weights = reader.get_weights("model")?;

    let backend = create_best_backend()?;
    let plan = graph.compile_to_plan_with_weights(&backend, weights)?;
    Ok(PlanExecutor::new(plan, &*backend)?)
}
```

### 6.3 A/B Testing with Multiple Models

```rust
struct ModelRegistry {
    models: HashMap<String, Arc<Mutex<PlanExecutor>>>,
}

impl ModelRegistry {
    fn new() -> Self {
        Self { models: HashMap::new() }
    }

    fn register_model(&mut self, name: &str, executor: PlanExecutor) {
        self.models.insert(name.to_string(), Arc::new(Mutex::new(executor)));
    }

    fn predict(&self, model_name: &str, input: &[f32]) -> Result<Vec<f32>, Error> {
        let executor = self.models.get(model_name)
            .ok_or("Model not found")?;

        let mut executor = executor.lock().unwrap();
        let mut output = vec![0.0f32; output_size];
        executor.execute(&[input.as_bytes()], &mut [output.as_bytes_mut()])?;

        Ok(output)
    }
}

// Usage: A/B test model_v1 vs model_v2
let mut registry = ModelRegistry::new();
registry.register_model("model_v1", load_model("v1.holp")?);
registry.register_model("model_v2", load_model("v2.holp")?);

// Route 50/50 to each model
let model_name = if rand::random::<f32>() < 0.5 { "model_v1" } else { "model_v2" };
let prediction = registry.predict(model_name, &input)?;
```

---

## 7. Example: End-to-End Transformer

### 7.1 Model Definition

```rust
struct TransformerConfig {
    vocab_size: usize,
    d_model: usize,
    num_heads: usize,
    num_layers: usize,
    d_ff: usize,
    max_seq_len: usize,
}

fn build_transformer(
    config: &TransformerConfig,
    weights: &HashMap<String, Vec<u8>>,
) -> Result<CompileGraph, Error> {
    let mut builder = GraphBuilder::new();

    // Input: [batch, seq_len]
    let input_ids = builder.add_input("input_ids", vec![0, config.max_seq_len]);

    // Embedding: [batch, seq_len, d_model]
    let embedding_table = builder.add_constant(
        "embedding_table",
        vec![config.vocab_size, config.d_model],
        weights["embedding.weight"].clone(),
    );
    let embeddings = builder.add_op(
        OpNode::Gather(Gather::new(0)),
        vec![embedding_table, input_ids],
    );

    // Positional encoding
    let pos_encoding = builder.add_constant(
        "pos_encoding",
        vec![config.max_seq_len, config.d_model],
        weights["pos_encoding.weight"].clone(),
    );
    let mut x = builder.add_op(
        OpNode::Add(Add::new(config.max_seq_len * config.d_model)),
        vec![embeddings, pos_encoding],
    );

    // Transformer layers
    for layer_idx in 0..config.num_layers {
        let prefix = format!("layer_{}", layer_idx);

        // Multi-head self-attention
        let attn_output = build_self_attention(
            &mut builder,
            x,
            load_weight(&mut builder, &weights, &format!("{}.attn.wq", prefix))?,
            load_weight(&mut builder, &weights, &format!("{}.attn.wk", prefix))?,
            load_weight(&mut builder, &weights, &format!("{}.attn.wv", prefix))?,
            load_weight(&mut builder, &weights, &format!("{}.attn.wo", prefix))?,
            0,  // batch (dynamic)
            config.max_seq_len,
            config.d_model,
            config.d_model / config.num_heads,
        );

        // Add & Norm
        let add1 = builder.add_op(
            OpNode::Add(Add::new(config.max_seq_len * config.d_model)),
            vec![x, attn_output],
        );
        let norm1 = build_layer_norm(
            &mut builder,
            add1,
            load_weight(&mut builder, &weights, &format!("{}.norm1.weight", prefix))?,
            load_weight(&mut builder, &weights, &format!("{}.norm1.bias", prefix))?,
            config.d_model,
        );

        // Feed-forward network
        let ffn_output = build_ffn(
            &mut builder,
            norm1,
            load_weight(&mut builder, &weights, &format!("{}.ffn.w1", prefix))?,
            load_weight(&mut builder, &weights, &format!("{}.ffn.b1", prefix))?,
            load_weight(&mut builder, &weights, &format!("{}.ffn.w2", prefix))?,
            load_weight(&mut builder, &weights, &format!("{}.ffn.b2", prefix))?,
            0,  // batch (dynamic)
            config.max_seq_len,
            config.d_model,
            config.d_ff,
        );

        // Add & Norm
        let add2 = builder.add_op(
            OpNode::Add(Add::new(config.max_seq_len * config.d_model)),
            vec![norm1, ffn_output],
        );
        let norm2 = build_layer_norm(
            &mut builder,
            add2,
            load_weight(&mut builder, &weights, &format!("{}.norm2.weight", prefix))?,
            load_weight(&mut builder, &weights, &format!("{}.norm2.bias", prefix))?,
            config.d_model,
        );

        x = norm2;
    }

    // Output projection: [batch, seq_len, d_model] -> [batch, seq_len, vocab_size]
    let output_proj = builder.add_constant(
        "output_proj.weight",
        vec![config.d_model, config.vocab_size],
        weights["output_proj.weight"].clone(),
    );
    let logits = builder.add_op(
        OpNode::MatMul(MatMul::new(0, config.d_model, config.vocab_size)),
        vec![x, output_proj],
    );

    builder.add_output("logits", logits);

    Ok(builder.build()?)
}
```

### 7.2 Training Pipeline

```rust
fn train_transformer(
    config: &TransformerConfig,
    train_data: &DataLoader,
    num_epochs: usize,
) -> Result<(), Error> {
    // Initialize weights
    let mut weights = initialize_weights(config);

    // Optimizer
    let mut optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8, &weights);

    // Training loop
    for epoch in 0..num_epochs {
        let mut total_loss = 0.0;

        for (batch_idx, batch) in train_data.iter().enumerate() {
            // Forward pass
            let graph = build_transformer(config, &weights)?;
            let backend = create_best_backend()?;
            let plan = graph.compile_to_plan(&backend)?;
            let mut executor = PlanExecutor::new(plan, &*backend)?;

            let input_ids = batch.input_ids.as_bytes();
            let mut logits_output = vec![0.0f32; batch.batch_size * config.max_seq_len * config.vocab_size];

            executor.execute(&[input_ids], &mut [logits_output.as_bytes_mut()])?;

            // Compute loss (cross-entropy)
            let loss = cross_entropy_loss(&logits_output, &batch.labels);
            total_loss += loss;

            // Backward pass (hologram-ai autograd)
            let gradients = compute_gradients(&logits_output, &batch.labels, &weights)?;

            // Update weights
            optimizer.step(&mut weights, &gradients);

            if batch_idx % 100 == 0 {
                println!("Epoch {}, Batch {}, Loss: {:.4}", epoch, batch_idx, loss);
            }
        }

        println!("Epoch {} complete, Avg Loss: {:.4}", epoch, total_loss / train_data.len() as f32);

        // Save checkpoint
        save_checkpoint(epoch, &weights, &optimizer)?;
    }

    Ok(())
}
```

### 7.3 Inference Pipeline

```rust
fn inference_transformer(
    config: &TransformerConfig,
    weights: &HashMap<String, Vec<u8>>,
    input_text: &str,
) -> Result<String, Error> {
    // Tokenize input
    let tokenizer = Tokenizer::new()?;
    let input_ids = tokenizer.encode(input_text)?;

    // Build and compile model
    let graph = build_transformer(config, weights)?;
    let backend = create_best_backend()?;
    let plan = graph.compile_to_plan(&backend)?;

    // Cache compiled plan for repeated inference
    std::fs::write("transformer.holp", plan.to_bytes()?)?;

    // Create executor
    let mut executor = PlanExecutor::new(plan, &*backend)?;

    // Run inference
    let mut logits_output = vec![0.0f32; input_ids.len() * config.vocab_size];
    executor.execute(&[input_ids.as_bytes()], &mut [logits_output.as_bytes_mut()])?;

    // Decode output
    let output_ids = argmax_along_dim(&logits_output, -1);
    let output_text = tokenizer.decode(&output_ids)?;

    Ok(output_text)
}
```

---

## 8. Benchmarking

### 8.1 Comparison with PyTorch/TensorFlow

#### Benchmark Setup

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");

    // Hologram
    group.bench_function("hologram", |b| {
        let mut builder = GraphBuilder::new();
        let x = builder.add_input("x", vec![1024, 512]);
        let w = builder.add_constant("W", vec![512, 256], weight_data.clone());
        let mm = builder.add_op(OpNode::MatMul(MatMul::new(1024, 512, 256)), vec![x, w]);
        builder.add_output("y", mm);

        let graph = builder.build().unwrap();
        let backend = create_best_backend().unwrap();
        let plan = graph.compile_to_plan(&backend).unwrap();
        let mut executor = PlanExecutor::new(plan, &*backend).unwrap();

        let input = vec![0.0f32; 1024 * 512];
        let mut output = vec![0.0f32; 1024 * 256];

        b.iter(|| {
            executor.execute(
                &[black_box(input.as_bytes())],
                &mut [black_box(output.as_bytes_mut())],
            ).unwrap();
        });
    });

    // PyTorch (via PyO3)
    group.bench_function("pytorch", |b| {
        Python::with_gil(|py| {
            let torch = py.import("torch").unwrap();
            let x = torch.call_method1("randn", (1024, 512)).unwrap();
            let w = torch.call_method1("randn", (512, 256)).unwrap();

            b.iter(|| {
                black_box(torch.call_method1("matmul", (x, w)).unwrap());
            });
        });
    });

    group.finish();
}

criterion_group!(benches, benchmark_matmul);
criterion_main!(benches);
```

#### Expected Results

| Operation | Hologram | PyTorch (CPU) | Speedup |
|-----------|----------|---------------|---------|
| MatMul (1024×512×256) | 1.2ms | 2.4ms | 2.0x |
| Conv2D (3×3, 128 filters) | 3.5ms | 8.2ms | 2.3x (Winograd) |
| Add (1M elements) | 0.15ms | 0.22ms | 1.5x (SIMD) |
| Transformer Block | 8.5ms | 18.3ms | 2.2x |

**Note:** Speedups depend on CPU (AVX2/AVX-512 support), model size, and batch size.

### 8.2 Profiling and Optimization

#### Profiling Execution Time

```rust
use std::time::Instant;

fn profile_model(executor: &mut PlanExecutor, input: &[f32], iterations: usize) {
    // Warm up
    for _ in 0..10 {
        executor.execute(&[input.as_bytes()], &mut [output.as_bytes_mut()]).unwrap();
    }

    // Profile
    let start = Instant::now();
    for _ in 0..iterations {
        executor.execute(&[input.as_bytes()], &mut [output.as_bytes_mut()]).unwrap();
    }
    let elapsed = start.elapsed();

    println!("Avg execution: {:?}", elapsed / iterations as u32);
    println!("Throughput: {:.2} samples/sec", iterations as f64 / elapsed.as_secs_f64());
}
```

#### Per-Operation Profiling

```rust
// Enable profiling mode
let plan = graph.compile_to_plan_with_options(&backend, CompileOptions {
    enable_profiling: true,
    ..Default::default()
})?;

let mut executor = PlanExecutor::new(plan, &*backend)?;
executor.execute(&[input], &mut [output])?;

// Get profiling results
let profile = executor.get_profile()?;
for (op_idx, timing) in profile.op_timings.iter().enumerate() {
    println!("Op {}: {:?}", op_idx, timing);
}
```

#### Memory Profiling

```rust
// Before execution
let rss_before = get_rss()?;

// Load model and execute
let executor = PlanExecutor::with_external_constants(plan, &*backend, weights_path)?;
executor.execute(&[input], &mut [output])?;

// After execution
let rss_after = get_rss()?;

println!("RSS increase: {} MB", (rss_after - rss_before) / 1_000_000);
println!("Model size: {} MB", std::fs::metadata(weights_path)?.len() / 1_000_000);
println!("Memory efficiency: {:.2}%",
    (rss_after - rss_before) as f64 / std::fs::metadata(weights_path)?.len() as f64 * 100.0);
```

### 8.3 Optimization Checklist

- [ ] **Use pre-compiled plans** (.holp files) for production inference
- [ ] **Memory-map weights** for large models (> 1GB)
- [ ] **Batch inputs** with same shape to avoid workspace reallocation
- [ ] **Cache compiled plans** per input shape for dynamic models
- [ ] **Use fixed batch size** for training (better performance)
- [ ] **Enable epilogue fusion** (automatic, but verify in plan)
- [ ] **Verify Winograd** is used for 3×3 convolutions
- [ ] **Profile per-operation** timing to identify bottlenecks
- [ ] **Monitor memory usage** (RSS) to validate lazy loading

---

## Appendix A: Migration from PyTorch

### A.1 PyTorch Model → Hologram

```python
# PyTorch model
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Save weights
model = SimpleModel()
torch.save(model.state_dict(), "model.pth")
```

```rust
// Hologram equivalent
use hologram::compiler::{GraphBuilder, OpNode, MatMul};

fn build_simple_model(weights: &HashMap<String, Vec<u8>>) -> Result<CompileGraph, Error> {
    let mut builder = GraphBuilder::new();

    let x = builder.add_input("x", vec![1, 784]);

    // fc1
    let w1 = builder.add_constant("fc1.weight", vec![784, 512], weights["fc1.weight"].clone());
    let b1 = builder.add_constant("fc1.bias", vec![512], weights["fc1.bias"].clone());
    let fc1 = builder.add_op(OpNode::MatMul(MatMul::new(1, 784, 512)), vec![x, w1]);
    let add1 = builder.add_op(OpNode::Add(Add::new(512)), vec![fc1, b1]);

    // relu
    let relu = builder.add_op(OpNode::FusedActivation(FusedActivation::relu()), vec![add1]);

    // fc2
    let w2 = builder.add_constant("fc2.weight", vec![512, 10], weights["fc2.weight"].clone());
    let b2 = builder.add_constant("fc2.bias", vec![10], weights["fc2.bias"].clone());
    let fc2 = builder.add_op(OpNode::MatMul(MatMul::new(1, 512, 10)), vec![relu, w2]);
    let add2 = builder.add_op(OpNode::Add(Add::new(10)), vec![fc2, b2]);

    builder.add_output("logits", add2);

    Ok(builder.build()?)
}
```

### A.2 Loading PyTorch Weights

```rust
use safetensors::SafeTensors;

fn load_pytorch_weights(path: &str) -> Result<HashMap<String, Vec<u8>>, Error> {
    let bytes = std::fs::read(path)?;
    let tensors = SafeTensors::deserialize(&bytes)?;

    let mut weights = HashMap::new();
    for (name, tensor) in tensors.tensors() {
        // Convert PyTorch format to Hologram format
        let data = tensor.data().to_vec();
        weights.insert(name.to_string(), data);
    }

    Ok(weights)
}
```

---

## Appendix B: API Quick Reference

### Key Functions

```rust
// Graph building
let mut builder = GraphBuilder::new();
let input = builder.add_input(name, shape);
let const_id = builder.add_constant(name, shape, data);
let op_id = builder.add_op(op_node, inputs);
builder.add_output(name, node_id);
let graph = builder.build()?;

// Compilation
let plan = graph.compile_to_plan(&backend)?;
let bytes = plan.to_bytes()?;
let plan = BackendPlan::from_bytes(&bytes)?;

// Execution
let mut executor = PlanExecutor::new(plan, &*backend)?;
let mut executor = PlanExecutor::with_external_constants(plan, &*backend, weights_path)?;
executor.register_input_shape(input_id, shape)?;
executor.execute(&inputs, &mut outputs)?;

// Archive handling
let reader = ArchiveReader::from_reader(file)?;
let manifest = reader.manifest();
let graph = reader.get_layer(name)?;
let weights = reader.get_weights(name)?;
```

### Common Operation Constructors

```rust
// Linear algebra
MatMul::new(m, k, n)
Gemm::new(m, k, n, alpha, beta, trans_a, trans_b)

// Convolution
Conv2D::new(h, w, kh, kw, in_c, out_c, stride)

// Binary ops
Add::new(size)
Sub::new(size)
Mul::new(size)
Div::new(size)

// Activations
FusedActivation::relu()
FusedActivation::sigmoid()
FusedActivation::tanh()
FusedActivation::gelu()

// Reductions
ReduceSum::new(axis, input_shape)
ReduceMean::new(axis, input_shape)

// Tensor ops
Reshape::new(new_shape)
Transpose::new(perm)
Concat::new(axis)
Split::new(axis, num_splits)
```

---

## Additional Resources

- **General Integration Guide:** [integration-guide.md](integration-guide.md)
- **Format Specification:** [../holo-format.md](../holo-format.md)
- **Operation Reference:** [../../crates/compiler/src/graph/ops/](../../crates/compiler/src/graph/ops/)
- **Examples:** [../../examples/](../../examples/)

---

**Questions or Issues?** Open an issue at: https://github.com/your-org/hologram/issues
