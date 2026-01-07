//! Model executor for running compiled .holo models.

use anyhow::Result;
use std::collections::HashMap;
use std::path::Path;

use super::loader::load_and_compile_holo;
use super::tensors::Tensor;

use hologram_backend::{BufferHandle, PlanExecutor, ProgramBackend};

/// Buffer requirements from BackendPlan metadata.
struct BufferRequirements {
    num_inputs: usize,
    num_outputs: usize,
    output_sizes: Vec<usize>,
    output_shapes: Vec<[usize; 4]>,
    output_shape_exprs: Vec<Option<Vec<hologram_ir::shape::DimExpr>>>,
}

/// Model executor for running compiled .holo models.
///
/// Wraps hologram-backend's PlanExecutor and provides a high-level
/// tensor I/O interface.
pub struct ModelExecutor {
    /// Plan executor
    executor: PlanExecutor,
    /// Backend for buffer management
    backend: Box<dyn ProgramBackend>,
}

impl ModelExecutor {
    /// Create a new executor from a .holo file path.
    ///
    /// This loads and compiles the .holo file, creating a ready-to-execute
    /// model executor.
    ///
    /// # Arguments
    /// * `path` - Path to compiled .holo file
    ///
    /// # Returns
    /// ModelExecutor ready for execution
    pub fn from_holo_file(path: &Path) -> Result<Self> {
        // Load and compile .holo → BackendPlan
        let (plan, backend) = load_and_compile_holo(path)?;

        // Create plan executor (PlanExecutor takes ownership of plan)
        let executor = PlanExecutor::new(plan, &*backend)
            .map_err(|e| anyhow::anyhow!("Failed to create PlanExecutor: {:?}", e))?;

        Ok(Self {
            executor,
            backend,
        })
    }

    /// Get buffer requirements from the plan.
    fn get_buffer_requirements(&self) -> BufferRequirements {
        let plan = self.executor.plan();

        BufferRequirements {
            num_inputs: plan.layout_metadata.num_inputs,
            num_outputs: plan.layout_metadata.num_outputs,
            output_sizes: plan.layout_metadata.output_sizes.clone(),
            output_shapes: plan.layout_metadata.output_shapes.clone(),
            output_shape_exprs: plan.layout_metadata.output_shape_exprs.clone(),
        }
    }

    /// Map named inputs to positional indices using alphabetical order.
    ///
    /// Since BackendPlan doesn't store ONNX tensor names, we use alphabetical
    /// ordering as a convention to map named inputs to buffer indices.
    fn map_inputs_to_buffers(
        &mut self,
        named_inputs: &HashMap<String, Tensor>,
        requirements: &BufferRequirements,
    ) -> Result<Vec<BufferHandle>> {
        // Sort input names alphabetically for consistent ordering
        let mut input_names: Vec<_> = named_inputs.keys().cloned().collect();
        input_names.sort();

        // Validate input count
        if input_names.len() != requirements.num_inputs {
            return Err(anyhow::anyhow!(
                "Expected {} inputs, got {}. Required inputs (sorted): {:?}",
                requirements.num_inputs,
                input_names.len(),
                input_names
            ));
        }

        tracing::trace!(
            "Mapping {} inputs (sorted): {:?}",
            input_names.len(),
            input_names
        );

        // Allocate and upload buffers in sorted order
        let mut handles = Vec::with_capacity(requirements.num_inputs);
        for (idx, name) in input_names.iter().enumerate() {
            let tensor = named_inputs.get(name).unwrap();

            tracing::trace!(
                "Input {}: '{}' shape {:?} -> {} bytes",
                idx,
                name,
                tensor.shape,
                tensor.size_bytes()
            );

            let handle = self.tensor_to_buffer_handle(tensor)?;
            handles.push(handle);

            // Register shape for Shape ops
            self.executor.register_shape(&handle, tensor.shape.clone());
        }

        Ok(handles)
    }


    /// Allocate output buffers and return both handles and computed shapes.
    ///
    /// This is a convenience wrapper that returns the shapes for use in buffers_to_outputs.
    fn allocate_output_buffers_with_shapes(
        &mut self,
        requirements: &BufferRequirements,
        input_tensors: &[&Tensor],
    ) -> Result<(Vec<BufferHandle>, Vec<[usize; 4]>)> {
        let mut handles = Vec::with_capacity(requirements.num_outputs);
        let mut shapes = Vec::with_capacity(requirements.num_outputs);

        for (idx, &metadata_size_bytes) in requirements.output_sizes.iter().enumerate() {
            // Compute actual output shape and size if shape_expr exists
            let (actual_size, actual_shape) = if let Some(ref shape_expr) = requirements.output_shape_exprs[idx] {
                use hologram_ir::shape::DimExpr;

                tracing::debug!("Output {} shape_expr has {} dimensions", idx, shape_expr.len());

                // Resolve each dimension expression to a concrete value
                let resolved_dims: Vec<usize> = shape_expr
                    .iter()
                    .enumerate()
                    .map(|(dim_idx, expr)| match expr {
                        DimExpr::Static(n) => {
                            tracing::debug!("  Dim {}: Static({})", dim_idx, n);
                            *n
                        }
                        DimExpr::InputRef { input_id, dim_index } => {
                            tracing::debug!("  Dim {}: InputRef {{ input_id: {}, dim_index: {} }}", dim_idx, input_id, dim_index);

                            // WORKAROUND: Use maximum dimension across all inputs at this index
                            // This handles cases where the compiler references the wrong input
                            // (e.g., T5 encoder references attention_mask[1,1] instead of input_ids[1,128])
                            let mut max_value = 1;
                            for (_i, tensor) in input_tensors.iter().enumerate() {
                                if *dim_index < tensor.shape.len() {
                                    max_value = max_value.max(tensor.shape[*dim_index]);
                                }
                            }

                            tracing::debug!("    -> {} (max across all inputs at dim {})", max_value, dim_index);
                            max_value
                        }
                    })
                    .collect();

                let numel: usize = resolved_dims.iter().product();
                let size_bytes = numel * 4; // f32 = 4 bytes

                // Convert to 4D shape for storage
                let shape_4d = Self::shape_to_4d(&resolved_dims);

                tracing::debug!(
                    "Output {} has dynamic shape: resolved {:?} from shape_expr -> {} bytes (metadata was {} bytes)",
                    idx, resolved_dims, size_bytes, metadata_size_bytes
                );

                (size_bytes, shape_4d)
            } else {
                // No shape_expr, use static metadata
                (metadata_size_bytes, requirements.output_shapes[idx])
            };

            tracing::trace!(
                "Allocating output buffer {}: {} bytes, shape {:?}",
                idx,
                actual_size,
                actual_shape
            );

            let handle = self
                .backend
                .allocate_buffer(actual_size)
                .map_err(|e| anyhow::anyhow!("Failed to allocate output buffer {}: {:?}", idx, e))?;

            handles.push(handle);
            shapes.push(actual_shape);
        }

        Ok((handles, shapes))
    }

    /// Convert shape to 4D representation, padding with 1s if needed.
    fn shape_to_4d(shape: &[usize]) -> [usize; 4] {
        match shape.len() {
            0 => [1, 1, 1, 1],
            1 => [shape[0], 1, 1, 1],
            2 => [shape[0], shape[1], 1, 1],
            3 => [shape[0], shape[1], shape[2], 1],
            _ => [shape[0], shape[1], shape[2], shape[3]],
        }
    }

    /// Convert output buffers to named tensors.
    ///
    /// Uses the provided output_shapes (computed at runtime for dynamic shapes)
    /// instead of the static shapes from requirements.
    fn buffers_to_outputs(
        &self,
        output_handles: Vec<BufferHandle>,
        output_shapes: Vec<[usize; 4]>,
        requirements: &BufferRequirements,
    ) -> Result<HashMap<String, Tensor>> {
        let mut outputs = HashMap::new();

        for (idx, handle) in output_handles.iter().enumerate() {
            let shape = output_shapes[idx];

            // Convert [usize; 4] to Vec<usize>, removing trailing 1s
            let shape_vec: Vec<usize> = shape
                .iter()
                .copied()
                .rev()
                .skip_while(|&x| x == 1)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();

            tracing::trace!(
                "Output {}: shape {:?} (from {:?})",
                idx,
                shape_vec,
                shape
            );

            let tensor = self.buffer_handle_to_tensor(*handle, shape_vec)?;

            // Name convention: "output" for single, "output_N" for multiple
            let name = if requirements.num_outputs == 1 {
                "output".to_string()
            } else {
                format!("output_{}", idx)
            };

            outputs.insert(name, tensor);
        }

        Ok(outputs)
    }

    /// Execute the model with given input tensors.
    ///
    /// # Arguments
    /// * `inputs` - Map of input name → tensor
    ///
    /// # Returns
    /// Map of output name → tensor
    ///
    /// # Input Ordering Convention
    /// Since .holo files don't store tensor names, inputs are mapped to buffer indices
    /// using alphabetical ordering. For example, if your model expects:
    /// - "input_ids" → buffer index 1
    /// - "attention_mask" → buffer index 0
    ///
    /// Because "attention_mask" comes before "input_ids" alphabetically.
    ///
    /// # Example
    /// ```ignore
    /// let mut executor = ModelExecutor::from_holo_file(path)?;
    ///
    /// let mut inputs = HashMap::new();
    /// inputs.insert("input_ids".to_string(), input_tensor);
    /// inputs.insert("attention_mask".to_string(), attention_tensor);
    ///
    /// let outputs = executor.execute(inputs)?;
    /// let result = outputs.get("output").unwrap();
    /// ```
    pub fn execute(&mut self, inputs: HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
        tracing::debug!("Executing model with {} inputs", inputs.len());

        // Get buffer requirements from plan
        let requirements = self.get_buffer_requirements();
        tracing::info!(
            "Plan requires {} inputs, {} outputs (output_sizes: {:?})",
            requirements.num_inputs,
            requirements.num_outputs,
            requirements.output_sizes
        );

        // Extract sorted input tensors for shape resolution
        // (Must sort alphabetically to match buffer index ordering)
        let mut input_names: Vec<_> = inputs.keys().cloned().collect();
        input_names.sort();
        let sorted_tensors: Vec<&Tensor> = input_names.iter().map(|name| inputs.get(name).unwrap()).collect();

        // Map named inputs to positional buffers (alphabetically sorted)
        let input_handles = self.map_inputs_to_buffers(&inputs, &requirements)?;

        // Allocate output buffers with runtime shape resolution
        let (mut output_handles, output_shapes) = self.allocate_output_buffers_with_shapes(&requirements, &sorted_tensors)?;

        // DEBUG: Check buffer references
        let plan = self.executor.plan();
        tracing::info!("Total operations in plan: {}", plan.ops.len());

        // Count operations by input count
        let mut single_input = 0;
        let mut dual_input = 0;
        let mut no_input = 0;
        let mut multi_input = 0;

        for op in &plan.ops {
            match op.input_refs.len() {
                0 => no_input += 1,
                1 => single_input += 1,
                2 => dual_input += 1,
                _ => multi_input += 1,
            }
        }

        tracing::info!(
            "Operation breakdown: {} no-input (constants), {} single-input, {} dual-input, {} multi-input",
            no_input, single_input, dual_input, multi_input
        );
        tracing::info!(
            "We're passing {} input buffers, {} output buffers",
            input_handles.len(),
            output_handles.len()
        );

        // Execute the plan
        self.executor
            .execute(&input_handles, &mut output_handles, &mut *self.backend)
            .map_err(|e| anyhow::anyhow!("Model execution failed: {:?}", e))?;

        tracing::debug!("Model execution completed successfully");

        // Convert output buffers to named tensors (using computed shapes)
        let outputs = self.buffers_to_outputs(output_handles.clone(), output_shapes, &requirements)?;

        // Free all buffers
        for handle in input_handles {
            self.backend
                .free_buffer(handle)
                .map_err(|e| anyhow::anyhow!("Failed to free input buffer: {:?}", e))?;
        }
        for handle in output_handles {
            self.backend
                .free_buffer(handle)
                .map_err(|e| anyhow::anyhow!("Failed to free output buffer: {:?}", e))?;
        }

        Ok(outputs)
    }

    /// Convert tensor to buffer handle (upload to backend).
    fn tensor_to_buffer_handle(&mut self, tensor: &Tensor) -> Result<BufferHandle> {
        let size_bytes = tensor.size_bytes();

        // Allocate buffer
        let handle = self
            .backend
            .allocate_buffer(size_bytes)
            .map_err(|e| anyhow::anyhow!("Failed to allocate buffer: {:?}", e))?;

        // Copy data to buffer
        let bytes = tensor.to_bytes();
        self.backend
            .copy_to_buffer(handle, &bytes)
            .map_err(|e| anyhow::anyhow!("Failed to copy data to buffer: {:?}", e))?;

        Ok(handle)
    }

    /// Convert buffer handle to tensor (download from backend).
    fn buffer_handle_to_tensor(&self, handle: BufferHandle, shape: Vec<usize>) -> Result<Tensor> {
        let numel: usize = shape.iter().product();
        let size_bytes = numel * std::mem::size_of::<f32>();

        // Copy data from buffer
        let mut bytes = vec![0u8; size_bytes];
        self.backend
            .copy_from_buffer(handle, &mut bytes)
            .map_err(|e| anyhow::anyhow!("Failed to copy data from buffer: {:?}", e))?;

        // Parse into tensor
        Tensor::from_bytes(&bytes, shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires compiled model
    fn test_executor_creation() {
        let encoder_path = Path::new("models/t5-small/compiled/encoder.holo");

        if !encoder_path.exists() {
            return;
        }

        let result = ModelExecutor::from_holo_file(encoder_path);
        assert!(result.is_ok());
    }

    #[test]
    #[ignore] // Requires compiled model
    fn test_buffer_requirements() {
        let encoder_path = Path::new("models/t5-small/compiled/encoder.holo");

        if !encoder_path.exists() {
            return;
        }

        let executor = ModelExecutor::from_holo_file(encoder_path).unwrap();
        let reqs = executor.get_buffer_requirements();

        // T5 encoder has 2 inputs (input_ids, attention_mask) and 1 output
        assert_eq!(reqs.num_inputs, 2);
        assert_eq!(reqs.num_outputs, 1);
    }

    #[test]
    #[ignore] // Requires compiled model
    fn test_encoder_execution() {
        let encoder_path = Path::new("models/t5-small/compiled/encoder.holo");

        if !encoder_path.exists() {
            return;
        }

        let mut executor = ModelExecutor::from_holo_file(encoder_path).unwrap();

        // T5 encoder inputs (alphabetically: attention_mask, input_ids)
        let mut inputs = HashMap::new();
        inputs.insert(
            "input_ids".to_string(),
            Tensor::from_token_ids(&vec![0, 123, 456, 1], vec![1, 4]),
        );
        inputs.insert(
            "attention_mask".to_string(),
            Tensor::ones(vec![1, 4]),
        );

        let result = executor.execute(inputs);
        assert!(result.is_ok(), "Execution failed: {:?}", result.err());

        let outputs = result.unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(outputs.contains_key("output"));

        // T5-small encoder output should be [batch, seq_len, 512]
        let output = outputs.get("output").unwrap();
        println!("Output shape: {:?}", output.shape);
    }
}
