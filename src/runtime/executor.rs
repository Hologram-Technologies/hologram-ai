//! Model executor for running compiled .holo models.

use anyhow::Result;
use std::collections::HashMap;
use std::path::Path;

use super::loader::load_and_compile_holo;
use super::tensors::Tensor;

use hologram_backend::{BufferHandle, PlanExecutor, ProgramBackend};

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

    /// Execute the model with given input tensors.
    ///
    /// # Arguments
    /// * `inputs` - Map of input name → tensor
    ///
    /// # Returns
    /// Map of output name → tensor
    ///
    /// # Example
    /// ```ignore
    /// let mut executor = ModelExecutor::from_holo_file(path)?;
    ///
    /// let mut inputs = HashMap::new();
    /// inputs.insert("input_ids".to_string(), input_tensor);
    ///
    /// let outputs = executor.execute(inputs)?;
    /// let result = outputs.get("last_hidden_state").unwrap();
    /// ```
    pub fn execute(&mut self, inputs: HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
        tracing::debug!("Executing model with {} inputs", inputs.len());

        // Convert input tensors to buffer handles
        let input_handles: Result<Vec<BufferHandle>> = inputs
            .values()
            .map(|tensor| self.tensor_to_buffer_handle(tensor))
            .collect();
        let input_handles = input_handles?;

        // Allocate output buffers
        // NOTE: For MVP, we'll assume output size same as input size
        // In a full implementation, we'd query the plan for output shapes
        let output_size = inputs.values().next().map(|t| t.size_bytes()).unwrap_or(0);
        let output_handle = self
            .backend
            .allocate_buffer(output_size)
            .map_err(|e| anyhow::anyhow!("Failed to allocate output buffer: {:?}", e))?;

        let mut output_handles = vec![output_handle];

        // Execute the plan
        self.executor
            .execute(&input_handles, &mut output_handles, &mut *self.backend)
            .map_err(|e| anyhow::anyhow!("Model execution failed: {:?}", e))?;

        tracing::debug!("Model execution completed successfully");

        // Convert output buffers back to tensors
        let mut outputs = HashMap::new();

        // For MVP, we'll use a simple output naming scheme
        // In a full implementation, we'd get output names from the plan
        if let Some(output_handle) = output_handles.first() {
            // Infer output shape based on input (simplified)
            let input_shape: Vec<usize> = inputs.values().next().map(|t| t.shape.clone()).unwrap_or_default();
            let output_tensor = self.buffer_handle_to_tensor(*output_handle, input_shape)?;

            // Use first output name from plan or default
            outputs.insert("output".to_string(), output_tensor);
        }

        // Free input buffers
        for handle in input_handles {
            self.backend
                .free_buffer(handle)
                .map_err(|e| anyhow::anyhow!("Failed to free input buffer: {:?}", e))?;
        }

        // Note: Output buffers are kept allocated for result

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
    fn test_encoder_execution() {
        let encoder_path = Path::new("models/t5-small/compiled/encoder.holo");

        if !encoder_path.exists() {
            return;
        }

        let mut executor = ModelExecutor::from_holo_file(encoder_path).unwrap();

        // Create sample input
        let input_ids = Tensor::from_token_ids(&vec![0, 123, 456, 1], vec![1, 4]);

        let mut inputs = HashMap::new();
        inputs.insert("input_ids".to_string(), input_ids);

        let result = executor.execute(inputs);
        assert!(result.is_ok(), "Execution failed: {:?}", result.err());
    }
}
