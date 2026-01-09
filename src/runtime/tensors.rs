//! Tensor types and I/O conversions for runtime execution.

use anyhow::Result;

/// Tensor data type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    /// 32-bit float (primary type for hologram)
    F32,
    /// 64-bit signed integer (for token IDs)
    I64,
}

/// Tensor wrapper for runtime I/O.
#[derive(Debug, Clone)]
pub struct Tensor {
    /// Flattened tensor data (stored as f32)
    pub data: Vec<f32>,
    /// Tensor shape
    pub shape: Vec<usize>,
    /// Data type
    pub dtype: DType,
}

impl Tensor {
    /// Create a new tensor from f32 data.
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self {
            data,
            shape,
            dtype: DType::F32,
        }
    }

    /// Create tensor from token IDs (u32 → f32 conversion).
    ///
    /// # Arguments
    /// * `ids` - Token IDs as u32
    /// * `shape` - Tensor shape (e.g., [batch=1, seq_len])
    pub fn from_token_ids(ids: &[u32], shape: Vec<usize>) -> Self {
        let data: Vec<f32> = ids.iter().map(|&id| id as f32).collect();
        Self {
            data,
            shape,
            dtype: DType::I64, // Mark as I64 semantically, but stored as f32
        }
    }

    /// Get data as f32 slice.
    pub fn to_f32(&self) -> &[f32] {
        &self.data
    }

    /// Convert data to token IDs (f32 → u32 conversion).
    pub fn to_token_ids(&self) -> Vec<u32> {
        self.data.iter().map(|&f| f as u32).collect()
    }

    /// Get number of elements.
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// Reshape tensor (validates size matches).
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Self> {
        let new_numel: usize = new_shape.iter().product();
        anyhow::ensure!(
            new_numel == self.numel(),
            "Reshape size mismatch: {} → {}",
            self.numel(),
            new_numel
        );

        Ok(Self {
            data: self.data.clone(),
            shape: new_shape,
            dtype: self.dtype,
        })
    }

    /// Get tensor size in bytes (f32 representation).
    pub fn size_bytes(&self) -> usize {
        match self.dtype {
            DType::F32 => self.data.len() * std::mem::size_of::<f32>(),
            DType::I64 => self.data.len() * std::mem::size_of::<i64>(),
        }
    }

    /// Serialize tensor data to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        match self.dtype {
            DType::F32 => self
                .data
                .iter()
                .flat_map(|&f| f.to_le_bytes())
                .collect(),
            DType::I64 => self
                .data
                .iter()
                .flat_map(|&f| (f as i64).to_le_bytes())
                .collect(),
        }
    }

    /// Deserialize tensor data from bytes.
    pub fn from_bytes(bytes: &[u8], shape: Vec<usize>) -> Result<Self> {
        anyhow::ensure!(
            bytes.len().is_multiple_of(4),
            "Byte length {} not divisible by 4",
            bytes.len()
        );

        let data: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        let numel: usize = shape.iter().product();
        anyhow::ensure!(
            data.len() == numel,
            "Shape mismatch: {} elements from bytes, {} from shape",
            data.len(),
            numel
        );

        Ok(Self {
            data,
            shape,
            dtype: DType::F32,
        })
    }
}

/// Infer tensor shape from data and input name hints.
///
/// For T5 models:
/// - input_ids: [batch=1, seq_len]
/// - attention_mask: [batch=1, seq_len]
/// - encoder_hidden_states: [batch=1, seq_len, 512]
/// - decoder_input_ids: [batch=1, decoder_seq_len]
pub fn infer_tensor_shape(data: &[f32], input_name: &str) -> Result<Vec<usize>> {
    if input_name.contains("input_ids") || input_name.contains("attention_mask") {
        // Assume batch=1, infer seq_len
        Ok(vec![1, data.len()])
    } else if input_name.contains("hidden_states") {
        // Assume [1, seq_len, 512] - infer seq_len
        let hidden_dim = 512;
        let seq_len = data.len() / hidden_dim;
        anyhow::ensure!(
            data.len().is_multiple_of(hidden_dim),
            "Hidden state size {} not divisible by hidden_dim {}",
            data.len(),
            hidden_dim
        );
        Ok(vec![1, seq_len, hidden_dim])
    } else {
        // Default: 2D tensor with batch=1
        Ok(vec![1, data.len()])
    }
}

/// Infer tensor dtype from input name conventions.
///
/// Returns F32 for all inputs because the hologram backend works with f32 internally.
/// ONNX models may declare int64 for token IDs, but hologram processes them as f32
/// (the values are still integer token IDs, just stored as f32).
pub fn infer_tensor_dtype(_input_name: &str) -> DType {
    // Always use F32 since hologram backend uses f32 internally.
    // Token IDs like [817, 140, 3, ...] are stored as [817.0, 140.0, 3.0, ...].
    DType::F32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::new(data.clone(), vec![2, 2]);

        assert_eq!(tensor.data, data);
        assert_eq!(tensor.shape, vec![2, 2]);
        assert_eq!(tensor.dtype, DType::F32);
        assert_eq!(tensor.numel(), 4);
    }

    #[test]
    fn test_token_id_conversion() {
        let token_ids = vec![0, 123, 456, 789];
        let tensor = Tensor::from_token_ids(&token_ids, vec![1, 4]);

        assert_eq!(tensor.shape, vec![1, 4]);
        assert_eq!(tensor.dtype, DType::I64);
        assert_eq!(tensor.to_token_ids(), token_ids);
    }

    #[test]
    fn test_reshape() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let reshaped = tensor.reshape(vec![2, 2]).unwrap();

        assert_eq!(reshaped.shape, vec![2, 2]);
        assert_eq!(reshaped.data, tensor.data);
    }

    #[test]
    fn test_reshape_invalid() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let result = tensor.reshape(vec![2, 3]);

        assert!(result.is_err());
    }

    #[test]
    fn test_bytes_roundtrip() {
        let tensor = Tensor::new(vec![1.5, 2.5, 3.5], vec![3]);
        let bytes = tensor.to_bytes();
        let restored = Tensor::from_bytes(&bytes, vec![3]).unwrap();

        assert_eq!(restored.data, tensor.data);
        assert_eq!(restored.shape, tensor.shape);
    }

    #[test]
    fn test_infer_shape_input_ids() {
        let data = vec![1.0; 128];
        let shape = infer_tensor_shape(&data, "input_ids").unwrap();

        assert_eq!(shape, vec![1, 128]);
    }

    #[test]
    fn test_infer_shape_hidden_states() {
        let data = vec![1.0; 512 * 10]; // 10 tokens, 512 hidden_dim
        let shape = infer_tensor_shape(&data, "encoder_hidden_states").unwrap();

        assert_eq!(shape, vec![1, 10, 512]);
    }

    #[test]
    fn test_infer_tensor_dtype() {
        assert_eq!(infer_tensor_dtype("input_ids"), DType::I64);
        assert_eq!(infer_tensor_dtype("attention_mask"), DType::I64);
        assert_eq!(infer_tensor_dtype("encoder_hidden_states"), DType::F32);
    }
}
