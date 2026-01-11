//! Weight map for storing and retrieving model weights.

use crate::error::{CommonError, Result};
use std::collections::HashMap;

/// Data type for weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightDtype {
    /// 32-bit floating point.
    F32,
    /// 16-bit floating point.
    F16,
    /// Brain floating point 16.
    BF16,
}

/// A single weight tensor with metadata.
#[derive(Debug, Clone)]
pub struct WeightTensor {
    /// Raw weight data as bytes.
    pub data: Vec<u8>,
    /// Shape of the tensor.
    pub shape: Vec<usize>,
    /// Data type.
    pub dtype: WeightDtype,
}

impl WeightTensor {
    /// Create a new weight tensor from F32 data.
    pub fn from_f32(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        Self {
            data: bytes,
            shape,
            dtype: WeightDtype::F32,
        }
    }

    /// Create a new weight tensor from F16 data.
    pub fn from_f16(data: Vec<half::f16>, shape: Vec<usize>) -> Self {
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        Self {
            data: bytes,
            shape,
            dtype: WeightDtype::F16,
        }
    }

    /// Get the number of elements in the tensor.
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Convert weight data to F32 slice (only valid if dtype is F32).
    ///
    /// # Safety
    /// This reinterprets the underlying bytes as f32. Only call if dtype is F32.
    pub fn as_f32_slice(&self) -> Option<&[f32]> {
        if self.dtype != WeightDtype::F32 {
            return None;
        }
        // SAFETY: Data was created from f32 values with proper alignment
        let ptr = self.data.as_ptr() as *const f32;
        let len = self.data.len() / 4;
        Some(unsafe { std::slice::from_raw_parts(ptr, len) })
    }

    /// Convert weight to F32 Vec, dequantizing if necessary.
    pub fn to_f32_vec(&self) -> Vec<f32> {
        match self.dtype {
            WeightDtype::F32 => self.as_f32_slice().unwrap().to_vec(),
            WeightDtype::F16 => {
                let ptr = self.data.as_ptr() as *const half::f16;
                let len = self.data.len() / 2;
                let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
                slice.iter().map(|h| h.to_f32()).collect()
            }
            WeightDtype::BF16 => {
                let ptr = self.data.as_ptr() as *const half::bf16;
                let len = self.data.len() / 2;
                let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
                slice.iter().map(|h| h.to_f32()).collect()
            }
        }
    }
}

/// A map of weight names to weight tensors.
#[derive(Debug, Default)]
pub struct WeightMap {
    weights: HashMap<String, WeightTensor>,
}

impl WeightMap {
    /// Create a new empty weight map.
    pub fn new() -> Self {
        Self {
            weights: HashMap::new(),
        }
    }

    /// Insert a weight tensor.
    pub fn insert(&mut self, name: String, tensor: WeightTensor) {
        self.weights.insert(name, tensor);
    }

    /// Get a weight tensor by name.
    pub fn get(&self, name: &str) -> Option<&WeightTensor> {
        self.weights.get(name)
    }

    /// Get a weight tensor by name, returning an error if not found.
    pub fn get_required(&self, name: &str) -> Result<&WeightTensor> {
        self.weights
            .get(name)
            .ok_or_else(|| CommonError::WeightNotFound(name.to_string()))
    }

    /// Check if a weight exists.
    pub fn contains(&self, name: &str) -> bool {
        self.weights.contains_key(name)
    }

    /// Get the number of weights.
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    /// Check if the weight map is empty.
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }

    /// Iterate over all weights.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &WeightTensor)> {
        self.weights.iter()
    }

    /// Get all weight names.
    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.weights.keys()
    }

    /// Total size of all weights in bytes.
    pub fn total_size_bytes(&self) -> usize {
        self.weights.values().map(|w| w.data.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_tensor_from_f32() {
        let data = vec![1.0_f32, 2.0, 3.0, 4.0];
        let tensor = WeightTensor::from_f32(data.clone(), vec![2, 2]);

        assert_eq!(tensor.shape, vec![2, 2]);
        assert_eq!(tensor.dtype, WeightDtype::F32);
        assert_eq!(tensor.num_elements(), 4);

        let slice = tensor.as_f32_slice().unwrap();
        assert_eq!(slice, &data[..]);
    }

    #[test]
    fn test_weight_tensor_from_f16() {
        let data: Vec<half::f16> = vec![1.0, 2.0, 3.0]
            .into_iter()
            .map(half::f16::from_f32)
            .collect();
        let tensor = WeightTensor::from_f16(data, vec![3]);

        assert_eq!(tensor.shape, vec![3]);
        assert_eq!(tensor.dtype, WeightDtype::F16);
        assert_eq!(tensor.num_elements(), 3);
    }

    #[test]
    fn test_weight_tensor_to_f32_vec() {
        // F32 tensor
        let f32_tensor = WeightTensor::from_f32(vec![1.0, 2.0, 3.0], vec![3]);
        let f32_vec = f32_tensor.to_f32_vec();
        assert_eq!(f32_vec, vec![1.0, 2.0, 3.0]);

        // F16 tensor
        let f16_data: Vec<half::f16> = vec![1.0, 2.0, 3.0]
            .into_iter()
            .map(half::f16::from_f32)
            .collect();
        let f16_tensor = WeightTensor::from_f16(f16_data, vec![3]);
        let converted = f16_tensor.to_f32_vec();
        assert!((converted[0] - 1.0).abs() < 0.01);
        assert!((converted[1] - 2.0).abs() < 0.01);
        assert!((converted[2] - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_weight_map_operations() {
        let mut map = WeightMap::new();
        assert!(map.is_empty());
        assert_eq!(map.len(), 0);

        let tensor = WeightTensor::from_f32(vec![1.0, 2.0], vec![2]);
        map.insert("weight1".to_string(), tensor);

        assert!(!map.is_empty());
        assert_eq!(map.len(), 1);
        assert!(map.contains("weight1"));
        assert!(!map.contains("weight2"));

        let retrieved = map.get("weight1").unwrap();
        assert_eq!(retrieved.shape, vec![2]);

        let required = map.get_required("weight1").unwrap();
        assert_eq!(required.shape, vec![2]);

        let not_found = map.get_required("nonexistent");
        assert!(not_found.is_err());
    }

    #[test]
    fn test_weight_map_iteration() {
        let mut map = WeightMap::new();
        map.insert("a".to_string(), WeightTensor::from_f32(vec![1.0], vec![1]));
        map.insert(
            "b".to_string(),
            WeightTensor::from_f32(vec![2.0, 3.0], vec![2]),
        );

        let names: Vec<_> = map.names().collect();
        assert_eq!(names.len(), 2);

        let total_bytes = map.total_size_bytes();
        assert_eq!(total_bytes, 4 + 8); // 1 f32 + 2 f32s
    }

    #[test]
    fn test_weight_dtype_equality() {
        assert_eq!(WeightDtype::F32, WeightDtype::F32);
        assert_ne!(WeightDtype::F32, WeightDtype::F16);
        assert_ne!(WeightDtype::F16, WeightDtype::BF16);
    }
}
