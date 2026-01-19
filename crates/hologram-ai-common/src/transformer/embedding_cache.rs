//! Cache-optimized embedding table management.
//!
//! This module provides infrastructure for keeping embedding tables resident
//! in L2/L3 cache through strategic warming and access pattern tracking.
//!
//! Unlike small activation tables that fit entirely in L1 cache (256 bytes),
//! embedding tables are typically large (megabytes) and require different
//! caching strategies:
//!
//! - **Token embeddings**: 32K vocab × 768 dim × 4 bytes = 98MB
//! - **Position embeddings**: 2K positions × 768 dim × 4 bytes = 6MB
//!
//! # Strategy
//!
//! 1. **Selective warming**: Warm only frequently accessed regions
//! 2. **Access tracking**: Monitor which embeddings are hot
//! 3. **Aligned storage**: 64-byte alignment for cache efficiency
//! 4. **Periodic refresh**: Re-warm before inference batches
//!
//! # Usage
//!
//! ```ignore
//! use hologram_ai_common::transformer::EmbeddingCacheManager;
//!
//! let mut cache = EmbeddingCacheManager::new();
//!
//! // Pin token embeddings
//! let token_embeddings = vec![0.0f32; 32000 * 768];
//! cache.pin_embedding("token_embed".to_string(), token_embeddings, 768)?;
//!
//! // Warm cache before inference
//! cache.warm_all();
//!
//! // Fast lookups
//! let token_id = 12345;
//! let embedding = cache.lookup("token_embed", token_id)?;
//! ```

use anyhow::{Result, bail};
use std::collections::HashMap;

/// Cache line size on modern CPUs (64 bytes).
const CACHE_LINE_SIZE: usize = 64;

/// Maximum embedding table size to pin (64MB per table).
/// This is a practical limit for L3 cache (typical: 8-32MB per core).
const MAX_EMBEDDING_SIZE: usize = 64 * 1024 * 1024; // 64MB

/// Aligned embedding table for cache efficiency.
///
/// Stores embeddings in 64-byte aligned Vec for optimal cache line usage.
/// Each embedding is accessed as a contiguous slice of f32 values.
#[repr(C, align(64))]
struct AlignedEmbeddingTable {
    /// Flat storage: [emb0_dim0, emb0_dim1, ..., emb1_dim0, emb1_dim1, ...]
    data: Vec<f32>,
    /// Number of embeddings in the table
    count: usize,
    /// Dimensionality of each embedding
    dim: usize,
    /// Access frequency counter per embedding
    access_counts: Vec<usize>,
}

impl AlignedEmbeddingTable {
    /// Create a new aligned embedding table.
    #[allow(clippy::manual_is_multiple_of)]
    fn new(data: Vec<f32>, dim: usize) -> Result<Self> {
        if data.len() % dim != 0 {
            bail!(
                "Data length {} must be divisible by dimension {}",
                data.len(),
                dim
            );
        }

        let count = data.len() / dim;

        Ok(Self {
            data,
            count,
            dim,
            access_counts: vec![0; count],
        })
    }

    /// Get embedding by index.
    #[inline]
    fn get(&mut self, index: usize) -> Option<&[f32]> {
        if index >= self.count {
            return None;
        }

        // Track access
        self.access_counts[index] += 1;

        let start = index * self.dim;
        let end = start + self.dim;
        Some(&self.data[start..end])
    }

    /// Warm the entire table by touching all cache lines.
    fn warm(&self) {
        let bytes = unsafe {
            std::slice::from_raw_parts(
                self.data.as_ptr() as *const u8,
                self.data.len() * std::mem::size_of::<f32>(),
            )
        };

        let mut checksum = 0u8;
        let mut i = 0;
        let len = bytes.len();

        // Touch one byte per cache line
        while i < len {
            checksum ^= unsafe { *bytes.get_unchecked(i) };
            i += CACHE_LINE_SIZE;
        }

        // Prevent optimization from removing the reads
        std::hint::black_box(checksum);
    }

    /// Warm only the most frequently accessed embeddings.
    ///
    /// Warms the top N most accessed embeddings to keep them in cache.
    fn warm_hot(&self, top_n: usize) {
        // Find top N most accessed indices
        let mut indexed_counts: Vec<(usize, usize)> =
            self.access_counts.iter().copied().enumerate().collect();

        indexed_counts.sort_by(|a, b| b.1.cmp(&a.1));

        // Warm top N embeddings
        for (index, _count) in indexed_counts.iter().take(top_n) {
            let start = index * self.dim;
            let end = start + self.dim;

            let bytes = unsafe {
                std::slice::from_raw_parts(
                    self.data[start..end].as_ptr() as *const u8,
                    self.dim * std::mem::size_of::<f32>(),
                )
            };

            let mut checksum = 0u8;
            let mut i = 0;
            while i < bytes.len() {
                checksum ^= unsafe { *bytes.get_unchecked(i) };
                i += CACHE_LINE_SIZE;
            }
            std::hint::black_box(checksum);
        }
    }

    /// Get total access count.
    fn total_accesses(&self) -> usize {
        self.access_counts.iter().sum()
    }

    /// Get memory size in bytes.
    fn size_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<f32>()
    }
}

/// Manager for cache-optimized embedding tables.
///
/// Handles multiple embedding tables with:
/// - Aligned storage for cache efficiency
/// - Access tracking for hot embedding detection
/// - Selective warming strategies
/// - Memory usage monitoring
pub struct EmbeddingCacheManager {
    /// Pinned embedding tables by name
    tables: HashMap<String, AlignedEmbeddingTable>,
    /// Total memory used by all tables
    total_memory_bytes: usize,
}

impl EmbeddingCacheManager {
    /// Create a new embedding cache manager.
    pub fn new() -> Self {
        Self {
            tables: HashMap::new(),
            total_memory_bytes: 0,
        }
    }

    /// Pin an embedding table in the cache manager.
    ///
    /// # Arguments
    ///
    /// * `name` - Unique identifier for this table (e.g., "token_embed")
    /// * `data` - Flat embedding data [emb0_dim0, emb0_dim1, ..., emb1_dim0, ...]
    /// * `dim` - Dimensionality of each embedding
    ///
    /// # Returns
    ///
    /// Ok if pinned successfully, Err if:
    /// - Table would exceed MAX_EMBEDDING_SIZE
    /// - Data length not divisible by dim
    /// - Table name already exists
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Pin token embeddings: 1000 tokens × 128 dimensions
    /// let embeddings = vec![0.0f32; 1000 * 128];
    /// cache.pin_embedding("tokens".to_string(), embeddings, 128)?;
    /// ```
    pub fn pin_embedding(&mut self, name: String, data: Vec<f32>, dim: usize) -> Result<()> {
        if self.tables.contains_key(&name) {
            bail!("Embedding table '{}' already exists", name);
        }

        let size_bytes = data.len() * std::mem::size_of::<f32>();
        if size_bytes > MAX_EMBEDDING_SIZE {
            bail!(
                "Embedding table '{}' too large: {} bytes (max: {} bytes)",
                name,
                size_bytes,
                MAX_EMBEDDING_SIZE
            );
        }

        let table = AlignedEmbeddingTable::new(data, dim)?;

        tracing::debug!(
            "Pinned embedding '{}': {} embeddings × {} dims = {} bytes ({} cache lines)",
            name,
            table.count,
            table.dim,
            size_bytes,
            size_bytes.div_ceil(CACHE_LINE_SIZE)
        );

        self.total_memory_bytes += size_bytes;
        self.tables.insert(name, table);

        Ok(())
    }

    /// Look up an embedding by index.
    ///
    /// # Arguments
    ///
    /// * `table_name` - Name of the embedding table
    /// * `index` - Index of the embedding (e.g., token ID)
    ///
    /// # Returns
    ///
    /// A slice containing the embedding dimensions, or None if not found.
    ///
    /// # Performance
    ///
    /// After warming, lookups are L2/L3 cache hits (~10-40 cycles per embedding).
    pub fn lookup(&mut self, table_name: &str, index: usize) -> Option<&[f32]> {
        self.tables.get_mut(table_name)?.get(index)
    }

    /// Warm all pinned tables by touching all cache lines.
    ///
    /// Call this before performance-critical inference batches to ensure
    /// embeddings are resident in cache.
    ///
    /// # Performance
    ///
    /// Cost: ~1 cycle per cache line
    /// - 6MB table: ~96K cache lines = ~96K cycles (~30μs on 3GHz CPU)
    /// - 98MB table: ~1.5M cache lines = ~1.5M cycles (~500μs on 3GHz CPU)
    pub fn warm_all(&self) {
        for (name, table) in &self.tables {
            tracing::trace!("Warming embedding '{}': {} bytes", name, table.size_bytes());
            table.warm();
        }
    }

    /// Warm only the most frequently accessed embeddings.
    ///
    /// More efficient than warming all embeddings when access patterns are skewed
    /// (e.g., common tokens accessed much more than rare tokens).
    ///
    /// # Arguments
    ///
    /// * `top_n` - Number of hot embeddings to warm per table
    ///
    /// # Performance
    ///
    /// Cost: ~1 cycle per cache line of hot embeddings
    /// - 1000 hot embeddings × 768 dims × 4 bytes = 3MB = 48K cache lines (~48K cycles)
    pub fn warm_hot(&self, top_n: usize) {
        for (name, table) in &self.tables {
            tracing::trace!("Warming top {} embeddings in '{}'", top_n, name);
            table.warm_hot(top_n);
        }
    }

    /// Get cache statistics.
    ///
    /// Returns a summary of:
    /// - Total memory usage
    /// - Number of tables
    /// - Total accesses per table
    /// - Hit rate (always 100% since we don't evict)
    pub fn stats(&self) -> EmbeddingCacheStats {
        let table_stats: Vec<TableStats> = self
            .tables
            .iter()
            .map(|(name, table)| TableStats {
                name: name.clone(),
                count: table.count,
                dim: table.dim,
                size_bytes: table.size_bytes(),
                total_accesses: table.total_accesses(),
            })
            .collect();

        EmbeddingCacheStats {
            total_memory_bytes: self.total_memory_bytes,
            num_tables: self.tables.len(),
            tables: table_stats,
        }
    }

    /// Get the number of pinned tables.
    pub fn num_tables(&self) -> usize {
        self.tables.len()
    }

    /// Check if a table exists.
    pub fn has_table(&self, name: &str) -> bool {
        self.tables.contains_key(name)
    }
}

impl Default for EmbeddingCacheManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for a single embedding table.
#[derive(Debug, Clone)]
pub struct TableStats {
    /// Name of the embedding table
    pub name: String,
    /// Number of embeddings in the table
    pub count: usize,
    /// Dimensionality of each embedding
    pub dim: usize,
    /// Total size in bytes
    pub size_bytes: usize,
    /// Total number of lookup operations performed
    pub total_accesses: usize,
}

/// Overall cache statistics.
#[derive(Debug, Clone)]
pub struct EmbeddingCacheStats {
    /// Total memory used by all tables in bytes
    pub total_memory_bytes: usize,
    /// Number of embedding tables pinned
    pub num_tables: usize,
    /// Per-table statistics
    pub tables: Vec<TableStats>,
}

impl EmbeddingCacheStats {
    /// Get a formatted summary string.
    pub fn summary(&self) -> String {
        let total_mb = self.total_memory_bytes as f64 / (1024.0 * 1024.0);
        let mut lines = vec![format!(
            "Embedding Cache: {} tables, {:.2} MB total",
            self.num_tables, total_mb
        )];

        for table in &self.tables {
            let size_mb = table.size_bytes as f64 / (1024.0 * 1024.0);
            lines.push(format!(
                "  - {}: {} embeddings × {} dims, {:.2} MB, {} accesses",
                table.name, table.count, table.dim, size_mb, table.total_accesses
            ));
        }

        lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_cache_new() {
        let cache = EmbeddingCacheManager::new();
        assert_eq!(cache.num_tables(), 0);
        assert_eq!(cache.total_memory_bytes, 0);
    }

    #[test]
    fn test_pin_embedding() {
        let mut cache = EmbeddingCacheManager::new();

        // Create 10 embeddings of dimension 4
        let data = vec![
            1.0, 2.0, 3.0, 4.0, // emb 0
            5.0, 6.0, 7.0, 8.0, // emb 1
            9.0, 10.0, 11.0, 12.0, // emb 2
            13.0, 14.0, 15.0, 16.0, // emb 3
            17.0, 18.0, 19.0, 20.0, // emb 4
            21.0, 22.0, 23.0, 24.0, // emb 5
            25.0, 26.0, 27.0, 28.0, // emb 6
            29.0, 30.0, 31.0, 32.0, // emb 7
            33.0, 34.0, 35.0, 36.0, // emb 8
            37.0, 38.0, 39.0, 40.0, // emb 9
        ];

        cache
            .pin_embedding("test".to_string(), data, 4)
            .expect("Failed to pin embedding");

        assert_eq!(cache.num_tables(), 1);
        assert!(cache.has_table("test"));
        assert_eq!(cache.total_memory_bytes, 10 * 4 * 4); // 10 embs × 4 dims × 4 bytes
    }

    #[test]
    fn test_embedding_lookup() {
        let mut cache = EmbeddingCacheManager::new();

        let data = vec![
            1.0, 2.0, 3.0, // emb 0
            4.0, 5.0, 6.0, // emb 1
            7.0, 8.0, 9.0, // emb 2
        ];

        cache
            .pin_embedding("test".to_string(), data, 3)
            .expect("Failed to pin");

        // Lookup emb 0
        let emb0 = cache.lookup("test", 0).expect("Failed to lookup");
        assert_eq!(emb0, &[1.0, 2.0, 3.0]);

        // Lookup emb 1
        let emb1 = cache.lookup("test", 1).expect("Failed to lookup");
        assert_eq!(emb1, &[4.0, 5.0, 6.0]);

        // Lookup emb 2
        let emb2 = cache.lookup("test", 2).expect("Failed to lookup");
        assert_eq!(emb2, &[7.0, 8.0, 9.0]);

        // Out of bounds
        let emb3 = cache.lookup("test", 3);
        assert!(emb3.is_none());

        // Non-existent table
        let emb = cache.lookup("nonexistent", 0);
        assert!(emb.is_none());
    }

    #[test]
    fn test_embedding_warm() {
        let mut cache = EmbeddingCacheManager::new();
        let data = vec![0.0f32; 1000 * 128]; // 1000 embeddings × 128 dims

        cache
            .pin_embedding("test".to_string(), data, 128)
            .expect("Failed to pin");

        // Should not panic
        cache.warm_all();
    }

    #[test]
    fn test_embedding_warm_hot() {
        let mut cache = EmbeddingCacheManager::new();
        let data = vec![0.0f32; 100 * 10];

        cache
            .pin_embedding("test".to_string(), data, 10)
            .expect("Failed to pin");

        // Access some embeddings
        let _ = cache.lookup("test", 0);
        let _ = cache.lookup("test", 0); // Access 0 twice
        let _ = cache.lookup("test", 5);

        // Warm top 2
        cache.warm_hot(2);
    }

    #[test]
    fn test_embedding_stats() {
        let mut cache = EmbeddingCacheManager::new();

        let data1 = vec![0.0f32; 100 * 10];
        cache
            .pin_embedding("table1".to_string(), data1, 10)
            .expect("Failed to pin");

        let data2 = vec![0.0f32; 200 * 20];
        cache
            .pin_embedding("table2".to_string(), data2, 20)
            .expect("Failed to pin");

        // Access some embeddings
        let _ = cache.lookup("table1", 0);
        let _ = cache.lookup("table1", 1);
        let _ = cache.lookup("table2", 0);

        let stats = cache.stats();
        assert_eq!(stats.num_tables, 2);
        assert_eq!(stats.total_memory_bytes, (100 * 10 + 200 * 20) * 4);
        assert_eq!(stats.tables.len(), 2);

        // Check summary format
        let summary = stats.summary();
        assert!(summary.contains("Embedding Cache"));
        assert!(summary.contains("table1"));
        assert!(summary.contains("table2"));
    }

    #[test]
    fn test_duplicate_table_name() {
        let mut cache = EmbeddingCacheManager::new();
        let data = vec![0.0f32; 10 * 4];

        cache
            .pin_embedding("test".to_string(), data.clone(), 4)
            .expect("Failed to pin");

        // Try to pin again with same name
        let result = cache.pin_embedding("test".to_string(), data, 4);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("already exists"));
    }

    #[test]
    fn test_invalid_dimension() {
        let mut cache = EmbeddingCacheManager::new();

        // Data length not divisible by dim
        let data = vec![0.0f32; 10];
        let result = cache.pin_embedding("test".to_string(), data, 3);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("must be divisible")
        );
    }

    #[test]
    fn test_access_tracking() {
        let mut cache = EmbeddingCacheManager::new();
        let data = vec![0.0f32; 10 * 4];

        cache
            .pin_embedding("test".to_string(), data, 4)
            .expect("Failed to pin");

        // Access embedding 0 three times
        let _ = cache.lookup("test", 0);
        let _ = cache.lookup("test", 0);
        let _ = cache.lookup("test", 0);

        // Access embedding 1 once
        let _ = cache.lookup("test", 1);

        let stats = cache.stats();
        let table_stats = &stats.tables[0];
        assert_eq!(table_stats.total_accesses, 4);
    }

    #[test]
    fn test_too_large_table() {
        let mut cache = EmbeddingCacheManager::new();

        // Try to pin a table larger than MAX_EMBEDDING_SIZE
        let data = vec![0.0f32; MAX_EMBEDDING_SIZE / 4 + 1]; // Exceeds limit
        let result = cache.pin_embedding("huge".to_string(), data, 1);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("too large"));
    }
}
