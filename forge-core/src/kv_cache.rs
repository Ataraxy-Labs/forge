use anyhow::{anyhow, Result};
use candle_core::{Device, Tensor, DType, IndexOp};
use parking_lot::Mutex;
use std::sync::Arc;

/// Size of each KV cache page in tokens
pub const PAGE_SIZE: usize = 16;

/// Paged KV Cache manager
/// Implements memory-efficient KV caching by allocating in fixed-size pages
/// rather than pre-allocating the full context window per request
pub struct PagedKVCache {
    /// Number of layers in the model
    _num_layers: usize,
    /// Number of attention heads
    num_heads: usize,
    /// Size of each attention head
    head_dim: usize,
    /// Data type for cache
    dtype: DType,
    /// Device (CPU/CUDA/Metal)
    device: Device,
    /// Pool of available pages
    free_pages: Arc<Mutex<Vec<usize>>>,
    /// Total number of pages allocated
    total_pages: usize,
    /// Actual K/V tensors per layer
    /// Shape: [num_layers, total_pages, PAGE_SIZE, num_heads, head_dim]
    _k_cache: Vec<Tensor>,
    _v_cache: Vec<Tensor>,
}

impl PagedKVCache {
    /// Create a new paged KV cache
    pub fn new(
        num_layers: usize,
        num_heads: usize,
        head_dim: usize,
        total_pages: usize,
        dtype: DType,
        device: Device,
    ) -> Result<Self> {
        let mut k_cache = Vec::with_capacity(num_layers);
        let mut v_cache = Vec::with_capacity(num_layers);

        // Pre-allocate all pages for all layers
        for _ in 0..num_layers {
            let k = Tensor::zeros(
                (total_pages, PAGE_SIZE, num_heads, head_dim),
                dtype,
                &device,
            )?;
            let v = Tensor::zeros(
                (total_pages, PAGE_SIZE, num_heads, head_dim),
                dtype,
                &device,
            )?;
            k_cache.push(k);
            v_cache.push(v);
        }

        // Initialize free page pool (all pages available initially)
        let free_pages = (0..total_pages).collect();

        Ok(Self {
            _num_layers: num_layers,
            num_heads,
            head_dim,
            dtype,
            device,
            free_pages: Arc::new(Mutex::new(free_pages)),
            total_pages,
            _k_cache: k_cache,
            _v_cache: v_cache,
        })
    }

    /// Allocate a page from the pool
    pub fn allocate_page(&self) -> Option<usize> {
        self.free_pages.lock().pop()
    }

    /// Free a page back to the pool
    pub fn free_page(&self, page_id: usize) {
        self.free_pages.lock().push(page_id);
    }

    /// Free multiple pages
    pub fn free_pages(&self, page_ids: &[usize]) {
        let mut free_pages = self.free_pages.lock();
        free_pages.extend_from_slice(page_ids);
    }

    /// Get number of available pages
    pub fn num_free_pages(&self) -> usize {
        self.free_pages.lock().len()
    }

    /// Write K/V values to a specific page at given token positions
    ///
    /// # Arguments
    /// * `layer_idx` - The layer index to write to
    /// * `page_id` - The page ID to write to
    /// * `token_offset` - The starting token offset within the page (0 to PAGE_SIZE-1)
    /// * `k` - Key tensor, shape: [num_tokens, num_heads, head_dim]
    /// * `v` - Value tensor, shape: [num_tokens, num_heads, head_dim]
    pub fn write_kv(
        &mut self,
        layer_idx: usize,
        page_id: usize,
        token_offset: usize,
        k: &Tensor,
        v: &Tensor,
    ) -> Result<()> {
        // Validate inputs
        if layer_idx >= self._k_cache.len() {
            return Err(anyhow!("Layer index {} out of bounds", layer_idx));
        }
        if page_id >= self.total_pages {
            return Err(anyhow!("Page ID {} out of bounds", page_id));
        }
        if token_offset >= PAGE_SIZE {
            return Err(anyhow!("Token offset {} out of bounds", token_offset));
        }

        let num_tokens = k.dim(0)?;
        if token_offset + num_tokens > PAGE_SIZE {
            return Err(anyhow!(
                "Write would exceed page boundary: offset {} + tokens {} > PAGE_SIZE {}",
                token_offset, num_tokens, PAGE_SIZE
            ));
        }

        // Cache shape: [total_pages, PAGE_SIZE, num_heads, head_dim]
        // k, v shape: [num_tokens, num_heads, head_dim]

        // Get the current cache tensors
        let k_cache = &self._k_cache[layer_idx];
        let v_cache = &self._v_cache[layer_idx];

        // Extract the page we're writing to
        // Shape: [PAGE_SIZE, num_heads, head_dim]
        let k_page = k_cache.i(page_id)?;
        let v_page = v_cache.i(page_id)?;

        // Create the updated page by concatenating:
        // [existing before] + [new values] + [existing after]
        let k_before = if token_offset > 0 {
            Some(k_page.narrow(0, 0, token_offset)?)
        } else {
            None
        };
        let k_after = if token_offset + num_tokens < PAGE_SIZE {
            Some(k_page.narrow(0, token_offset + num_tokens, PAGE_SIZE - token_offset - num_tokens)?)
        } else {
            None
        };

        let v_before = if token_offset > 0 {
            Some(v_page.narrow(0, 0, token_offset)?)
        } else {
            None
        };
        let v_after = if token_offset + num_tokens < PAGE_SIZE {
            Some(v_page.narrow(0, token_offset + num_tokens, PAGE_SIZE - token_offset - num_tokens)?)
        } else {
            None
        };

        // Build the new page
        let new_k_page = match (k_before, k_after) {
            (Some(before), Some(after)) => Tensor::cat(&[&before, k, &after], 0)?,
            (Some(before), None) => Tensor::cat(&[&before, k], 0)?,
            (None, Some(after)) => Tensor::cat(&[k, &after], 0)?,
            (None, None) => k.clone(),
        };

        let new_v_page = match (v_before, v_after) {
            (Some(before), Some(after)) => Tensor::cat(&[&before, v, &after], 0)?,
            (Some(before), None) => Tensor::cat(&[&before, v], 0)?,
            (None, Some(after)) => Tensor::cat(&[v, &after], 0)?,
            (None, None) => v.clone(),
        };

        // Reconstruct the full cache with the updated page
        // This is not the most efficient approach, but it works
        // A better approach would use slice_scatter when available
        let mut k_pages = Vec::with_capacity(self.total_pages);
        let mut v_pages = Vec::with_capacity(self.total_pages);

        for i in 0..self.total_pages {
            if i == page_id {
                k_pages.push(new_k_page.unsqueeze(0)?);
                v_pages.push(new_v_page.unsqueeze(0)?);
            } else {
                k_pages.push(k_cache.i(i)?.unsqueeze(0)?);
                v_pages.push(v_cache.i(i)?.unsqueeze(0)?);
            }
        }

        let k_refs: Vec<&Tensor> = k_pages.iter().collect();
        let v_refs: Vec<&Tensor> = v_pages.iter().collect();

        self._k_cache[layer_idx] = Tensor::cat(&k_refs, 0)?;
        self._v_cache[layer_idx] = Tensor::cat(&v_refs, 0)?;

        Ok(())
    }

    /// Read K/V values for attention computation
    /// Returns concatenated K/V from all pages used by a request
    ///
    /// # Arguments
    /// * `layer_idx` - The layer index to read from
    /// * `page_ids` - List of page IDs to read from (in order)
    /// * `seq_len` - Total sequence length to read
    ///
    /// # Returns
    /// Tuple of (K, V) tensors with shape [seq_len, num_heads, head_dim]
    pub fn read_kv(
        &self,
        layer_idx: usize,
        page_ids: &[usize],
        seq_len: usize,
    ) -> Result<(Tensor, Tensor)> {
        if layer_idx >= self._k_cache.len() {
            return Err(anyhow!("Layer index {} out of bounds", layer_idx));
        }

        if page_ids.is_empty() {
            // Return empty tensors
            let k = Tensor::zeros(
                (0, self.num_heads, self.head_dim),
                self.dtype,
                &self.device,
            )?;
            let v = Tensor::zeros(
                (0, self.num_heads, self.head_dim),
                self.dtype,
                &self.device,
            )?;
            return Ok((k, v));
        }

        // Cache shape: [total_pages, PAGE_SIZE, num_heads, head_dim]
        let k_cache = &self._k_cache[layer_idx];
        let v_cache = &self._v_cache[layer_idx];

        // Gather tokens from pages
        let mut k_slices = Vec::with_capacity(page_ids.len());
        let mut v_slices = Vec::with_capacity(page_ids.len());

        let mut tokens_remaining = seq_len;

        for &page_id in page_ids {
            if page_id >= self.total_pages {
                return Err(anyhow!("Page ID {} out of bounds", page_id));
            }

            // Determine how many tokens to read from this page
            let tokens_in_page = std::cmp::min(tokens_remaining, PAGE_SIZE);

            if tokens_in_page > 0 {
                // Extract the page
                let k_page = k_cache.i(page_id)?;
                let v_page = v_cache.i(page_id)?;

                // Narrow to the tokens we need
                let k_slice = k_page.narrow(0, 0, tokens_in_page)?;
                let v_slice = v_page.narrow(0, 0, tokens_in_page)?;

                k_slices.push(k_slice);
                v_slices.push(v_slice);

                tokens_remaining -= tokens_in_page;
            }

            if tokens_remaining == 0 {
                break;
            }
        }

        // Concatenate all slices
        if k_slices.is_empty() {
            let k = Tensor::zeros(
                (seq_len, self.num_heads, self.head_dim),
                self.dtype,
                &self.device,
            )?;
            let v = Tensor::zeros(
                (seq_len, self.num_heads, self.head_dim),
                self.dtype,
                &self.device,
            )?;
            return Ok((k, v));
        }

        let k_refs: Vec<&Tensor> = k_slices.iter().collect();
        let v_refs: Vec<&Tensor> = v_slices.iter().collect();

        let k = Tensor::cat(&k_refs, 0)?;
        let v = Tensor::cat(&v_refs, 0)?;

        Ok((k, v))
    }

    /// Convenience method to write a single token's K/V values
    pub fn write_single_token(
        &mut self,
        layer_idx: usize,
        page_id: usize,
        token_idx_in_page: usize,
        k: &Tensor,
        v: &Tensor,
    ) -> Result<()> {
        // k, v shape should be [1, num_heads, head_dim] or [num_heads, head_dim]
        let k = if k.dims().len() == 2 {
            k.unsqueeze(0)?
        } else {
            k.clone()
        };
        let v = if v.dims().len() == 2 {
            v.unsqueeze(0)?
        } else {
            v.clone()
        };

        self.write_kv(layer_idx, page_id, token_idx_in_page, &k, &v)
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            total_pages: self.total_pages,
            free_pages: self.num_free_pages(),
            used_pages: self.total_pages - self.num_free_pages(),
            memory_per_page_mb: self.page_size_mb(),
        }
    }

    fn page_size_mb(&self) -> f64 {
        let elements_per_page = PAGE_SIZE * self.num_heads * self.head_dim;
        let bytes_per_element = match self.dtype {
            DType::F16 | DType::BF16 => 2,
            DType::F32 => 4,
            _ => 2,
        };
        let bytes_per_page = elements_per_page * bytes_per_element * 2; // *2 for K and V
        bytes_per_page as f64 / 1024.0 / 1024.0
    }
}

#[derive(Debug, Clone)]
pub struct CacheStats {
    pub total_pages: usize,
    pub free_pages: usize,
    pub used_pages: usize,
    pub memory_per_page_mb: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_page_allocation() {
        let cache = PagedKVCache::new(4, 8, 64, 10, DType::F32, Device::Cpu).unwrap();

        // All pages should be free initially
        assert_eq!(cache.num_free_pages(), 10);

        // Allocate a page
        let page_id = cache.allocate_page().unwrap();
        assert_eq!(cache.num_free_pages(), 9);

        // Free the page
        cache.free_page(page_id);
        assert_eq!(cache.num_free_pages(), 10);
    }

    #[test]
    fn test_write_and_read_kv() {
        let num_layers = 2;
        let num_heads = 4;
        let head_dim = 8;
        let total_pages = 4;

        let mut cache = PagedKVCache::new(
            num_layers, num_heads, head_dim, total_pages,
            DType::F32, Device::Cpu
        ).unwrap();

        // Create test K/V tensors for 3 tokens
        // Shape: [num_tokens, num_heads, head_dim]
        let k_data: Vec<f32> = (0..3 * num_heads * head_dim)
            .map(|i| i as f32 / 100.0)
            .collect();
        let v_data: Vec<f32> = (0..3 * num_heads * head_dim)
            .map(|i| (i as f32 + 100.0) / 100.0)
            .collect();

        let k = Tensor::from_vec(k_data, (3, num_heads, head_dim), &Device::Cpu).unwrap();
        let v = Tensor::from_vec(v_data, (3, num_heads, head_dim), &Device::Cpu).unwrap();

        // Write to layer 0, page 0, starting at offset 0
        cache.write_kv(0, 0, 0, &k, &v).unwrap();

        // Read back
        let (k_read, v_read) = cache.read_kv(0, &[0], 3).unwrap();

        // Verify shapes
        assert_eq!(k_read.dims(), &[3, num_heads, head_dim]);
        assert_eq!(v_read.dims(), &[3, num_heads, head_dim]);

        // Verify values (first element)
        let k_orig_val: Vec<f32> = k.flatten_all().unwrap().to_vec1().unwrap();
        let k_read_val: Vec<f32> = k_read.flatten_all().unwrap().to_vec1().unwrap();

        for i in 0..k_orig_val.len() {
            assert!((k_orig_val[i] - k_read_val[i]).abs() < 1e-6,
                "K mismatch at {}: {} vs {}", i, k_orig_val[i], k_read_val[i]);
        }
    }

    #[test]
    fn test_write_at_offset() {
        let num_layers = 1;
        let num_heads = 2;
        let head_dim = 4;
        let total_pages = 2;

        let mut cache = PagedKVCache::new(
            num_layers, num_heads, head_dim, total_pages,
            DType::F32, Device::Cpu
        ).unwrap();

        // Write 2 tokens at offset 5
        let k = Tensor::ones((2, num_heads, head_dim), DType::F32, &Device::Cpu).unwrap();
        let v = Tensor::ones((2, num_heads, head_dim), DType::F32, &Device::Cpu).unwrap();

        cache.write_kv(0, 0, 5, &k, &v).unwrap();

        // Read back the full page and check
        let (k_read, _) = cache.read_kv(0, &[0], PAGE_SIZE).unwrap();

        // Tokens at offset 5 and 6 should be 1.0
        let k_vals: Vec<f32> = k_read.flatten_all().unwrap().to_vec1().unwrap();

        // Check that positions 5 and 6 have value 1.0
        let vals_per_token = num_heads * head_dim;
        for i in 5 * vals_per_token..(5 + 2) * vals_per_token {
            assert!((k_vals[i] - 1.0).abs() < 1e-6, "Expected 1.0 at position {}", i);
        }

        // Positions before 5 should be 0
        for i in 0..5 * vals_per_token {
            assert!((k_vals[i] - 0.0).abs() < 1e-6, "Expected 0.0 at position {}", i);
        }
    }

    #[test]
    fn test_read_across_pages() {
        let num_layers = 1;
        let num_heads = 2;
        let head_dim = 4;
        let total_pages = 3;

        let mut cache = PagedKVCache::new(
            num_layers, num_heads, head_dim, total_pages,
            DType::F32, Device::Cpu
        ).unwrap();

        // Fill page 0 with ones
        let k1 = Tensor::ones((PAGE_SIZE, num_heads, head_dim), DType::F32, &Device::Cpu).unwrap();
        let v1 = Tensor::ones((PAGE_SIZE, num_heads, head_dim), DType::F32, &Device::Cpu).unwrap();
        cache.write_kv(0, 0, 0, &k1, &v1).unwrap();

        // Fill page 1 with twos
        let k2 = (Tensor::ones((PAGE_SIZE, num_heads, head_dim), DType::F32, &Device::Cpu).unwrap() * 2.0).unwrap();
        let v2 = (Tensor::ones((PAGE_SIZE, num_heads, head_dim), DType::F32, &Device::Cpu).unwrap() * 2.0).unwrap();
        cache.write_kv(0, 1, 0, &k2, &v2).unwrap();

        // Read 24 tokens across pages 0 and 1
        let seq_len = PAGE_SIZE + 8;  // 16 + 8 = 24 tokens
        let (k_read, _) = cache.read_kv(0, &[0, 1], seq_len).unwrap();

        assert_eq!(k_read.dims(), &[seq_len, num_heads, head_dim]);

        let k_vals: Vec<f32> = k_read.flatten_all().unwrap().to_vec1().unwrap();
        let vals_per_token = num_heads * head_dim;

        // First PAGE_SIZE tokens should be 1.0
        for i in 0..PAGE_SIZE * vals_per_token {
            assert!((k_vals[i] - 1.0).abs() < 1e-6);
        }

        // Next 8 tokens should be 2.0
        for i in PAGE_SIZE * vals_per_token..(PAGE_SIZE + 8) * vals_per_token {
            assert!((k_vals[i] - 2.0).abs() < 1e-6);
        }
    }
}
