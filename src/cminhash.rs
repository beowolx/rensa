//! Implementation of C-MinHash, an optimized `MinHash` variant.
//! This algorithm provides an efficient way to estimate Jaccard similarity
//! between sets, using a technique that rigorously reduces K permutations to two.
//!
//! - C-MinHash: Rigorously Reducing K Permutations to Two.
//!   Ping Li, Arnd Christian König.
//!   [arXiv:2109.03337](https://arxiv.org/abs/2109.03337)
//!
//! The implementation focuses on high single-threaded performance through
//! optimized memory access patterns and batch processing. It uses two main
//! hash transformations:
//!   - An initial permutation σ applied to item hashes.
//!   - A second set of parameters π used to generate the `num_perm` signature values.
//!
//! The `update` method processes items in batches to improve cache utilization,
//! and the Jaccard calculation is optimized using chunked operations.

use crate::utils::calculate_hash_fast;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyIterator};
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;
use serde::{Deserialize, Serialize};

/// `CMinHash` implements an optimized version of `C-MinHash` with better memory access patterns
/// and aggressive optimizations for maximum single-threaded performance.
#[derive(Serialize, Deserialize, Clone)]
#[pyclass(module = "rensa")]
pub struct CMinHash {
  num_perm: usize,
  seed: u64,
  hash_values: Vec<u64>,
  // Permutation σ parameters (a, b)
  sigma_a: u64,
  sigma_b: u64,
  // Permutation π parameters (c, d)
  pi_c: u64,
  pi_d: u64,
  // Precomputed pi_c * k + pi_d for k in 0..num_perm
  pi_precomputed: Vec<u64>,
}

impl CMinHash {
  fn update_internal(&mut self, items: Vec<String>) {
    // Batch hash computation
    const BATCH_SIZE: usize = 32;
    let mut hash_batch = Vec::with_capacity(BATCH_SIZE);

    for chunk in items.chunks(BATCH_SIZE) {
      hash_batch.clear();

      // First pass: compute all hashes
      for item in chunk {
        let h = calculate_hash_fast(item.as_bytes());
        let sigma_h = self.sigma_a.wrapping_mul(h).wrapping_add(self.sigma_b);
        hash_batch.push(sigma_h);
      }

      // Second pass: update hash values
      // Process in blocks of 16 for better vectorization
      let chunks_iter = self.hash_values.chunks_exact_mut(16);
      let pi_chunks_iter = self.pi_precomputed.chunks_exact(16);

      // Process complete chunks of 16
      for (hash_chunk, pi_chunk) in chunks_iter.zip(pi_chunks_iter) {
        let mut current = [0u64; 16];
        current.copy_from_slice(hash_chunk);

        for &sigma_h in &hash_batch {
          let base = self.pi_c.wrapping_mul(sigma_h);

          for i in 0..16 {
            let pi_value = base.wrapping_add(pi_chunk[i]);
            current[i] = current[i].min(pi_value);
          }
        }

        hash_chunk.copy_from_slice(&current);
      }

      // Handle remainder (elements not in chunks of 16)
      let remainder_start = (self.num_perm / 16) * 16;
      if remainder_start < self.num_perm {
        let hash_remainder = &mut self.hash_values[remainder_start..];
        let pi_remainder = &self.pi_precomputed[remainder_start..];

        for &sigma_h in &hash_batch {
          let base = self.pi_c.wrapping_mul(sigma_h);

          for (hash_val, &pi_val) in
            hash_remainder.iter_mut().zip(pi_remainder.iter())
          {
            let pi_value = base.wrapping_add(pi_val);
            *hash_val = (*hash_val).min(pi_value);
          }
        }
      }
    }
  }

  /// Updates the `CMinHash` with a new set of items from a vector of strings.
  pub fn update_vec(&mut self, items: Vec<String>) {
    self.update_internal(items);
  }
}

#[pymethods]
impl CMinHash {
  /// Creates a new `CMinHash` instance.
  ///
  /// # Arguments
  ///
  /// * `num_perm` - The number of permutations to use in the `MinHash` algorithm.
  /// * `seed` - A seed value for the random number generator.
  #[new]
  #[must_use]
  pub fn new(num_perm: usize, seed: u64) -> Self {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);

    let sigma_a = rng.random::<u64>() | 1;
    let sigma_b = rng.random::<u64>();
    let pi_c = rng.random::<u64>() | 1;
    let pi_d = rng.random::<u64>();

    // Precompute pi_c * k + pi_d for all k values
    let pi_precomputed: Vec<u64> = (0..num_perm)
      .map(|k| pi_c.wrapping_mul(k as u64).wrapping_add(pi_d))
      .collect();

    Self {
      num_perm,
      seed,
      hash_values: vec![u64::MAX; num_perm],
      sigma_a,
      sigma_b,
      pi_c,
      pi_d,
      pi_precomputed,
    }
  }

  /// Updates the `CMinHash` with a new set of items.
  ///
  /// # Arguments
  ///
  /// * `items` - An iterable of strings to be hashed and incorporated into the `MinHash`.
  ///
  /// # Errors
  ///
  /// Returns an error if `items` is not iterable or contains non-string elements.
  #[pyo3(signature = (items))]
  pub fn update(&mut self, items: Bound<'_, PyAny>) -> PyResult<()> {
    let iterator = PyIterator::from_object(&items)?;
    let mut collected_items = Vec::new();

    for item in iterator {
      collected_items.push(item?.extract::<String>()?);
    }

    self.update_internal(collected_items);
    Ok(())
  }

  /// Returns the current `MinHash` digest as u32 values for compatibility.
  #[inline]
  #[must_use]
  pub fn digest(&self) -> Vec<u32> {
    self.hash_values.iter().map(|&v| (v >> 32) as u32).collect()
  }

  /// Returns the current `MinHash` digest as u64 values.
  #[inline]
  #[must_use]
  pub fn digest_u64(&self) -> Vec<u64> {
    self.hash_values.clone()
  }

  /// Calculates the Jaccard similarity between this `CMinHash` and another.
  #[inline]
  #[must_use]
  pub fn jaccard(&self, other: &Self) -> f64 {
    let mut equal_count = 0usize;

    // Process in chunks of 8 for CPU-friendly operations
    let chunks_a = self.hash_values.chunks_exact(8);
    let chunks_b = other.hash_values.chunks_exact(8);

    // Process complete chunks
    for (chunk_a, chunk_b) in chunks_a.zip(chunks_b) {
      // Unroll manually for better performance
      equal_count += usize::from(chunk_a[0] == chunk_b[0]);
      equal_count += usize::from(chunk_a[1] == chunk_b[1]);
      equal_count += usize::from(chunk_a[2] == chunk_b[2]);
      equal_count += usize::from(chunk_a[3] == chunk_b[3]);
      equal_count += usize::from(chunk_a[4] == chunk_b[4]);
      equal_count += usize::from(chunk_a[5] == chunk_b[5]);
      equal_count += usize::from(chunk_a[6] == chunk_b[6]);
      equal_count += usize::from(chunk_a[7] == chunk_b[7]);
    }

    // Handle remainder
    let remainder_start = (self.num_perm / 8) * 8;
    if remainder_start < self.num_perm {
      equal_count += self.hash_values[remainder_start..]
        .iter()
        .zip(&other.hash_values[remainder_start..])
        .filter(|&(&a, &b)| a == b)
        .count();
    }

    equal_count as f64 / self.num_perm as f64
  }

  fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) {
    *self = bincode::serde::decode_from_slice(
      state.as_bytes(),
      bincode::config::standard(),
    )
    .unwrap()
    .0;
  }

  fn __getstate__<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
    PyBytes::new(
      py,
      &bincode::serde::encode_to_vec(self, bincode::config::standard())
        .unwrap(),
    )
  }

  const fn __getnewargs__(&self) -> (usize, u64) {
    (self.num_perm, self.seed)
  }

  fn __reduce__(&self) -> PyResult<(PyObject, (usize, u64), PyObject)> {
    Python::with_gil(|py| {
      let type_obj = py.get_type::<Self>().into();
      let state = self.__getstate__(py).into();
      Ok((type_obj, (self.num_perm, self.seed), state))
    })
  }
}
