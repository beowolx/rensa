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

use crate::py_input::{hash_single_bufferlike, hash_token};
use crate::utils::calculate_hash_fast;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyIterator};
use rand_core::{RngCore, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use serde::{Deserialize, Serialize};

const HASH_BATCH_SIZE: usize = 32;
type ReduceResult = (Py<PyAny>, (usize, u64), Py<PyAny>);

/// `CMinHash` implements an optimized version of `C-MinHash` with better memory access patterns
/// and aggressive optimizations for maximum single-threaded performance.
#[derive(Serialize, Deserialize, Clone)]
#[pyclass(module = "rensa", skip_from_py_object)]
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
  fn build_pi_precomputed(num_perm: usize, pi_c: u64, pi_d: u64) -> Vec<u64> {
    (0..num_perm)
      .map(|k| pi_c.wrapping_mul(k as u64).wrapping_add(pi_d))
      .collect()
  }

  #[inline]
  pub(crate) const fn num_perm(&self) -> usize {
    self.num_perm
  }

  fn validate_num_perm(num_perm: usize) -> PyResult<()> {
    if num_perm == 0 {
      return Err(PyValueError::new_err("num_perm must be greater than 0"));
    }
    Ok(())
  }

  fn validate_state(&self) -> PyResult<()> {
    Self::validate_num_perm(self.num_perm)?;
    if self.hash_values.len() != self.num_perm {
      return Err(PyValueError::new_err(format!(
        "invalid CMinHash state: hash_values length {} does not match num_perm {}",
        self.hash_values.len(),
        self.num_perm
      )));
    }
    if !self.pi_precomputed.is_empty()
      && self.pi_precomputed.len() != self.num_perm
    {
      return Err(PyValueError::new_err(format!(
        "invalid CMinHash state: pi_precomputed length {} does not match num_perm {} (or be compacted to 0)",
        self.pi_precomputed.len(),
        self.num_perm
      )));
    }
    Ok(())
  }

  pub(crate) fn ensure_compatible_for_jaccard(
    &self,
    other: &Self,
  ) -> PyResult<()> {
    self.validate_state()?;
    other.validate_state()?;
    if self.num_perm != other.num_perm {
      return Err(PyValueError::new_err(format!(
        "num_perm mismatch: left is {}, right is {}",
        self.num_perm, other.num_perm
      )));
    }
    Ok(())
  }

  #[inline]
  const fn sigma_transform(&self, hash: u64) -> u64 {
    self.sigma_a.wrapping_mul(hash).wrapping_add(self.sigma_b)
  }

  fn ensure_pi_precomputed(&mut self) {
    if self.pi_precomputed.len() != self.num_perm {
      self.pi_precomputed =
        Self::build_pi_precomputed(self.num_perm, self.pi_c, self.pi_d);
    }
  }

  fn release_update_state(&mut self) {
    self.pi_precomputed = Vec::new();
  }

  #[inline]
  pub(crate) fn jaccard_unchecked(&self, other: &Self) -> f64 {
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

  fn apply_sigma_batch(&mut self, sigma_batch: &[u64]) {
    // Update hash values in blocks of 16 for better vectorization
    let chunks_iter = self.hash_values.chunks_exact_mut(16);
    let pi_chunks_iter = self.pi_precomputed.chunks_exact(16);

    // Process complete chunks of 16
    for (hash_chunk, pi_chunk) in chunks_iter.zip(pi_chunks_iter) {
      let mut current = [0u64; 16];
      current.copy_from_slice(hash_chunk);

      for &sigma_h in sigma_batch {
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

      for &sigma_h in sigma_batch {
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

  fn update_internal(&mut self, items: Vec<String>) {
    self.ensure_pi_precomputed();
    let mut sigma_batch = Vec::with_capacity(HASH_BATCH_SIZE);

    for chunk in items.chunks(HASH_BATCH_SIZE) {
      sigma_batch.clear();

      for item in chunk {
        let h = calculate_hash_fast(item.as_bytes());
        sigma_batch.push(self.sigma_transform(h));
      }

      self.apply_sigma_batch(&sigma_batch);
    }

    self.release_update_state();
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
  pub fn new(num_perm: usize, seed: u64) -> PyResult<Self> {
    Self::validate_num_perm(num_perm)?;

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);

    let sigma_a = rng.next_u64() | 1;
    let sigma_b = rng.next_u64();
    let pi_c = rng.next_u64() | 1;
    let pi_d = rng.next_u64();

    Ok(Self {
      num_perm,
      seed,
      hash_values: vec![u64::MAX; num_perm],
      sigma_a,
      sigma_b,
      pi_c,
      pi_d,
      pi_precomputed: Self::build_pi_precomputed(num_perm, pi_c, pi_d),
    })
  }

  /// Updates the `CMinHash` with a new set of items.
  ///
  /// # Arguments
  ///
  /// * `items` - `str`, bytes-like object, or iterable of `str`/bytes-like tokens.
  ///
  /// # Errors
  ///
  /// Returns an error if `items` is neither a supported bytes-like object
  /// nor an iterable of supported token types.
  #[pyo3(signature = (items))]
  pub fn update(&mut self, items: Bound<'_, PyAny>) -> PyResult<()> {
    self.ensure_pi_precomputed();

    let result = (|| -> PyResult<()> {
      if let Some(single_hash) = hash_single_bufferlike(&items)? {
        self.apply_sigma_batch(&[self.sigma_transform(single_hash)]);
        return Ok(());
      }

      let iterator = PyIterator::from_object(&items)?;
      let mut sigma_batch = Vec::with_capacity(HASH_BATCH_SIZE);

      for item in iterator {
        let hash = hash_token(&item?)?;
        sigma_batch.push(self.sigma_transform(hash));
        if sigma_batch.len() == HASH_BATCH_SIZE {
          self.apply_sigma_batch(&sigma_batch);
          sigma_batch.clear();
        }
      }

      if !sigma_batch.is_empty() {
        self.apply_sigma_batch(&sigma_batch);
      }

      Ok(())
    })();

    self.release_update_state();
    result
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
  pub fn jaccard(&self, other: &Self) -> PyResult<f64> {
    self.ensure_compatible_for_jaccard(other)?;
    Ok(self.jaccard_unchecked(other))
  }

  fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> {
    let decoded: Self =
      postcard::from_bytes(state.as_bytes()).map_err(|err| {
        PyValueError::new_err(format!(
          "failed to deserialize CMinHash state: {err}"
        ))
      })?;
    decoded.validate_state()?;
    *self = decoded;
    Ok(())
  }

  fn __getstate__<'py>(
    &self,
    py: Python<'py>,
  ) -> PyResult<Bound<'py, PyBytes>> {
    let encoded = postcard::to_allocvec(self).map_err(|err| {
      PyValueError::new_err(format!(
        "failed to serialize CMinHash state: {err}"
      ))
    })?;
    Ok(PyBytes::new(py, &encoded))
  }

  const fn __getnewargs__(&self) -> (usize, u64) {
    (self.num_perm, self.seed)
  }

  fn __reduce__(&self) -> PyResult<ReduceResult> {
    Python::attach(|py| {
      let type_obj = py.get_type::<Self>().into();
      let state = self.__getstate__(py)?.into();
      Ok((type_obj, (self.num_perm, self.seed), state))
    })
  }
}
