//! Implementation of the R-MinHash algorithm, a novel variant of `MinHash`.
//! R-MinHash is designed for high-performance similarity estimation and deduplication
//! of large datasets, forming a core component of the Rensa library.
//!
//! This algorithm represents an original approach developed for Rensa. It draws
//! inspiration from traditional `MinHash` techniques and concepts discussed in
//! the context of algorithms like C-MinHash, but implements a distinct and
//! simplified method for generating `MinHash` signatures.
//!
//! For context on related advanced `MinHash` techniques, see:
//! - C-MinHash: Rigorously Reducing K Permutations to Two.
//!   Ping Li, Arnd Christian König. [arXiv:2109.03337](https://arxiv.org/abs/2109.03337)
//!
//! Key characteristics of R-MinHash:
//! - Simulates `num_perm` independent hash functions using unique pairs of random
//!   numbers (a, b) for each permutation, applied on-the-fly. This avoids
//!   storing full permutations or complex derivation schemes.
//! - Optimized for speed using batch processing of input items and leveraging
//!   efficient hash computations.
//! - Provides a practical balance between performance and accuracy for large-scale
//!   similarity tasks.
//!
//! Usage:
//! - Create an instance with `RMinHash::new(num_perm, seed)`.
//! - Process data items with `rminhash.update(items)`.
//! - Obtain the signature with `rminhash.digest()`.
//! - Estimate Jaccard similarity with `rminhash.jaccard(&other_rminhash)`.

use crate::py_input::{hash_single_bufferlike, hash_token};
use crate::utils::{calculate_hash_fast, permute_hash};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyIterator};
use rand_core::{RngCore, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use serde::{Deserialize, Serialize};

const PERM_CHUNK_SIZE: usize = 16;
const HASH_BATCH_SIZE: usize = 32;
type ReduceResult = (Py<PyAny>, (usize, u64), Py<PyAny>);

/// `RMinHash` implements the `MinHash` algorithm for efficient similarity estimation.
#[derive(Serialize, Deserialize, Clone)]
#[pyclass(module = "rensa", skip_from_py_object)]
pub struct RMinHash {
  num_perm: usize,
  seed: u64,
  hash_values: Vec<u32>,
  permutations: Vec<(u64, u64)>,
}

impl RMinHash {
  fn build_permutations(num_perm: usize, seed: u64) -> Vec<(u64, u64)> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    (0..num_perm)
      .map(|_| {
        // Ensure odd multiplier for better distribution
        let a = rng.next_u64() | 1;
        let b = rng.next_u64();
        (a, b)
      })
      .collect()
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
        "invalid RMinHash state: hash_values length {} does not match num_perm {}",
        self.hash_values.len(),
        self.num_perm
      )));
    }
    if !self.permutations.is_empty() && self.permutations.len() != self.num_perm
    {
      return Err(PyValueError::new_err(format!(
        "invalid RMinHash state: permutations length {} does not match num_perm {} (or be compacted to 0)",
        self.permutations.len(),
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
  pub(crate) fn hash_values(&self) -> &[u32] {
    &self.hash_values
  }

  #[inline]
  pub(crate) const fn num_perm(&self) -> usize {
    self.num_perm
  }

  fn ensure_permutations(&mut self) {
    if self.permutations.len() != self.num_perm {
      self.permutations = Self::build_permutations(self.num_perm, self.seed);
    }
  }

  fn release_update_state(&mut self) {
    self.permutations = Vec::new();
  }

  #[inline]
  pub(crate) fn jaccard_unchecked(&self, other: &Self) -> f64 {
    let mut equal_count = 0usize;

    // Process in chunks of 8 for CPU-friendly operations
    let chunks_a = self.hash_values.chunks_exact(8);
    let chunks_b = other.hash_values.chunks_exact(8);

    for (chunk_a, chunk_b) in chunks_a.zip(chunks_b) {
      // Manual unrolling for better performance
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

  fn apply_hash_batch(&mut self, hash_batch: &[u64]) {
    // Update hash values in chunks for better vectorization
    let perm_chunks_iter = self.permutations.chunks_exact(PERM_CHUNK_SIZE);
    let hash_chunks_iter = self.hash_values.chunks_exact_mut(PERM_CHUNK_SIZE);

    // Process complete chunks
    for (perm_chunk, hash_chunk) in perm_chunks_iter.zip(hash_chunks_iter) {
      let mut current = [0u32; PERM_CHUNK_SIZE];
      current.copy_from_slice(hash_chunk);

      for &item_hash in hash_batch {
        for i in 0..PERM_CHUNK_SIZE {
          let (a, b) = perm_chunk[i];
          let hash = permute_hash(item_hash, a, b);
          current[i] = current[i].min(hash);
        }
      }

      hash_chunk.copy_from_slice(&current);
    }

    // Handle remainder
    let remainder_start = (self.num_perm / PERM_CHUNK_SIZE) * PERM_CHUNK_SIZE;
    if remainder_start < self.num_perm {
      let perm_remainder = &self.permutations[remainder_start..];
      let hash_remainder = &mut self.hash_values[remainder_start..];

      for &item_hash in hash_batch {
        for (i, &(a, b)) in perm_remainder.iter().enumerate() {
          let hash = permute_hash(item_hash, a, b);
          hash_remainder[i] = hash_remainder[i].min(hash);
        }
      }
    }
  }

  fn update_internal(&mut self, items: Vec<String>) {
    self.ensure_permutations();
    let mut hash_batch = Vec::with_capacity(HASH_BATCH_SIZE);

    // Process items in batches for better cache utilization
    for chunk in items.chunks(HASH_BATCH_SIZE) {
      hash_batch.clear();

      // First pass: compute all hashes
      for item in chunk {
        hash_batch.push(calculate_hash_fast(item.as_bytes()));
      }

      self.apply_hash_batch(&hash_batch);
    }

    self.release_update_state();
  }

  /// Updates the `MinHash` with a new set of items from a vector of strings.
  pub fn update_vec(&mut self, items: Vec<String>) {
    self.update_internal(items);
  }
}

#[pymethods]
impl RMinHash {
  /// Creates a new `RMinHash` instance.
  ///
  /// # Arguments
  ///
  /// * `num_perm` - The number of permutations to use in the `MinHash` algorithm.
  /// * `seed` - A seed value for the random number generator.
  #[new]
  pub fn new(num_perm: usize, seed: u64) -> PyResult<Self> {
    Self::validate_num_perm(num_perm)?;

    Ok(Self {
      num_perm,
      seed,
      hash_values: vec![u32::MAX; num_perm],
      permutations: Self::build_permutations(num_perm, seed),
    })
  }

  /// Updates the `MinHash` with a new set of items.
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
    self.ensure_permutations();

    let result = (|| -> PyResult<()> {
      if let Some(single_hash) = hash_single_bufferlike(&items)? {
        self.apply_hash_batch(&[single_hash]);
        return Ok(());
      }

      let iterator = PyIterator::from_object(&items)?;
      let mut hash_batch = Vec::with_capacity(HASH_BATCH_SIZE);

      for item in iterator {
        hash_batch.push(hash_token(&item?)?);
        if hash_batch.len() == HASH_BATCH_SIZE {
          self.apply_hash_batch(&hash_batch);
          hash_batch.clear();
        }
      }

      if !hash_batch.is_empty() {
        self.apply_hash_batch(&hash_batch);
      }

      Ok(())
    })();

    self.release_update_state();
    result
  }

  /// Returns the current `MinHash` digest.
  ///
  /// # Returns
  ///
  /// A vector of u32 values representing the `MinHash` signature.
  #[must_use]
  pub fn digest(&self) -> Vec<u32> {
    self.hash_values.clone()
  }

  /// Calculates the Jaccard similarity between this `MinHash` and another.
  ///
  /// # Arguments
  ///
  /// * `other` - Another `RMinHash` instance to compare with.
  ///
  /// # Returns
  ///
  /// A float value representing the estimated `Jaccard` similarity.
  ///
  /// # Errors
  ///
  /// Returns an error when instances have incompatible parameters.
  pub fn jaccard(&self, other: &Self) -> PyResult<f64> {
    self.ensure_compatible_for_jaccard(other)?;
    Ok(self.jaccard_unchecked(other))
  }

  fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> {
    let decoded: Self =
      postcard::from_bytes(state.as_bytes()).map_err(|err| {
        PyValueError::new_err(format!(
          "failed to deserialize RMinHash state: {err}"
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
        "failed to serialize RMinHash state: {err}"
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
