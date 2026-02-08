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
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;
use serde::{Deserialize, Serialize};

const PERM_CHUNK_SIZE: usize = 16;
const HASH_BATCH_SIZE: usize = 32;

/// `RMinHash` implements the `MinHash` algorithm for efficient similarity estimation.
#[derive(Serialize, Deserialize, Clone)]
#[pyclass(module = "rensa")]
pub struct RMinHash {
  num_perm: usize,
  seed: u64,
  hash_values: Vec<u32>,
  permutations: Vec<(u64, u64)>,
}

impl RMinHash {
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
  #[must_use]
  pub fn new(num_perm: usize, seed: u64) -> Self {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let permutations: Vec<(u64, u64)> = (0..num_perm)
      .map(|_| {
        // Ensure odd multiplier for better distribution
        let a = rng.random::<u64>() | 1;
        let b = rng.random::<u64>();
        (a, b)
      })
      .collect();

    Self {
      num_perm,
      seed,
      hash_values: vec![u32::MAX; num_perm],
      permutations,
    }
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
  #[must_use]
  pub fn jaccard(&self, other: &Self) -> f64 {
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

  fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> {
    let decoded: Self =
      postcard::from_bytes(state.as_bytes()).map_err(|err| {
        PyValueError::new_err(format!(
          "failed to deserialize RMinHash state: {err}"
        ))
      })?;
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

  fn __reduce__(&self) -> PyResult<(PyObject, (usize, u64), PyObject)> {
    Python::with_gil(|py| {
      let type_obj = py.get_type::<Self>().into();
      let state = self.__getstate__(py)?.into();
      Ok((type_obj, (self.num_perm, self.seed), state))
    })
  }
}
