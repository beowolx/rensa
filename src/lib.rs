#![deny(clippy::all)]
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]
#![allow(clippy::unsafe_derive_deserialize)]
#![allow(clippy::cast_precision_loss)]

use bincode;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rand::prelude::*;
use rustc_hash::FxHasher;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// RMinHash implements the MinHash algorithm for efficient similarity estimation.
#[derive(Serialize, Deserialize)]
#[pyclass(module = "rensa")]
struct RMinHash {
  num_perm: usize,
  seed: u64,
  hash_values: Vec<u32>,
  permutations: Vec<(u64, u64)>,
}

#[pymethods]
impl RMinHash {
  /// Creates a new RMinHash instance.
  ///
  /// # Arguments
  ///
  /// * `num_perm` - The number of permutations to use in the MinHash algorithm.
  /// * `seed` - A seed value for the random number generator.
  #[new]
  fn new(num_perm: usize, seed: u64) -> Self {
    let mut rng = StdRng::seed_from_u64(seed);
    let permutations: Vec<(u64, u64)> = (0..num_perm)
      .map(|_| (rng.random(), rng.random()))
      .collect();

    Self {
      num_perm,
      seed,
      hash_values: vec![u32::MAX; num_perm],
      permutations,
    }
  }

  /// Updates the MinHash with a new set of items.
  ///
  /// # Arguments
  ///
  /// * `items` - A vector of strings to be hashed and incorporated into the MinHash.
  fn update(&mut self, items: Vec<String>) {
    for item in items {
      let item_hash = calculate_hash(&item);
      for (i, &(a, b)) in self.permutations.iter().enumerate() {
        let hash = permute_hash(item_hash, a, b);
        self.hash_values[i] = self.hash_values[i].min(hash);
      }
    }
  }

  /// Returns the current MinHash digest.
  ///
  /// # Returns
  ///
  /// A vector of u32 values representing the MinHash signature.
  fn digest(&self) -> Vec<u32> {
    self.hash_values.clone()
  }

  /// Calculates the Jaccard similarity between this MinHash and another.
  ///
  /// # Arguments
  ///
  /// * `other` - Another RMinHash instance to compare with.
  ///
  /// # Returns
  ///
  /// A float value representing the estimated Jaccard similarity.
  fn jaccard(&self, other: &Self) -> f64 {
    let equal_count = self
      .hash_values
      .iter()
      .zip(&other.hash_values)
      .filter(|&(&a, &b)| a == b)
      .count();
    // Safe because self.num_perm is expected to be << 2^53.
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
      &bincode::serde::encode_to_vec(&self, bincode::config::standard())
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

/// CMinHash implements an optimized version of C-MinHash with better memory access patterns
/// and aggressive optimizations for maximum single-threaded performance.
#[derive(Serialize, Deserialize)]
#[pyclass(module = "rensa")]
struct CMinHash {
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

#[pymethods]
impl CMinHash {
  /// Creates a new CMinHash instance.
  ///
  /// # Arguments
  ///
  /// * `num_perm` - The number of permutations to use in the MinHash algorithm.
  /// * `seed` - A seed value for the random number generator.
  #[new]
  fn new(num_perm: usize, seed: u64) -> Self {
    let mut rng = StdRng::seed_from_u64(seed);

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

  /// Updates the CMinHash with a new set of items using optimized loops.
  ///
  /// # Arguments
  ///
  /// * `items` - A vector of strings to be hashed and incorporated into the MinHash.
  fn update(&mut self, items: Vec<String>) {
    // Process multiple items together to improve cache utilization
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

  /// Returns the current MinHash digest as u32 values for compatibility.
  #[inline(always)]
  fn digest(&self) -> Vec<u32> {
    self.hash_values.iter().map(|&v| (v >> 32) as u32).collect()
  }

  /// Returns the current MinHash digest as u64 values.
  #[inline(always)]
  fn digest_u64(&self) -> Vec<u64> {
    self.hash_values.clone()
  }

  /// Calculates the Jaccard similarity between this CMinHash and another.
  #[inline(always)]
  fn jaccard(&self, other: &Self) -> f64 {
    // Use chunks for better auto-vectorization
    let mut equal_count = 0usize;

    // Process in chunks of 8 for CPU-friendly operations
    let chunks_a = self.hash_values.chunks_exact(8);
    let chunks_b = other.hash_values.chunks_exact(8);

    // Process complete chunks
    for (chunk_a, chunk_b) in chunks_a.zip(chunks_b) {
      // Unroll manually for better performance
      equal_count += (chunk_a[0] == chunk_b[0]) as usize;
      equal_count += (chunk_a[1] == chunk_b[1]) as usize;
      equal_count += (chunk_a[2] == chunk_b[2]) as usize;
      equal_count += (chunk_a[3] == chunk_b[3]) as usize;
      equal_count += (chunk_a[4] == chunk_b[4]) as usize;
      equal_count += (chunk_a[5] == chunk_b[5]) as usize;
      equal_count += (chunk_a[6] == chunk_b[6]) as usize;
      equal_count += (chunk_a[7] == chunk_b[7]) as usize;
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
      &bincode::serde::encode_to_vec(&self, bincode::config::standard())
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

/// RMinHashLSH implements Locality-Sensitive Hashing using MinHash for efficient similarity search.
#[derive(Serialize, Deserialize)]
#[pyclass(module = "rensa")]
struct RMinHashLSH {
  threshold: f64,
  num_perm: usize,
  num_bands: usize,
  band_size: usize,
  hash_tables: Vec<HashMap<u64, Vec<usize>>>,
}

#[pymethods]
impl RMinHashLSH {
  /// Creates a new RMinHashLSH instance.
  ///
  /// # Arguments
  ///
  /// * `threshold` - The similarity threshold for considering items as similar.
  /// * `num_perm` - The number of permutations used in the MinHash algorithm.
  /// * `num_bands` - The number of bands for the LSH algorithm.
  #[new]
  fn new(threshold: f64, num_perm: usize, num_bands: usize) -> Self {
    Self {
      threshold,
      num_perm,
      num_bands,
      band_size: num_perm / num_bands,
      hash_tables: vec![HashMap::new(); num_bands],
    }
  }

  /// Inserts a MinHash into the LSH index.
  ///
  /// # Arguments
  ///
  /// * `key` - A unique identifier for the MinHash.
  /// * `minhash` - The RMinHash instance to be inserted.
  fn insert(&mut self, key: usize, minhash: &RMinHash) {
    let digest = minhash.digest();
    for (i, table) in self.hash_tables.iter_mut().enumerate() {
      let start = i * self.band_size;
      let end = start + self.band_size;
      let band_hash = calculate_band_hash(&digest[start..end]);
      table.entry(band_hash).or_insert_with(Vec::new).push(key);
    }
  }

  /// Queries the LSH index for similar items.
  ///
  /// # Arguments
  ///
  /// * `minhash` - The RMinHash instance to query for.
  ///
  /// # Returns
  ///
  /// A vector of keys (usize) of potentially similar items.
  fn query(&self, minhash: &RMinHash) -> Vec<usize> {
    let digest = minhash.digest();
    let mut candidates = Vec::new();
    for (i, table) in self.hash_tables.iter().enumerate() {
      let start = i * self.band_size;
      let end = start + self.band_size;
      let band_hash = calculate_band_hash(&digest[start..end]);
      if let Some(keys) = table.get(&band_hash) {
        candidates.extend(keys);
      }
    }
    candidates.sort_unstable();
    candidates.dedup();
    candidates
  }

  /// Checks if two MinHashes are similar based on the LSH threshold.
  ///
  /// # Arguments
  ///
  /// * `minhash1` - The first RMinHash instance.
  /// * `minhash2` - The second RMinHash instance.
  ///
  /// # Returns
  ///
  /// A boolean indicating whether the MinHashes are considered similar.
  fn is_similar(&self, minhash1: &RMinHash, minhash2: &RMinHash) -> bool {
    minhash1.jaccard(minhash2) >= self.threshold
  }

  /// Returns the number of permutations used in the LSH index.
  const fn get_num_perm(&self) -> usize {
    self.num_perm
  }

  /// Returns the number of bands used in the LSH index.
  const fn get_num_bands(&self) -> usize {
    self.num_bands
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
      &bincode::serde::encode_to_vec(&self, bincode::config::standard())
        .unwrap(),
    )
  }

  const fn __getnewargs__(&self) -> (f64, usize, usize) {
    (self.threshold, self.num_perm, self.num_bands)
  }
}

/// Calculates a hash value for a given item.
#[inline(always)]
fn calculate_hash<T: Hash>(t: &T) -> u64 {
  let mut s = FxHasher::default();
  t.hash(&mut s);
  s.finish()
}

/// Fast hash function for byte arrays
#[inline(always)]
fn calculate_hash_fast(data: &[u8]) -> u64 {
  // Use a simplified version of FxHash for byte arrays
  let mut hash = 0xcbf29ce484222325u64;

  // Process 8 bytes at a time
  let chunks = data.chunks_exact(8);
  let remainder = chunks.remainder();

  for chunk in chunks {
    // Safe conversion using try_into with unwrap (chunk is guaranteed to be 8 bytes)
    let val = u64::from_le_bytes(chunk.try_into().unwrap());
    hash = hash.wrapping_mul(0x100000001b3).wrapping_add(val);
  }

  // Handle remainder bytes
  for &byte in remainder {
    hash = hash.wrapping_mul(0x100000001b3).wrapping_add(byte as u64);
  }

  hash
}

/// Applies a permutation to a hash value.
#[inline(always)]
const fn permute_hash(hash: u64, a: u64, b: u64) -> u32 {
  ((a.wrapping_mul(hash).wrapping_add(b)) >> 32) as u32
}

/// Calculates a hash value for a band of `MinHash` values.
#[inline(always)]
fn calculate_band_hash(band: &[u32]) -> u64 {
  let mut hasher = FxHasher::default();
  for &value in band {
    hasher.write_u32(value);
  }
  hasher.finish()
}

/// Python module for MinHash and LSH implementations
///
/// # Errors
/// Returns an error if the module initialization fails or classes cannot be added
#[pymodule]
pub fn rensa(m: &Bound<'_, PyModule>) -> PyResult<()> {
  m.add_class::<RMinHash>()?;
  m.add_class::<CMinHash>()?;
  m.add_class::<RMinHashLSH>()?;
  Ok(())
}
