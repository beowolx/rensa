//! Locality-Sensitive Hashing (LSH) for `MinHash`.
//!
//! This module implements `RMinHashLSH`, a Locality-Sensitive Hashing scheme
//! that uses `RMinHash` (Rensa's novel `MinHash` variant) to efficiently find
//! approximate nearest neighbors in large datasets. It's designed for identifying
//! items with high Jaccard similarity.
//!
//! The core idea of LSH is to hash input items such that similar items are mapped
//! to the same "buckets" with high probability, while dissimilar items are not.
//! This implementation achieves this by:
//! 1. Generating `MinHash` signatures for items using `RMinHash`.
//! 2. Dividing these signatures into several "bands".
//! 3. For each band, hashing its portion of the signature.
//! 4. Items are considered candidates for similarity if they share the same hash
//!    value in at least one band.
//!
//! This approach allows for querying similar items much faster than pairwise
//! comparisons, especially for large numbers of items.
//!
//! ## Usage:
//!
//! An `RMinHashLSH` index is initialized with a Jaccard similarity threshold, the number of
//! permutations for the `MinHash` signatures, and the number of bands to use for LSH.
//! `RMinHash` objects (representing items) are inserted into the index. Queries with an
//! `RMinHash` object will return a set of keys of potentially similar items.
//!
//! Key methods include:
//! - `new(threshold, num_perm, num_bands)`: Initializes a new LSH index.
//! - `insert(key, minhash)`: Inserts an item's `MinHash` signature into the index.
//! - `remove(key)`: Removes a previously inserted key from the index.
//! - `query(minhash)`: Retrieves candidate keys that are potentially similar to the query `MinHash`.
//! - `is_similar(minhash1, minhash2)`: Directly checks if two `MinHashes` meet the similarity threshold.
//!
//! This LSH implementation is particularly useful for tasks such as near-duplicate detection,
//! document clustering, etc.

use crate::rminhash::RMinHash;
use crate::utils::calculate_band_hash;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rustc_hash::FxHasher;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::BuildHasherDefault;

/// `RMinHashLSH` implements Locality-Sensitive Hashing using `MinHash` for efficient similarity search.
#[derive(Serialize, Deserialize)]
#[pyclass(module = "rensa")]
pub struct RMinHashLSH {
  threshold: f64,
  num_perm: usize,
  num_bands: usize,
  band_size: usize,
  hash_tables: Vec<HashMap<u64, Vec<usize>, BuildHasherDefault<FxHasher>>>,
  #[serde(default)]
  key_bands: HashMap<usize, Vec<u64>, BuildHasherDefault<FxHasher>>,
}

impl RMinHashLSH {
  fn digest_band_hashes(&self, digest: &[u32]) -> Vec<u64> {
    let mut band_hashes = Vec::with_capacity(self.num_bands);
    for i in 0..self.num_bands {
      band_hashes.push(calculate_band_hash(
        &digest[i * self.band_size..(i + 1) * self.band_size],
      ));
    }
    band_hashes
  }

  fn remove_key_from_bands(&mut self, key: usize, band_hashes: &[u64]) {
    for (table, &band_hash) in
      self.hash_tables.iter_mut().zip(band_hashes.iter())
    {
      if let Some(keys) = table.get_mut(&band_hash) {
        keys.retain(|&stored_key| stored_key != key);
        if keys.is_empty() {
          table.remove(&band_hash);
        }
      }
    }
  }
}

#[pymethods]
impl RMinHashLSH {
  /// Creates a new `RMinHashLSH` instance.
  ///
  /// # Arguments
  ///
  /// * `threshold` - The similarity threshold for considering items as similar.
  /// * `num_perm` - The number of permutations used in the `MinHash` algorithm.
  /// * `num_bands` - The number of bands for the LSH algorithm.
  #[new]
  #[must_use]
  pub fn new(threshold: f64, num_perm: usize, num_bands: usize) -> Self {
    let hasher = BuildHasherDefault::<FxHasher>::default();
    let hash_tables = (0..num_bands)
      .map(|_| HashMap::with_hasher(hasher.clone()))
      .collect();

    Self {
      threshold,
      num_perm,
      num_bands,
      band_size: num_perm / num_bands,
      hash_tables,
      key_bands: HashMap::with_hasher(hasher),
    }
  }

  /// Inserts a `MinHash` into the LSH index.
  ///
  /// # Arguments
  ///
  /// * `key` - A unique identifier for the `MinHash`.
  /// * `minhash` - The `RMinHash` instance to be inserted.
  ///
  /// # Panics
  ///
  /// Panics if the `MinHash` has a different number of permutations than expected by the LSH index.
  pub fn insert(&mut self, key: usize, minhash: &RMinHash) {
    let digest = minhash.hash_values();

    assert_eq!(
      digest.len(),
      self.num_perm,
      "MinHash has {} permutations but LSH expects {}",
      digest.len(),
      self.num_perm
    );

    if let Some(previous_band_hashes) = self.key_bands.get(&key).cloned() {
      self.remove_key_from_bands(key, &previous_band_hashes);
    }

    let band_hashes = self.digest_band_hashes(digest);
    for (table, &band_hash) in
      self.hash_tables.iter_mut().zip(band_hashes.iter())
    {
      table.entry(band_hash).or_default().push(key);
    }

    self.key_bands.insert(key, band_hashes);
  }

  /// Removes a key from all LSH bands.
  ///
  /// # Returns
  ///
  /// `true` if the key existed and was removed, `false` otherwise.
  pub fn remove(&mut self, key: usize) -> bool {
    let Some(band_hashes) = self.key_bands.remove(&key) else {
      return false;
    };

    self.remove_key_from_bands(key, &band_hashes);
    true
  }

  /// Queries the LSH index for similar items.
  ///
  /// # Arguments
  ///
  /// * `minhash` - The `RMinHash` instance to query for.
  ///
  /// # Returns
  ///
  /// A vector of keys (usize) of potentially similar items.
  ///
  /// # Panics
  ///
  /// Panics if the `MinHash` has a different number of permutations than expected by the LSH index.
  #[must_use]
  pub fn query(&self, minhash: &RMinHash) -> Vec<usize> {
    let digest = minhash.hash_values();

    assert_eq!(
      digest.len(),
      self.num_perm,
      "MinHash has {} permutations but LSH expects {}",
      digest.len(),
      self.num_perm
    );

    let mut candidates = Vec::new();
    for (i, table) in self.hash_tables.iter().enumerate() {
      if let Some(keys) = table.get(&calculate_band_hash(
        &digest[i * self.band_size..(i + 1) * self.band_size],
      )) {
        candidates.extend(keys);
      }
    }

    candidates.sort_unstable();
    candidates.dedup();
    candidates
  }

  /// Checks if two `MinHashes` are similar based on the LSH threshold.
  ///
  /// # Arguments
  ///
  /// * `minhash1` - The first `RMinHash` instance.
  /// * `minhash2` - The second `RMinHash` instance.
  ///
  /// # Returns
  ///
  /// A boolean indicating whether the `MinHashes` are considered similar.
  #[must_use]
  pub fn is_similar(&self, minhash1: &RMinHash, minhash2: &RMinHash) -> bool {
    minhash1.jaccard(minhash2) >= self.threshold
  }

  /// Returns the number of permutations used in the LSH index.
  #[must_use]
  pub const fn get_num_perm(&self) -> usize {
    self.num_perm
  }

  /// Returns the number of bands used in the LSH index.
  #[must_use]
  pub const fn get_num_bands(&self) -> usize {
    self.num_bands
  }

  fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> {
    let decoded: Self =
      postcard::from_bytes(state.as_bytes()).map_err(|err| {
        PyValueError::new_err(format!(
          "failed to deserialize RMinHashLSH state: {err}"
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
        "failed to serialize RMinHashLSH state: {err}"
      ))
    })?;
    Ok(PyBytes::new(py, &encoded))
  }

  const fn __getnewargs__(&self) -> (f64, usize, usize) {
    (self.threshold, self.num_perm, self.num_bands)
  }
}
