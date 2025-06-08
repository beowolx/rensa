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
//! - `query(minhash)`: Retrieves candidate keys that are potentially similar to the query `MinHash`.
//! - `is_similar(minhash1, minhash2)`: Directly checks if two `MinHashes` meet the similarity threshold.
//!
//! This LSH implementation is particularly useful for tasks such as near-duplicate detection,
//! document clustering, etc.

use crate::rminhash::RMinHash;
use crate::utils::calculate_band_hash;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rustc_hash::FxHasher;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::BuildHasherDefault;

/// RMinHashLSH implements Locality-Sensitive Hashing using MinHash for efficient similarity search.
#[derive(Serialize, Deserialize)]
#[pyclass(module = "rensa")]
pub struct RMinHashLSH {
  threshold: f64,
  num_perm: usize,
  num_bands: usize,
  band_size: usize,
  hash_tables: Vec<HashMap<u64, Vec<usize>, BuildHasherDefault<FxHasher>>>,
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
    }
  }

  /// Inserts a MinHash into the LSH index.
  ///
  /// # Arguments
  ///
  /// * `key` - A unique identifier for the MinHash.
  /// * `minhash` - The RMinHash instance to be inserted.
  ///
  /// # Panics
  ///
  /// Panics if the MinHash has a different number of permutations than expected by the LSH index.
  pub fn insert(&mut self, key: usize, minhash: &RMinHash) {
    let digest = minhash.digest();

    assert_eq!(
      digest.len(),
      self.num_perm,
      "MinHash has {} permutations but LSH expects {}",
      digest.len(),
      self.num_perm
    );

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
  ///
  /// # Panics
  ///
  /// Panics if the MinHash has a different number of permutations than expected by the LSH index.
  #[must_use]
  pub fn query(&self, minhash: &RMinHash) -> Vec<usize> {
    let digest = minhash.digest();

    assert_eq!(
      digest.len(),
      self.num_perm,
      "MinHash has {} permutations but LSH expects {}",
      digest.len(),
      self.num_perm
    );

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

  const fn __getnewargs__(&self) -> (f64, usize, usize) {
    (self.threshold, self.num_perm, self.num_bands)
  }
}
