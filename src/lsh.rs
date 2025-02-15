use bincode::{deserialize, serialize};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::rminhash::RMinHash;
use crate::utils::calculate_band_hash;

/// LSH index for MinHash signatures
///
/// Implements the band technique for LSH where signatures are split into bands
/// and hashed to buckets. Similar items are likely to share at least one bucket.
#[pyclass(module = "rensa")]
#[derive(Serialize, Deserialize)]
pub struct RMinHashLSH {
  threshold: f64,
  num_perm: usize,
  num_bands: usize,
  band_size: usize,
  hash_tables: Vec<HashMap<u64, Vec<usize>>>,
}

#[pymethods]
impl RMinHashLSH {
  /// Creates a new LSH index
  ///
  /// # Arguments
  /// * `threshold` - Similarity threshold for considering items as similar
  /// * `num_perm` - Number of permutations in the MinHash signatures
  /// * `num_bands` - Number of bands to split signatures into
  ///
  /// # Returns
  /// A new LSH index configured with the specified parameters
  #[new]
  pub fn new(threshold: f64, num_perm: usize, num_bands: usize) -> Self {
    Self {
      threshold,
      num_perm,
      num_bands,
      band_size: num_perm / num_bands,
      hash_tables: vec![HashMap::new(); num_bands],
    }
  }

  /// Inserts a MinHash signature into the index
  ///
  /// # Arguments
  /// * `key` - Unique identifier for the signature
  /// * `minhash` - MinHash signature to insert
  pub fn insert(&mut self, key: usize, minhash: &RMinHash) {
    let digest = minhash.digest();
    for (i, table) in self.hash_tables.iter_mut().enumerate() {
      let start = i * self.band_size;
      let end = start + self.band_size;
      let band_hash = calculate_band_hash(&digest[start..end]);
      table.entry(band_hash).or_default().push(key);
    }
  }

  /// Queries the index for similar items
  ///
  /// # Arguments
  /// * `minhash` - MinHash signature to query with
  ///
  /// # Returns
  /// Vector of keys for potentially similar items
  pub fn query(&self, minhash: &RMinHash) -> Vec<usize> {
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

  pub fn is_similar(&self, minhash1: &RMinHash, minhash2: &RMinHash) -> bool {
    minhash1.jaccard(minhash2) >= self.threshold
  }

  pub const fn get_num_perm(&self) -> usize {
    self.num_perm
  }

  pub const fn get_num_bands(&self) -> usize {
    self.num_bands
  }

  pub fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) {
    *self = deserialize(state.as_bytes()).unwrap();
  }

  pub fn __getstate__<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
    PyBytes::new(py, &serialize(&self).unwrap())
  }

  pub const fn __getnewargs__(&self) -> (f64, usize, usize) {
    (self.threshold, self.num_perm, self.num_bands)
  }
}
