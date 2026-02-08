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
  fn validate_params(
    threshold: f64,
    num_perm: usize,
    num_bands: usize,
  ) -> PyResult<()> {
    if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
      return Err(PyValueError::new_err(
        "threshold must be a finite value between 0.0 and 1.0",
      ));
    }
    if num_perm == 0 {
      return Err(PyValueError::new_err("num_perm must be greater than 0"));
    }
    if num_bands == 0 {
      return Err(PyValueError::new_err("num_bands must be greater than 0"));
    }
    if num_bands > num_perm {
      return Err(PyValueError::new_err(format!(
        "num_bands ({num_bands}) must be less than or equal to num_perm ({num_perm})",
      )));
    }
    if !num_perm.is_multiple_of(num_bands) {
      return Err(PyValueError::new_err(format!(
        "num_perm ({num_perm}) must be divisible by num_bands ({num_bands})",
      )));
    }
    Ok(())
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

  fn validate_digest_len(&self, digest_len: usize) -> PyResult<()> {
    if digest_len != self.num_perm {
      return Err(PyValueError::new_err(format!(
        "MinHash has {digest_len} permutations but LSH expects {}",
        self.num_perm
      )));
    }
    Ok(())
  }

  fn validate_state(&self) -> PyResult<()> {
    Self::validate_params(self.threshold, self.num_perm, self.num_bands)?;
    if self.band_size != self.num_perm / self.num_bands {
      return Err(PyValueError::new_err(format!(
        "Invalid state: band_size {} does not match num_perm/num_bands ({})",
        self.band_size,
        self.num_perm / self.num_bands
      )));
    }
    if self.hash_tables.len() != self.num_bands {
      return Err(PyValueError::new_err(format!(
        "Invalid state: hash_tables length {} does not match num_bands {}",
        self.hash_tables.len(),
        self.num_bands
      )));
    }
    if self
      .key_bands
      .values()
      .any(|band_hashes| band_hashes.len() != self.num_bands)
    {
      return Err(PyValueError::new_err(
        "Invalid state: key_bands entries must have one hash per band",
      ));
    }
    Ok(())
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
  ///
  /// # Errors
  ///
  /// Returns an error if any input invariant is violated.
  #[new]
  pub fn new(
    threshold: f64,
    num_perm: usize,
    num_bands: usize,
  ) -> PyResult<Self> {
    Self::validate_params(threshold, num_perm, num_bands)?;

    let hasher = BuildHasherDefault::<FxHasher>::default();
    let hash_tables = (0..num_bands)
      .map(|_| HashMap::with_hasher(hasher.clone()))
      .collect();

    Ok(Self {
      threshold,
      num_perm,
      num_bands,
      band_size: num_perm / num_bands,
      hash_tables,
      key_bands: HashMap::with_hasher(hasher),
    })
  }

  /// Inserts a `MinHash` into the LSH index.
  ///
  /// # Arguments
  ///
  /// * `key` - A unique identifier for the `MinHash`.
  /// * `minhash` - The `RMinHash` instance to be inserted.
  ///
  /// # Errors
  ///
  /// Returns an error if the `MinHash` has a different number of permutations than expected by the LSH index.
  pub fn insert(&mut self, key: usize, minhash: &RMinHash) -> PyResult<()> {
    let digest = minhash.digest_slice();
    self.validate_digest_len(digest.len())?;

    if let Some(previous_band_hashes) = self.key_bands.get(&key).cloned() {
      self.remove_key_from_bands(key, &previous_band_hashes);
    }

    let mut band_hashes = Vec::with_capacity(self.num_bands);
    for (i, table) in self.hash_tables.iter_mut().enumerate() {
      let start = i * self.band_size;
      let end = start + self.band_size;
      let band_hash = calculate_band_hash(&digest[start..end]);
      table.entry(band_hash).or_default().push(key);
      band_hashes.push(band_hash);
    }

    self.key_bands.insert(key, band_hashes);
    Ok(())
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
  /// # Errors
  ///
  /// Returns an error if the `MinHash` has a different number of permutations than expected by the LSH index.
  pub fn query(&self, minhash: &RMinHash) -> PyResult<Vec<usize>> {
    let digest = minhash.digest_slice();
    self.validate_digest_len(digest.len())?;

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
    Ok(candidates)
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
  ///
  /// # Errors
  ///
  /// Returns an error if the signatures are incompatible.
  pub fn is_similar(
    &self,
    minhash1: &RMinHash,
    minhash2: &RMinHash,
  ) -> PyResult<bool> {
    Ok(minhash1.jaccard(minhash2)? >= self.threshold)
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
        PyValueError::new_err(format!("Failed to decode state: {err}"))
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
        "failed to serialize RMinHashLSH state: {err}"
      ))
    })?;
    Ok(PyBytes::new(py, &encoded))
  }

  const fn __getnewargs__(&self) -> (f64, usize, usize) {
    (self.threshold, self.num_perm, self.num_bands)
  }
}
