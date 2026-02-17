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

use crate::rminhash::{RMinHash, RMinHashDigestMatrix};
use crate::utils::calculate_band_hash;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyIterator};
use rustc_hash::{FxHashMap, FxHashSet, FxHasher};
use serde::{Deserialize, Serialize};
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::hash::BuildHasherDefault;

#[cfg(target_pointer_width = "64")]
const FX_POLY_K: usize = 0xf135_7aea_2e62_a9c5;
#[cfg(target_pointer_width = "32")]
const FX_POLY_K: usize = 0x93d7_65dd;

#[cfg(target_pointer_width = "64")]
const FX_FINISH_ROTATE: u32 = 26;
#[cfg(target_pointer_width = "32")]
const FX_FINISH_ROTATE: u32 = 15;

const DEFAULT_RHO_SPARSE_OCCUPANCY_THRESHOLD_BASE: usize = 56;
const MIN_RHO_SPARSE_OCCUPANCY_THRESHOLD_BASE: usize = 1;
const MAX_RHO_SPARSE_OCCUPANCY_THRESHOLD_BASE: usize = 512;
const DEFAULT_RHO_SPARSE_REQUIRED_BAND_MATCHES: usize = 2;
const MIN_RHO_SPARSE_REQUIRED_BAND_MATCHES: usize = 1;
const DEFAULT_RHO_SPARSE_VERIFY_THRESHOLD: f64 = 0.75;
const MIN_RHO_SPARSE_VERIFY_THRESHOLD: f64 = 0.0;
const MAX_RHO_SPARSE_VERIFY_THRESHOLD: f64 = 1.0;
const DEFAULT_RHO_SPARSE_VERIFY_MAX_CANDIDATES: usize = 16;
const MIN_RHO_SPARSE_VERIFY_MAX_CANDIDATES: usize = 1;
const MAX_RHO_SPARSE_VERIFY_MAX_CANDIDATES: usize = 512;
const DEFAULT_RHO_BAND_FOLD: usize = 2;
const MIN_RHO_BAND_FOLD: usize = 1;
const DEFAULT_RHO_RECALL_RESCUE_MIN_TOKENS: usize = 17;
const MIN_RHO_RECALL_RESCUE_MIN_TOKENS: usize = 1;
const MAX_RHO_RECALL_RESCUE_MIN_TOKENS: usize = 65_536;
const DEFAULT_RHO_RECALL_RESCUE_MAX_TOKENS: usize = 96;
const MIN_RHO_RECALL_RESCUE_MAX_TOKENS: usize = 1;
const MAX_RHO_RECALL_RESCUE_MAX_TOKENS: usize = 65_536;
const DEFAULT_RHO_RECALL_RESCUE_REQUIRED_BAND_MATCHES: usize = 2;
const MIN_RHO_RECALL_RESCUE_REQUIRED_BAND_MATCHES: usize = 1;

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
  #[serde(skip, default)]
  last_one_shot_sparse_verify_checks: usize,
  #[serde(skip, default)]
  last_one_shot_sparse_verify_passes: usize,
}

impl RMinHashLSH {
  #[inline]
  const fn fx_poly_steps(len_u32: usize) -> usize {
    // `calculate_band_hash` packs 4x u32 into 2x u64 writes, then writes any
    // remainder u32 values. The polynomial state multiplies by K per write.
    (len_u32 / 4) * 2 + (len_u32 % 4)
  }

  #[inline]
  fn fx_poly_k_pow(steps: usize) -> usize {
    let mut result = 1_usize;
    for _ in 0..steps {
      result = result.wrapping_mul(FX_POLY_K);
    }
    result
  }

  fn read_env_usize_clamped(
    key: &str,
    default: usize,
    min: usize,
    max: usize,
  ) -> usize {
    std::env::var(key)
      .ok()
      .and_then(|value| value.parse::<usize>().ok())
      .map_or(default, |value| value.clamp(min, max))
  }

  fn read_env_f64_clamped(key: &str, default: f64, min: f64, max: f64) -> f64 {
    std::env::var(key)
      .ok()
      .and_then(|value| value.parse::<f64>().ok())
      .map_or(default, |value| value.clamp(min, max))
  }

  fn rho_sparse_occupancy_threshold(num_perm: usize) -> usize {
    let base = Self::read_env_usize_clamped(
      "RENSA_RHO_SPARSE_OCCUPANCY_THRESHOLD",
      DEFAULT_RHO_SPARSE_OCCUPANCY_THRESHOLD_BASE,
      MIN_RHO_SPARSE_OCCUPANCY_THRESHOLD_BASE,
      MAX_RHO_SPARSE_OCCUPANCY_THRESHOLD_BASE,
    );
    let scaled = base
      .saturating_mul(num_perm)
      .saturating_add(64)
      .saturating_div(128);
    scaled.clamp(1, num_perm.max(1))
  }

  fn rho_sparse_required_band_matches(num_bands: usize) -> usize {
    Self::read_env_usize_clamped(
      "RENSA_RHO_SPARSE_REQUIRED_BAND_MATCHES",
      DEFAULT_RHO_SPARSE_REQUIRED_BAND_MATCHES,
      MIN_RHO_SPARSE_REQUIRED_BAND_MATCHES,
      num_bands.max(1),
    )
  }

  fn rho_sparse_verify_enabled() -> bool {
    std::env::var("RENSA_RHO_SPARSE_VERIFY_ENABLE")
      .ok()
      .is_none_or(|value| value != "0")
  }

  fn rho_sparse_verify_threshold() -> f64 {
    Self::read_env_f64_clamped(
      "RENSA_RHO_SPARSE_VERIFY_THRESHOLD",
      DEFAULT_RHO_SPARSE_VERIFY_THRESHOLD,
      MIN_RHO_SPARSE_VERIFY_THRESHOLD,
      MAX_RHO_SPARSE_VERIFY_THRESHOLD,
    )
  }

  fn rho_sparse_verify_max_candidates() -> usize {
    Self::read_env_usize_clamped(
      "RENSA_RHO_SPARSE_VERIFY_MAX_CANDIDATES",
      DEFAULT_RHO_SPARSE_VERIFY_MAX_CANDIDATES,
      MIN_RHO_SPARSE_VERIFY_MAX_CANDIDATES,
      MAX_RHO_SPARSE_VERIFY_MAX_CANDIDATES,
    )
  }

  fn rho_band_fold(num_bands: usize) -> usize {
    let max_fold = num_bands.max(1);
    Self::read_env_usize_clamped(
      "RENSA_RHO_BAND_FOLD",
      DEFAULT_RHO_BAND_FOLD,
      MIN_RHO_BAND_FOLD,
      max_fold,
    )
    .max(1)
  }

  fn rho_recall_rescue_enabled() -> bool {
    std::env::var("RENSA_RHO_RECALL_RESCUE_ENABLE")
      .ok()
      .is_none_or(|value| value != "0")
  }

  fn rho_recall_rescue_min_tokens() -> usize {
    Self::read_env_usize_clamped(
      "RENSA_RHO_RECALL_RESCUE_MIN_TOKENS",
      DEFAULT_RHO_RECALL_RESCUE_MIN_TOKENS,
      MIN_RHO_RECALL_RESCUE_MIN_TOKENS,
      MAX_RHO_RECALL_RESCUE_MIN_TOKENS,
    )
  }

  fn rho_recall_rescue_max_tokens() -> usize {
    Self::read_env_usize_clamped(
      "RENSA_RHO_RECALL_RESCUE_MAX_TOKENS",
      DEFAULT_RHO_RECALL_RESCUE_MAX_TOKENS,
      MIN_RHO_RECALL_RESCUE_MAX_TOKENS,
      MAX_RHO_RECALL_RESCUE_MAX_TOKENS,
    )
  }

  fn rho_recall_rescue_required_band_matches(num_bands: usize) -> usize {
    Self::read_env_usize_clamped(
      "RENSA_RHO_RECALL_RESCUE_REQUIRED_BAND_MATCHES",
      DEFAULT_RHO_RECALL_RESCUE_REQUIRED_BAND_MATCHES,
      MIN_RHO_RECALL_RESCUE_REQUIRED_BAND_MATCHES,
      num_bands.max(1),
    )
  }

  fn sparse_verify_similarity(signature_a: &[u32], signature_b: &[u32]) -> f64 {
    if signature_a.is_empty() || signature_a.len() != signature_b.len() {
      return 0.0;
    }
    let equal = signature_a
      .iter()
      .zip(signature_b.iter())
      .filter(|(left, right)| left == right)
      .count();
    equal as f64 / signature_a.len() as f64
  }

  fn validate_threshold(threshold: f64) -> PyResult<()> {
    if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
      return Err(PyValueError::new_err(
        "threshold must be a finite value between 0.0 and 1.0",
      ));
    }
    Ok(())
  }

  fn validate_params(
    threshold: f64,
    num_perm: usize,
    num_bands: usize,
  ) -> PyResult<usize> {
    Self::validate_threshold(threshold)?;
    if num_perm == 0 {
      return Err(PyValueError::new_err("num_perm must be greater than 0"));
    }
    if num_bands == 0 {
      return Err(PyValueError::new_err("num_bands must be greater than 0"));
    }
    if num_bands > num_perm {
      return Err(PyValueError::new_err(format!(
        "num_bands ({num_bands}) must be less than or equal to num_perm ({num_perm})"
      )));
    }
    if num_perm % num_bands != 0 {
      return Err(PyValueError::new_err(format!(
        "num_perm ({num_perm}) must be divisible by num_bands ({num_bands})"
      )));
    }

    Ok(num_perm / num_bands)
  }

  pub(crate) fn from_validated(
    threshold: f64,
    num_perm: usize,
    num_bands: usize,
  ) -> Self {
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
      last_one_shot_sparse_verify_checks: 0,
      last_one_shot_sparse_verify_passes: 0,
    }
  }

  fn validate_state(&self) -> PyResult<()> {
    let expected_band_size =
      Self::validate_params(self.threshold, self.num_perm, self.num_bands)?;
    if self.band_size != expected_band_size {
      return Err(PyValueError::new_err(format!(
        "invalid RMinHashLSH state: band_size {} does not match expected {}",
        self.band_size, expected_band_size
      )));
    }
    if self.hash_tables.len() != self.num_bands {
      return Err(PyValueError::new_err(format!(
        "invalid RMinHashLSH state: hash_tables length {} does not match num_bands {}",
        self.hash_tables.len(),
        self.num_bands
      )));
    }
    for (key, band_hashes) in &self.key_bands {
      if band_hashes.len() != self.num_bands {
        return Err(PyValueError::new_err(format!(
          "invalid RMinHashLSH state: key {key} stores {} band hashes, expected {}",
          band_hashes.len(),
          self.num_bands
        )));
      }
    }
    Ok(())
  }

  fn ensure_digest_len(&self, digest_len: usize) -> PyResult<()> {
    if digest_len != self.num_perm {
      return Err(PyValueError::new_err(format!(
        "MinHash has {digest_len} permutations but LSH expects {}",
        self.num_perm
      )));
    }
    Ok(())
  }

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

  fn collect_unique_candidates(
    &self,
    digest: &[u32],
    seen: &mut FxHashSet<usize>,
    candidates: &mut Vec<usize>,
  ) {
    seen.clear();
    candidates.clear();

    for (i, table) in self.hash_tables.iter().enumerate() {
      let band_hash = calculate_band_hash(
        &digest[i * self.band_size..(i + 1) * self.band_size],
      );
      if let Some(keys) = table.get(&band_hash) {
        for &key in keys {
          if seen.insert(key) {
            candidates.push(key);
          }
        }
      }
    }
  }

  fn has_multiple_candidates(
    &self,
    digest: &[u32],
    seen: &mut FxHashSet<usize>,
  ) -> bool {
    seen.clear();

    for (i, table) in self.hash_tables.iter().enumerate() {
      let band_hash = calculate_band_hash(
        &digest[i * self.band_size..(i + 1) * self.band_size],
      );
      if let Some(keys) = table.get(&band_hash) {
        for &key in keys {
          if seen.insert(key) && seen.len() > 1 {
            return true;
          }
        }
      }
    }
    false
  }

  fn insert_digest(&mut self, key: usize, digest: &[u32]) -> PyResult<()> {
    self.ensure_digest_len(digest.len())?;

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
  /// Returns an error when threshold or banding parameters are invalid.
  #[new]
  pub fn new(
    threshold: f64,
    num_perm: usize,
    num_bands: usize,
  ) -> PyResult<Self> {
    Self::validate_params(threshold, num_perm, num_bands)?;
    Ok(Self::from_validated(threshold, num_perm, num_bands))
  }

  /// Inserts a `MinHash` into the LSH index.
  ///
  /// # Arguments
  ///
  /// * `key` - A unique identifier for the `MinHash`.
  /// * `minhash` - The `RMinHash` instance to be inserted.
  ///
  ///
  /// # Errors
  ///
  /// Returns an error if the supplied `MinHash` has a different number of permutations.
  pub fn insert(&mut self, key: usize, minhash: &RMinHash) -> PyResult<()> {
    self.insert_digest(key, minhash.hash_values())
  }

  /// Inserts many `(key, minhash)` pairs.
  ///
  /// # Errors
  ///
  /// Returns an error if `entries` is not iterable, if an entry is malformed,
  /// or if a `minhash` has incompatible parameters.
  pub fn insert_pairs(&mut self, entries: Bound<'_, PyAny>) -> PyResult<()> {
    let iterator = PyIterator::from_object(&entries)?;

    for entry in iterator {
      let entry = entry?;
      let (key, minhash): (usize, PyRef<'_, RMinHash>) = entry.extract()?;
      self.insert(key, &minhash)?;
    }

    Ok(())
  }

  /// Inserts many `RMinHash` values using sequential keys.
  ///
  /// Keys are assigned as `start_key + offset`.
  ///
  /// # Errors
  ///
  /// Returns an error if `minhashes` is not iterable, if an item is not an
  /// `RMinHash`, or if a `minhash` has incompatible parameters.
  #[pyo3(signature = (minhashes, start_key=0))]
  pub fn insert_many(
    &mut self,
    minhashes: Bound<'_, PyAny>,
    start_key: usize,
  ) -> PyResult<()> {
    let iterator = PyIterator::from_object(&minhashes)?;
    for (offset, minhash) in iterator.enumerate() {
      let minhash: PyRef<'_, RMinHash> = minhash?.extract()?;
      self.insert_digest(start_key + offset, minhash.hash_values())?;
    }
    Ok(())
  }

  /// Inserts all rows from a compact digest matrix.
  ///
  /// Keys are assigned as `start_key + offset`.
  ///
  /// # Errors
  ///
  /// Returns an error when `num_perm` does not match this index.
  #[pyo3(signature = (digest_matrix, start_key=0))]
  pub fn insert_matrix(
    &mut self,
    digest_matrix: &RMinHashDigestMatrix,
    start_key: usize,
  ) -> PyResult<()> {
    self.ensure_digest_len(digest_matrix.num_perm())?;
    self.key_bands.reserve(digest_matrix.rows());
    for table in &mut self.hash_tables {
      table.reserve(digest_matrix.rows());
    }
    for offset in 0..digest_matrix.rows() {
      self.insert_digest(start_key + offset, digest_matrix.row(offset))?;
    }
    Ok(())
  }

  /// Inserts matrix rows and returns duplicate flags in one pass.
  ///
  /// Flags are `true` for any row that shares at least one LSH band hash with
  /// another row in this insertion call or an already-indexed key.
  ///
  /// # Errors
  ///
  /// Returns an error when `num_perm` does not match this index.
  #[pyo3(signature = (digest_matrix, start_key=0))]
  pub fn insert_matrix_and_query_duplicate_flags(
    &mut self,
    digest_matrix: &RMinHashDigestMatrix,
    start_key: usize,
  ) -> PyResult<Vec<bool>> {
    self.ensure_digest_len(digest_matrix.num_perm())?;
    let rows = digest_matrix.rows();
    self.key_bands.reserve(rows);
    for table in &mut self.hash_tables {
      table.reserve(rows);
    }

    let mut flags = vec![false; rows];

    for offset in 0..rows {
      let key = start_key + offset;
      if let Some(previous_band_hashes) = self.key_bands.get(&key).cloned() {
        self.remove_key_from_bands(key, &previous_band_hashes);
      }

      let digest = digest_matrix.row(offset);
      let band_hashes = self.digest_band_hashes(digest);

      for (table, &band_hash) in
        self.hash_tables.iter_mut().zip(band_hashes.iter())
      {
        let keys = table.entry(band_hash).or_default();
        if let Some(&first_key) = keys.first() {
          flags[offset] = true;
          if keys.len() == 1 && first_key >= start_key {
            let other_offset = first_key - start_key;
            if other_offset < rows {
              flags[other_offset] = true;
            }
          }
        }
        keys.push(key);
      }

      self.key_bands.insert(key, band_hashes);
    }

    Ok(flags)
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
  ///
  /// # Errors
  ///
  /// Returns an error if the supplied `MinHash` has a different number of permutations.
  pub fn query(&self, minhash: &RMinHash) -> PyResult<Vec<usize>> {
    let digest = minhash.hash_values();
    self.ensure_digest_len(digest.len())?;

    let mut seen = FxHashSet::default();
    let mut candidates = Vec::new();
    self.collect_unique_candidates(digest, &mut seen, &mut candidates);
    Ok(candidates)
  }

  /// Queries candidates for all supplied `RMinHash` values.
  ///
  /// # Errors
  ///
  /// Returns an error if `minhashes` is not iterable, if an item is not an
  /// `RMinHash`, or if a `minhash` has incompatible parameters.
  pub fn query_all(
    &self,
    minhashes: Bound<'_, PyAny>,
  ) -> PyResult<Vec<Vec<usize>>> {
    let capacity = minhashes.len().unwrap_or_default();
    let iterator = PyIterator::from_object(&minhashes)?;
    let mut all_candidates = Vec::with_capacity(capacity);
    let mut seen = FxHashSet::default();
    let mut candidates = Vec::new();

    for minhash in iterator {
      let minhash: PyRef<'_, RMinHash> = minhash?.extract()?;
      let digest = minhash.hash_values();
      self.ensure_digest_len(digest.len())?;
      self.collect_unique_candidates(digest, &mut seen, &mut candidates);
      all_candidates.push(candidates.clone());
    }

    Ok(all_candidates)
  }

  /// Returns duplicate flags for all supplied `RMinHash` values.
  ///
  /// Each output flag is `true` when the query has more than one unique
  /// LSH candidate key, equivalent to `len(query(minhash)) > 1`.
  ///
  /// # Errors
  ///
  /// Returns an error if `minhashes` is not iterable, if an item is not an
  /// `RMinHash`, or if a `minhash` has incompatible parameters.
  pub fn query_duplicate_flags(
    &self,
    minhashes: Bound<'_, PyAny>,
  ) -> PyResult<Vec<bool>> {
    let capacity = minhashes.len().unwrap_or_default();
    let iterator = PyIterator::from_object(&minhashes)?;
    let mut flags = Vec::with_capacity(capacity);
    let mut seen = FxHashSet::default();

    for minhash in iterator {
      let minhash: PyRef<'_, RMinHash> = minhash?.extract()?;
      let digest = minhash.hash_values();
      self.ensure_digest_len(digest.len())?;
      flags.push(self.has_multiple_candidates(digest, &mut seen));
    }

    Ok(flags)
  }

  /// Returns duplicate flags for all rows in a compact digest matrix.
  ///
  /// # Errors
  ///
  /// Returns an error when `num_perm` does not match this index.
  pub fn query_duplicate_flags_matrix(
    &self,
    digest_matrix: &RMinHashDigestMatrix,
  ) -> PyResult<Vec<bool>> {
    self.ensure_digest_len(digest_matrix.num_perm())?;
    let mut flags = Vec::with_capacity(digest_matrix.rows());
    let mut seen = FxHashSet::default();

    for row_index in 0..digest_matrix.rows() {
      flags.push(
        self.has_multiple_candidates(digest_matrix.row(row_index), &mut seen),
      );
    }

    Ok(flags)
  }

  /// Returns duplicate flags for matrix rows without mutating this index.
  ///
  /// A row is flagged when it shares at least one band hash with either:
  /// - an existing key already present in this index, or
  /// - another row in the provided matrix.
  ///
  /// This is useful for one-shot batch deduplication workflows where inserting
  /// all rows into the index is unnecessary.
  ///
  /// # Errors
  ///
  /// Returns an error when `num_perm` does not match this index.
  #[allow(clippy::too_many_lines)]
  #[allow(clippy::cast_possible_truncation)]
  pub fn query_duplicate_flags_matrix_one_shot(
    &mut self,
    digest_matrix: &RMinHashDigestMatrix,
  ) -> PyResult<Vec<bool>> {
    self.ensure_digest_len(digest_matrix.num_perm())?;
    let rows = digest_matrix.rows();
    let mut band_match_counts = vec![0usize; rows];
    let has_existing_entries =
      self.hash_tables.iter().any(|table| !table.is_empty());
    let rho_sidecar_present =
      digest_matrix.rho_sparse_occupancy_threshold().is_some();
    let mut rho_band_fold = if rho_sidecar_present && !has_existing_entries {
      Self::rho_band_fold(self.num_bands)
    } else {
      1
    };
    if self.num_bands % rho_band_fold != 0 {
      rho_band_fold = 1;
    }
    let effective_num_bands = self.num_bands / rho_band_fold;
    let effective_band_size = self.band_size * rho_band_fold;
    let sparse_occupancy_threshold = digest_matrix
      .rho_sparse_occupancy_threshold()
      .unwrap_or_else(|| Self::rho_sparse_occupancy_threshold(self.num_perm));
    let sparse_required_band_matches =
      Self::rho_sparse_required_band_matches(effective_num_bands);
    let mut required_band_matches = vec![1usize; rows];
    for (row_index, required) in required_band_matches.iter_mut().enumerate() {
      if digest_matrix
        .rho_non_empty_count(row_index)
        .is_some_and(|count| count < sparse_occupancy_threshold)
      {
        *required = sparse_required_band_matches;
      }
    }

    let sparse_verify_enabled = Self::rho_sparse_verify_enabled()
      && digest_matrix.rho_sparse_verify_perm() > 0;
    let sparse_verify_threshold = Self::rho_sparse_verify_threshold();
    let sparse_verify_max_candidates = Self::rho_sparse_verify_max_candidates();
    let recall_rescue_enabled = rho_sidecar_present
      && rho_band_fold > 1
      && !has_existing_entries
      && Self::rho_recall_rescue_enabled();
    let recall_rescue_min_tokens = Self::rho_recall_rescue_min_tokens();
    let recall_rescue_max_tokens =
      Self::rho_recall_rescue_max_tokens().max(recall_rescue_min_tokens);
    let recall_rescue_required_band_matches =
      Self::rho_recall_rescue_required_band_matches(self.num_bands);
    let mut sparse_verify_checks = 0usize;
    let mut sparse_verify_passes = 0usize;
    let any_sparse_rows =
      required_band_matches.iter().any(|&required| required > 1);

    // Folding bands via polynomial composition only matches the current
    // `calculate_band_hash` implementation when each band boundary is aligned
    // to a 4-u32 chunk. This holds when `band_size % 4 == 0` (common when
    // `num_perm` is a multiple of 128 and `num_bands` divides evenly).
    let precompute_band_hashes =
      recall_rescue_enabled && rho_band_fold > 1 && self.band_size % 4 == 0;
    let (band_hashes, fold_k_pow) = if precompute_band_hashes {
      let mut hashes = vec![0u64; rows.saturating_mul(self.num_bands)];
      for row_index in 0..rows {
        let row = digest_matrix.row(row_index);
        for band_idx in 0..self.num_bands {
          let start = band_idx * self.band_size;
          let end = start + self.band_size;
          hashes[row_index * self.num_bands + band_idx] =
            calculate_band_hash(&row[start..end]);
        }
      }
      let steps = Self::fx_poly_steps(self.band_size);
      (Some(hashes), Some(Self::fx_poly_k_pow(steps)))
    } else {
      (None, None)
    };

    if !any_sparse_rows && !sparse_verify_enabled && !recall_rescue_enabled {
      let mut flags = vec![false; rows];
      for band_idx in 0..effective_num_bands {
        let table = if rho_band_fold == 1 {
          self.hash_tables.get(band_idx)
        } else {
          None
        };
        let mut first_row_by_hash: FxHashMap<u64, usize> = FxHashMap::default();
        first_row_by_hash.reserve(rows);
        for row_index in 0..rows {
          let row = digest_matrix.row(row_index);
          let start = band_idx * effective_band_size;
          let end = start + effective_band_size;
          let band_hash = calculate_band_hash(&row[start..end]);
          if has_existing_entries
            && table
              .is_some_and(|band_table| band_table.contains_key(&band_hash))
          {
            flags[row_index] = true;
          }
          if let Some(&first_row) = first_row_by_hash.get(&band_hash) {
            flags[row_index] = true;
            flags[first_row] = true;
          } else {
            first_row_by_hash.insert(band_hash, row_index);
          }
        }
      }
      self.last_one_shot_sparse_verify_checks = 0;
      self.last_one_shot_sparse_verify_passes = 0;
      return Ok(flags);
    }

    for band_idx in 0..effective_num_bands {
      let table = if rho_band_fold == 1 {
        self.hash_tables.get(band_idx)
      } else {
        None
      };
      let mut first_row_by_hash: FxHashMap<u64, usize> = FxHashMap::default();
      first_row_by_hash.reserve(rows);
      let mut collisions_by_hash: FxHashMap<u64, Vec<usize>> =
        FxHashMap::default();

      for (row_index, band_match_count) in
        band_match_counts.iter_mut().enumerate().take(rows)
      {
        let band_hash = if rho_band_fold == 1 {
          let row = digest_matrix.row(row_index);
          let start = band_idx * effective_band_size;
          let end = start + effective_band_size;
          calculate_band_hash(&row[start..end])
        } else if let (Some(hashes), Some(k_pow)) =
          (band_hashes.as_ref(), fold_k_pow)
        {
          let start_band = band_idx * rho_band_fold;
          let base_offset = row_index * self.num_bands + start_band;
          let mut state =
            (hashes[base_offset] as usize).rotate_right(FX_FINISH_ROTATE);
          for offset in 1..rho_band_fold {
            let next_state = (hashes[base_offset + offset] as usize)
              .rotate_right(FX_FINISH_ROTATE);
            state = state.wrapping_mul(k_pow).wrapping_add(next_state);
          }
          state.rotate_left(FX_FINISH_ROTATE) as u64
        } else {
          let row = digest_matrix.row(row_index);
          let start = band_idx * effective_band_size;
          let end = start + effective_band_size;
          calculate_band_hash(&row[start..end])
        };

        if has_existing_entries
          && table.is_some_and(|band_table| band_table.contains_key(&band_hash))
        {
          *band_match_count = band_match_count.saturating_add(1);
        }
        match first_row_by_hash.entry(band_hash) {
          Entry::Vacant(entry) => {
            entry.insert(row_index);
          }
          Entry::Occupied(entry) => {
            let first_row = *entry.get();
            collisions_by_hash
              .entry(band_hash)
              .or_insert_with(|| {
                let mut rows = Vec::with_capacity(2);
                rows.push(first_row);
                rows
              })
              .push(row_index);
          }
        }
      }

      for row_indices in collisions_by_hash.values() {
        if row_indices.len() < 2 {
          continue;
        }
        for &row_index in row_indices {
          let row_sparse = required_band_matches[row_index] > 1;
          let mut matched = false;
          let mut checked_candidates = 0usize;

          for &other_row in row_indices {
            if other_row == row_index {
              continue;
            }
            let other_sparse = required_band_matches[other_row] > 1;
            if sparse_verify_enabled && (row_sparse || other_sparse) {
              if checked_candidates >= sparse_verify_max_candidates {
                break;
              }
              checked_candidates += 1;
              sparse_verify_checks = sparse_verify_checks.saturating_add(1);
              let passes = match (
                digest_matrix.rho_sparse_verify_signature(row_index),
                digest_matrix.rho_sparse_verify_signature(other_row),
              ) {
                (Some(left), Some(right)) => {
                  Self::sparse_verify_similarity(left, right)
                    >= sparse_verify_threshold
                }
                _ => true,
              };
              if passes {
                sparse_verify_passes = sparse_verify_passes.saturating_add(1);
                matched = true;
                break;
              }
            } else {
              matched = true;
              break;
            }
          }

          if matched {
            band_match_counts[row_index] =
              band_match_counts[row_index].saturating_add(1);
          }
        }
      }
    }

    if recall_rescue_enabled {
      let mut rescue_candidate_mask = vec![0u8; rows];
      let mut rescue_candidate_count = 0usize;
      for row_index in 0..rows {
        if band_match_counts[row_index] != 0
          || required_band_matches[row_index] > 1
        {
          continue;
        }
        if !digest_matrix.rho_source_token_count(row_index).is_some_and(
          |token_count| {
            token_count >= recall_rescue_min_tokens
              && token_count <= recall_rescue_max_tokens
          },
        ) {
          continue;
        }
        rescue_candidate_mask[row_index] = 1;
        rescue_candidate_count = rescue_candidate_count.saturating_add(1);
      }

      if rescue_candidate_count > 0 {
        let mut rescue_band_match_counts = vec![0u8; rows];
        let mut bucket_state_by_hash: FxHashMap<u64, (usize, bool)> =
          FxHashMap::default();
        bucket_state_by_hash.reserve(rows);
        for band_idx in 0..self.num_bands {
          bucket_state_by_hash.clear();
          for row_index in 0..rows {
            let band_hash = if let Some(hashes) = band_hashes.as_ref() {
              hashes[row_index * self.num_bands + band_idx]
            } else {
              let row = digest_matrix.row(row_index);
              let start = band_idx * self.band_size;
              let end = start + self.band_size;
              calculate_band_hash(&row[start..end])
            };
            if let Some((first_row, collided)) =
              bucket_state_by_hash.get_mut(&band_hash)
            {
              if !*collided {
                *collided = true;
                if rescue_candidate_mask[*first_row] == 1 {
                  rescue_band_match_counts[*first_row] =
                    rescue_band_match_counts[*first_row].saturating_add(1);
                }
              }
              if rescue_candidate_mask[row_index] == 1 {
                rescue_band_match_counts[row_index] =
                  rescue_band_match_counts[row_index].saturating_add(1);
              }
            } else {
              bucket_state_by_hash.insert(band_hash, (row_index, false));
            }
          }
        }

        let required_matches_u8 =
          u8::try_from(recall_rescue_required_band_matches).unwrap_or(u8::MAX);
        for row_index in 0..rows {
          if rescue_candidate_mask[row_index] == 1
            && rescue_band_match_counts[row_index] >= required_matches_u8
          {
            band_match_counts[row_index] = required_band_matches[row_index];
          }
        }
      }
    }

    let flags = band_match_counts
      .iter()
      .zip(required_band_matches.iter())
      .map(|(matches, required)| *matches >= *required)
      .collect();

    self.last_one_shot_sparse_verify_checks = sparse_verify_checks;
    self.last_one_shot_sparse_verify_passes = sparse_verify_passes;

    Ok(flags)
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
  /// Returns an error when the `MinHash` instances are incompatible.
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

  #[must_use]
  pub const fn get_last_one_shot_sparse_verify_checks(&self) -> usize {
    self.last_one_shot_sparse_verify_checks
  }

  #[must_use]
  pub const fn get_last_one_shot_sparse_verify_passes(&self) -> usize {
    self.last_one_shot_sparse_verify_passes
  }

  fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> {
    let decoded: Self =
      postcard::from_bytes(state.as_bytes()).map_err(|err| {
        PyValueError::new_err(format!(
          "failed to deserialize RMinHashLSH state: {err}"
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
        "failed to serialize RMinHashLSH state: {err}"
      ))
    })?;
    Ok(PyBytes::new(py, &encoded))
  }

  const fn __getnewargs__(&self) -> (f64, usize, usize) {
    (self.threshold, self.num_perm, self.num_bands)
  }
}

#[cfg(test)]
mod tests {
  use crate::lsh::{RMinHashLSH, FX_FINISH_ROTATE};
  use crate::utils::calculate_band_hash;

  #[test]
  #[allow(clippy::cast_possible_truncation)]
  fn folded_band_hash_matches_direct_hashing() {
    // Deterministic pseudo-random generator (splitmix64-ish).
    fn next_u32(state: &mut u64) -> u32 {
      *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
      let mut z = *state;
      z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
      z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
      (z ^ (z >> 31)) as u32
    }

    for band_size in [4_usize, 8, 12, 16, 20, 28, 32, 64] {
      let steps = RMinHashLSH::fx_poly_steps(band_size);
      let k_pow = RMinHashLSH::fx_poly_k_pow(steps);
      let mut rng = 0x1234_5678_9abc_def0_u64 ^ (band_size as u64);

      for _ in 0..50 {
        let mut values = vec![0u32; band_size * 2];
        for slot in &mut values {
          *slot = next_u32(&mut rng);
        }

        let direct = calculate_band_hash(&values);
        let left = calculate_band_hash(&values[..band_size]);
        let right = calculate_band_hash(&values[band_size..]);
        let left_state = (left as usize).rotate_right(FX_FINISH_ROTATE);
        let right_state = (right as usize).rotate_right(FX_FINISH_ROTATE);

        let combined_state =
          left_state.wrapping_mul(k_pow).wrapping_add(right_state);
        let combined = (combined_state.rotate_left(FX_FINISH_ROTATE)) as u64;

        assert_eq!(combined, direct);
      }
    }
  }
}
