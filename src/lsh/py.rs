use crate::lsh::RMinHashLSH;
use crate::rminhash::{RMinHash, RMinHashDigestMatrix};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyIterator};
use rustc_hash::FxHashSet;

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
  pub fn insert_pairs(&mut self, entries: &Bound<'_, PyAny>) -> PyResult<()> {
    let iterator = PyIterator::from_object(entries)?;

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
    minhashes: &Bound<'_, PyAny>,
    start_key: usize,
  ) -> PyResult<()> {
    let iterator = PyIterator::from_object(minhashes)?;
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
      if let Some(previous_band_hashes) = self.key_bands.remove(&key) {
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
    minhashes: &Bound<'_, PyAny>,
  ) -> PyResult<Vec<Vec<usize>>> {
    let capacity = minhashes.len().unwrap_or_default();
    let iterator = PyIterator::from_object(minhashes)?;
    let mut all_candidates = Vec::with_capacity(capacity);
    let mut seen = FxHashSet::default();

    for minhash in iterator {
      let minhash: PyRef<'_, RMinHash> = minhash?.extract()?;
      let digest = minhash.hash_values();
      self.ensure_digest_len(digest.len())?;
      let mut candidates = Vec::new();
      self.collect_unique_candidates(digest, &mut seen, &mut candidates);
      all_candidates.push(candidates);
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
    minhashes: &Bound<'_, PyAny>,
  ) -> PyResult<Vec<bool>> {
    let capacity = minhashes.len().unwrap_or_default();
    let iterator = PyIterator::from_object(minhashes)?;
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
  pub fn query_duplicate_flags_matrix_one_shot(
    &mut self,
    digest_matrix: &RMinHashDigestMatrix,
  ) -> PyResult<Vec<bool>> {
    self.query_duplicate_flags_matrix_one_shot_inner(digest_matrix)
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
