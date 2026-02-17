//! Inline deduplication support for continuous duplicate detection.
//!
//! This module provides functionality to check new records against an existing
//! dataset for duplicates in real-time, supporting all `MinHash` variants.

use crate::cminhash::CMinHash;
use crate::lsh::RMinHashLSH;
use crate::py_input::extend_token_hashes_from_document;
use crate::rminhash::RMinHash;
use crate::simd::dispatch::PermutationSoA;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyIterator, PyTuple};
use rustc_hash::FxHashMap;

fn validate_threshold(threshold: f64) -> PyResult<()> {
  if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
    return Err(PyValueError::new_err(
      "threshold must be a finite value between 0.0 and 1.0",
    ));
  }
  Ok(())
}

#[pyclass(module = "rensa")]
pub struct RMinHashDeduplicator {
  threshold: f64,
  num_perm: usize,
  seed: u64,
  entries_by_id: FxHashMap<usize, (String, RMinHash)>,
  lsh_index: Option<RMinHashLSH>,
  next_id: usize,
  key_to_id: FxHashMap<String, usize>,
}

impl RMinHashDeduplicator {
  fn validate_input_minhash(&self, minhash: &RMinHash) -> PyResult<()> {
    if minhash.num_perm() != self.num_perm {
      return Err(PyValueError::new_err(format!(
        "num_perm mismatch: deduplicator expects {}, received {}",
        self.num_perm,
        minhash.num_perm()
      )));
    }
    Ok(())
  }
}

#[pymethods]
impl RMinHashDeduplicator {
  #[new]
  #[pyo3(signature = (threshold, num_perm, use_lsh, num_bands=None, seed=42))]
  /// # Errors
  ///
  /// Returns an error when `threshold`, `num_perm`, or LSH parameters are invalid.
  pub fn new(
    threshold: f64,
    num_perm: usize,
    use_lsh: bool,
    num_bands: Option<usize>,
    seed: u64,
  ) -> PyResult<Self> {
    validate_threshold(threshold)?;
    if num_perm == 0 {
      return Err(PyValueError::new_err("num_perm must be greater than 0"));
    }

    let lsh_index = if use_lsh {
      let default_bands = if threshold >= 0.9 {
        4
      } else if threshold >= 0.8 {
        8
      } else if threshold >= 0.7 {
        16
      } else if threshold >= 0.5 {
        32
      } else {
        (num_perm / 2).max(1)
      };
      let selected_bands = num_bands.unwrap_or(default_bands);

      let bands = if num_bands.is_some() || num_perm % selected_bands == 0 {
        selected_bands
      } else {
        (1..=num_perm)
          .rev()
          .find(|&b| num_perm % b == 0 && b <= selected_bands)
          .unwrap_or(1)
      };

      Some(RMinHashLSH::new(threshold, num_perm, bands)?)
    } else {
      None
    };

    Ok(Self {
      threshold,
      num_perm,
      seed,
      entries_by_id: FxHashMap::default(),
      lsh_index,
      next_id: 0,
      key_to_id: FxHashMap::default(),
    })
  }

  /// Add a new item to the deduplicator. Returns true if added (not a duplicate), false if duplicate.
  ///
  /// # Errors
  ///
  /// Returns an error when the supplied `RMinHash` has an incompatible configuration.
  pub fn add(&mut self, key: String, minhash: &RMinHash) -> PyResult<bool> {
    self.validate_input_minhash(minhash)?;
    if self.is_duplicate(&key, minhash)? {
      return Ok(false);
    }

    let next_id = self.next_id;
    self
      .entries_by_id
      .insert(next_id, (key.clone(), minhash.clone()));
    self.key_to_id.insert(key, next_id);

    if let Some(ref mut lsh) = self.lsh_index {
      lsh.insert(next_id, minhash)?;
    }

    self.next_id += 1;
    Ok(true)
  }

  /// Adds many `(key, minhash_or_tokens)` pairs.
  ///
  /// # Errors
  ///
  /// Returns an error if `entries` is not iterable, if an entry is malformed,
  /// or if a `minhash` has incompatible parameters.
  pub fn add_pairs(
    &mut self,
    entries: Bound<'_, PyAny>,
  ) -> PyResult<Vec<bool>> {
    let capacity = entries.len().unwrap_or_default();
    let iterator = PyIterator::from_object(&entries)?;
    let permutations = RMinHash::build_permutations(self.num_perm, self.seed);
    let permutations_soa = PermutationSoA::from_permutations(&permutations);
    let mut scratch = RMinHash::new_compact(self.num_perm, self.seed)?;
    let mut token_hashes = Vec::with_capacity(32);
    let mut outcomes = Vec::with_capacity(capacity);

    for entry in iterator {
      let entry = entry?;
      let pair = entry.cast::<PyTuple>().map_err(|_| {
        PyTypeError::new_err(
          "each entry must be a tuple of (str, minhash_or_tokens)",
        )
      })?;
      if pair.len() != 2 {
        return Err(PyTypeError::new_err(
          "each entry must be a tuple of (str, minhash_or_tokens)",
        ));
      }

      let key: String = pair.get_item(0)?.extract()?;
      let value = pair.get_item(1)?;
      if let Ok(minhash) = value.extract::<PyRef<'_, RMinHash>>() {
        outcomes.push(self.add(key, &minhash)?);
      } else {
        token_hashes.clear();
        extend_token_hashes_from_document(&value, &mut token_hashes)?;
        scratch.reset_from_token_hashes_with_permutations(
          &token_hashes,
          &permutations,
          &permutations_soa,
        );
        outcomes.push(self.add(key, &scratch)?);
      }
    }

    Ok(outcomes)
  }

  /// Check if an item is a duplicate without adding it
  ///
  /// # Errors
  ///
  /// Returns an error when the supplied `RMinHash` has an incompatible configuration.
  pub fn is_duplicate(&self, key: &str, minhash: &RMinHash) -> PyResult<bool> {
    self.validate_input_minhash(minhash)?;
    if self.key_to_id.contains_key(key) {
      return Ok(true);
    }

    if let Some(ref lsh) = self.lsh_index {
      let candidates = lsh.query(minhash)?;
      for candidate_id in candidates {
        if let Some((_, candidate_minhash)) =
          self.entries_by_id.get(&candidate_id)
        {
          if minhash.jaccard_unchecked(candidate_minhash) >= self.threshold {
            return Ok(true);
          }
        }
      }
    } else {
      for (_, existing_minhash) in self.entries_by_id.values() {
        if minhash.jaccard_unchecked(existing_minhash) >= self.threshold {
          return Ok(true);
        }
      }
    }

    Ok(false)
  }

  /// Checks duplicate status for many `(key, minhash_or_tokens)` pairs.
  ///
  /// # Errors
  ///
  /// Returns an error if `entries` is not iterable, if an entry is malformed,
  /// or if a `minhash` has incompatible parameters.
  pub fn is_duplicate_pairs(
    &self,
    entries: Bound<'_, PyAny>,
  ) -> PyResult<Vec<bool>> {
    let capacity = entries.len().unwrap_or_default();
    let iterator = PyIterator::from_object(&entries)?;
    let permutations = RMinHash::build_permutations(self.num_perm, self.seed);
    let permutations_soa = PermutationSoA::from_permutations(&permutations);
    let mut scratch = RMinHash::new_compact(self.num_perm, self.seed)?;
    let mut token_hashes = Vec::with_capacity(32);
    let mut outcomes = Vec::with_capacity(capacity);

    for entry in iterator {
      let entry = entry?;
      let pair = entry.cast::<PyTuple>().map_err(|_| {
        PyTypeError::new_err(
          "each entry must be a tuple of (str, minhash_or_tokens)",
        )
      })?;
      if pair.len() != 2 {
        return Err(PyTypeError::new_err(
          "each entry must be a tuple of (str, minhash_or_tokens)",
        ));
      }

      let key: String = pair.get_item(0)?.extract()?;
      let value = pair.get_item(1)?;
      if let Ok(minhash) = value.extract::<PyRef<'_, RMinHash>>() {
        outcomes.push(self.is_duplicate(&key, &minhash)?);
      } else {
        token_hashes.clear();
        extend_token_hashes_from_document(&value, &mut token_hashes)?;
        scratch.reset_from_token_hashes_with_permutations(
          &token_hashes,
          &permutations,
          &permutations_soa,
        );
        outcomes.push(self.is_duplicate(&key, &scratch)?);
      }
    }

    Ok(outcomes)
  }

  /// Get duplicate candidates for a given `MinHash`
  ///
  /// # Errors
  ///
  /// Returns an error when the supplied `RMinHash` has an incompatible configuration.
  pub fn get_duplicates(&self, minhash: &RMinHash) -> PyResult<Vec<String>> {
    self.validate_input_minhash(minhash)?;
    let mut duplicates = Vec::new();

    if let Some(ref lsh) = self.lsh_index {
      let candidates = lsh.query(minhash)?;
      for candidate_id in candidates {
        if let Some((candidate_key, candidate_minhash)) =
          self.entries_by_id.get(&candidate_id)
        {
          if minhash.jaccard_unchecked(candidate_minhash) >= self.threshold {
            duplicates.push(candidate_key.clone());
          }
        }
      }
    } else {
      for (key, existing_minhash) in self.entries_by_id.values() {
        if minhash.jaccard_unchecked(existing_minhash) >= self.threshold {
          duplicates.push(key.clone());
        }
      }
    }

    Ok(duplicates)
  }

  /// Gets duplicate candidate key sets for many `RMinHash` or token-set values.
  ///
  /// # Errors
  ///
  /// Returns an error if `minhashes` is not iterable, if an item is not an
  /// `RMinHash`, or if a `minhash` has incompatible parameters.
  pub fn get_duplicate_sets(
    &self,
    minhashes: Bound<'_, PyAny>,
  ) -> PyResult<Vec<Vec<String>>> {
    let capacity = minhashes.len().unwrap_or_default();
    let iterator = PyIterator::from_object(&minhashes)?;
    let permutations = RMinHash::build_permutations(self.num_perm, self.seed);
    let permutations_soa = PermutationSoA::from_permutations(&permutations);
    let mut scratch = RMinHash::new_compact(self.num_perm, self.seed)?;
    let mut token_hashes = Vec::with_capacity(32);
    let mut duplicate_sets = Vec::with_capacity(capacity);

    for minhash in iterator {
      let minhash = minhash?;
      if let Ok(minhash) = minhash.extract::<PyRef<'_, RMinHash>>() {
        duplicate_sets.push(self.get_duplicates(&minhash)?);
      } else {
        token_hashes.clear();
        extend_token_hashes_from_document(&minhash, &mut token_hashes)?;
        scratch.reset_from_token_hashes_with_permutations(
          &token_hashes,
          &permutations,
          &permutations_soa,
        );
        duplicate_sets.push(self.get_duplicates(&scratch)?);
      }
    }

    Ok(duplicate_sets)
  }

  /// Remove an item from the deduplicator
  pub fn remove(&mut self, key: &str) -> bool {
    let Some(id) = self.key_to_id.remove(key) else {
      return false;
    };

    if let Some(ref mut lsh) = self.lsh_index {
      let _ = lsh.remove(id);
    }

    self.entries_by_id.remove(&id).is_some()
  }

  /// Get the number of items in the deduplicator
  #[must_use]
  pub fn len(&self) -> usize {
    self.entries_by_id.len()
  }

  /// Check if the deduplicator is empty
  #[must_use]
  pub fn is_empty(&self) -> bool {
    self.entries_by_id.is_empty()
  }

  /// Clear all items from the deduplicator
  pub fn clear(&mut self) {
    self.entries_by_id.clear();
    self.key_to_id.clear();
    self.next_id = 0;
    if let Some(ref lsh) = self.lsh_index {
      self.lsh_index = Some(RMinHashLSH::from_validated(
        self.threshold,
        lsh.get_num_perm(),
        lsh.get_num_bands(),
      ));
    }
  }
}

/// `InlineDeduplicator` for `CMinHash`
#[pyclass(module = "rensa")]
pub struct CMinHashDeduplicator {
  threshold: f64,
  existing_signatures: FxHashMap<String, CMinHash>,
  num_perm: Option<usize>,
  seed: u64,
}

impl CMinHashDeduplicator {
  fn validate_input_minhash(&self, minhash: &CMinHash) -> PyResult<()> {
    if let Some(expected_num_perm) = self.num_perm {
      if minhash.num_perm() != expected_num_perm {
        return Err(PyValueError::new_err(format!(
          "num_perm mismatch: deduplicator expects {}, received {}",
          expected_num_perm,
          minhash.num_perm()
        )));
      }
    }
    Ok(())
  }
}

#[pymethods]
impl CMinHashDeduplicator {
  #[new]
  #[pyo3(signature = (threshold, num_perm=None, seed=42))]
  /// # Errors
  ///
  /// Returns an error when `threshold` is not in the inclusive range `0.0..=1.0`.
  pub fn new(
    threshold: f64,
    num_perm: Option<usize>,
    seed: u64,
  ) -> PyResult<Self> {
    validate_threshold(threshold)?;
    if let Some(value) = num_perm {
      if value == 0 {
        return Err(PyValueError::new_err("num_perm must be greater than 0"));
      }
    }

    Ok(Self {
      threshold,
      existing_signatures: FxHashMap::default(),
      num_perm,
      seed,
    })
  }

  /// # Errors
  ///
  /// Returns an error when the supplied `CMinHash` has an incompatible configuration.
  pub fn add(&mut self, key: String, minhash: &CMinHash) -> PyResult<bool> {
    self.validate_input_minhash(minhash)?;
    if self.is_duplicate(&key, minhash)? {
      return Ok(false);
    }

    if self.num_perm.is_none() {
      self.num_perm = Some(minhash.num_perm());
    }
    self.existing_signatures.insert(key, minhash.clone());
    Ok(true)
  }

  /// Adds many `(key, minhash_or_tokens)` pairs.
  ///
  /// # Errors
  ///
  /// Returns an error if `entries` is not iterable, if an entry is malformed,
  /// or if a `minhash` has incompatible parameters.
  pub fn add_pairs(
    &mut self,
    entries: Bound<'_, PyAny>,
  ) -> PyResult<Vec<bool>> {
    let capacity = entries.len().unwrap_or_default();
    let iterator = PyIterator::from_object(&entries)?;
    let mut template: Option<CMinHash> = None;
    let mut scratch: Option<CMinHash> = None;
    let mut token_hashes = Vec::with_capacity(32);
    let mut outcomes = Vec::with_capacity(capacity);

    for entry in iterator {
      let entry = entry?;
      let pair = entry.cast::<PyTuple>().map_err(|_| {
        PyTypeError::new_err(
          "each entry must be a tuple of (str, minhash_or_tokens)",
        )
      })?;
      if pair.len() != 2 {
        return Err(PyTypeError::new_err(
          "each entry must be a tuple of (str, minhash_or_tokens)",
        ));
      }

      let key: String = pair.get_item(0)?.extract()?;
      let value = pair.get_item(1)?;
      if let Ok(minhash) = value.extract::<PyRef<'_, CMinHash>>() {
        outcomes.push(self.add(key, &minhash)?);
      } else {
        if template.is_none() {
          let num_perm = self.num_perm.ok_or_else(|| {
            PyValueError::new_err(
              "num_perm is not configured; initialize CMinHashDeduplicator with num_perm to add token-set entries",
            )
          })?;
          let built = CMinHash::new(num_perm, self.seed)?;
          scratch = Some(CMinHash::compact_from_template(&built));
          template = Some(built);
        }

        token_hashes.clear();
        extend_token_hashes_from_document(&value, &mut token_hashes)?;
        let (Some(template_ref), Some(scratch_ref)) =
          (template.as_ref(), scratch.as_mut())
        else {
          return Err(PyValueError::new_err(
            "internal error: token-set template initialization failed",
          ));
        };
        scratch_ref
          .reset_from_token_hashes_with_template(&token_hashes, template_ref);
        outcomes.push(self.add(key, scratch_ref)?);
      }
    }

    Ok(outcomes)
  }

  /// # Errors
  ///
  /// Returns an error when the supplied `CMinHash` has an incompatible configuration.
  pub fn is_duplicate(&self, key: &str, minhash: &CMinHash) -> PyResult<bool> {
    self.validate_input_minhash(minhash)?;
    if self.existing_signatures.contains_key(key) {
      return Ok(true);
    }

    for existing_minhash in self.existing_signatures.values() {
      if minhash.jaccard_unchecked(existing_minhash) >= self.threshold {
        return Ok(true);
      }
    }

    Ok(false)
  }

  /// Checks duplicate status for many `(key, minhash_or_tokens)` pairs.
  ///
  /// # Errors
  ///
  /// Returns an error if `entries` is not iterable, if an entry is malformed,
  /// or if a `minhash` has incompatible parameters.
  pub fn is_duplicate_pairs(
    &self,
    entries: Bound<'_, PyAny>,
  ) -> PyResult<Vec<bool>> {
    let capacity = entries.len().unwrap_or_default();
    let iterator = PyIterator::from_object(&entries)?;
    let mut template: Option<CMinHash> = None;
    let mut scratch: Option<CMinHash> = None;
    let mut token_hashes = Vec::with_capacity(32);
    let mut outcomes = Vec::with_capacity(capacity);

    for entry in iterator {
      let entry = entry?;
      let pair = entry.cast::<PyTuple>().map_err(|_| {
        PyTypeError::new_err(
          "each entry must be a tuple of (str, minhash_or_tokens)",
        )
      })?;
      if pair.len() != 2 {
        return Err(PyTypeError::new_err(
          "each entry must be a tuple of (str, minhash_or_tokens)",
        ));
      }

      let key: String = pair.get_item(0)?.extract()?;
      let value = pair.get_item(1)?;
      if let Ok(minhash) = value.extract::<PyRef<'_, CMinHash>>() {
        outcomes.push(self.is_duplicate(&key, &minhash)?);
      } else {
        if template.is_none() {
          let num_perm = self.num_perm.ok_or_else(|| {
            PyValueError::new_err(
              "num_perm is not configured; initialize CMinHashDeduplicator with num_perm to check token-set entries",
            )
          })?;
          let built = CMinHash::new(num_perm, self.seed)?;
          scratch = Some(CMinHash::compact_from_template(&built));
          template = Some(built);
        }

        token_hashes.clear();
        extend_token_hashes_from_document(&value, &mut token_hashes)?;
        let (Some(template_ref), Some(scratch_ref)) =
          (template.as_ref(), scratch.as_mut())
        else {
          return Err(PyValueError::new_err(
            "internal error: token-set template initialization failed",
          ));
        };
        scratch_ref
          .reset_from_token_hashes_with_template(&token_hashes, template_ref);
        outcomes.push(self.is_duplicate(&key, scratch_ref)?);
      }
    }

    Ok(outcomes)
  }

  /// # Errors
  ///
  /// Returns an error when the supplied `CMinHash` has an incompatible configuration.
  pub fn get_duplicates(&self, minhash: &CMinHash) -> PyResult<Vec<String>> {
    self.validate_input_minhash(minhash)?;
    let mut duplicates = Vec::new();

    for (key, existing_minhash) in &self.existing_signatures {
      if minhash.jaccard_unchecked(existing_minhash) >= self.threshold {
        duplicates.push(key.clone());
      }
    }

    Ok(duplicates)
  }

  /// Gets duplicate candidate key sets for many `CMinHash` or token-set values.
  ///
  /// # Errors
  ///
  /// Returns an error if `minhashes` is not iterable, if an item is not a
  /// `CMinHash`, or if a `minhash` has incompatible parameters.
  pub fn get_duplicate_sets(
    &self,
    minhashes: Bound<'_, PyAny>,
  ) -> PyResult<Vec<Vec<String>>> {
    let capacity = minhashes.len().unwrap_or_default();
    let iterator = PyIterator::from_object(&minhashes)?;
    let mut template: Option<CMinHash> = None;
    let mut scratch: Option<CMinHash> = None;
    let mut token_hashes = Vec::with_capacity(32);
    let mut duplicate_sets = Vec::with_capacity(capacity);

    for minhash in iterator {
      let minhash = minhash?;
      if let Ok(minhash) = minhash.extract::<PyRef<'_, CMinHash>>() {
        duplicate_sets.push(self.get_duplicates(&minhash)?);
      } else {
        if template.is_none() {
          let num_perm = self.num_perm.ok_or_else(|| {
            PyValueError::new_err(
              "num_perm is not configured; initialize CMinHashDeduplicator with num_perm to query token-set entries",
            )
          })?;
          let built = CMinHash::new(num_perm, self.seed)?;
          scratch = Some(CMinHash::compact_from_template(&built));
          template = Some(built);
        }

        token_hashes.clear();
        extend_token_hashes_from_document(&minhash, &mut token_hashes)?;
        let (Some(template_ref), Some(scratch_ref)) =
          (template.as_ref(), scratch.as_mut())
        else {
          return Err(PyValueError::new_err(
            "internal error: token-set template initialization failed",
          ));
        };
        scratch_ref
          .reset_from_token_hashes_with_template(&token_hashes, template_ref);
        duplicate_sets.push(self.get_duplicates(scratch_ref)?);
      }
    }

    Ok(duplicate_sets)
  }

  pub fn remove(&mut self, key: &str) -> bool {
    let removed = self.existing_signatures.remove(key).is_some();
    if self.existing_signatures.is_empty() {
      self.num_perm = None;
    }
    removed
  }

  #[must_use]
  pub fn len(&self) -> usize {
    self.existing_signatures.len()
  }

  #[must_use]
  pub fn is_empty(&self) -> bool {
    self.existing_signatures.is_empty()
  }

  pub fn clear(&mut self) {
    self.existing_signatures.clear();
    self.num_perm = None;
  }
}
