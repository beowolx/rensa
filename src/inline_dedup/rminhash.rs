use crate::inline_dedup::common::{validate_threshold, PAIR_ENTRY_ERROR};
use crate::inline_dedup::RMinHashDeduplicator;
use crate::lsh::RMinHashLSH;
use crate::py_input::extend_token_hashes_from_document;
use crate::rminhash::RMinHash;
use crate::simd::dispatch::PermutationSoA;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyIterator, PyTuple};
use rustc_hash::FxHashMap;

fn default_lsh_bands(threshold: f64, num_perm: usize) -> usize {
  if threshold >= 0.9 {
    4
  } else if threshold >= 0.8 {
    8
  } else if threshold >= 0.7 {
    16
  } else if threshold >= 0.5 {
    32
  } else {
    (num_perm / 2).max(1)
  }
}

fn select_lsh_bands(
  threshold: f64,
  num_perm: usize,
  requested: Option<usize>,
) -> usize {
  let Some(requested) = requested else {
    let selected = default_lsh_bands(threshold, num_perm);
    if num_perm % selected == 0 {
      return selected;
    }

    let max_candidate = selected.min(num_perm);
    return (1..=max_candidate)
      .rev()
      .find(|&bands| num_perm % bands == 0)
      .unwrap_or(1);
  };

  requested
}

fn map_rminhash_pairs<T, F>(
  entries: &Bound<'_, PyAny>,
  num_perm: usize,
  seed: u64,
  mut handler: F,
) -> PyResult<Vec<T>>
where
  F: FnMut(String, &RMinHash) -> PyResult<T>,
{
  let capacity = entries.len().unwrap_or_default();
  let iterator = PyIterator::from_object(entries)?;
  let permutations = RMinHash::build_permutations(num_perm, seed);
  let permutations_soa = PermutationSoA::from_permutations(&permutations);
  let mut scratch = RMinHash::new_compact(num_perm, seed)?;
  let mut token_hashes = Vec::with_capacity(32);
  let mut outcomes = Vec::with_capacity(capacity);

  for entry in iterator {
    let entry = entry?;
    let pair = entry
      .cast::<PyTuple>()
      .map_err(|_| PyTypeError::new_err(PAIR_ENTRY_ERROR))?;
    if pair.len() != 2 {
      return Err(PyTypeError::new_err(PAIR_ENTRY_ERROR));
    }

    let key: String = pair.get_item(0)?.extract()?;
    let value = pair.get_item(1)?;
    if let Ok(minhash) = value.extract::<PyRef<'_, RMinHash>>() {
      outcomes.push(handler(key, &minhash)?);
      continue;
    }

    token_hashes.clear();
    extend_token_hashes_from_document(&value, &mut token_hashes)?;
    scratch.reset_from_token_hashes_with_permutations(
      &token_hashes,
      &permutations,
      &permutations_soa,
    );
    outcomes.push(handler(key, &scratch)?);
  }

  Ok(outcomes)
}

fn map_rminhash_values<T, F>(
  values: &Bound<'_, PyAny>,
  num_perm: usize,
  seed: u64,
  mut handler: F,
) -> PyResult<Vec<T>>
where
  F: FnMut(&RMinHash) -> PyResult<T>,
{
  let capacity = values.len().unwrap_or_default();
  let iterator = PyIterator::from_object(values)?;
  let permutations = RMinHash::build_permutations(num_perm, seed);
  let permutations_soa = PermutationSoA::from_permutations(&permutations);
  let mut scratch = RMinHash::new_compact(num_perm, seed)?;
  let mut token_hashes = Vec::with_capacity(32);
  let mut outcomes = Vec::with_capacity(capacity);

  for value in iterator {
    let value = value?;
    if let Ok(minhash) = value.extract::<PyRef<'_, RMinHash>>() {
      outcomes.push(handler(&minhash)?);
      continue;
    }

    token_hashes.clear();
    extend_token_hashes_from_document(&value, &mut token_hashes)?;
    scratch.reset_from_token_hashes_with_permutations(
      &token_hashes,
      &permutations,
      &permutations_soa,
    );
    outcomes.push(handler(&scratch)?);
  }

  Ok(outcomes)
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
      let bands = select_lsh_bands(threshold, num_perm, num_bands);
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
    entries: &Bound<'_, PyAny>,
  ) -> PyResult<Vec<bool>> {
    map_rminhash_pairs(entries, self.num_perm, self.seed, |key, minhash| {
      self.add(key, minhash)
    })
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
    entries: &Bound<'_, PyAny>,
  ) -> PyResult<Vec<bool>> {
    map_rminhash_pairs(entries, self.num_perm, self.seed, |key, minhash| {
      self.is_duplicate(&key, minhash)
    })
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
    minhashes: &Bound<'_, PyAny>,
  ) -> PyResult<Vec<Vec<String>>> {
    map_rminhash_values(minhashes, self.num_perm, self.seed, |minhash| {
      self.get_duplicates(minhash)
    })
  }

  /// Remove an item from the deduplicator
  pub fn remove(&mut self, key: &str) -> bool {
    let Some(id) = self.key_to_id.remove(key) else {
      return false;
    };

    if let Some(lsh) = self.lsh_index.as_mut() {
      lsh.remove(id);
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
    let lsh_config = self
      .lsh_index
      .as_ref()
      .map(|lsh| (lsh.get_num_perm(), lsh.get_num_bands()));

    self.entries_by_id.clear();
    self.key_to_id.clear();
    self.next_id = 0;

    if let Some((num_perm, num_bands)) = lsh_config {
      self.lsh_index = Some(RMinHashLSH::from_validated(
        self.threshold,
        num_perm,
        num_bands,
      ));
    }
  }
}
