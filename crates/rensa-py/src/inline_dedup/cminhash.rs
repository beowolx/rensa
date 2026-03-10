use crate::cminhash::CMinHash;
use crate::inline_dedup::common::{validate_threshold, PAIR_ENTRY_ERROR};
use crate::inline_dedup::CMinHashDeduplicator;
use crate::py_input::extend_token_hashes_from_document;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyIterator, PyTuple};
use rustc_hash::FxHashMap;

const TOKEN_SET_TEMPLATE_ERROR: &str =
  "internal error: token-set template initialization failed";
const NUM_PERM_ADD_TOKEN_ENTRIES_ERROR: &str =
  "num_perm is not configured; initialize CMinHashDeduplicator with num_perm to add token-set entries";
const NUM_PERM_CHECK_TOKEN_ENTRIES_ERROR: &str =
  "num_perm is not configured; initialize CMinHashDeduplicator with num_perm to check token-set entries";
const NUM_PERM_QUERY_TOKEN_ENTRIES_ERROR: &str =
  "num_perm is not configured; initialize CMinHashDeduplicator with num_perm to query token-set entries";

#[derive(Default)]
struct CMinHashTokenScratch {
  template: Option<CMinHash>,
  scratch: Option<CMinHash>,
}

impl CMinHashTokenScratch {
  fn reset_from_document(
    &mut self,
    document: &Bound<'_, PyAny>,
    num_perm: usize,
    seed: u64,
    token_hashes: &mut Vec<u64>,
  ) -> PyResult<&CMinHash> {
    if self.template.is_none() {
      let built = CMinHash::new(num_perm, seed)?;
      self.scratch = Some(CMinHash::compact_from_template(&built));
      self.template = Some(built);
    }

    token_hashes.clear();
    extend_token_hashes_from_document(document, token_hashes)?;
    let Some(template_ref) = self.template.as_ref() else {
      return Err(PyValueError::new_err(TOKEN_SET_TEMPLATE_ERROR));
    };
    let Some(scratch_ref) = self.scratch.as_mut() else {
      return Err(PyValueError::new_err(TOKEN_SET_TEMPLATE_ERROR));
    };
    scratch_ref
      .reset_from_token_hashes_with_template(token_hashes, template_ref);
    Ok(scratch_ref)
  }
}

fn map_cminhash_pairs<T, F>(
  entries: &Bound<'_, PyAny>,
  mut num_perm: Option<usize>,
  seed: u64,
  learn_num_perm: bool,
  token_error: &'static str,
  mut handler: F,
) -> PyResult<Vec<T>>
where
  F: FnMut(String, &CMinHash) -> PyResult<T>,
{
  let capacity = entries.len().unwrap_or_default();
  let iterator = PyIterator::from_object(entries)?;
  let mut token_hashes = Vec::with_capacity(32);
  let mut token_scratch = CMinHashTokenScratch::default();
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
    if let Ok(minhash) = value.extract::<PyRef<'_, CMinHash>>() {
      outcomes.push(handler(key, &minhash)?);
      if learn_num_perm && num_perm.is_none() {
        num_perm = Some(minhash.num_perm());
      }
      continue;
    }

    let configured =
      num_perm.ok_or_else(|| PyValueError::new_err(token_error))?;
    let scratch = token_scratch.reset_from_document(
      &value,
      configured,
      seed,
      &mut token_hashes,
    )?;
    outcomes.push(handler(key, scratch)?);
  }

  Ok(outcomes)
}

fn map_cminhash_values<T, F>(
  values: &Bound<'_, PyAny>,
  num_perm: Option<usize>,
  seed: u64,
  token_error: &'static str,
  mut handler: F,
) -> PyResult<Vec<T>>
where
  F: FnMut(&CMinHash) -> PyResult<T>,
{
  let capacity = values.len().unwrap_or_default();
  let iterator = PyIterator::from_object(values)?;
  let mut token_hashes = Vec::with_capacity(32);
  let mut token_scratch = CMinHashTokenScratch::default();
  let mut outcomes = Vec::with_capacity(capacity);

  for value in iterator {
    let value = value?;
    if let Ok(minhash) = value.extract::<PyRef<'_, CMinHash>>() {
      outcomes.push(handler(&minhash)?);
      continue;
    }

    let configured =
      num_perm.ok_or_else(|| PyValueError::new_err(token_error))?;
    let scratch = token_scratch.reset_from_document(
      &value,
      configured,
      seed,
      &mut token_hashes,
    )?;
    outcomes.push(handler(scratch)?);
  }

  Ok(outcomes)
}

impl CMinHashDeduplicator {
  #[inline]
  fn validate_input_minhash(&self, minhash: &CMinHash) -> PyResult<()> {
    if minhash.seed() != self.seed {
      return Err(PyValueError::new_err(format!(
        "seed mismatch: deduplicator expects {}, received {}",
        self.seed,
        minhash.seed()
      )));
    }
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
    entries: &Bound<'_, PyAny>,
  ) -> PyResult<Vec<bool>> {
    map_cminhash_pairs(
      entries,
      self.num_perm,
      self.seed,
      true,
      NUM_PERM_ADD_TOKEN_ENTRIES_ERROR,
      |key, minhash| self.add(key, minhash),
    )
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
    entries: &Bound<'_, PyAny>,
  ) -> PyResult<Vec<bool>> {
    map_cminhash_pairs(
      entries,
      self.num_perm,
      self.seed,
      false,
      NUM_PERM_CHECK_TOKEN_ENTRIES_ERROR,
      |key, minhash| self.is_duplicate(&key, minhash),
    )
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
    minhashes: &Bound<'_, PyAny>,
  ) -> PyResult<Vec<Vec<String>>> {
    map_cminhash_values(
      minhashes,
      self.num_perm,
      self.seed,
      NUM_PERM_QUERY_TOKEN_ENTRIES_ERROR,
      |minhash| self.get_duplicates(minhash),
    )
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
