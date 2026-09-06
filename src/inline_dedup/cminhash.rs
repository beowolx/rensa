use crate::cminhash::CMinHash;
use crate::inline_dedup::common::{validate_threshold, PAIR_ENTRY_ERROR};
use crate::inline_dedup::CMinHashDeduplicator;
use crate::py_input::extend_token_hashes_from_document;
use crate::utils::ratio_usize;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyIterator, PyTuple};
use rustc_hash::{FxHashMap, FxHashSet, FxHasher};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

// Below this size, a linear scan is cheaper than constructing the band index.
const MIN_INDEX_ENTRIES: usize = 128;

const NUM_PERM_ADD_TOKEN_ENTRIES_ERROR: &str =
  "num_perm is not configured; initialize CMinHashDeduplicator with num_perm to add token-set entries";
const NUM_PERM_CHECK_TOKEN_ENTRIES_ERROR: &str =
  "num_perm is not configured; initialize CMinHashDeduplicator with num_perm to check token-set entries";
const NUM_PERM_QUERY_TOKEN_ENTRIES_ERROR: &str =
  "num_perm is not configured; initialize CMinHashDeduplicator with num_perm to query token-set entries";

#[derive(Default)]
struct CMinHashTokenScratch {
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
    let scratch = match &mut self.scratch {
      Some(scratch) => scratch,
      empty => empty.insert(CMinHash::new(num_perm, seed)?),
    };

    token_hashes.clear();
    extend_token_hashes_from_document(document, token_hashes)?;
    scratch.reset_from_token_hashes(token_hashes);
    Ok(scratch)
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

// Most bands have one entry, so allocate a vector only when bands collide.
pub(super) enum SignatureBucket {
  One(Arc<str>),
  Many(Vec<Arc<str>>),
}

impl SignatureBucket {
  fn push(&mut self, key: Arc<str>) {
    match self {
      Self::One(existing) => {
        *self = Self::Many(vec![Arc::clone(existing), key]);
      }
      Self::Many(keys) => keys.push(key),
    }
  }

  fn as_slice(&self) -> &[Arc<str>] {
    match self {
      Self::One(key) => std::slice::from_ref(key),
      Self::Many(keys) => keys,
    }
  }

  // Return whether the bucket can be removed from the index.
  fn remove(&mut self, key: &str) -> bool {
    match self {
      Self::One(existing) => existing.as_ref() == key,
      Self::Many(keys) => {
        keys.retain(|existing| existing.as_ref() != key);
        match keys.as_slice() {
          [] => true,
          [remaining] => {
            *self = Self::One(Arc::clone(remaining));
            false
          }
          _ => false,
        }
      }
    }
  }
}

fn insert_signature_bands(
  index: &mut [FxHashMap<u64, SignatureBucket>],
  key: &str,
  signature: &CMinHash,
) {
  let key: Arc<str> = key.into();
  let hashes = signature_band_hashes(signature.signature_values(), index.len());
  for (band, hash) in index.iter_mut().zip(hashes) {
    band
      .entry(hash)
      .and_modify(|bucket| bucket.push(Arc::clone(&key)))
      .or_insert_with(|| SignatureBucket::One(Arc::clone(&key)));
  }
}

// Find the first accepted integer match count with the exact comparison used
// by verification. Multiplying threshold by num_perm can round differently.
fn exact_band_count(num_perm: usize, threshold: f64) -> usize {
  let mut low = 0;
  let mut high = num_perm;
  while low < high {
    let middle = low + (high - low) / 2;
    if ratio_usize(middle, num_perm) >= threshold {
      high = middle;
    } else {
      low = middle + 1;
    }
  }
  if low <= 1 {
    // Zero matches accepts everything; one match offers only single-lane bands.
    return 0;
  }
  num_perm - low + 1
}

fn signature_band_hashes(
  values: &[u64],
  band_count: usize,
) -> impl Iterator<Item = u64> + '_ {
  let width = values.len() / band_count;
  let extra = values.len() % band_count;
  let mut start = 0;
  (0..band_count).map(move |band| {
    let end = start + width + usize::from(band < extra);
    let mut hasher = FxHasher::default();
    values[start..end].hash(&mut hasher);
    start = end;
    hasher.finish()
  })
}

impl CMinHashDeduplicator {
  fn has_indexed_duplicate(&self, minhash: &CMinHash) -> bool {
    // With d allowed mismatches, d+1 disjoint bands guarantee at least one
    // entirely matching band. Hash collisions only add candidates to verify.
    let mut seen = FxHashSet::default();
    let hashes =
      signature_band_hashes(minhash.signature_values(), self.band_count);
    for (band, hash) in self.signature_bands.iter().zip(hashes) {
      let Some(keys) = band.get(&hash) else {
        continue;
      };
      for candidate in keys.as_slice() {
        if !seen.insert(candidate.as_ref()) {
          continue;
        }
        let Some(existing) = self.existing_signatures.get(candidate.as_ref())
        else {
          continue;
        };
        if minhash.jaccard_at_least_unchecked(existing, self.threshold) {
          return true;
        }
      }
    }
    false
  }

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
      signature_bands: Vec::new(),
      band_count: 0,
      num_perm,
      configured_num_perm: num_perm,
      seed,
    })
  }

  /// # Errors
  ///
  /// Returns an error when the supplied `CMinHash` has an incompatible configuration.
  pub fn add(&mut self, key: String, minhash: &CMinHash) -> PyResult<bool> {
    if self.is_duplicate(&key, minhash)? {
      return Ok(false);
    }

    if self.num_perm.is_none() {
      self.num_perm = Some(minhash.num_perm());
    }
    if self.existing_signatures.is_empty() {
      self.band_count = exact_band_count(minhash.num_perm(), self.threshold);
    }
    if !self.signature_bands.is_empty() {
      insert_signature_bands(&mut self.signature_bands, &key, minhash);
    }
    self.existing_signatures.insert(key, minhash.clone());
    if self.band_count > 0
      && self.signature_bands.is_empty()
      && self.existing_signatures.len() >= MIN_INDEX_ENTRIES
    {
      self.signature_bands =
        (0..self.band_count).map(|_| FxHashMap::default()).collect();
      for (key, signature) in &self.existing_signatures {
        insert_signature_bands(&mut self.signature_bands, key, signature);
      }
    }
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

    if !self.signature_bands.is_empty() {
      return Ok(self.has_indexed_duplicate(minhash));
    }

    for existing_minhash in self.existing_signatures.values() {
      if minhash.jaccard_at_least_unchecked(existing_minhash, self.threshold) {
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
      if minhash.jaccard_at_least_unchecked(existing_minhash, self.threshold) {
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
    let Some(removed) = self.existing_signatures.remove(key) else {
      return false;
    };
    if !self.signature_bands.is_empty() {
      let hashes =
        signature_band_hashes(removed.signature_values(), self.band_count);
      for (band, hash) in self.signature_bands.iter_mut().zip(hashes) {
        if let Some(keys) = band.get_mut(&hash) {
          if keys.remove(key) {
            band.remove(&hash);
          }
        }
      }
    }
    if self.existing_signatures.is_empty() {
      self.num_perm = self.configured_num_perm;
      self.band_count = 0;
      self.signature_bands.clear();
    }
    true
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
    self.signature_bands.clear();
    self.band_count = 0;
    self.num_perm = self.configured_num_perm;
  }
}

#[cfg(test)]
mod tests {
  use crate::inline_dedup::cminhash::{
    exact_band_count, signature_band_hashes,
  };
  use crate::utils::ratio_usize;

  #[test]
  fn signature_bucket_handles_collisions_and_removal() {
    use crate::inline_dedup::cminhash::SignatureBucket;
    use std::sync::Arc;

    let mut bucket = SignatureBucket::One(Arc::from("first"));
    assert!(!bucket.remove("missing"));
    bucket.push(Arc::from("second"));
    bucket.push(Arc::from("third"));
    assert_eq!(bucket.as_slice().len(), 3);
    assert!(!bucket.remove("second"));
    assert_eq!(bucket.as_slice().len(), 2);
    assert!(!bucket.remove("first"));
    assert!(matches!(bucket, SignatureBucket::One(_)));
    assert_eq!(bucket.as_slice()[0].as_ref(), "third");
    assert!(bucket.remove("third"));
  }

  #[test]
  fn exact_bands_never_exclude_an_accepted_binary_signature() {
    for width in 1..=6 {
      for matching in 0..=width {
        let boundary = ratio_usize(matching, width);
        let below = f64::from_bits(boundary.to_bits().saturating_sub(1));
        let above = f64::from_bits(boundary.to_bits() + 1);
        for threshold in [below, boundary, above.min(1.0)] {
          let band_count = exact_band_count(width, threshold);
          let required = (0..=width)
            .find(|&count| ratio_usize(count, width) >= threshold)
            .unwrap();
          assert_eq!(
            band_count,
            if required <= 1 {
              0
            } else {
              width - required + 1
            }
          );
          if band_count == 0 {
            continue;
          }
          let signatures: Vec<Vec<u64>> = (0..1usize << width)
            .map(|bits| {
              (0..width).map(|lane| ((bits >> lane) & 1) as u64).collect()
            })
            .collect();
          let bands: Vec<Vec<_>> = signatures
            .iter()
            .map(|values| signature_band_hashes(values, band_count).collect())
            .collect();
          for (left_index, left) in signatures.iter().enumerate() {
            for (right_index, right) in signatures.iter().enumerate() {
              let matches =
                left.iter().zip(right).filter(|(a, b)| a == b).count();
              if ratio_usize(matches, width) >= threshold {
                assert!(bands[left_index]
                  .iter()
                  .zip(&bands[right_index])
                  .any(|(a, b)| a == b));
              }
            }
          }
        }
      }
    }
  }
}
