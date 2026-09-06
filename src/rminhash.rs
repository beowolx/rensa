//! Seeded `MinHash` sketches for similarity estimation and deduplication.
//!
//! R-MinHash uses Rensa's practical pseudorandom hash family. Its scalar and
//! batch entry points produce the same signature and support incremental
//! updates. Rho is a separate bucket sketch used by the fast deduplication
//! pipeline; its output is not interchangeable with an R-MinHash signature.
//!
//! Algorithm version 2 changes R-MinHash signatures. Persisted version 1
//! sketches and indexes must be rebuilt from their source tokens.

use crate::py_input::for_each_token_hash;
use crate::simd::dispatch::PermutationSoA;
use crate::utils::{calculate_hash_fast, ratio_usize};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyIterator, PyList, PyTuple};
use rand_core::{RngCore, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::sync::Arc;

mod bucket;
mod config;
mod matrix;
mod permutation_cache;
mod pipeline;
mod py;
mod rho;
#[cfg(not(any(PyPy, GraalPy)))]
mod rho_raw;
mod send_ptr;
mod token;

use config::DigestBuildConfig;
use matrix::RhoDigestSidecar;
pub use permutation_cache::{shared_permutations, SharedPermutations};

pub const STATE_VERSION: u32 = 0x524d_4802;
const PY_STATE_PREFIX: &[u8] = b"Rensa:RMinHash:2\0";
const HASH_BATCH_SIZE: usize = 32;
const MAX_HASH_BATCH_SIZE: usize = 4096;
const DEFAULT_DOC_CHUNK_SIZE: usize = 2048;
const MIN_DOC_CHUNK_SIZE: usize = 256;
const MAX_DOC_CHUNK_SIZE: usize = 8192;
const DEFAULT_DOC_PAR_BATCH_SIZE: usize = 256;
const MIN_DOC_PAR_BATCH_SIZE: usize = 32;
const MAX_DOC_PAR_BATCH_SIZE: usize = 2048;
const DEFAULT_PIPELINE_QUEUE_CAP: usize = 2;
const MIN_PIPELINE_QUEUE_CAP: usize = 0;
const MAX_PIPELINE_QUEUE_CAP: usize = 16;
const DEFAULT_PERM_CACHE_MIN_FREQUENCY: usize = 3;
const DEFAULT_MAX_PERM_CACHE_HASHES: usize = 0;
const MIN_MAX_PERM_CACHE_HASHES: usize = 0;
const MAX_MAX_PERM_CACHE_HASHES: usize = 200_000;
const DEFAULT_RHO_PROBES: usize = 4;
const MIN_RHO_PROBES: usize = 1;
const MAX_RHO_PROBES: usize = 4;
const DEFAULT_RHO_TOKEN_BUDGET_MIN: usize = 15;
const MAX_RHO_TOKEN_BUDGET: usize = 4096;
const DEFAULT_RHO_SHORT_FULL_TOKEN_THRESHOLD: usize = 32;
const DEFAULT_RHO_MEDIUM_TOKEN_THRESHOLD: usize = 96;
const MIN_RHO_MEDIUM_TOKEN_THRESHOLD: usize = 33;
const MAX_RHO_MEDIUM_TOKEN_THRESHOLD: usize = 65_536;
const DEFAULT_RHO_MEDIUM_TOKEN_BUDGET: usize = 64;
const MIN_RHO_MEDIUM_TOKEN_BUDGET: usize = 1;
const MAX_RHO_MEDIUM_TOKEN_BUDGET: usize = MAX_RHO_TOKEN_BUDGET;
const DEFAULT_RHO_SPARSE_OCCUPANCY_THRESHOLD_BASE: usize = 56;
const MIN_RHO_SPARSE_OCCUPANCY_THRESHOLD_BASE: usize = 1;
const MAX_RHO_SPARSE_OCCUPANCY_THRESHOLD_BASE: usize = 512;
const DEFAULT_RHO_SPARSE_VERIFY_PERM: usize = 8;
const MIN_RHO_SPARSE_VERIFY_PERM: usize = 1;
const MAX_RHO_SPARSE_VERIFY_PERM: usize = 64;
const DEFAULT_RHO_LONG_DOC_FACTOR: usize = 4;
const MIN_RHO_LONG_DOC_THRESHOLD: usize = 64;
const MAX_RHO_LONG_DOC_THRESHOLD: usize = 8192;
const EMPTY_BUCKET: u32 = u32::MAX;
const FLAT_ROW_OFFSETS_ERROR: &str = "row_offsets must start at 0, be non-decreasing, and end at token_hashes length";
const FLAT_ROW_OFFSET_TYPE_ERROR: &str =
  "row_offsets must be an unsigned integer iterable or C-contiguous unsigned integer buffer";
type ReduceResult = (Py<PyAny>, (usize, u64), Py<PyAny>);

#[derive(Clone)]
#[pyclass(module = "rensa", skip_from_py_object)]
pub struct RMinHashDigestMatrix {
  num_perm: usize,
  rows: usize,
  data: Vec<u32>,
  rho_sidecar: Option<RhoDigestSidecar>,
}

/// `RMinHash` implements the `MinHash` algorithm for efficient similarity estimation.
#[derive(Clone)]
#[pyclass(module = "rensa", skip_from_py_object)]
pub struct RMinHash {
  num_perm: usize,
  seed: u64,
  hash_values: Vec<u32>,
  permutations: Arc<[(u64, u64)]>,
  permutations_soa: Arc<PermutationSoA>,
}

#[derive(Deserialize)]
struct RMinHashState {
  #[serde(rename = "version", deserialize_with = "deserialize_state_version")]
  _version: (),
  num_perm: usize,
  seed: u64,
  hash_values: Vec<u32>,
}

pub fn deserialize_state_version<'de, D>(
  deserializer: D,
) -> Result<(), D::Error>
where
  D: Deserializer<'de>,
{
  if u32::deserialize(deserializer)? != STATE_VERSION {
    return Err(serde::de::Error::custom(
      "unsupported RMinHash state version; rebuild sketches and indexes from their tokens",
    ));
  }
  Ok(())
}

impl Serialize for RMinHash {
  fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
  where
    S: Serializer,
  {
    let mut state = serializer.serialize_struct("RMinHash", 4)?;
    state.serialize_field("version", &STATE_VERSION)?;
    state.serialize_field("num_perm", &self.num_perm)?;
    state.serialize_field("seed", &self.seed)?;
    state.serialize_field("hash_values", &self.hash_values)?;
    state.end()
  }
}

impl<'de> Deserialize<'de> for RMinHash {
  fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
  where
    D: Deserializer<'de>,
  {
    let state = RMinHashState::deserialize(deserializer)?;
    if state.num_perm == 0 || state.hash_values.len() != state.num_perm {
      return Err(serde::de::Error::custom(
        "invalid RMinHash state: num_perm must equal the nonempty hash_values length",
      ));
    }
    Ok(Self {
      num_perm: state.num_perm,
      seed: state.seed,
      hash_values: state.hash_values,
      permutations: Arc::default(),
      permutations_soa: Arc::default(),
    })
  }
}

impl RMinHash {
  pub(crate) fn build_permutations(
    num_perm: usize,
    seed: u64,
  ) -> Vec<(u64, u64)> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    (0..num_perm)
      .map(|_| {
        let a = rng.next_u64() | 1;
        let b = rng.next_u64();
        (a, b)
      })
      .collect()
  }

  fn validate_num_perm(num_perm: usize) -> PyResult<()> {
    if num_perm == 0 {
      return Err(PyValueError::new_err("num_perm must be greater than 0"));
    }
    Ok(())
  }

  fn token_sets_capacity(token_sets: &Bound<'_, PyAny>) -> usize {
    if let Ok(py_list) = token_sets.cast::<PyList>() {
      return py_list.len();
    }
    if let Ok(py_tuple) = token_sets.cast::<PyTuple>() {
      return py_tuple.len();
    }
    token_sets.len().unwrap_or_default()
  }

  fn for_each_document<F>(
    token_sets: &Bound<'_, PyAny>,
    mut visitor: F,
  ) -> PyResult<()>
  where
    F: FnMut(Bound<'_, PyAny>) -> PyResult<()>,
  {
    if let Ok(py_list) = token_sets.cast::<PyList>() {
      for document in py_list.iter() {
        visitor(document)?;
      }
      return Ok(());
    }

    if let Ok(py_tuple) = token_sets.cast::<PyTuple>() {
      for document in py_tuple.iter() {
        visitor(document)?;
      }
      return Ok(());
    }

    let iterator = PyIterator::from_object(token_sets)?;
    for document in iterator {
      visitor(document?)?;
    }
    Ok(())
  }

  fn validate_state(&self) -> PyResult<()> {
    Self::validate_num_perm(self.num_perm)?;
    if self.hash_values.len() != self.num_perm {
      return Err(PyValueError::new_err(format!(
        "invalid RMinHash state: hash_values length {} does not match num_perm {}",
        self.hash_values.len(),
        self.num_perm
      )));
    }
    if !self.permutations.is_empty() && self.permutations.len() != self.num_perm
    {
      return Err(PyValueError::new_err(format!(
        "invalid RMinHash state: permutations length {} does not match num_perm {} (or be compacted to 0)",
        self.permutations.len(),
        self.num_perm
      )));
    }
    if !self.permutations_soa.is_empty()
      && self.permutations_soa.len() != self.num_perm
    {
      return Err(PyValueError::new_err(format!(
        "invalid RMinHash state: permutations_soa length {} does not match num_perm {} (or be compacted to 0)",
        self.permutations_soa.len(),
        self.num_perm
      )));
    }
    Ok(())
  }

  pub(crate) fn ensure_compatible_for_jaccard(
    &self,
    other: &Self,
  ) -> PyResult<()> {
    self.validate_state()?;
    other.validate_state()?;
    if self.num_perm != other.num_perm {
      return Err(PyValueError::new_err(format!(
        "num_perm mismatch: left is {}, right is {}",
        self.num_perm, other.num_perm
      )));
    }
    if self.seed != other.seed {
      return Err(PyValueError::new_err(format!(
        "seed mismatch: left is {}, right is {}",
        self.seed, other.seed
      )));
    }
    Ok(())
  }

  #[inline]
  pub(crate) fn hash_values(&self) -> &[u32] {
    &self.hash_values
  }

  #[inline]
  pub(crate) const fn num_perm(&self) -> usize {
    self.num_perm
  }

  #[inline]
  pub(crate) const fn seed(&self) -> u64 {
    self.seed
  }

  fn ensure_permutations(&mut self) {
    if self.permutations.len() != self.num_perm
      || self.permutations_soa.len() != self.num_perm
    {
      let shared = shared_permutations(self.num_perm, self.seed);
      self.permutations = shared.pairs;
      self.permutations_soa = shared.soa;
    }
  }

  #[inline]
  pub(crate) fn jaccard_unchecked(&self, other: &Self) -> f64 {
    let mut equal_count = 0usize;

    let chunks_a = self.hash_values.chunks_exact(8);
    let chunks_b = other.hash_values.chunks_exact(8);

    for (chunk_a, chunk_b) in chunks_a.zip(chunks_b) {
      equal_count += usize::from(chunk_a[0] == chunk_b[0]);
      equal_count += usize::from(chunk_a[1] == chunk_b[1]);
      equal_count += usize::from(chunk_a[2] == chunk_b[2]);
      equal_count += usize::from(chunk_a[3] == chunk_b[3]);
      equal_count += usize::from(chunk_a[4] == chunk_b[4]);
      equal_count += usize::from(chunk_a[5] == chunk_b[5]);
      equal_count += usize::from(chunk_a[6] == chunk_b[6]);
      equal_count += usize::from(chunk_a[7] == chunk_b[7]);
    }

    let remainder_start = (self.num_perm / 8) * 8;
    if remainder_start < self.num_perm {
      equal_count += self.hash_values[remainder_start..]
        .iter()
        .zip(&other.hash_values[remainder_start..])
        .filter(|&(&a, &b)| a == b)
        .count();
    }

    ratio_usize(equal_count, self.num_perm)
  }

  fn apply_hash_batch(&mut self, hash_batch: &[u64]) {
    Self::apply_token_hashes_to_values(
      &mut self.hash_values,
      &self.permutations,
      &self.permutations_soa,
      hash_batch,
    );
  }

  fn apply_token_hashes_to_values(
    hash_values: &mut [u32],
    permutations: &[(u64, u64)],
    permutations_soa: &PermutationSoA,
    token_hashes: &[u64],
  ) {
    bucket::apply(hash_values, permutations, permutations_soa, token_hashes);
  }

  fn apply_document_to_values(
    document: &Bound<'_, PyAny>,
    hash_values: &mut [u32],
    permutations: &[(u64, u64)],
    permutations_soa: &PermutationSoA,
    hash_batch: &mut Vec<u64>,
  ) -> PyResult<()> {
    hash_batch.clear();
    for_each_token_hash(document, |token_hash| {
      hash_batch.push(token_hash);
      if hash_batch.len() == MAX_HASH_BATCH_SIZE {
        Self::apply_token_hashes_to_values(
          hash_values,
          permutations,
          permutations_soa,
          hash_batch,
        );
        hash_batch.clear();
      }
      Ok(())
    })?;
    if !hash_batch.is_empty() {
      Self::apply_token_hashes_to_values(
        hash_values,
        permutations,
        permutations_soa,
        hash_batch,
      );
      hash_batch.clear();
    }
    Ok(())
  }

  fn update_internal<I, S>(&mut self, items: I)
  where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
  {
    self.ensure_permutations();
    let mut hash_batch = Vec::with_capacity(HASH_BATCH_SIZE);

    for item in items {
      hash_batch.push(calculate_hash_fast(item.as_ref().as_bytes()));
      if hash_batch.len() == MAX_HASH_BATCH_SIZE {
        self.apply_hash_batch(&hash_batch);
        hash_batch.clear();
      }
    }

    if !hash_batch.is_empty() {
      self.apply_hash_batch(&hash_batch);
    }
  }

  /// Updates the `MinHash` with items from any iterable of string-like values.
  pub fn update_iter<I, S>(&mut self, items: I)
  where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
  {
    self.update_internal(items);
  }

  /// Updates the `MinHash` with a new set of items from a vector of strings.
  pub fn update_vec(&mut self, items: Vec<String>) {
    self.update_iter(items);
  }
}

#[cfg(any(PyPy, GraalPy))]
impl RMinHash {
  #[allow(clippy::unnecessary_wraps)]
  pub(in crate::rminhash) const fn try_build_rho_digest_matrix_raw_parallel(
    _token_sets: &Bound<'_, PyAny>,
    _num_perm: usize,
    _seed: u64,
    _probes: usize,
  ) -> PyResult<Option<RMinHashDigestMatrix>> {
    Ok(None)
  }
}

#[cfg(test)]
mod serialization_tests {
  use crate::lsh::RMinHashLSH;
  use crate::rminhash::{RMinHash, STATE_VERSION};

  #[test]
  fn postcard_restores_sketch_updates_and_index_queries() -> pyo3::PyResult<()>
  {
    let mut original = RMinHash::new(128, 42)?;
    original.update_iter(["first", "second"]);
    let bytes = postcard::to_allocvec(&original).unwrap();
    let mut restored: RMinHash = postcard::from_bytes(&bytes).unwrap();
    assert_eq!(restored.digest(), original.digest());
    original.update_iter(["third"]);
    restored.update_iter(["third"]);
    assert_eq!(restored.digest(), original.digest());

    let mut index = RMinHashLSH::new(0.8, 128, 8)?;
    index.insert(0, &original)?;
    let bytes = postcard::to_allocvec(&index).unwrap();
    let restored_index: RMinHashLSH = postcard::from_bytes(&bytes).unwrap();
    assert_eq!(restored_index.query(&restored)?, vec![0]);
    Ok(())
  }

  #[test]
  fn postcard_rejects_legacy_nested_and_invalid_sketches() {
    let legacy = (2_usize, 42_u64, vec![1_u32, 2]);
    let bytes = postcard::to_allocvec(&legacy).unwrap();
    assert!(postcard::from_bytes::<RMinHash>(&bytes).is_err());
    let bytes = postcard::to_allocvec(&vec![legacy]).unwrap();
    assert!(postcard::from_bytes::<Vec<RMinHash>>(&bytes).is_err());
    for (version, num_perm, values) in [
      (STATE_VERSION + 1, 1_usize, vec![0_u32]),
      (STATE_VERSION, 0, vec![]),
      (STATE_VERSION, 2, vec![0]),
    ] {
      let bytes =
        postcard::to_allocvec(&(version, num_perm, 42_u64, values)).unwrap();
      assert!(postcard::from_bytes::<RMinHash>(&bytes).is_err());
    }
  }
}
