//! Implementation of the R-MinHash algorithm, a novel variant of `MinHash`.
//! R-MinHash is designed for high-performance similarity estimation and deduplication
//! of large datasets, forming a core component of the Rensa library.
//!
//! This algorithm represents an original approach developed for Rensa. It draws
//! inspiration from traditional `MinHash` techniques and concepts discussed in
//! the context of algorithms like C-MinHash, but implements a distinct and
//! simplified method for generating `MinHash` signatures.
//!
//! For context on related advanced `MinHash` techniques, see:
//! - C-MinHash: Rigorously Reducing K Permutations to Two.
//!   Ping Li, Arnd Christian König. [arXiv:2109.03337](https://arxiv.org/abs/2109.03337)
//!
//! Key characteristics of R-MinHash:
//! - Simulates `num_perm` independent hash functions using unique pairs of random
//!   numbers (a, b) for each permutation, applied on-the-fly. This avoids
//!   storing full permutations or complex derivation schemes.
//! - Optimized for speed using batch processing of input items and leveraging
//!   efficient hash computations.
//! - Provides a practical balance between performance and accuracy for large-scale
//!   similarity tasks.
//!
//! Usage:
//! - Create an instance with `RMinHash::new(num_perm, seed)`.
//! - Process data items with `rminhash.update(items)`.
//! - Obtain the signature with `rminhash.digest()`.
//! - Estimate Jaccard similarity with `rminhash.jaccard(&other_rminhash)`.

use crate::py_input::for_each_token_hash;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyIterator, PyList, PyTuple};
use rensa_core::rminhash::{apply_token_hashes_to_values, RMinHashContext};
use rensa_core::utils::{calculate_hash_fast, ratio_usize};
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::sync::Arc;

mod config;
mod matrix;
mod permutation_cache;
mod pipeline;
mod py;
mod rho;
mod send_ptr;
mod token;

use config::DigestBuildConfig;
use matrix::RhoDigestSidecar;

const HASH_BATCH_SIZE: usize = 32;
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
  seed: u64,
  rows: usize,
  data: Vec<u32>,
  rho_sidecar: Option<RhoDigestSidecar>,
}

/// `RMinHash` implements the `MinHash` algorithm for efficient similarity estimation.
#[derive(Clone)]
#[pyclass(module = "rensa", skip_from_py_object)]
pub struct RMinHash {
  context: Arc<RMinHashContext>,
  hash_values: Vec<u32>,
}

#[derive(Deserialize)]
struct RMinHashState {
  num_perm: usize,
  seed: u64,
  hash_values: Vec<u32>,
}

impl<'de> Deserialize<'de> for RMinHash {
  fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
  where
    D: Deserializer<'de>,
  {
    let state = RMinHashState::deserialize(deserializer)?;
    let context = RMinHashContext::new(state.num_perm, state.seed)
      .map_err(serde::de::Error::custom)?;
    Ok(Self {
      context: Arc::new(context),
      hash_values: state.hash_values,
    })
  }
}

impl Serialize for RMinHash {
  fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
  where
    S: Serializer,
  {
    let mut state = serializer.serialize_struct("RMinHash", 3)?;
    state.serialize_field("num_perm", &self.context.num_perm())?;
    state.serialize_field("seed", &self.context.seed())?;
    state.serialize_field("hash_values", &self.hash_values)?;
    state.end()
  }
}

impl RMinHash {
  pub(crate) fn build_permutations(
    num_perm: usize,
    seed: u64,
  ) -> Vec<(u64, u64)> {
    rensa_core::rminhash::build_permutations(num_perm, seed)
  }

  fn build_context(
    num_perm: usize,
    seed: u64,
  ) -> PyResult<Arc<RMinHashContext>> {
    RMinHashContext::new(num_perm, seed)
      .map(Arc::new)
      .map_err(|err| {
        PyValueError::new_err(format!(
          "failed to build RMinHash context: {err}"
        ))
      })
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
    let num_perm = self.context.num_perm();
    Self::validate_num_perm(num_perm)?;
    if self.hash_values.len() != num_perm {
      return Err(PyValueError::new_err(format!(
        "invalid RMinHash state: hash_values length {} does not match num_perm {}",
        self.hash_values.len(),
        num_perm,
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
    if self.num_perm() != other.num_perm() {
      return Err(PyValueError::new_err(format!(
        "num_perm mismatch: left is {}, right is {}",
        self.num_perm(),
        other.num_perm()
      )));
    }
    if self.seed() != other.seed() {
      return Err(PyValueError::new_err(format!(
        "seed mismatch: left is {}, right is {}",
        self.seed(),
        other.seed()
      )));
    }
    Ok(())
  }

  #[inline]
  pub(crate) fn hash_values(&self) -> &[u32] {
    &self.hash_values
  }

  #[inline]
  pub(crate) fn num_perm(&self) -> usize {
    self.context.num_perm()
  }

  #[inline]
  pub(crate) fn seed(&self) -> u64 {
    self.context.seed()
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

    let remainder_start = (self.num_perm() / 8) * 8;
    if remainder_start < self.num_perm() {
      equal_count += self.hash_values[remainder_start..]
        .iter()
        .zip(&other.hash_values[remainder_start..])
        .filter(|&(&a, &b)| a == b)
        .count();
    }

    ratio_usize(equal_count, self.num_perm())
  }

  fn apply_hash_batch(&mut self, hash_batch: &[u64]) {
    apply_token_hashes_to_values(
      &mut self.hash_values,
      self.context.permutations(),
      self.context.permutations_soa(),
      hash_batch,
    );
  }

  fn apply_document_to_values(
    document: &Bound<'_, PyAny>,
    hash_values: &mut [u32],
    context: &RMinHashContext,
    hash_batch: &mut Vec<u64>,
  ) -> PyResult<()> {
    hash_batch.clear();
    for_each_token_hash(document, |token_hash| {
      hash_batch.push(token_hash);
      if hash_batch.len() == HASH_BATCH_SIZE {
        apply_token_hashes_to_values(
          hash_values,
          context.permutations(),
          context.permutations_soa(),
          hash_batch,
        );
        hash_batch.clear();
      }
      Ok(())
    })?;
    if !hash_batch.is_empty() {
      apply_token_hashes_to_values(
        hash_values,
        context.permutations(),
        context.permutations_soa(),
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
    let mut hash_batch = Vec::with_capacity(HASH_BATCH_SIZE);

    for item in items {
      hash_batch.push(calculate_hash_fast(item.as_ref().as_bytes()));
      if hash_batch.len() == HASH_BATCH_SIZE {
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
