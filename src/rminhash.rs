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

use crate::py_input::{
  extend_byte_token_hashes_from_document,
  extend_prehashed_token_values_from_document,
  extend_token_hashes_from_document,
  extend_token_hashes_from_document_with_limit, fast_sequence_length,
  for_each_token_hash,
};
use crate::simd::dispatch::{
  apply_hash_batch_to_values, kernel_kind, kernel_kind_name, PermutationSoA,
};
use crate::utils::calculate_hash_fast;
use pyo3::buffer::{Element, PyBuffer};
use pyo3::exceptions::PyTypeError;
use pyo3::exceptions::PyValueError;
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyIterator, PyList, PyTuple, PyType};
use rand_core::{RngCore, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use std::fs;
use std::os::raw::c_char;
use std::sync::mpsc;
use std::sync::mpsc::TrySendError;
use std::sync::OnceLock;
use std::thread;
use std::time::{Duration, Instant};

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
const DIGEST_CACHE_MAGIC: [u8; 4] = *b"RDC1";
const FLAT_ROW_OFFSETS_ERROR: &str = "row_offsets must start at 0, be non-decreasing, and end at token_hashes length";
const FLAT_ROW_OFFSET_TYPE_ERROR: &str =
  "row_offsets must be an unsigned integer iterable or C-contiguous unsigned integer buffer";
const SEQUENCE_SIZE_ERROR: &str = "sequence size does not fit in usize";
type ReduceResult = (Py<PyAny>, (usize, u64), Py<PyAny>);

#[derive(Clone, Copy)]
struct DigestBuildConfig {
  doc_chunk_size: usize,
  doc_par_batch_size: usize,
  pipeline_queue_cap: usize,
  perm_cache_min_frequency: usize,
  max_perm_cache_hashes: usize,
}

impl DigestBuildConfig {
  fn from_env() -> Self {
    Self {
      doc_chunk_size: read_env_usize_clamped(
        "RENSA_DOC_CHUNK_SIZE",
        DEFAULT_DOC_CHUNK_SIZE,
        MIN_DOC_CHUNK_SIZE,
        MAX_DOC_CHUNK_SIZE,
      ),
      doc_par_batch_size: read_env_usize_clamped(
        "RENSA_DOC_PAR_BATCH_SIZE",
        DEFAULT_DOC_PAR_BATCH_SIZE,
        MIN_DOC_PAR_BATCH_SIZE,
        MAX_DOC_PAR_BATCH_SIZE,
      ),
      pipeline_queue_cap: read_env_usize_clamped(
        "RENSA_PIPELINE_QUEUE_CAP",
        DEFAULT_PIPELINE_QUEUE_CAP,
        MIN_PIPELINE_QUEUE_CAP,
        MAX_PIPELINE_QUEUE_CAP,
      ),
      perm_cache_min_frequency: DEFAULT_PERM_CACHE_MIN_FREQUENCY,
      max_perm_cache_hashes: read_env_usize_clamped(
        "RENSA_MAX_PERM_CACHE_HASHES",
        DEFAULT_MAX_PERM_CACHE_HASHES,
        MIN_MAX_PERM_CACHE_HASHES,
        MAX_MAX_PERM_CACHE_HASHES,
      ),
    }
  }
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
    .map_or(default, |parsed| parsed.clamp(min, max))
}

#[inline]
const fn splitmix64(mut value: u64) -> u64 {
  value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
  value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
  value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
  value ^ (value >> 31)
}

#[inline]
const fn mix_u32(mut value: u32) -> u32 {
  value ^= value >> 16;
  value = value.wrapping_mul(0x7feb_352d);
  value ^= value >> 15;
  value = value.wrapping_mul(0x846c_a68b);
  value ^ (value >> 16)
}

#[inline]
fn parse_rho_probes(probes: Option<usize>) -> usize {
  probes
    .unwrap_or(DEFAULT_RHO_PROBES)
    .clamp(MIN_RHO_PROBES, MAX_RHO_PROBES)
}

fn rho_token_budget(num_perm: usize) -> Option<usize> {
  let default_budget = (num_perm / 9).max(DEFAULT_RHO_TOKEN_BUDGET_MIN);
  if let Ok(raw) = std::env::var("RENSA_RHO_TOKEN_BUDGET") {
    return raw
      .parse::<usize>()
      .ok()
      .map_or(Some(default_budget), |value| {
        let clamped = value.min(MAX_RHO_TOKEN_BUDGET);
        if clamped == 0 {
          None
        } else {
          Some(clamped)
        }
      });
  }
  Some(default_budget)
}

fn rho_token_budget_env_override_is_set() -> bool {
  std::env::var_os("RENSA_RHO_TOKEN_BUDGET").is_some()
}

#[inline]
fn py_ssize_to_usize(value: ffi::Py_ssize_t) -> PyResult<usize> {
  usize::try_from(value).map_err(|_| PyTypeError::new_err(SEQUENCE_SIZE_ERROR))
}

#[derive(Clone, Copy)]
struct TokenBytesRef {
  ptr: *const u8,
  len: usize,
}

// SAFETY: `TokenBytesRef` is only constructed from immutable Python `str`/`bytes`
// buffers. Pointers are only read while the GIL is held (even though work
// runs on Rayon threads), which prevents other Python threads from mutating or
// dropping the backing objects.
unsafe impl Send for TokenBytesRef {}
unsafe impl Sync for TokenBytesRef {}

#[allow(clippy::cast_sign_loss)]
#[allow(clippy::inline_always)]
#[inline(always)]
unsafe fn token_bytes_ref_from_unicode_ptr(
  py: Python<'_>,
  object_ptr: *mut ffi::PyObject,
) -> PyResult<TokenBytesRef> {
  // SAFETY: caller ensures object_ptr is a unicode object.
  if unsafe { ffi::PyUnicode_IS_READY(object_ptr) } == 0
    && unsafe { ffi::PyUnicode_READY(object_ptr) } == -1
  {
    return Err(PyErr::fetch(py));
  }

  // SAFETY: unicode object is ready.
  if unsafe { ffi::PyUnicode_IS_ASCII(object_ptr) } != 0 {
    // SAFETY: `PyUnicode_GET_LENGTH` is always non-negative and fits in `usize`
    // on supported platforms.
    let length = unsafe { ffi::PyUnicode_GET_LENGTH(object_ptr) } as usize;
    let data = unsafe { ffi::PyUnicode_1BYTE_DATA(object_ptr) }.cast::<u8>();
    let ptr = if length == 0 {
      std::ptr::NonNull::<u8>::dangling().as_ptr()
    } else {
      data
    };
    return Ok(TokenBytesRef { ptr, len: length });
  }

  let mut length: ffi::Py_ssize_t = 0;
  // SAFETY: unicode object is valid.
  let utf8_ptr =
    unsafe { ffi::PyUnicode_AsUTF8AndSize(object_ptr, &raw mut length) };
  if utf8_ptr.is_null() {
    return Err(PyErr::fetch(py));
  }
  // SAFETY: `PyUnicode_AsUTF8AndSize` returns a non-negative length.
  let utf8_len = length as usize;
  let ptr = if utf8_len == 0 {
    std::ptr::NonNull::<u8>::dangling().as_ptr()
  } else {
    utf8_ptr.cast::<u8>()
  };
  Ok(TokenBytesRef { ptr, len: utf8_len })
}

#[allow(clippy::cast_sign_loss)]
#[allow(clippy::inline_always)]
#[inline(always)]
unsafe fn token_bytes_ref_from_bytes_ptr(
  py: Python<'_>,
  object_ptr: *mut ffi::PyObject,
) -> PyResult<TokenBytesRef> {
  let mut data_ptr: *mut c_char = std::ptr::null_mut();
  let mut length: ffi::Py_ssize_t = 0;
  // SAFETY: caller ensures object_ptr is bytes.
  if unsafe {
    ffi::PyBytes_AsStringAndSize(object_ptr, &raw mut data_ptr, &raw mut length)
  } == -1
  {
    return Err(PyErr::fetch(py));
  }
  // SAFETY: bytes length is always non-negative and fits in `usize` on supported
  // platforms.
  let byte_len = length as usize;
  let ptr = if byte_len == 0 {
    std::ptr::NonNull::<u8>::dangling().as_ptr()
  } else {
    data_ptr.cast::<u8>()
  };
  Ok(TokenBytesRef { ptr, len: byte_len })
}

fn token_bytes_ref_from_token_ptr(
  py: Python<'_>,
  object_ptr: *mut ffi::PyObject,
) -> PyResult<Option<TokenBytesRef>> {
  // SAFETY: object_ptr is a borrowed Python pointer under the GIL.
  unsafe {
    if ffi::PyUnicode_Check(object_ptr) != 0 {
      return token_bytes_ref_from_unicode_ptr(py, object_ptr).map(Some);
    }
    if ffi::PyBytes_Check(object_ptr) != 0 {
      return token_bytes_ref_from_bytes_ptr(py, object_ptr).map(Some);
    }
  }
  Ok(None)
}

fn rho_medium_token_threshold() -> usize {
  read_env_usize_clamped(
    "RENSA_RHO_MEDIUM_TOKEN_THRESHOLD",
    DEFAULT_RHO_MEDIUM_TOKEN_THRESHOLD,
    MIN_RHO_MEDIUM_TOKEN_THRESHOLD,
    MAX_RHO_MEDIUM_TOKEN_THRESHOLD,
  )
}

fn rho_medium_token_budget() -> usize {
  read_env_usize_clamped(
    "RENSA_RHO_MEDIUM_TOKEN_BUDGET",
    DEFAULT_RHO_MEDIUM_TOKEN_BUDGET,
    MIN_RHO_MEDIUM_TOKEN_BUDGET,
    MAX_RHO_MEDIUM_TOKEN_BUDGET,
  )
}

#[inline]
fn saturating_u16(value: usize) -> u16 {
  u16::try_from(value).unwrap_or(u16::MAX)
}

fn rho_adaptive_token_budget_for_row(
  source_token_count: Option<usize>,
  default_budget: Option<usize>,
) -> Option<usize> {
  if rho_token_budget_env_override_is_set() {
    return default_budget;
  }

  let Some(source_token_count) = source_token_count else {
    return default_budget;
  };

  if source_token_count <= DEFAULT_RHO_SHORT_FULL_TOKEN_THRESHOLD {
    return None;
  }
  if source_token_count <= rho_medium_token_threshold() {
    return Some(rho_medium_token_budget());
  }
  default_budget
}

fn rho_sparse_occupancy_threshold(num_perm: usize) -> usize {
  let base = read_env_usize_clamped(
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

fn rho_sparse_verify_enabled() -> bool {
  std::env::var("RENSA_RHO_SPARSE_VERIFY_ENABLE")
    .ok()
    .is_none_or(|value| value != "0")
}

fn rho_sparse_verify_perm(num_perm: usize) -> usize {
  read_env_usize_clamped(
    "RENSA_RHO_SPARSE_VERIFY_PERM",
    DEFAULT_RHO_SPARSE_VERIFY_PERM,
    MIN_RHO_SPARSE_VERIFY_PERM,
    MAX_RHO_SPARSE_VERIFY_PERM.min(num_perm.max(1)),
  )
}

fn rho_adaptive_probes_enabled() -> bool {
  static ENABLED: OnceLock<bool> = OnceLock::new();
  *ENABLED.get_or_init(|| {
    std::env::var("RENSA_RHO_ADAPTIVE_PROBES")
      .ok()
      .is_some_and(|value| value != "0")
  })
}

fn rho_long_doc_threshold(num_perm: usize) -> usize {
  static OVERRIDE: OnceLock<Option<usize>> = OnceLock::new();
  let default_threshold = num_perm
    .saturating_mul(DEFAULT_RHO_LONG_DOC_FACTOR)
    .clamp(MIN_RHO_LONG_DOC_THRESHOLD, MAX_RHO_LONG_DOC_THRESHOLD);
  OVERRIDE
    .get_or_init(|| {
      std::env::var("RENSA_RHO_LONG_DOC_THRESHOLD")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
    })
    .map_or(default_threshold, |value| {
      value.clamp(MIN_RHO_LONG_DOC_THRESHOLD, MAX_RHO_LONG_DOC_THRESHOLD)
    })
}

#[inline]
fn effective_rho_probes(
  configured_probes: usize,
  source_token_count: usize,
  num_perm: usize,
) -> usize {
  if configured_probes <= 1 || !rho_adaptive_probes_enabled() {
    return configured_probes;
  }
  let long_doc_threshold = rho_long_doc_threshold(num_perm);
  if source_token_count >= long_doc_threshold {
    configured_probes
  } else {
    configured_probes.saturating_sub(1).max(1)
  }
}

fn rho_densify_enabled() -> bool {
  static ENABLED: OnceLock<bool> = OnceLock::new();
  *ENABLED.get_or_init(|| {
    std::env::var("RENSA_RHO_DENSIFY")
      .ok()
      .is_some_and(|value| value != "0")
  })
}

fn digest_cache_dir() -> std::path::PathBuf {
  if let Some(value) = std::env::var_os("RENSA_DIGEST_MATRIX_CACHE_DIR") {
    if !value.is_empty() {
      return std::path::PathBuf::from(value);
    }
  }
  std::env::temp_dir().join("rensa_digest_matrix_cache")
}

fn digest_cache_key_env() -> Option<String> {
  let value = std::env::var("RENSA_DIGEST_MATRIX_CACHE_KEY").ok()?;
  let trimmed = value.trim();
  if trimmed.is_empty() {
    return None;
  }
  Some(trimmed.to_string())
}

fn sanitize_cache_key_component(input: &str) -> String {
  let mut out = String::with_capacity(input.len());
  for ch in input.chars() {
    if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.') {
      out.push(ch);
    } else {
      out.push('_');
    }
  }
  if out.is_empty() {
    out.push('_');
  }
  out
}

struct DigestBuildProfile {
  token_hash_extract: Duration,
  digest_kernel: Duration,
  matrix_build: Duration,
  pipeline_send_wait: Duration,
  pipeline_recv_wait: Duration,
  chunk_jobs: usize,
  pipeline_mode: &'static str,
  doc_chunk_size: usize,
  doc_par_batch_size: usize,
  pipeline_queue_cap: usize,
  perm_cache_entries: usize,
  perm_cache_hits: usize,
}

impl Default for DigestBuildProfile {
  fn default() -> Self {
    Self {
      token_hash_extract: Duration::default(),
      digest_kernel: Duration::default(),
      matrix_build: Duration::default(),
      pipeline_send_wait: Duration::default(),
      pipeline_recv_wait: Duration::default(),
      chunk_jobs: 0,
      pipeline_mode: "sequential",
      doc_chunk_size: DEFAULT_DOC_CHUNK_SIZE,
      doc_par_batch_size: DEFAULT_DOC_PAR_BATCH_SIZE,
      pipeline_queue_cap: DEFAULT_PIPELINE_QUEUE_CAP,
      perm_cache_entries: 0,
      perm_cache_hits: 0,
    }
  }
}

struct DigestChunkJob {
  flat: Vec<u64>,
  ranges: Vec<(usize, usize)>,
  output_ptr_addr: usize,
  output_len: usize,
}

struct DigestChunkResult {
  digest_kernel: Duration,
  perm_cache_entries: usize,
  perm_cache_hits: usize,
}

struct AdaptivePermutationCache {
  digests: FxHashMap<u64, Box<[u32]>>,
  seen_counts: FxHashMap<u64, u8>,
  min_frequency: usize,
  max_hashes: usize,
  max_tracked_seen_hashes: usize,
}

impl AdaptivePermutationCache {
  fn new(min_frequency: usize, max_hashes: usize) -> Self {
    Self {
      digests: FxHashMap::default(),
      seen_counts: FxHashMap::default(),
      min_frequency,
      max_hashes,
      max_tracked_seen_hashes: max_hashes.saturating_mul(4).max(8_192),
    }
  }

  #[inline]
  fn entry_len(&self) -> usize {
    self.digests.len()
  }
}

#[derive(Clone, Copy)]
struct DigestComputeContext<'a> {
  num_perm: usize,
  permutations: &'a [(u64, u64)],
  permutations_soa: &'a PermutationSoA,
  config: DigestBuildConfig,
}

#[derive(Clone)]
struct RhoDigestSidecar {
  non_empty_counts: Vec<u16>,
  source_token_counts: Vec<u16>,
  sparse_occupancy_threshold: usize,
  sparse_verify_perm: usize,
  sparse_verify_signatures: Vec<u32>,
  sparse_verify_active: Vec<u8>,
}

#[derive(Clone)]
#[pyclass(module = "rensa", skip_from_py_object)]
pub struct RMinHashDigestMatrix {
  num_perm: usize,
  rows: usize,
  data: Vec<u32>,
  rho_sidecar: Option<RhoDigestSidecar>,
}

impl RMinHashDigestMatrix {
  #[inline]
  pub(crate) const fn num_perm(&self) -> usize {
    self.num_perm
  }

  #[inline]
  pub(crate) const fn rows(&self) -> usize {
    self.rows
  }

  #[inline]
  pub(crate) fn row(&self, row_index: usize) -> &[u32] {
    let start = row_index * self.num_perm;
    &self.data[start..start + self.num_perm]
  }

  pub(crate) fn rho_non_empty_count(&self, row_index: usize) -> Option<usize> {
    let sidecar = self.rho_sidecar.as_ref()?;
    sidecar
      .non_empty_counts
      .get(row_index)
      .copied()
      .map(usize::from)
  }

  pub(crate) fn rho_source_token_count(
    &self,
    row_index: usize,
  ) -> Option<usize> {
    let sidecar = self.rho_sidecar.as_ref()?;
    sidecar
      .source_token_counts
      .get(row_index)
      .copied()
      .map(usize::from)
  }

  pub(crate) fn rho_sparse_occupancy_threshold(&self) -> Option<usize> {
    self
      .rho_sidecar
      .as_ref()
      .map(|sidecar| sidecar.sparse_occupancy_threshold)
  }

  pub(crate) fn rho_sparse_verify_perm(&self) -> usize {
    self
      .rho_sidecar
      .as_ref()
      .map_or(0, |sidecar| sidecar.sparse_verify_perm)
  }

  pub(crate) fn rho_sparse_verify_signature(
    &self,
    row_index: usize,
  ) -> Option<&[u32]> {
    let sidecar = self.rho_sidecar.as_ref()?;
    if sidecar.sparse_verify_perm == 0 {
      return None;
    }
    if sidecar.sparse_verify_active.get(row_index).copied() != Some(1) {
      return None;
    }
    let start = row_index.saturating_mul(sidecar.sparse_verify_perm);
    let end = start.saturating_add(sidecar.sparse_verify_perm);
    sidecar.sparse_verify_signatures.get(start..end)
  }
}

#[pymethods]
impl RMinHashDigestMatrix {
  #[must_use]
  pub const fn len(&self) -> usize {
    self.rows
  }

  #[must_use]
  pub const fn is_empty(&self) -> bool {
    self.rows == 0
  }

  #[must_use]
  pub const fn get_num_perm(&self) -> usize {
    self.num_perm
  }

  #[must_use]
  pub fn to_rows(&self) -> Vec<Vec<u32>> {
    self
      .data
      .chunks_exact(self.num_perm)
      .map(std::borrow::ToOwned::to_owned)
      .collect()
  }

  #[must_use]
  pub fn get_rho_non_empty_counts(&self) -> Option<Vec<usize>> {
    let sidecar = self.rho_sidecar.as_ref()?;
    Some(
      sidecar
        .non_empty_counts
        .iter()
        .copied()
        .map(usize::from)
        .collect(),
    )
  }

  #[must_use]
  pub fn get_rho_source_token_counts(&self) -> Option<Vec<usize>> {
    let sidecar = self.rho_sidecar.as_ref()?;
    Some(
      sidecar
        .source_token_counts
        .iter()
        .copied()
        .map(usize::from)
        .collect(),
    )
  }

  #[must_use]
  pub fn get_rho_sparse_occupancy_threshold(&self) -> Option<usize> {
    self
      .rho_sidecar
      .as_ref()
      .map(|sidecar| sidecar.sparse_occupancy_threshold)
  }

  #[must_use]
  pub fn get_rho_sparse_row_rate(&self) -> Option<f64> {
    let sidecar = self.rho_sidecar.as_ref()?;
    if sidecar.non_empty_counts.is_empty() {
      return Some(0.0);
    }
    let sparse_rows = sidecar
      .non_empty_counts
      .iter()
      .filter(|&&count| usize::from(count) < sidecar.sparse_occupancy_threshold)
      .count();
    Some(sparse_rows as f64 / sidecar.non_empty_counts.len() as f64)
  }
}

/// `RMinHash` implements the `MinHash` algorithm for efficient similarity estimation.
#[derive(Serialize, Deserialize, Clone)]
#[pyclass(module = "rensa", skip_from_py_object)]
pub struct RMinHash {
  num_perm: usize,
  seed: u64,
  hash_values: Vec<u32>,
  #[serde(skip, default)]
  permutations: Vec<(u64, u64)>,
  #[serde(skip, default)]
  permutations_soa: PermutationSoA,
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

  fn ensure_permutations(&mut self) {
    if self.permutations.len() != self.num_perm {
      self.permutations = Self::build_permutations(self.num_perm, self.seed);
    }
    if self.permutations_soa.len() != self.num_perm {
      self.permutations_soa =
        PermutationSoA::from_permutations(&self.permutations);
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

    equal_count as f64 / self.num_perm as f64
  }

  fn profile_enabled() -> bool {
    std::env::var_os("RENSA_PROFILE_BATCH").is_some()
  }

  fn emit_profile(profile: &DigestBuildProfile, rows: usize, num_perm: usize) {
    let kernel = kernel_kind_name(kernel_kind());
    eprintln!(
      "rensa.profile rows={rows} num_perm={num_perm} kernel={kernel} mode={} chunk_jobs={} doc_chunk_size={} doc_par_batch_size={} pipeline_queue_cap={} perm_cache_entries={} perm_cache_hits={} token_hash_extract={:.6}s digest_kernel={:.6}s matrix_build={:.6}s pipeline_send_wait={:.6}s pipeline_recv_wait={:.6}s",
      profile.pipeline_mode,
      profile.chunk_jobs,
      profile.doc_chunk_size,
      profile.doc_par_batch_size,
      profile.pipeline_queue_cap,
      profile.perm_cache_entries,
      profile.perm_cache_hits,
      profile.token_hash_extract.as_secs_f64(),
      profile.digest_kernel.as_secs_f64(),
      profile.matrix_build.as_secs_f64(),
      profile.pipeline_send_wait.as_secs_f64(),
      profile.pipeline_recv_wait.as_secs_f64(),
    );
  }

  fn digest_cache_path(
    cache_key: &str,
    cache_domain: &str,
    num_perm: usize,
    seed: u64,
  ) -> std::path::PathBuf {
    let sanitized_key = sanitize_cache_key_component(cache_key);
    digest_cache_dir().join(format!(
      "{cache_domain}_{sanitized_key}_np{num_perm}_s{seed}.bin"
    ))
  }

  fn try_load_cached_digest_matrix(
    cache_key: &str,
    cache_domain: &str,
    num_perm: usize,
    seed: u64,
  ) -> Option<RMinHashDigestMatrix> {
    let path = Self::digest_cache_path(cache_key, cache_domain, num_perm, seed);
    let bytes = fs::read(path).ok()?;
    let header_len = 4 + (8 * 4);
    if bytes.len() < header_len {
      return None;
    }
    if bytes[0..4] != DIGEST_CACHE_MAGIC {
      return None;
    }
    let mut offset = 4usize;
    let read_u64 = |payload: &[u8], cursor: &mut usize| -> Option<u64> {
      let end = cursor.checked_add(8)?;
      let mut array = [0u8; 8];
      array.copy_from_slice(payload.get(*cursor..end)?);
      *cursor = end;
      Some(u64::from_le_bytes(array))
    };

    let cached_num_perm =
      usize::try_from(read_u64(&bytes, &mut offset)?).ok()?;
    let cached_rows = usize::try_from(read_u64(&bytes, &mut offset)?).ok()?;
    let cached_seed = read_u64(&bytes, &mut offset)?;
    let cached_len = usize::try_from(read_u64(&bytes, &mut offset)?).ok()?;
    if cached_num_perm != num_perm || cached_seed != seed {
      return None;
    }
    if cached_rows.checked_mul(cached_num_perm)? != cached_len {
      return None;
    }
    let payload_len = cached_len.checked_mul(std::mem::size_of::<u32>())?;
    if bytes.len() != offset.checked_add(payload_len)? {
      return None;
    }

    let mut data = vec![0u32; cached_len];
    #[cfg(target_endian = "little")]
    unsafe {
      std::ptr::copy_nonoverlapping(
        bytes.as_ptr().add(offset),
        data.as_mut_ptr().cast::<u8>(),
        payload_len,
      );
    }
    #[cfg(not(target_endian = "little"))]
    {
      for (chunk, value) in bytes[offset..].chunks_exact(4).zip(data.iter_mut())
      {
        let mut array = [0u8; 4];
        array.copy_from_slice(chunk);
        *value = u32::from_le_bytes(array);
      }
    }

    Some(RMinHashDigestMatrix {
      num_perm: cached_num_perm,
      rows: cached_rows,
      data,
      rho_sidecar: None,
    })
  }

  fn store_cached_digest_matrix(
    cache_key: &str,
    cache_domain: &str,
    num_perm: usize,
    seed: u64,
    matrix: &RMinHashDigestMatrix,
  ) {
    if matrix.num_perm != num_perm {
      return;
    }
    let cache_dir = digest_cache_dir();
    if fs::create_dir_all(&cache_dir).is_err() {
      return;
    }
    let path = Self::digest_cache_path(cache_key, cache_domain, num_perm, seed);
    let data_len = matrix.data.len();
    let Some(payload_len) = data_len.checked_mul(std::mem::size_of::<u32>())
    else {
      return;
    };
    let mut bytes = Vec::with_capacity(4 + (8 * 4) + payload_len);
    bytes.extend_from_slice(&DIGEST_CACHE_MAGIC);
    bytes.extend_from_slice(&(num_perm as u64).to_le_bytes());
    bytes.extend_from_slice(&(matrix.rows as u64).to_le_bytes());
    bytes.extend_from_slice(&seed.to_le_bytes());
    bytes.extend_from_slice(&(data_len as u64).to_le_bytes());

    #[cfg(target_endian = "little")]
    unsafe {
      let start = bytes.len();
      bytes.resize(start + payload_len, 0);
      std::ptr::copy_nonoverlapping(
        matrix.data.as_ptr().cast::<u8>(),
        bytes.as_mut_ptr().add(start),
        payload_len,
      );
    }
    #[cfg(not(target_endian = "little"))]
    for &value in &matrix.data {
      bytes.extend_from_slice(&value.to_le_bytes());
    }

    let tmp_path = path.with_extension(format!("tmp-{}", std::process::id()));
    if fs::write(&tmp_path, &bytes).is_ok()
      && fs::rename(&tmp_path, &path).is_err()
    {
      let _ = fs::remove_file(&tmp_path);
    }
  }

  fn apply_hash_batch_to_values(
    hash_values: &mut [u32],
    permutations: &[(u64, u64)],
    permutations_soa: &PermutationSoA,
    hash_batch: &[u64],
  ) {
    apply_hash_batch_to_values(
      hash_values,
      permutations,
      permutations_soa,
      hash_batch,
    );
  }

  fn apply_hash_batch(&mut self, hash_batch: &[u64]) {
    Self::apply_hash_batch_to_values(
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
    Self::apply_hash_batch_to_values(
      hash_values,
      permutations,
      permutations_soa,
      token_hashes,
    );
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
      if hash_batch.len() == HASH_BATCH_SIZE {
        Self::apply_hash_batch_to_values(
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
      Self::apply_hash_batch_to_values(
        hash_values,
        permutations,
        permutations_soa,
        hash_batch,
      );
      hash_batch.clear();
    }
    Ok(())
  }

  #[allow(clippy::cast_possible_truncation)]
  #[inline]
  const fn rho_bucket_index(
    mixed_hash: u64,
    _num_perm: usize,
    num_perm_u64: u64,
    is_power_of_two: bool,
  ) -> usize {
    if is_power_of_two {
      (mixed_hash & (num_perm_u64 - 1)) as usize
    } else {
      (mixed_hash % num_perm_u64) as usize
    }
  }

  #[inline]
  fn apply_rho_probes_to_row(
    digest_row: &mut [u32],
    token_hash: u64,
    seed: u64,
    probes: usize,
    num_perm_u64: u64,
    is_power_of_two: bool,
  ) {
    const RHO_SALTS: [u64; 4] = [
      0x517c_c1b7_2722_0a95,
      0x6eed_0e9d_a4d9_4a4f,
      0x9e37_79b9_7f4a_7c15,
      0xbf58_476d_1ce4_e5b9,
    ];
    let mut mixed = splitmix64(token_hash ^ seed ^ RHO_SALTS[0]);
    let mut probe = 0usize;
    while probe < probes {
      let bucket = Self::rho_bucket_index(
        mixed,
        digest_row.len(),
        num_perm_u64,
        is_power_of_two,
      );
      let value = (mixed >> 32) as u32;
      digest_row[bucket] = digest_row[bucket].min(value);
      if probe + 1 < probes {
        mixed = splitmix64(mixed ^ RHO_SALTS[(probe + 1) & 3]);
      }
      probe += 1;
    }
  }

  #[allow(clippy::cast_possible_truncation)]
  fn densify_rho_row(digest_row: &mut [u32], seed: u64) {
    if digest_row.is_empty()
      || digest_row.iter().all(|&value| value == EMPTY_BUCKET)
    {
      return;
    }
    let len = digest_row.len();
    let mut next_non_empty = vec![len; len];
    let mut next = len;
    for rev_index in (0..(len * 2)).rev() {
      let index = rev_index % len;
      if digest_row[index] != EMPTY_BUCKET {
        next = index;
      }
      if rev_index < len {
        next_non_empty[index] = next;
      }
    }

    for index in 0..len {
      if digest_row[index] != EMPTY_BUCKET {
        continue;
      }
      let candidate = next_non_empty[index];
      if candidate < len {
        let value = digest_row[candidate];
        let probe = if candidate >= index {
          candidate - index
        } else {
          len - index + candidate
        };
        let index_mix = (index as u32).wrapping_mul(0x9e37_79b9);
        let probe_mix = (probe as u32).wrapping_mul(0x85eb_ca6b);
        let seed_mix = (seed as u32).wrapping_mul(0xc2b2_ae35);
        digest_row[index] = mix_u32(value ^ index_mix ^ probe_mix ^ seed_mix);
      } else {
        digest_row[index] =
          mix_u32((seed as u32) ^ (index as u32).wrapping_mul(0x27d4_eb2d));
      }
    }
  }

  #[allow(clippy::cast_possible_truncation)]
  #[inline]
  const fn sampled_index(
    sample_idx: usize,
    total: usize,
    limit: usize,
  ) -> usize {
    let denom = (limit as u64).wrapping_mul(2);
    if denom == 0 {
      return 0;
    }
    let factor = (sample_idx as u64).wrapping_mul(2).wrapping_add(1);
    if let Some(numer) = (total as u64).checked_mul(factor) {
      return (numer / denom) as usize;
    }
    (((sample_idx as u128 * 2 + 1) * total as u128) / (limit as u128 * 2))
      as usize
  }

  #[inline]
  fn count_non_empty_buckets(digest_row: &[u32]) -> usize {
    digest_row
      .iter()
      .filter(|&&value| value != EMPTY_BUCKET)
      .count()
  }

  const fn sparse_verify_signature_seed(seed: u64, index: usize) -> u64 {
    splitmix64(
      seed
        ^ 0x243f_6a88_85a3_08d3
        ^ ((index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)),
    )
  }

  fn compute_sparse_verify_signature_into(
    signature_row: &mut [u32],
    token_hashes: &[u64],
    seed: u64,
  ) {
    signature_row.fill(u32::MAX);
    if token_hashes.is_empty() {
      return;
    }
    for (perm_idx, value) in signature_row.iter_mut().enumerate() {
      let perm_seed = Self::sparse_verify_signature_seed(seed, perm_idx);
      let mut minimum = u32::MAX;
      for &token_hash in token_hashes {
        let mixed = splitmix64(token_hash ^ perm_seed);
        minimum = minimum.min((mixed >> 32) as u32);
      }
      *value = minimum;
    }
  }

  fn percentile_u16(
    values: &[u16],
    numerator: usize,
    denominator: usize,
  ) -> usize {
    if values.is_empty() {
      return 0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let length = sorted.len();
    let idx =
      (length.saturating_sub(1)).saturating_mul(numerator) / denominator;
    usize::from(sorted[idx])
  }

  fn compute_rho_digest_from_token_hashes_into(
    digest_row: &mut [u32],
    token_hashes: &[u64],
    seed: u64,
    probes: usize,
    token_budget: Option<usize>,
  ) {
    digest_row.fill(EMPTY_BUCKET);
    if digest_row.is_empty() || token_hashes.is_empty() {
      return;
    }
    let num_perm_u64 = digest_row.len() as u64;
    let is_power_of_two = digest_row.len().is_power_of_two();
    if let Some(limit) =
      token_budget.filter(|&budget| budget > 0 && token_hashes.len() > budget)
    {
      for sample_idx in 0..limit {
        let index = Self::sampled_index(sample_idx, token_hashes.len(), limit);
        Self::apply_rho_probes_to_row(
          digest_row,
          token_hashes[index],
          seed,
          probes,
          num_perm_u64,
          is_power_of_two,
        );
      }
    } else {
      for &token_hash in token_hashes {
        Self::apply_rho_probes_to_row(
          digest_row,
          token_hash,
          seed,
          probes,
          num_perm_u64,
          is_power_of_two,
        );
      }
    }
    if rho_densify_enabled() {
      Self::densify_rho_row(digest_row, seed);
    }
  }

  #[allow(clippy::too_many_lines)]
  #[allow(clippy::cast_possible_truncation)]
  #[allow(clippy::cast_possible_wrap)]
  #[allow(clippy::cast_sign_loss)]
  #[allow(clippy::items_after_statements)]
  #[allow(clippy::struct_field_names)]
  #[allow(clippy::option_if_let_else)]
  #[allow(clippy::use_self)]
  #[allow(clippy::missing_const_for_fn)]
  fn try_build_rho_digest_matrix_from_token_sets_parallel(
    token_sets: &Bound<'_, PyAny>,
    num_perm: usize,
    seed: u64,
    probes: usize,
  ) -> PyResult<Option<RMinHashDigestMatrix>> {
    if rayon::current_num_threads() <= 1 {
      return Ok(None);
    }

    let Some(rows) = fast_sequence_length(token_sets)? else {
      return Ok(None);
    };

    let config = DigestBuildConfig::from_env();
    if rows < config.doc_par_batch_size {
      return Ok(None);
    }

    let object_ptr = token_sets.as_ptr();
    // SAFETY: type checks under the GIL.
    let outer_is_list = unsafe { ffi::PyList_Check(object_ptr) != 0 };
    let outer_is_tuple = unsafe { ffi::PyTuple_Check(object_ptr) != 0 };
    if !outer_is_list && !outer_is_tuple {
      return Ok(None);
    }

    let probes = parse_rho_probes(Some(probes));
    let default_token_budget = rho_token_budget(num_perm);
    let sparse_occupancy_threshold = rho_sparse_occupancy_threshold(num_perm);
    let sparse_verify_perm = if rho_sparse_verify_enabled() {
      rho_sparse_verify_perm(num_perm)
    } else {
      0
    };

    let matrix_len = rows.saturating_mul(num_perm);
    let mut matrix_data = Vec::<u32>::with_capacity(matrix_len);

    let mut non_empty_counts = Vec::<u16>::with_capacity(rows);

    let mut source_token_counts = Vec::<u16>::with_capacity(rows);
    let mut sparse_verify_signatures = if sparse_verify_perm > 0 {
      vec![u32::MAX; rows.saturating_mul(sparse_verify_perm)]
    } else {
      Vec::new()
    };
    let mut sparse_verify_active = if sparse_verify_perm > 0 {
      vec![0u8; rows]
    } else {
      Vec::new()
    };

    let profile_enabled = Self::profile_enabled();
    let mut extract_elapsed = Duration::default();
    let mut digest_elapsed = Duration::default();
    let sketch_start = Instant::now();
    let py = token_sets.py();

    let chunk_size = config.doc_chunk_size.max(1);
    let num_perm_u64 = num_perm as u64;
    let is_power_of_two = num_perm.is_power_of_two();

    struct RhoWorkChunk {
      row_start: usize,
      row_end: usize,
      token_refs: Vec<TokenBytesRef>,
      row_token_offsets: Vec<usize>,
      source_token_counts: Vec<usize>,
    }

    #[derive(Clone, Copy)]
    struct MidpointSampler {
      q: usize,
      r: usize,
      step_div: usize,
      step_mod: usize,
      denom: usize,
    }

    impl MidpointSampler {
      #[inline]
      fn new(total: usize, limit: usize) -> Self {
        debug_assert!(limit > 0);
        debug_assert!(total >= limit);

        let denom = limit * 2;
        let total_div = total / limit;
        let total_rem = total - total_div * limit;
        let q = total_div / 2;
        let r = if (total_div & 1) == 0 {
          total_rem
        } else {
          limit + total_rem
        };
        let step_div = total_div;
        let step_mod = total_rem * 2;

        Self {
          q,
          r,
          step_div,
          step_mod,
          denom,
        }
      }

      #[inline]
      fn next(&mut self) -> usize {
        let index = self.q;
        self.r += self.step_mod;
        self.q += self.step_div;
        if self.r >= self.denom {
          self.r -= self.denom;
          self.q += 1;
        }
        index
      }
    }

    #[derive(Clone, Copy)]
    struct SendPtr<T>(*mut T);

    // SAFETY: this is only used for pointers into Rust-owned allocations that
    // outlive the pipeline worker thread.
    unsafe impl<T: Send> Send for SendPtr<T> {}
    unsafe impl<T: Sync> Sync for SendPtr<T> {}

    impl<T> SendPtr<T> {
      #[inline]
      fn new(ptr: *mut T) -> Self {
        Self(ptr)
      }

      #[inline]
      unsafe fn add(self, count: usize) -> *mut T {
        // SAFETY: caller ensures `count` stays within the allocated object.
        unsafe { self.0.add(count) }
      }
    }

    #[derive(Clone, Copy)]
    struct RhoOutputPointers {
      matrix_ptr: SendPtr<u32>,
      non_empty_ptr: SendPtr<u16>,
      source_counts_ptr: SendPtr<u16>,
      sparse_active_ptr: Option<SendPtr<u8>>,
      sparse_sig_ptr: Option<SendPtr<u32>>,
    }

    let output_ptrs = RhoOutputPointers {
      matrix_ptr: SendPtr::new(matrix_data.as_mut_ptr()),
      non_empty_ptr: SendPtr::new(non_empty_counts.as_mut_ptr()),
      source_counts_ptr: SendPtr::new(source_token_counts.as_mut_ptr()),
      sparse_active_ptr: if sparse_verify_perm > 0 {
        Some(SendPtr::new(sparse_verify_active.as_mut_ptr()))
      } else {
        None
      },
      sparse_sig_ptr: if sparse_verify_perm > 0 {
        Some(SendPtr::new(sparse_verify_signatures.as_mut_ptr()))
      } else {
        None
      },
    };

    let (work_tx, work_rx) =
      mpsc::sync_channel::<RhoWorkChunk>(config.pipeline_queue_cap);
    let (result_tx, result_rx) = mpsc::channel::<Duration>();
    let densify_enabled = rho_densify_enabled();

    let worker = thread::spawn(move || {
      for chunk in work_rx {
        let digest_start = Instant::now();
        let row_count = chunk.row_end.saturating_sub(chunk.row_start);

        // SAFETY: output pointers refer to preallocated vectors which remain alive
        // until the worker thread is joined. Each chunk writes to a disjoint row range.
        let matrix_chunk = unsafe {
          std::slice::from_raw_parts_mut(
            output_ptrs
              .matrix_ptr
              .add(chunk.row_start.saturating_mul(num_perm)),
            row_count.saturating_mul(num_perm),
          )
        };
        let non_empty_chunk = unsafe {
          std::slice::from_raw_parts_mut(
            output_ptrs.non_empty_ptr.add(chunk.row_start),
            row_count,
          )
        };
        let source_token_chunk = unsafe {
          std::slice::from_raw_parts_mut(
            output_ptrs.source_counts_ptr.add(chunk.row_start),
            row_count,
          )
        };

        if sparse_verify_perm == 0 {
          matrix_chunk
            .par_chunks_exact_mut(num_perm)
            .zip(non_empty_chunk.par_iter_mut())
	            .zip(source_token_chunk.par_iter_mut())
	            .enumerate()
	            .for_each_init(
	              || Vec::<u64>::with_capacity(64),
	              |token_hashes, (local_row_index, ((row, non_empty_out), source_out))| {
	                let source_token_count =
	                  chunk.source_token_counts[local_row_index];
	                *source_out =
	                  saturating_u16(source_token_count);

                let row_probes =
                  effective_rho_probes(probes, source_token_count, num_perm);
                row.fill(EMPTY_BUCKET);

	                token_hashes.clear();
	                let token_start = chunk.row_token_offsets[local_row_index];
	                let token_end = chunk.row_token_offsets[local_row_index + 1];
	                for token_index in token_start..token_end {
	                  let token_ref = chunk.token_refs[token_index];
	                  let token_bytes = if token_ref.len == 0 {
	                    &[][..]
	                  } else {
	                    // SAFETY: token bytes pointers come from CPython `str`/`bytes` objects,
	                    // The caller holds the GIL while this worker runs.
	                    unsafe {
	                      std::slice::from_raw_parts(token_ref.ptr, token_ref.len)
	                    }
	                  };
	                  let token_hash = calculate_hash_fast(token_bytes);
	                  token_hashes.push(token_hash);
	                  RMinHash::apply_rho_probes_to_row(
	                    row,
	                    token_hash,
                    seed,
                    row_probes,
                    num_perm_u64,
                    is_power_of_two,
                  );
                }
                if densify_enabled {
                  RMinHash::densify_rho_row(row, seed);
                }

                let non_empty = Self::count_non_empty_buckets(row);
                *non_empty_out = saturating_u16(non_empty);
              },
	            );
        } else {
          let Some(sparse_active_ptr) = output_ptrs.sparse_active_ptr else {
            break;
          };
          let Some(sparse_sig_ptr) = output_ptrs.sparse_sig_ptr else {
            break;
          };
          let sparse_active_chunk = unsafe {
            std::slice::from_raw_parts_mut(
              sparse_active_ptr.add(chunk.row_start),
              row_count,
            )
          };
          let sparse_sig_chunk = unsafe {
            std::slice::from_raw_parts_mut(
              sparse_sig_ptr
                .add(chunk.row_start.saturating_mul(sparse_verify_perm)),
              row_count.saturating_mul(sparse_verify_perm),
            )
          };

          matrix_chunk
            .par_chunks_exact_mut(num_perm)
            .zip(non_empty_chunk.par_iter_mut())
            .zip(source_token_chunk.par_iter_mut())
            .zip(sparse_active_chunk.par_iter_mut())
            .zip(sparse_sig_chunk.par_chunks_exact_mut(sparse_verify_perm))
            .enumerate()
            .for_each_init(
              || Vec::<u64>::with_capacity(64),
              |token_hashes,
               (
                local_row_index,
                (
                  (((row, non_empty_out), source_out), sparse_active_out),
                  signature_row,
                ),
              )| {
                let source_token_count =
                  chunk.source_token_counts[local_row_index];
                *source_out = saturating_u16(source_token_count);

                let row_probes =
                  effective_rho_probes(probes, source_token_count, num_perm);
                row.fill(EMPTY_BUCKET);

                token_hashes.clear();
                let token_start = chunk.row_token_offsets[local_row_index];
                let token_end = chunk.row_token_offsets[local_row_index + 1];
                for token_index in token_start..token_end {
                  let token_ref = chunk.token_refs[token_index];
                  let token_bytes = if token_ref.len == 0 {
                    &[][..]
                  } else {
                    // SAFETY: token bytes pointers come from CPython `str`/`bytes` objects,
                    // The caller holds the GIL while this worker runs.
                    unsafe {
                      std::slice::from_raw_parts(token_ref.ptr, token_ref.len)
                    }
                  };
                  let token_hash = calculate_hash_fast(token_bytes);
                  token_hashes.push(token_hash);
                  RMinHash::apply_rho_probes_to_row(
                    row,
                    token_hash,
                    seed,
                    row_probes,
                    num_perm_u64,
                    is_power_of_two,
                  );
                }
                if densify_enabled {
                  RMinHash::densify_rho_row(row, seed);
                }

                let non_empty = Self::count_non_empty_buckets(row);
                *non_empty_out = saturating_u16(non_empty);

                let is_sparse = non_empty < sparse_occupancy_threshold;
                *sparse_active_out = u8::from(is_sparse);
                if is_sparse {
                  RMinHash::compute_sparse_verify_signature_into(
                    signature_row,
                    token_hashes,
                    seed,
                  );
                }
              },
            );
        }

        let elapsed = digest_start.elapsed();
        if result_tx.send(elapsed).is_err() {
          break;
        }
      }
    });

    let mut chunks_sent = 0usize;
    let mut extraction_error: Option<PyErr> = None;
    let mut should_fallback = false;
    let mut row_start = 0usize;
    let medium_token_budget = rho_medium_token_budget();
    while row_start < rows {
      let row_end = (row_start + chunk_size).min(rows);

      let extract_prepare_start = Instant::now();
      let chunk_rows = row_end.saturating_sub(row_start);
      let max_take_per_row = default_token_budget
        .unwrap_or(0)
        .max(DEFAULT_RHO_SHORT_FULL_TOKEN_THRESHOLD)
        .max(medium_token_budget);
      let mut token_refs =
        Vec::with_capacity(chunk_rows.saturating_mul(max_take_per_row));
      let mut row_token_offsets =
        Vec::with_capacity(row_end.saturating_sub(row_start) + 1);
      row_token_offsets.push(0);
      let mut chunk_source_counts =
        Vec::with_capacity(row_end.saturating_sub(row_start));

      for row_index in row_start..row_end {
        // SAFETY: row indices are derived from a CPython `Py_ssize_t` length,
        // so they always fit into `Py_ssize_t`.
        let row_index_ssize = row_index as ffi::Py_ssize_t;
        // SAFETY: outer_is_list/outer_is_tuple guarantees GET_ITEM indexing.
        let document_ptr = unsafe {
          if outer_is_list {
            ffi::PyList_GET_ITEM(object_ptr, row_index_ssize)
          } else {
            ffi::PyTuple_GET_ITEM(object_ptr, row_index_ssize)
          }
        };

        // SAFETY: type checks under the GIL.
        let document_is_list = unsafe { ffi::PyList_Check(document_ptr) != 0 };
        let document_is_tuple =
          unsafe { ffi::PyTuple_Check(document_ptr) != 0 };
        if !document_is_list && !document_is_tuple {
          should_fallback = true;
          break;
        }

        let token_len_ssize = unsafe {
          if document_is_list {
            ffi::PyList_GET_SIZE(document_ptr)
          } else {
            ffi::PyTuple_GET_SIZE(document_ptr)
          }
        };
        let token_len = match py_ssize_to_usize(token_len_ssize) {
          Ok(value) => value,
          Err(err) => {
            extraction_error = Some(err);
            break;
          }
        };
        chunk_source_counts.push(token_len);

        let row_token_budget = rho_adaptive_token_budget_for_row(
          Some(token_len),
          default_token_budget,
        );
        let take =
          row_token_budget.map_or(token_len, |limit| token_len.min(limit));

        if take != 0 {
          if document_is_list {
            let first_item_ptr =
              unsafe { ffi::PyList_GET_ITEM(document_ptr, 0) };

            // SAFETY: type checks under the GIL.
            let first_is_unicode =
              unsafe { ffi::PyUnicode_Check(first_item_ptr) != 0 };
            let first_is_bytes =
              unsafe { ffi::PyBytes_Check(first_item_ptr) != 0 };
            let first_type_ptr = unsafe { ffi::Py_TYPE(first_item_ptr) };

            if take == token_len {
              for index in 0..take {
                // SAFETY: token_len is derived from a CPython `Py_ssize_t` length,
                // so sampled indices always fit back into `Py_ssize_t` as well.
                let index_ssize = index as ffi::Py_ssize_t;
                let item_ptr =
                  unsafe { ffi::PyList_GET_ITEM(document_ptr, index_ssize) };
                let item_type_ptr = unsafe { ffi::Py_TYPE(item_ptr) };

                let token_ref = if first_is_unicode
                  && item_type_ptr == first_type_ptr
                {
                  match unsafe {
                    token_bytes_ref_from_unicode_ptr(py, item_ptr)
                  } {
                    Ok(value) => Some(value),
                    Err(err) => {
                      extraction_error = Some(err);
                      break;
                    }
                  }
                } else if first_is_bytes && item_type_ptr == first_type_ptr {
                  match unsafe { token_bytes_ref_from_bytes_ptr(py, item_ptr) }
                  {
                    Ok(value) => Some(value),
                    Err(err) => {
                      extraction_error = Some(err);
                      break;
                    }
                  }
                } else {
                  match token_bytes_ref_from_token_ptr(py, item_ptr) {
                    Ok(value) => value,
                    Err(err) => {
                      extraction_error = Some(err);
                      break;
                    }
                  }
                };

                let Some(token_ref) = token_ref else {
                  should_fallback = true;
                  break;
                };
                token_refs.push(token_ref);
              }
            } else {
              let mut sampler = MidpointSampler::new(token_len, take);
              for _ in 0..take {
                let index = sampler.next();
                // SAFETY: token_len is derived from a CPython `Py_ssize_t` length,
                // so sampled indices always fit back into `Py_ssize_t` as well.
                let index_ssize = index as ffi::Py_ssize_t;
                let item_ptr =
                  unsafe { ffi::PyList_GET_ITEM(document_ptr, index_ssize) };
                let item_type_ptr = unsafe { ffi::Py_TYPE(item_ptr) };

                let token_ref = if first_is_unicode
                  && item_type_ptr == first_type_ptr
                {
                  match unsafe {
                    token_bytes_ref_from_unicode_ptr(py, item_ptr)
                  } {
                    Ok(value) => Some(value),
                    Err(err) => {
                      extraction_error = Some(err);
                      break;
                    }
                  }
                } else if first_is_bytes && item_type_ptr == first_type_ptr {
                  match unsafe { token_bytes_ref_from_bytes_ptr(py, item_ptr) }
                  {
                    Ok(value) => Some(value),
                    Err(err) => {
                      extraction_error = Some(err);
                      break;
                    }
                  }
                } else {
                  match token_bytes_ref_from_token_ptr(py, item_ptr) {
                    Ok(value) => value,
                    Err(err) => {
                      extraction_error = Some(err);
                      break;
                    }
                  }
                };

                let Some(token_ref) = token_ref else {
                  should_fallback = true;
                  break;
                };
                token_refs.push(token_ref);
              }
            }
          } else {
            let first_item_ptr =
              unsafe { ffi::PyTuple_GET_ITEM(document_ptr, 0) };

            // SAFETY: type checks under the GIL.
            let first_is_unicode =
              unsafe { ffi::PyUnicode_Check(first_item_ptr) != 0 };
            let first_is_bytes =
              unsafe { ffi::PyBytes_Check(first_item_ptr) != 0 };
            let first_type_ptr = unsafe { ffi::Py_TYPE(first_item_ptr) };

            if take == token_len {
              for index in 0..take {
                // SAFETY: token_len is derived from a CPython `Py_ssize_t` length,
                // so sampled indices always fit back into `Py_ssize_t` as well.
                let index_ssize = index as ffi::Py_ssize_t;
                let item_ptr =
                  unsafe { ffi::PyTuple_GET_ITEM(document_ptr, index_ssize) };
                let item_type_ptr = unsafe { ffi::Py_TYPE(item_ptr) };

                let token_ref = if first_is_unicode
                  && item_type_ptr == first_type_ptr
                {
                  match unsafe {
                    token_bytes_ref_from_unicode_ptr(py, item_ptr)
                  } {
                    Ok(value) => Some(value),
                    Err(err) => {
                      extraction_error = Some(err);
                      break;
                    }
                  }
                } else if first_is_bytes && item_type_ptr == first_type_ptr {
                  match unsafe { token_bytes_ref_from_bytes_ptr(py, item_ptr) }
                  {
                    Ok(value) => Some(value),
                    Err(err) => {
                      extraction_error = Some(err);
                      break;
                    }
                  }
                } else {
                  match token_bytes_ref_from_token_ptr(py, item_ptr) {
                    Ok(value) => value,
                    Err(err) => {
                      extraction_error = Some(err);
                      break;
                    }
                  }
                };

                let Some(token_ref) = token_ref else {
                  should_fallback = true;
                  break;
                };
                token_refs.push(token_ref);
              }
            } else {
              let mut sampler = MidpointSampler::new(token_len, take);
              for _ in 0..take {
                let index = sampler.next();
                // SAFETY: token_len is derived from a CPython `Py_ssize_t` length,
                // so sampled indices always fit back into `Py_ssize_t` as well.
                let index_ssize = index as ffi::Py_ssize_t;
                let item_ptr =
                  unsafe { ffi::PyTuple_GET_ITEM(document_ptr, index_ssize) };
                let item_type_ptr = unsafe { ffi::Py_TYPE(item_ptr) };

                let token_ref = if first_is_unicode
                  && item_type_ptr == first_type_ptr
                {
                  match unsafe {
                    token_bytes_ref_from_unicode_ptr(py, item_ptr)
                  } {
                    Ok(value) => Some(value),
                    Err(err) => {
                      extraction_error = Some(err);
                      break;
                    }
                  }
                } else if first_is_bytes && item_type_ptr == first_type_ptr {
                  match unsafe { token_bytes_ref_from_bytes_ptr(py, item_ptr) }
                  {
                    Ok(value) => Some(value),
                    Err(err) => {
                      extraction_error = Some(err);
                      break;
                    }
                  }
                } else {
                  match token_bytes_ref_from_token_ptr(py, item_ptr) {
                    Ok(value) => value,
                    Err(err) => {
                      extraction_error = Some(err);
                      break;
                    }
                  }
                };

                let Some(token_ref) = token_ref else {
                  should_fallback = true;
                  break;
                };
                token_refs.push(token_ref);
              }
            }
          }

          if extraction_error.is_some() || should_fallback {
            break;
          }
        }

        row_token_offsets.push(token_refs.len());
      }

      if profile_enabled {
        extract_elapsed += extract_prepare_start.elapsed();
      }

      if extraction_error.is_some() || should_fallback {
        break;
      }

      if let Err(err) = work_tx.send(RhoWorkChunk {
        row_start,
        row_end,
        token_refs,
        row_token_offsets,
        source_token_counts: chunk_source_counts,
      }) {
        extraction_error = Some(PyValueError::new_err(format!(
          "rho pipeline worker stopped unexpectedly: {err}"
        )));
        break;
      }
      chunks_sent += 1;
      row_start = row_end;
    }

    drop(work_tx);
    for _ in 0..chunks_sent {
      match result_rx.recv() {
        Ok(duration) => digest_elapsed += duration,
        Err(err) => {
          extraction_error = Some(PyValueError::new_err(format!(
            "rho pipeline worker failed to report results: {err}"
          )));
          break;
        }
      }
    }

    if worker.join().is_err() && extraction_error.is_none() {
      extraction_error =
        Some(PyValueError::new_err("rho pipeline worker panicked"));
    }

    if let Some(err) = extraction_error {
      return Err(err);
    }
    if should_fallback {
      return Ok(None);
    }
    // SAFETY: each slot is written by the worker before reaching this point.
    unsafe {
      matrix_data.set_len(matrix_len);
      non_empty_counts.set_len(rows);
      source_token_counts.set_len(rows);
    }

    if profile_enabled {
      let total_elapsed = sketch_start.elapsed();
      let total = extract_elapsed + digest_elapsed;
      let matrix_elapsed = total_elapsed.checked_sub(total).unwrap_or_default();
      let token_budget = default_token_budget.unwrap_or_default();
      let adaptive_probes = usize::from(rho_adaptive_probes_enabled());
      let long_doc_threshold = rho_long_doc_threshold(num_perm);
      let non_empty_p50 = Self::percentile_u16(&non_empty_counts, 50, 100);
      let non_empty_p90 = Self::percentile_u16(&non_empty_counts, 90, 100);
      let non_empty_p99 = Self::percentile_u16(&non_empty_counts, 99, 100);
      let sparse_rows = non_empty_counts
        .iter()
        .filter(|&&count| usize::from(count) < sparse_occupancy_threshold)
        .count();
      let sparse_row_rate = if rows == 0 {
        0.0
      } else {
        sparse_rows as f64 / rows as f64
      };
      eprintln!(
        "rensa.profile rows={rows} num_perm={num_perm} kernel=rho mode=parallel-rho probes={probes} adaptive_probes={adaptive_probes} long_doc_threshold={long_doc_threshold} token_budget={token_budget} rho_non_empty_p50={non_empty_p50} rho_non_empty_p90={non_empty_p90} rho_non_empty_p99={non_empty_p99} rho_sparse_row_rate={sparse_row_rate:.6} token_hash_extract={:.6}s digest_kernel={:.6}s matrix_build={:.6}s",
        extract_elapsed.as_secs_f64(),
        digest_elapsed.as_secs_f64(),
        matrix_elapsed.as_secs_f64(),
      );
    }

    let rho_sidecar = RhoDigestSidecar {
      non_empty_counts,
      source_token_counts,
      sparse_occupancy_threshold,
      sparse_verify_perm,
      sparse_verify_signatures,
      sparse_verify_active,
    };

    Ok(Some(RMinHashDigestMatrix {
      num_perm,
      rows,
      data: matrix_data,
      rho_sidecar: Some(rho_sidecar),
    }))
  }

  #[allow(clippy::too_many_lines)]
  fn build_rho_digest_matrix_from_token_sets_streaming(
    token_sets: &Bound<'_, PyAny>,
    num_perm: usize,
    seed: u64,
    probes: usize,
  ) -> PyResult<RMinHashDigestMatrix> {
    let capacity = Self::token_sets_capacity(token_sets);
    let mut matrix_data = Vec::with_capacity(capacity.saturating_mul(num_perm));
    let mut rows = 0usize;
    let probes = parse_rho_probes(Some(probes));
    let default_token_budget = rho_token_budget(num_perm);
    let sparse_occupancy_threshold = rho_sparse_occupancy_threshold(num_perm);
    let sparse_verify_perm = if rho_sparse_verify_enabled() {
      rho_sparse_verify_perm(num_perm)
    } else {
      0
    };
    let profile_enabled = Self::profile_enabled();
    let mut extract_elapsed = Duration::default();
    let mut digest_elapsed = Duration::default();
    let sketch_start = Instant::now();
    let mut token_hashes = Vec::new();
    let mut non_empty_counts = Vec::with_capacity(capacity);
    let mut source_token_counts = Vec::with_capacity(capacity);
    let mut sparse_verify_signatures =
      Vec::with_capacity(capacity.saturating_mul(sparse_verify_perm));
    let mut sparse_verify_active = Vec::with_capacity(capacity);

    Self::for_each_document(token_sets, |document| {
      let row_start = matrix_data.len();
      matrix_data.resize(row_start + num_perm, EMPTY_BUCKET);
      let row = &mut matrix_data[row_start..row_start + num_perm];
      let extract_start = Instant::now();
      let source_token_count = fast_sequence_length(&document)?;
      let row_token_budget = rho_adaptive_token_budget_for_row(
        source_token_count,
        default_token_budget,
      );
      token_hashes.clear();
      extend_token_hashes_from_document_with_limit(
        &document,
        &mut token_hashes,
        row_token_budget,
      )?;
      let row_source_token_count =
        source_token_count.unwrap_or(token_hashes.len());
      if profile_enabled {
        extract_elapsed += extract_start.elapsed();
      }
      let row_probes =
        effective_rho_probes(probes, row_source_token_count, num_perm);
      let digest_start = Instant::now();
      Self::compute_rho_digest_from_token_hashes_into(
        row,
        &token_hashes,
        seed,
        row_probes,
        row_token_budget,
      );
      if profile_enabled {
        digest_elapsed += digest_start.elapsed();
      }
      let non_empty_count = Self::count_non_empty_buckets(row);
      non_empty_counts.push(saturating_u16(non_empty_count));
      source_token_counts.push(saturating_u16(row_source_token_count));
      if sparse_verify_perm > 0 {
        let sparse = non_empty_count < sparse_occupancy_threshold;
        sparse_verify_active.push(u8::from(sparse));
        let signature_start = sparse_verify_signatures.len();
        sparse_verify_signatures
          .resize(signature_start + sparse_verify_perm, u32::MAX);
        if sparse {
          Self::compute_sparse_verify_signature_into(
            &mut sparse_verify_signatures
              [signature_start..signature_start + sparse_verify_perm],
            &token_hashes,
            seed,
          );
        }
      }
      rows += 1;
      Ok(())
    })?;

    if profile_enabled {
      let total_elapsed = sketch_start.elapsed();
      let total = extract_elapsed + digest_elapsed;
      let matrix_elapsed = total_elapsed.checked_sub(total).unwrap_or_default();
      let token_budget = default_token_budget.unwrap_or_default();
      let adaptive_probes = usize::from(rho_adaptive_probes_enabled());
      let long_doc_threshold = rho_long_doc_threshold(num_perm);
      let non_empty_p50 = Self::percentile_u16(&non_empty_counts, 50, 100);
      let non_empty_p90 = Self::percentile_u16(&non_empty_counts, 90, 100);
      let non_empty_p99 = Self::percentile_u16(&non_empty_counts, 99, 100);
      let sparse_rows = non_empty_counts
        .iter()
        .filter(|&&count| usize::from(count) < sparse_occupancy_threshold)
        .count();
      let sparse_row_rate = if rows == 0 {
        0.0
      } else {
        sparse_rows as f64 / rows as f64
      };
      eprintln!(
        "rensa.profile rows={rows} num_perm={num_perm} kernel=rho mode=streaming-rho probes={probes} adaptive_probes={adaptive_probes} long_doc_threshold={long_doc_threshold} token_budget={token_budget} rho_non_empty_p50={non_empty_p50} rho_non_empty_p90={non_empty_p90} rho_non_empty_p99={non_empty_p99} rho_sparse_row_rate={sparse_row_rate:.6} token_hash_extract={:.6}s digest_kernel={:.6}s matrix_build={:.6}s",
        extract_elapsed.as_secs_f64(),
        digest_elapsed.as_secs_f64(),
        matrix_elapsed.as_secs_f64(),
      );
    }

    let rho_sidecar = RhoDigestSidecar {
      non_empty_counts,
      source_token_counts,
      sparse_occupancy_threshold,
      sparse_verify_perm,
      sparse_verify_signatures,
      sparse_verify_active,
    };

    Ok(RMinHashDigestMatrix {
      num_perm,
      rows,
      data: matrix_data,
      rho_sidecar: Some(rho_sidecar),
    })
  }

  fn build_rho_digest_matrix_from_token_hash_sets(
    token_hash_sets: &Bound<'_, PyAny>,
    num_perm: usize,
    seed: u64,
    probes: usize,
  ) -> PyResult<RMinHashDigestMatrix> {
    let capacity = Self::token_sets_capacity(token_hash_sets);
    let mut matrix_data = Vec::with_capacity(capacity.saturating_mul(num_perm));
    let mut rows = 0usize;
    let probes = parse_rho_probes(Some(probes));
    let default_token_budget = rho_token_budget(num_perm);
    let sparse_occupancy_threshold = rho_sparse_occupancy_threshold(num_perm);
    let sparse_verify_perm = if rho_sparse_verify_enabled() {
      rho_sparse_verify_perm(num_perm)
    } else {
      0
    };
    let mut token_hashes = Vec::new();
    let mut non_empty_counts = Vec::with_capacity(capacity);
    let mut source_token_counts = Vec::with_capacity(capacity);
    let mut sparse_verify_signatures =
      Vec::with_capacity(capacity.saturating_mul(sparse_verify_perm));
    let mut sparse_verify_active = Vec::with_capacity(capacity);

    Self::for_each_document(token_hash_sets, |document| {
      token_hashes.clear();
      extend_prehashed_token_values_from_document(
        &document,
        &mut token_hashes,
      )?;
      let row_start = matrix_data.len();
      matrix_data.resize(row_start + num_perm, EMPTY_BUCKET);
      let row = &mut matrix_data[row_start..row_start + num_perm];
      let row_token_budget = rho_adaptive_token_budget_for_row(
        Some(token_hashes.len()),
        default_token_budget,
      );
      let row_probes =
        effective_rho_probes(probes, token_hashes.len(), num_perm);
      Self::compute_rho_digest_from_token_hashes_into(
        row,
        &token_hashes,
        seed,
        row_probes,
        row_token_budget,
      );
      let non_empty_count = Self::count_non_empty_buckets(row);
      non_empty_counts.push(saturating_u16(non_empty_count));
      source_token_counts.push(saturating_u16(token_hashes.len()));
      if sparse_verify_perm > 0 {
        let sparse = non_empty_count < sparse_occupancy_threshold;
        sparse_verify_active.push(u8::from(sparse));
        let signature_start = sparse_verify_signatures.len();
        sparse_verify_signatures
          .resize(signature_start + sparse_verify_perm, u32::MAX);
        if sparse {
          Self::compute_sparse_verify_signature_into(
            &mut sparse_verify_signatures
              [signature_start..signature_start + sparse_verify_perm],
            &token_hashes,
            seed,
          );
        }
      }
      rows += 1;
      Ok(())
    })?;

    let rho_sidecar = RhoDigestSidecar {
      non_empty_counts,
      source_token_counts,
      sparse_occupancy_threshold,
      sparse_verify_perm,
      sparse_verify_signatures,
      sparse_verify_active,
    };

    Ok(RMinHashDigestMatrix {
      num_perm,
      rows,
      data: matrix_data,
      rho_sidecar: Some(rho_sidecar),
    })
  }

  fn build_rho_digest_matrix_from_flat_token_hashes(
    token_hashes: &[u64],
    row_offsets: &[usize],
    num_perm: usize,
    seed: u64,
    probes: usize,
  ) -> PyResult<RMinHashDigestMatrix> {
    Self::validate_flat_row_offsets(row_offsets, token_hashes.len())?;
    let rows = row_offsets.len().saturating_sub(1);
    let mut matrix_data = vec![EMPTY_BUCKET; rows.saturating_mul(num_perm)];
    let probes = parse_rho_probes(Some(probes));
    let default_token_budget = rho_token_budget(num_perm);
    let sparse_occupancy_threshold = rho_sparse_occupancy_threshold(num_perm);
    let sparse_verify_perm = if rho_sparse_verify_enabled() {
      rho_sparse_verify_perm(num_perm)
    } else {
      0
    };
    let mut non_empty_counts = Vec::with_capacity(rows);
    let mut source_token_counts = Vec::with_capacity(rows);
    let mut sparse_verify_signatures =
      Vec::with_capacity(rows.saturating_mul(sparse_verify_perm));
    let mut sparse_verify_active = Vec::with_capacity(rows);
    for (row_index, row) in matrix_data.chunks_exact_mut(num_perm).enumerate() {
      let start = row_offsets[row_index];
      let end = row_offsets[row_index + 1];
      let token_count = end.saturating_sub(start);
      let row_token_budget = rho_adaptive_token_budget_for_row(
        Some(token_count),
        default_token_budget,
      );
      let row_probes = effective_rho_probes(probes, token_count, num_perm);
      Self::compute_rho_digest_from_token_hashes_into(
        row,
        &token_hashes[start..end],
        seed,
        row_probes,
        row_token_budget,
      );
      let non_empty_count = Self::count_non_empty_buckets(row);
      non_empty_counts.push(saturating_u16(non_empty_count));
      source_token_counts.push(saturating_u16(token_count));
      if sparse_verify_perm > 0 {
        let sparse = non_empty_count < sparse_occupancy_threshold;
        sparse_verify_active.push(u8::from(sparse));
        let signature_start = sparse_verify_signatures.len();
        sparse_verify_signatures
          .resize(signature_start + sparse_verify_perm, u32::MAX);
        if sparse {
          Self::compute_sparse_verify_signature_into(
            &mut sparse_verify_signatures
              [signature_start..signature_start + sparse_verify_perm],
            &token_hashes[start..end],
            seed,
          );
        }
      }
    }
    let rho_sidecar = RhoDigestSidecar {
      non_empty_counts,
      source_token_counts,
      sparse_occupancy_threshold,
      sparse_verify_perm,
      sparse_verify_signatures,
      sparse_verify_active,
    };
    Ok(RMinHashDigestMatrix {
      num_perm,
      rows,
      data: matrix_data,
      rho_sidecar: Some(rho_sidecar),
    })
  }

  fn build_cached_digest_for_token(
    num_perm: usize,
    permutations: &[(u64, u64)],
    permutations_soa: &PermutationSoA,
    token_hash: u64,
  ) -> Box<[u32]> {
    let mut digest_row = vec![u32::MAX; num_perm];
    Self::apply_token_hashes_to_values(
      &mut digest_row,
      permutations,
      permutations_soa,
      std::slice::from_ref(&token_hash),
    );
    digest_row.into_boxed_slice()
  }

  fn apply_token_hashes_with_cache(
    digest_row: &mut [u32],
    token_hashes: &[u64],
    permutations: &[(u64, u64)],
    permutations_soa: &PermutationSoA,
    cache: &mut AdaptivePermutationCache,
    miss_hashes: &mut Vec<u64>,
  ) -> usize {
    digest_row.fill(u32::MAX);
    if cache.max_hashes == 0 {
      Self::apply_token_hashes_to_values(
        digest_row,
        permutations,
        permutations_soa,
        token_hashes,
      );
      return 0;
    }

    miss_hashes.clear();
    let mut cache_hits = 0usize;
    for &token_hash in token_hashes {
      if let Some(cached_digest) = cache.digests.get(&token_hash) {
        for (value, &cached_value) in
          digest_row.iter_mut().zip(cached_digest.iter())
        {
          *value = (*value).min(cached_value);
        }
        cache_hits += 1;
      } else {
        if cache.digests.len() < cache.max_hashes {
          if let Some(count) = cache.seen_counts.get_mut(&token_hash) {
            *count = count.saturating_add(1);
            if usize::from(*count) >= cache.min_frequency {
              let cached_digest = Self::build_cached_digest_for_token(
                digest_row.len(),
                permutations,
                permutations_soa,
                token_hash,
              );
              cache.digests.insert(token_hash, cached_digest);
              cache.seen_counts.remove(&token_hash);
            }
          } else if cache.seen_counts.len() < cache.max_tracked_seen_hashes {
            cache.seen_counts.insert(token_hash, 1);
          }
        }
        miss_hashes.push(token_hash);
      }
    }
    if !miss_hashes.is_empty() {
      Self::apply_token_hashes_to_values(
        digest_row,
        permutations,
        permutations_soa,
        miss_hashes,
      );
    }
    cache_hits
  }

  fn compute_digest_from_token_hashes_into(
    digest_row: &mut [u32],
    token_hashes: &[u64],
    permutations: &[(u64, u64)],
    permutations_soa: &PermutationSoA,
  ) {
    digest_row.fill(u32::MAX);
    Self::apply_token_hashes_to_values(
      digest_row,
      permutations,
      permutations_soa,
      token_hashes,
    );
  }

  fn compute_digest_chunk(
    job: DigestChunkJob,
    num_perm: usize,
    permutations: &[(u64, u64)],
    permutations_soa: &PermutationSoA,
    config: DigestBuildConfig,
    permutation_cache: Option<&mut AdaptivePermutationCache>,
  ) -> DigestChunkResult {
    let row_count = job.ranges.len();
    let data = unsafe {
      std::slice::from_raw_parts_mut(
        job.output_ptr_addr as *mut u32,
        job.output_len,
      )
    };
    data.fill(u32::MAX);
    let digest_start = Instant::now();
    let use_parallel = row_count >= config.doc_par_batch_size
      && rayon::current_num_threads() > 1;
    let mut cache_hits = 0usize;
    let mut cache_entries = 0usize;
    if use_parallel {
      data
        .par_chunks_mut(num_perm)
        .zip(job.ranges.par_iter())
        .for_each(|(row, &(start, end))| {
          Self::compute_digest_from_token_hashes_into(
            row,
            &job.flat[start..end],
            permutations,
            permutations_soa,
          );
        });
    } else if let Some(cache) = permutation_cache {
      let start_entries = cache.entry_len();
      let mut misses = Vec::new();
      for (row, &(start, end)) in
        data.chunks_exact_mut(num_perm).zip(job.ranges.iter())
      {
        cache_hits += Self::apply_token_hashes_with_cache(
          row,
          &job.flat[start..end],
          permutations,
          permutations_soa,
          cache,
          &mut misses,
        );
      }
      cache_entries = cache.entry_len().saturating_sub(start_entries);
    } else {
      for (row, &(start, end)) in
        data.chunks_exact_mut(num_perm).zip(job.ranges.iter())
      {
        Self::compute_digest_from_token_hashes_into(
          row,
          &job.flat[start..end],
          permutations,
          permutations_soa,
        );
      }
    }
    DigestChunkResult {
      digest_kernel: digest_start.elapsed(),
      perm_cache_entries: cache_entries,
      perm_cache_hits: cache_hits,
    }
  }

  fn flush_digest_chunk(
    token_hashes_chunk: &mut Vec<u64>,
    token_hash_ranges: &mut Vec<(usize, usize)>,
    matrix_data: &mut Vec<u32>,
    context: DigestComputeContext<'_>,
    mut profile: Option<&mut DigestBuildProfile>,
  ) {
    if token_hash_ranges.is_empty() {
      return;
    }

    let flat = std::mem::take(token_hashes_chunk);
    let flat_capacity = flat.capacity();
    let ranges = std::mem::take(token_hash_ranges);
    let ranges_capacity = ranges.capacity();
    let rows = ranges.len();
    let matrix_start = matrix_data.len();
    let matrix_expand_start = Instant::now();
    matrix_data.resize(matrix_start + rows * context.num_perm, u32::MAX);
    if let Some(active_profile) = &mut profile {
      active_profile.matrix_build += matrix_expand_start.elapsed();
    }

    let job = DigestChunkJob {
      flat,
      ranges,
      output_ptr_addr: matrix_data[matrix_start..].as_mut_ptr() as usize,
      output_len: rows * context.num_perm,
    };
    let result = Python::attach(|py| {
      py.detach(|| {
        Self::compute_digest_chunk(
          job,
          context.num_perm,
          context.permutations,
          context.permutations_soa,
          context.config,
          None,
        )
      })
    });
    if let Some(active_profile) = &mut profile {
      active_profile.digest_kernel += result.digest_kernel;
      active_profile.chunk_jobs += 1;
      active_profile.perm_cache_entries += result.perm_cache_entries;
      active_profile.perm_cache_hits += result.perm_cache_hits;
    }

    *token_hashes_chunk = Vec::with_capacity(flat_capacity);
    *token_hash_ranges = Vec::with_capacity(ranges_capacity);
  }

  fn build_token_hash_rows(
    token_sets: &Bound<'_, PyAny>,
    document_hasher: fn(&Bound<'_, PyAny>, &mut Vec<u64>) -> PyResult<()>,
  ) -> PyResult<Vec<Vec<u64>>> {
    let capacity = Self::token_sets_capacity(token_sets);
    let mut rows = Vec::with_capacity(capacity);

    Self::for_each_document(token_sets, |document| {
      let mut hashes = Vec::new();
      document_hasher(&document, &mut hashes)?;
      rows.push(hashes);
      Ok(())
    })?;

    Ok(rows)
  }

  fn parse_flat_token_hashes(
    token_hashes: &Bound<'_, PyAny>,
  ) -> PyResult<Vec<u64>> {
    let mut values = Vec::new();
    extend_prehashed_token_values_from_document(token_hashes, &mut values)?;
    Ok(values)
  }

  fn extract_row_offset(item: &Bound<'_, PyAny>) -> PyResult<usize> {
    let value = item
      .extract::<u64>()
      .map_err(|_| PyValueError::new_err(FLAT_ROW_OFFSET_TYPE_ERROR))?;
    usize::try_from(value)
      .map_err(|_| PyValueError::new_err(FLAT_ROW_OFFSET_TYPE_ERROR))
  }

  fn extend_usize_values_from_buffer<T>(
    values: &Bound<'_, PyAny>,
    output: &mut Vec<usize>,
  ) -> PyResult<bool>
  where
    T: Element + Copy,
    usize: TryFrom<T>,
  {
    let Ok(buffer) = PyBuffer::<T>::get(values) else {
      return Ok(false);
    };
    if !buffer.is_c_contiguous() {
      return Err(PyValueError::new_err(FLAT_ROW_OFFSET_TYPE_ERROR));
    }
    let slice = unsafe {
      std::slice::from_raw_parts(
        buffer.buf_ptr().cast::<T>(),
        buffer.item_count(),
      )
    };
    output.reserve(slice.len());
    for &value in slice {
      output.push(
        usize::try_from(value)
          .map_err(|_| PyValueError::new_err(FLAT_ROW_OFFSET_TYPE_ERROR))?,
      );
    }
    Ok(true)
  }

  fn parse_row_offsets(row_offsets: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    let mut offsets = Vec::new();
    if Self::extend_usize_values_from_buffer::<usize>(
      row_offsets,
      &mut offsets,
    )? || Self::extend_usize_values_from_buffer::<u64>(
      row_offsets,
      &mut offsets,
    )? || Self::extend_usize_values_from_buffer::<u32>(
      row_offsets,
      &mut offsets,
    )? || Self::extend_usize_values_from_buffer::<u16>(
      row_offsets,
      &mut offsets,
    )? || Self::extend_usize_values_from_buffer::<u8>(
      row_offsets,
      &mut offsets,
    )? {
      return Ok(offsets);
    }

    if let Ok(py_list) = row_offsets.cast::<PyList>() {
      offsets.reserve(py_list.len());
      for item in py_list.iter() {
        offsets.push(Self::extract_row_offset(&item)?);
      }
      return Ok(offsets);
    }

    if let Ok(py_tuple) = row_offsets.cast::<PyTuple>() {
      offsets.reserve(py_tuple.len());
      for item in py_tuple.iter() {
        offsets.push(Self::extract_row_offset(&item)?);
      }
      return Ok(offsets);
    }

    let iterator = PyIterator::from_object(row_offsets)?;
    for item in iterator {
      offsets.push(Self::extract_row_offset(&item?)?);
    }
    Ok(offsets)
  }

  fn validate_flat_row_offsets(
    row_offsets: &[usize],
    token_hash_count: usize,
  ) -> PyResult<()> {
    if row_offsets.is_empty() {
      return Err(PyValueError::new_err(FLAT_ROW_OFFSETS_ERROR));
    }
    if row_offsets[0] != 0 {
      return Err(PyValueError::new_err(FLAT_ROW_OFFSETS_ERROR));
    }
    if row_offsets[row_offsets.len() - 1] != token_hash_count {
      return Err(PyValueError::new_err(FLAT_ROW_OFFSETS_ERROR));
    }
    for window in row_offsets.windows(2) {
      if window[0] > window[1] {
        return Err(PyValueError::new_err(FLAT_ROW_OFFSETS_ERROR));
      }
    }
    Ok(())
  }

  fn build_digest_matrix_from_flat_token_hashes(
    token_hashes: &[u64],
    row_offsets: &[usize],
    num_perm: usize,
    seed: u64,
  ) -> PyResult<RMinHashDigestMatrix> {
    Self::validate_flat_row_offsets(row_offsets, token_hashes.len())?;
    let rows = row_offsets.len().saturating_sub(1);
    let permutations = Self::build_permutations(num_perm, seed);
    let permutations_soa = PermutationSoA::from_permutations(&permutations);
    let config = DigestBuildConfig::from_env();
    let matrix_start = Instant::now();
    let mut matrix_data = vec![u32::MAX; rows.saturating_mul(num_perm)];
    let mut profile = Self::profile_enabled().then(DigestBuildProfile::default);
    if let Some(active_profile) = profile.as_mut() {
      active_profile.pipeline_mode = "flat";
      active_profile.doc_chunk_size = config.doc_chunk_size;
      active_profile.doc_par_batch_size = config.doc_par_batch_size;
      active_profile.pipeline_queue_cap = config.pipeline_queue_cap;
      active_profile.matrix_build += matrix_start.elapsed();
    }

    let digest_start = Instant::now();
    Python::attach(|py| {
      py.detach(|| {
        let use_parallel =
          rows >= config.doc_par_batch_size && rayon::current_num_threads() > 1;
        if use_parallel {
          matrix_data.par_chunks_mut(num_perm).enumerate().for_each(
            |(row_index, row)| {
              let start = row_offsets[row_index];
              let end = row_offsets[row_index + 1];
              Self::compute_digest_from_token_hashes_into(
                row,
                &token_hashes[start..end],
                &permutations,
                &permutations_soa,
              );
            },
          );
        } else {
          let mut permutation_cache = if config.max_perm_cache_hashes > 0 {
            Some(AdaptivePermutationCache::new(
              config.perm_cache_min_frequency,
              config.max_perm_cache_hashes,
            ))
          } else {
            None
          };
          let mut misses = Vec::new();
          for (row_index, row) in
            matrix_data.chunks_exact_mut(num_perm).enumerate()
          {
            let start = row_offsets[row_index];
            let end = row_offsets[row_index + 1];
            if let Some(cache) = permutation_cache.as_mut() {
              Self::apply_token_hashes_with_cache(
                row,
                &token_hashes[start..end],
                &permutations,
                &permutations_soa,
                cache,
                &mut misses,
              );
            } else {
              Self::compute_digest_from_token_hashes_into(
                row,
                &token_hashes[start..end],
                &permutations,
                &permutations_soa,
              );
            }
          }
        }
      });
    });
    if let Some(active_profile) = profile.as_mut() {
      active_profile.digest_kernel += digest_start.elapsed();
      if rows > 0 {
        active_profile.chunk_jobs = 1;
      }
    }

    if let Some(active_profile) = profile {
      Self::emit_profile(&active_profile, rows, num_perm);
    }

    Ok(RMinHashDigestMatrix {
      num_perm,
      rows,
      data: matrix_data,
      rho_sidecar: None,
    })
  }

  #[allow(clippy::too_many_lines)]
  fn build_digest_matrix_data_with_known_rows<'py, I>(
    rows: usize,
    documents: I,
    num_perm: usize,
    permutations: &[(u64, u64)],
    permutations_soa: &PermutationSoA,
    document_hasher: fn(&Bound<'_, PyAny>, &mut Vec<u64>) -> PyResult<()>,
  ) -> PyResult<RMinHashDigestMatrix>
  where
    I: Iterator<Item = Bound<'py, PyAny>>,
  {
    let config = DigestBuildConfig::from_env();
    let matrix_expand_start = Instant::now();
    let mut matrix_data = vec![u32::MAX; rows.saturating_mul(num_perm)];
    let mut token_hashes_chunk = Vec::new();
    let mut token_hash_ranges = Vec::with_capacity(config.doc_chunk_size);
    let mut chunk_row_start = 0usize;
    let mut jobs_sent = 0usize;
    let mut jobs_received = 0usize;
    let mut extraction_error: Option<PyErr> = None;
    let mut profile = Self::profile_enabled().then(DigestBuildProfile::default);
    if let Some(active_profile) = profile.as_mut() {
      active_profile.pipeline_mode = "pipelined";
      active_profile.matrix_build += matrix_expand_start.elapsed();
      active_profile.doc_chunk_size = config.doc_chunk_size;
      active_profile.doc_par_batch_size = config.doc_par_batch_size;
      active_profile.pipeline_queue_cap = config.pipeline_queue_cap;
    }

    if config.pipeline_queue_cap == 0 {
      if let Some(active_profile) = profile.as_mut() {
        active_profile.pipeline_mode = "sequential";
      }
      let mut permutation_cache = if config.max_perm_cache_hashes > 0 {
        Some(AdaptivePermutationCache::new(
          config.perm_cache_min_frequency,
          config.max_perm_cache_hashes,
        ))
      } else {
        None
      };
      for document in documents {
        let extract_start = Instant::now();
        let start = token_hashes_chunk.len();
        document_hasher(&document, &mut token_hashes_chunk)?;
        let end = token_hashes_chunk.len();
        token_hash_ranges.push((start, end));
        if let Some(active_profile) = profile.as_mut() {
          active_profile.token_hash_extract += extract_start.elapsed();
        }

        if token_hash_ranges.len() == config.doc_chunk_size {
          let job = DigestChunkJob {
            flat: std::mem::take(&mut token_hashes_chunk),
            ranges: std::mem::take(&mut token_hash_ranges),
            output_ptr_addr: unsafe {
              matrix_data.as_mut_ptr().add(chunk_row_start * num_perm) as usize
            },
            output_len: config.doc_chunk_size * num_perm,
          };
          let result = Self::compute_digest_chunk(
            job,
            num_perm,
            permutations,
            permutations_soa,
            config,
            permutation_cache.as_mut(),
          );
          if let Some(active_profile) = profile.as_mut() {
            active_profile.digest_kernel += result.digest_kernel;
            active_profile.chunk_jobs += 1;
            active_profile.perm_cache_entries += result.perm_cache_entries;
            active_profile.perm_cache_hits += result.perm_cache_hits;
          }
          chunk_row_start += config.doc_chunk_size;
        }
      }

      if !token_hash_ranges.is_empty() {
        let output_rows = rows - chunk_row_start;
        let job = DigestChunkJob {
          flat: std::mem::take(&mut token_hashes_chunk),
          ranges: std::mem::take(&mut token_hash_ranges),
          output_ptr_addr: unsafe {
            matrix_data.as_mut_ptr().add(chunk_row_start * num_perm) as usize
          },
          output_len: output_rows * num_perm,
        };
        let result = Self::compute_digest_chunk(
          job,
          num_perm,
          permutations,
          permutations_soa,
          config,
          permutation_cache.as_mut(),
        );
        if let Some(active_profile) = profile.as_mut() {
          active_profile.digest_kernel += result.digest_kernel;
          active_profile.chunk_jobs += 1;
          active_profile.perm_cache_entries += result.perm_cache_entries;
          active_profile.perm_cache_hits += result.perm_cache_hits;
        }
      }

      if let Some(active_profile) = profile {
        Self::emit_profile(&active_profile, rows, num_perm);
      }

      return Ok(RMinHashDigestMatrix {
        num_perm,
        rows,
        data: matrix_data,
        rho_sidecar: None,
      });
    }

    let (job_tx, job_rx) =
      mpsc::sync_channel::<DigestChunkJob>(config.pipeline_queue_cap);
    let (result_tx, result_rx) =
      mpsc::sync_channel::<DigestChunkResult>(config.pipeline_queue_cap);
    let permutations_owned = permutations.to_vec();
    let permutations_soa_owned = permutations_soa.clone();
    let worker_config = config;
    let worker = thread::Builder::new()
      .name(String::from("rensa-digest-worker"))
      .spawn(move || {
        let mut permutation_cache = if worker_config.max_perm_cache_hashes > 0 {
          Some(AdaptivePermutationCache::new(
            worker_config.perm_cache_min_frequency,
            worker_config.max_perm_cache_hashes,
          ))
        } else {
          None
        };
        while let Ok(job) = job_rx.recv() {
          let result = Self::compute_digest_chunk(
            job,
            num_perm,
            &permutations_owned,
            &permutations_soa_owned,
            worker_config,
            permutation_cache.as_mut(),
          );
          if result_tx.send(result).is_err() {
            break;
          }
        }
      })
      .map_err(|err| {
        PyValueError::new_err(format!("failed to spawn digest worker: {err}"))
      })?;

    for document in documents {
      let extract_start = Instant::now();
      let start = token_hashes_chunk.len();
      if let Err(err) = document_hasher(&document, &mut token_hashes_chunk) {
        extraction_error = Some(err);
        break;
      }
      let end = token_hashes_chunk.len();
      token_hash_ranges.push((start, end));
      if let Some(active_profile) = profile.as_mut() {
        active_profile.token_hash_extract += extract_start.elapsed();
      }

      if token_hash_ranges.len() == config.doc_chunk_size {
        let mut job = DigestChunkJob {
          flat: std::mem::take(&mut token_hashes_chunk),
          ranges: std::mem::take(&mut token_hash_ranges),
          output_ptr_addr: unsafe {
            matrix_data.as_mut_ptr().add(chunk_row_start * num_perm) as usize
          },
          output_len: config.doc_chunk_size * num_perm,
        };
        loop {
          let send_start = Instant::now();
          match job_tx.try_send(job) {
            Ok(()) => {
              if let Some(active_profile) = profile.as_mut() {
                active_profile.pipeline_send_wait += send_start.elapsed();
                active_profile.chunk_jobs += 1;
              }
              jobs_sent += 1;
              break;
            }
            Err(TrySendError::Full(returned_job)) => {
              if let Some(active_profile) = profile.as_mut() {
                active_profile.pipeline_send_wait += send_start.elapsed();
              }
              job = returned_job;
              let recv_start = Instant::now();
              let result = result_rx.recv().map_err(|_| {
                PyValueError::new_err(
                  "digest worker exited before all chunks completed",
                )
              })?;
              if let Some(active_profile) = profile.as_mut() {
                active_profile.pipeline_recv_wait += recv_start.elapsed();
                active_profile.digest_kernel += result.digest_kernel;
                active_profile.perm_cache_entries += result.perm_cache_entries;
                active_profile.perm_cache_hits += result.perm_cache_hits;
              }
              jobs_received += 1;
            }
            Err(TrySendError::Disconnected(_)) => {
              return Err(PyValueError::new_err(
                "digest worker channel closed unexpectedly",
              ));
            }
          }
        }
        chunk_row_start += config.doc_chunk_size;
      }
    }

    if extraction_error.is_none() && !token_hash_ranges.is_empty() {
      let mut job = DigestChunkJob {
        flat: std::mem::take(&mut token_hashes_chunk),
        ranges: std::mem::take(&mut token_hash_ranges),
        output_ptr_addr: unsafe {
          matrix_data.as_mut_ptr().add(chunk_row_start * num_perm) as usize
        },
        output_len: (rows - chunk_row_start) * num_perm,
      };
      loop {
        let send_start = Instant::now();
        match job_tx.try_send(job) {
          Ok(()) => {
            if let Some(active_profile) = profile.as_mut() {
              active_profile.pipeline_send_wait += send_start.elapsed();
              active_profile.chunk_jobs += 1;
            }
            jobs_sent += 1;
            break;
          }
          Err(TrySendError::Full(returned_job)) => {
            if let Some(active_profile) = profile.as_mut() {
              active_profile.pipeline_send_wait += send_start.elapsed();
            }
            job = returned_job;
            let recv_start = Instant::now();
            let result = result_rx.recv().map_err(|_| {
              PyValueError::new_err(
                "digest worker exited before all chunks completed",
              )
            })?;
            if let Some(active_profile) = profile.as_mut() {
              active_profile.pipeline_recv_wait += recv_start.elapsed();
              active_profile.digest_kernel += result.digest_kernel;
              active_profile.perm_cache_entries += result.perm_cache_entries;
              active_profile.perm_cache_hits += result.perm_cache_hits;
            }
            jobs_received += 1;
          }
          Err(TrySendError::Disconnected(_)) => {
            return Err(PyValueError::new_err(
              "digest worker channel closed unexpectedly",
            ));
          }
        }
      }
    }

    drop(job_tx);

    while jobs_received < jobs_sent {
      let recv_start = Instant::now();
      let result = result_rx.recv().map_err(|_| {
        PyValueError::new_err(
          "digest worker exited before all chunks completed",
        )
      })?;
      if let Some(active_profile) = profile.as_mut() {
        active_profile.pipeline_recv_wait += recv_start.elapsed();
        active_profile.digest_kernel += result.digest_kernel;
        active_profile.perm_cache_entries += result.perm_cache_entries;
        active_profile.perm_cache_hits += result.perm_cache_hits;
      }
      jobs_received += 1;
    }

    if worker.join().is_err() {
      return Err(PyValueError::new_err("digest worker thread panicked"));
    }

    if let Some(err) = extraction_error {
      return Err(err);
    }

    if let Some(active_profile) = profile {
      Self::emit_profile(&active_profile, rows, num_perm);
    }

    Ok(RMinHashDigestMatrix {
      num_perm,
      rows,
      data: matrix_data,
      rho_sidecar: None,
    })
  }

  fn build_digest_matrix_data(
    token_sets: &Bound<'_, PyAny>,
    num_perm: usize,
    permutations: &[(u64, u64)],
    permutations_soa: &PermutationSoA,
    document_hasher: fn(&Bound<'_, PyAny>, &mut Vec<u64>) -> PyResult<()>,
  ) -> PyResult<RMinHashDigestMatrix> {
    let config = DigestBuildConfig::from_env();
    if let Ok(py_list) = token_sets.cast::<PyList>() {
      return Self::build_digest_matrix_data_with_known_rows(
        py_list.len(),
        py_list.iter(),
        num_perm,
        permutations,
        permutations_soa,
        document_hasher,
      );
    }
    if let Ok(py_tuple) = token_sets.cast::<PyTuple>() {
      return Self::build_digest_matrix_data_with_known_rows(
        py_tuple.len(),
        py_tuple.iter(),
        num_perm,
        permutations,
        permutations_soa,
        document_hasher,
      );
    }

    let capacity = Self::token_sets_capacity(token_sets);
    let mut matrix_data = Vec::with_capacity(capacity.saturating_mul(num_perm));
    let mut token_hashes_chunk = Vec::new();
    let mut token_hash_ranges = Vec::with_capacity(config.doc_chunk_size);
    let context = DigestComputeContext {
      num_perm,
      permutations,
      permutations_soa,
      config,
    };
    let mut profile = Self::profile_enabled().then(DigestBuildProfile::default);
    if let Some(active_profile) = profile.as_mut() {
      active_profile.doc_chunk_size = config.doc_chunk_size;
      active_profile.doc_par_batch_size = config.doc_par_batch_size;
      active_profile.pipeline_queue_cap = config.pipeline_queue_cap;
    }
    Self::for_each_document(token_sets, |document| {
      let extract_start = Instant::now();
      let start = token_hashes_chunk.len();
      document_hasher(&document, &mut token_hashes_chunk)?;
      let end = token_hashes_chunk.len();
      token_hash_ranges.push((start, end));

      if let Some(active_profile) = profile.as_mut() {
        active_profile.token_hash_extract += extract_start.elapsed();
      }

      if token_hash_ranges.len() == config.doc_chunk_size {
        Self::flush_digest_chunk(
          &mut token_hashes_chunk,
          &mut token_hash_ranges,
          &mut matrix_data,
          context,
          profile.as_mut(),
        );
      }
      Ok(())
    })?;

    Self::flush_digest_chunk(
      &mut token_hashes_chunk,
      &mut token_hash_ranges,
      &mut matrix_data,
      context,
      profile.as_mut(),
    );

    let rows = if num_perm == 0 {
      0
    } else {
      matrix_data.len() / num_perm
    };
    if let Some(active_profile) = profile {
      Self::emit_profile(&active_profile, rows, num_perm);
    }

    Ok(RMinHashDigestMatrix {
      num_perm,
      rows,
      data: matrix_data,
      rho_sidecar: None,
    })
  }

  fn build_digest_matrix_from_token_sets(
    token_sets: &Bound<'_, PyAny>,
    num_perm: usize,
    seed: u64,
    cache_domain: &str,
    document_hasher: fn(&Bound<'_, PyAny>, &mut Vec<u64>) -> PyResult<()>,
  ) -> PyResult<RMinHashDigestMatrix> {
    let cache_key = digest_cache_key_env();
    if let Some(cache_key_value) = cache_key.as_deref() {
      if let Some(matrix) = Self::try_load_cached_digest_matrix(
        cache_key_value,
        cache_domain,
        num_perm,
        seed,
      ) {
        return Ok(matrix);
      }
    }
    let permutations = Self::build_permutations(num_perm, seed);
    let permutations_soa = PermutationSoA::from_permutations(&permutations);
    let matrix = Self::build_digest_matrix_data(
      token_sets,
      num_perm,
      &permutations,
      &permutations_soa,
      document_hasher,
    )?;
    if let Some(cache_key_value) = cache_key.as_deref() {
      Self::store_cached_digest_matrix(
        cache_key_value,
        cache_domain,
        num_perm,
        seed,
        &matrix,
      );
    }
    Ok(matrix)
  }

  fn digest_rows_from_matrix(matrix: &RMinHashDigestMatrix) -> Vec<Vec<u32>> {
    matrix
      .data
      .chunks_exact(matrix.num_perm)
      .map(std::borrow::ToOwned::to_owned)
      .collect()
  }

  fn from_matrix(matrix: RMinHashDigestMatrix, seed: u64) -> Vec<Self> {
    let mut minhashes = Vec::with_capacity(matrix.rows);
    for hash_values in matrix.data.chunks_exact(matrix.num_perm) {
      minhashes.push(Self {
        num_perm: matrix.num_perm,
        seed,
        hash_values: hash_values.to_vec(),
        permutations: Vec::new(),
        permutations_soa: PermutationSoA::default(),
      });
    }
    minhashes
  }

  pub(crate) fn new_compact(num_perm: usize, seed: u64) -> PyResult<Self> {
    Self::validate_num_perm(num_perm)?;
    Ok(Self {
      num_perm,
      seed,
      hash_values: vec![u32::MAX; num_perm],
      permutations: Vec::new(),
      permutations_soa: PermutationSoA::default(),
    })
  }

  pub(crate) fn reset_from_token_hashes_with_permutations(
    &mut self,
    token_hashes: &[u64],
    permutations: &[(u64, u64)],
    permutations_soa: &PermutationSoA,
  ) {
    self.hash_values.fill(u32::MAX);
    Self::apply_token_hashes_to_values(
      &mut self.hash_values,
      permutations,
      permutations_soa,
      token_hashes,
    );
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

#[pymethods]
impl RMinHash {
  /// Creates a new `RMinHash` instance.
  ///
  /// # Arguments
  ///
  /// * `num_perm` - The number of permutations to use in the `MinHash` algorithm.
  /// * `seed` - A seed value for the random number generator.
  ///
  /// # Errors
  ///
  /// Returns an error when `num_perm` is zero.
  #[new]
  pub fn new(num_perm: usize, seed: u64) -> PyResult<Self> {
    Self::validate_num_perm(num_perm)?;
    let permutations = Self::build_permutations(num_perm, seed);
    let permutations_soa = PermutationSoA::from_permutations(&permutations);

    Ok(Self {
      num_perm,
      seed,
      hash_values: vec![u32::MAX; num_perm],
      permutations,
      permutations_soa,
    })
  }

  /// Creates `RMinHash` objects from an iterable of token iterables.
  ///
  /// # Errors
  ///
  /// Returns an error if `num_perm` is zero, the outer input is not iterable,
  /// or any token has an unsupported type.
  #[classmethod]
  #[pyo3(signature = (token_sets, num_perm, seed))]
  pub fn from_token_sets(
    _cls: &Bound<'_, PyType>,
    token_sets: Bound<'_, PyAny>,
    num_perm: usize,
    seed: u64,
  ) -> PyResult<Vec<Self>> {
    Self::validate_num_perm(num_perm)?;
    let matrix = Self::build_digest_matrix_from_token_sets(
      &token_sets,
      num_perm,
      seed,
      "tokens",
      extend_token_hashes_from_document,
    )?;
    Ok(Self::from_matrix(matrix, seed))
  }

  /// Computes `RMinHash` digests from an iterable of token iterables.
  ///
  /// # Errors
  ///
  /// Returns an error if `num_perm` is zero, the outer input is not iterable,
  /// or any token has an unsupported type.
  #[classmethod]
  #[pyo3(signature = (token_sets, num_perm, seed))]
  pub fn digests_from_token_sets(
    _cls: &Bound<'_, PyType>,
    token_sets: Bound<'_, PyAny>,
    num_perm: usize,
    seed: u64,
  ) -> PyResult<Vec<Vec<u32>>> {
    Self::validate_num_perm(num_perm)?;
    let matrix = Self::build_digest_matrix_from_token_sets(
      &token_sets,
      num_perm,
      seed,
      "tokens",
      extend_token_hashes_from_document,
    )?;
    Ok(Self::digest_rows_from_matrix(&matrix))
  }

  /// Hashes token documents into `u64` token hashes.
  ///
  /// This method is intended for high-throughput workflows where token hashing
  /// is reused across multiple runs.
  ///
  /// # Errors
  ///
  /// Returns an error if the outer input is not iterable or any token has an
  /// unsupported type.
  #[classmethod]
  #[pyo3(signature = (token_sets))]
  pub fn hash_token_sets(
    _cls: &Bound<'_, PyType>,
    token_sets: Bound<'_, PyAny>,
  ) -> PyResult<Vec<Vec<u64>>> {
    Self::build_token_hash_rows(&token_sets, extend_token_hashes_from_document)
  }

  /// Computes `RMinHash` digests in a compact row-major matrix.
  ///
  /// This avoids building one Python object per document and is intended for
  /// high-throughput bulk workflows.
  ///
  /// # Errors
  ///
  /// Returns an error if `num_perm` is zero, the outer input is not iterable,
  /// or any token has an unsupported type.
  #[classmethod]
  #[pyo3(signature = (token_sets, num_perm, seed))]
  pub fn digest_matrix_from_token_sets(
    _cls: &Bound<'_, PyType>,
    token_sets: Bound<'_, PyAny>,
    num_perm: usize,
    seed: u64,
  ) -> PyResult<RMinHashDigestMatrix> {
    Self::validate_num_perm(num_perm)?;
    Self::build_digest_matrix_from_token_sets(
      &token_sets,
      num_perm,
      seed,
      "tokens",
      extend_token_hashes_from_document,
    )
  }

  /// Computes a compact row-major matrix using the Rho multi-probe sketch.
  ///
  /// This is an O(tokens) alternative sketching strategy that maps each token
  /// hash into one or more permutation buckets, followed by deterministic
  /// densification for empty buckets.
  ///
  /// # Errors
  ///
  /// Returns an error if `num_perm` is zero, the outer input is not iterable,
  /// or any token has an unsupported type.
  #[classmethod]
  #[pyo3(signature = (token_sets, num_perm, seed, probes=DEFAULT_RHO_PROBES))]
  pub fn digest_matrix_from_token_sets_rho(
    _cls: &Bound<'_, PyType>,
    token_sets: Bound<'_, PyAny>,
    num_perm: usize,
    seed: u64,
    probes: usize,
  ) -> PyResult<RMinHashDigestMatrix> {
    Self::validate_num_perm(num_perm)?;
    if let Some(matrix) =
      Self::try_build_rho_digest_matrix_from_token_sets_parallel(
        &token_sets,
        num_perm,
        seed,
        probes,
      )?
    {
      return Ok(matrix);
    }
    Self::build_rho_digest_matrix_from_token_sets_streaming(
      &token_sets,
      num_perm,
      seed,
      probes,
    )
  }

  /// Computes `RMinHash` digests in a compact row-major matrix from pre-hashed tokens.
  ///
  /// # Errors
  ///
  /// Returns an error if `num_perm` is zero, the outer input is not iterable,
  /// or any token hash is not an unsigned 64-bit integer.
  #[classmethod]
  #[pyo3(signature = (token_hash_sets, num_perm, seed))]
  pub fn digest_matrix_from_token_hash_sets(
    _cls: &Bound<'_, PyType>,
    token_hash_sets: Bound<'_, PyAny>,
    num_perm: usize,
    seed: u64,
  ) -> PyResult<RMinHashDigestMatrix> {
    Self::validate_num_perm(num_perm)?;
    Self::build_digest_matrix_from_token_sets(
      &token_hash_sets,
      num_perm,
      seed,
      "token_hashes",
      extend_prehashed_token_values_from_document,
    )
  }

  /// Computes a compact row-major matrix from pre-hashed tokens using the Rho sketch.
  ///
  /// # Errors
  ///
  /// Returns an error if `num_perm` is zero, the outer input is not iterable,
  /// or any token hash is not an unsigned 64-bit integer.
  #[classmethod]
  #[pyo3(signature = (token_hash_sets, num_perm, seed, probes=DEFAULT_RHO_PROBES))]
  pub fn digest_matrix_from_token_hash_sets_rho(
    _cls: &Bound<'_, PyType>,
    token_hash_sets: Bound<'_, PyAny>,
    num_perm: usize,
    seed: u64,
    probes: usize,
  ) -> PyResult<RMinHashDigestMatrix> {
    Self::validate_num_perm(num_perm)?;
    Self::build_rho_digest_matrix_from_token_hash_sets(
      &token_hash_sets,
      num_perm,
      seed,
      probes,
    )
  }

  /// Computes `RMinHash` digests in a compact row-major matrix from one flat
  /// token-hash buffer and row offsets.
  ///
  /// # Errors
  ///
  /// Returns an error if `num_perm` is zero, inputs are not unsigned-integer
  /// iterables/buffers, or `row_offsets` do not describe valid row boundaries.
  #[classmethod]
  #[pyo3(signature = (token_hashes, row_offsets, num_perm, seed))]
  pub fn digest_matrix_from_flat_token_hashes(
    _cls: &Bound<'_, PyType>,
    token_hashes: Bound<'_, PyAny>,
    row_offsets: Bound<'_, PyAny>,
    num_perm: usize,
    seed: u64,
  ) -> PyResult<RMinHashDigestMatrix> {
    Self::validate_num_perm(num_perm)?;
    let flat_hashes = Self::parse_flat_token_hashes(&token_hashes)?;
    let offsets = Self::parse_row_offsets(&row_offsets)?;
    Self::build_digest_matrix_from_flat_token_hashes(
      &flat_hashes,
      &offsets,
      num_perm,
      seed,
    )
  }

  /// Computes a compact row-major matrix from flat pre-hashed buffers using the Rho sketch.
  ///
  /// # Errors
  ///
  /// Returns an error if `num_perm` is zero, inputs are not unsigned-integer
  /// iterables/buffers, or `row_offsets` do not describe valid row boundaries.
  #[classmethod]
  #[pyo3(signature = (token_hashes, row_offsets, num_perm, seed, probes=DEFAULT_RHO_PROBES))]
  pub fn digest_matrix_from_flat_token_hashes_rho(
    _cls: &Bound<'_, PyType>,
    token_hashes: Bound<'_, PyAny>,
    row_offsets: Bound<'_, PyAny>,
    num_perm: usize,
    seed: u64,
    probes: usize,
  ) -> PyResult<RMinHashDigestMatrix> {
    Self::validate_num_perm(num_perm)?;
    let flat_hashes = Self::parse_flat_token_hashes(&token_hashes)?;
    let offsets = Self::parse_row_offsets(&row_offsets)?;
    Self::build_rho_digest_matrix_from_flat_token_hashes(
      &flat_hashes,
      &offsets,
      num_perm,
      seed,
      probes,
    )
  }

  /// Computes `RMinHash` digests in a compact row-major matrix from bytes-only tokens.
  ///
  /// # Errors
  ///
  /// Returns an error if `num_perm` is zero, the outer input is not iterable,
  /// or any token is not a bytes-like object.
  #[classmethod]
  #[pyo3(signature = (token_byte_sets, num_perm, seed))]
  pub fn digest_matrix_from_token_byte_sets(
    _cls: &Bound<'_, PyType>,
    token_byte_sets: Bound<'_, PyAny>,
    num_perm: usize,
    seed: u64,
  ) -> PyResult<RMinHashDigestMatrix> {
    Self::validate_num_perm(num_perm)?;
    Self::build_digest_matrix_from_token_sets(
      &token_byte_sets,
      num_perm,
      seed,
      "token_bytes",
      extend_byte_token_hashes_from_document,
    )
  }

  /// Updates the `MinHash` with a new set of items.
  ///
  /// # Arguments
  ///
  /// * `items` - `str`, bytes-like object, or iterable of `str`/bytes-like tokens.
  ///
  /// # Errors
  ///
  /// Returns an error if `items` is neither a supported bytes-like object
  /// nor an iterable of supported token types.
  #[pyo3(signature = (items))]
  pub fn update(&mut self, items: Bound<'_, PyAny>) -> PyResult<()> {
    self.ensure_permutations();
    let mut hash_batch = Vec::with_capacity(HASH_BATCH_SIZE);
    Self::apply_document_to_values(
      &items,
      &mut self.hash_values,
      &self.permutations,
      &self.permutations_soa,
      &mut hash_batch,
    )?;
    Ok(())
  }

  /// Returns the current `MinHash` digest.
  ///
  /// # Returns
  ///
  /// A vector of u32 values representing the `MinHash` signature.
  #[must_use]
  pub fn digest(&self) -> Vec<u32> {
    self.hash_values.clone()
  }

  /// Calculates the Jaccard similarity between this `MinHash` and another.
  ///
  /// # Arguments
  ///
  /// * `other` - Another `RMinHash` instance to compare with.
  ///
  /// # Returns
  ///
  /// A float value representing the estimated `Jaccard` similarity.
  ///
  /// # Errors
  ///
  /// Returns an error when instances have incompatible parameters.
  pub fn jaccard(&self, other: &Self) -> PyResult<f64> {
    self.ensure_compatible_for_jaccard(other)?;
    Ok(self.jaccard_unchecked(other))
  }

  fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> {
    let decoded: Self =
      postcard::from_bytes(state.as_bytes()).map_err(|err| {
        PyValueError::new_err(format!(
          "failed to deserialize RMinHash state: {err}"
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
        "failed to serialize RMinHash state: {err}"
      ))
    })?;
    Ok(PyBytes::new(py, &encoded))
  }

  const fn __getnewargs__(&self) -> (usize, u64) {
    (self.num_perm, self.seed)
  }

  fn __reduce__(&self) -> PyResult<ReduceResult> {
    Python::attach(|py| {
      let type_obj = py.get_type::<Self>().into();
      let state = self.__getstate__(py)?.into();
      Ok((type_obj, (self.num_perm, self.seed), state))
    })
  }
}
