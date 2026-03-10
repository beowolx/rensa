use crate::env::read_env_usize_clamped;
use crate::py_input::{
  extend_prehashed_token_values_from_document,
  extend_token_hashes_from_document_with_limit, fast_sequence_length,
};
use crate::rminhash::matrix::RhoDigestSidecar;
use crate::rminhash::send_ptr::SendPtr;
use crate::rminhash::token::{
  token_bytes_ref_from_bytes_ptr, token_bytes_ref_from_token_ptr,
  token_bytes_ref_from_unicode_ptr, TokenBytesKind, TokenBytesPtr,
  TokenBytesRef,
};
use crate::rminhash::{
  DigestBuildConfig, RMinHash, RMinHashDigestMatrix,
  DEFAULT_RHO_LONG_DOC_FACTOR, DEFAULT_RHO_MEDIUM_TOKEN_BUDGET,
  DEFAULT_RHO_MEDIUM_TOKEN_THRESHOLD, DEFAULT_RHO_SHORT_FULL_TOKEN_THRESHOLD,
  DEFAULT_RHO_SPARSE_OCCUPANCY_THRESHOLD_BASE, DEFAULT_RHO_SPARSE_VERIFY_PERM,
  DEFAULT_RHO_TOKEN_BUDGET_MIN, EMPTY_BUCKET, MAX_RHO_LONG_DOC_THRESHOLD,
  MAX_RHO_MEDIUM_TOKEN_BUDGET, MAX_RHO_MEDIUM_TOKEN_THRESHOLD, MAX_RHO_PROBES,
  MAX_RHO_SPARSE_OCCUPANCY_THRESHOLD_BASE, MAX_RHO_SPARSE_VERIFY_PERM,
  MAX_RHO_TOKEN_BUDGET, MIN_RHO_LONG_DOC_THRESHOLD,
  MIN_RHO_MEDIUM_TOKEN_BUDGET, MIN_RHO_MEDIUM_TOKEN_THRESHOLD, MIN_RHO_PROBES,
  MIN_RHO_SPARSE_OCCUPANCY_THRESHOLD_BASE, MIN_RHO_SPARSE_VERIFY_PERM,
};
use pyo3::exceptions::PyValueError;
use pyo3::ffi;
use pyo3::prelude::*;
use rayon::prelude::*;
use rensa_core::utils::calculate_hash_fast;
use std::sync::mpsc;
use std::thread;

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
fn parse_rho_probes(probes: usize) -> usize {
  probes.clamp(MIN_RHO_PROBES, MAX_RHO_PROBES)
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

fn checked_len_mul(left: usize, right: usize, label: &str) -> PyResult<usize> {
  left.checked_mul(right).ok_or_else(|| {
    PyValueError::new_err(format!(
      "{label} size overflow: left={left}, right={right}"
    ))
  })
}

const fn rho_adaptive_token_budget_for_row(
  source_token_count: Option<usize>,
  default_budget: Option<usize>,
  has_token_budget_override: bool,
  medium_token_threshold: usize,
  medium_token_budget: usize,
) -> Option<usize> {
  if has_token_budget_override {
    return default_budget;
  }

  let Some(source_token_count) = source_token_count else {
    return default_budget;
  };

  if source_token_count <= DEFAULT_RHO_SHORT_FULL_TOKEN_THRESHOLD {
    return None;
  }
  if source_token_count <= medium_token_threshold {
    return Some(medium_token_budget);
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

#[inline]
fn effective_rho_probes(
  configured_probes: usize,
  source_token_count: usize,
  adaptive_probes_enabled: bool,
  long_doc_threshold: usize,
) -> usize {
  if configured_probes <= 1 || !adaptive_probes_enabled {
    return configured_probes;
  }
  if source_token_count >= long_doc_threshold {
    configured_probes
  } else {
    configured_probes.saturating_sub(1).max(1)
  }
}

fn rho_sparse_verify_perm(num_perm: usize) -> usize {
  read_env_usize_clamped(
    "RENSA_RHO_SPARSE_VERIFY_PERM",
    DEFAULT_RHO_SPARSE_VERIFY_PERM,
    MIN_RHO_SPARSE_VERIFY_PERM,
    MAX_RHO_SPARSE_VERIFY_PERM.min(num_perm.max(1)),
  )
}

fn read_env_flag(name: &str, default_enabled: bool) -> bool {
  std::env::var(name)
    .ok()
    .map_or(default_enabled, |value| value != "0")
}

fn rho_long_doc_threshold(num_perm: usize) -> usize {
  let default_threshold = num_perm
    .saturating_mul(DEFAULT_RHO_LONG_DOC_FACTOR)
    .clamp(MIN_RHO_LONG_DOC_THRESHOLD, MAX_RHO_LONG_DOC_THRESHOLD);
  std::env::var("RENSA_RHO_LONG_DOC_THRESHOLD")
    .ok()
    .and_then(|value| value.parse::<usize>().ok())
    .map_or(default_threshold, |value| {
      value.clamp(MIN_RHO_LONG_DOC_THRESHOLD, MAX_RHO_LONG_DOC_THRESHOLD)
    })
}

#[derive(Clone, Copy)]
struct RhoSketchConfig {
  probes: usize,
  default_token_budget: Option<usize>,
  has_token_budget_override: bool,
  medium_token_threshold: usize,
  medium_token_budget: usize,
  sparse_occupancy_threshold: usize,
  sparse_verify_perm: usize,
  adaptive_probes_enabled: bool,
  long_doc_threshold: usize,
  densify_enabled: bool,
}

impl RhoSketchConfig {
  fn from_env(num_perm: usize, probes: usize) -> Self {
    // Snapshot Rho tuning once per build call so a digest run does not mix
    // cached and live env semantics halfway through the same matrix build.
    let probes = parse_rho_probes(probes);
    let default_token_budget = rho_token_budget(num_perm);
    let has_token_budget_override = rho_token_budget_env_override_is_set();
    let (medium_token_threshold, medium_token_budget) =
      if has_token_budget_override {
        (0, 0)
      } else {
        (rho_medium_token_threshold(), rho_medium_token_budget())
      };
    let sparse_occupancy_threshold = rho_sparse_occupancy_threshold(num_perm);
    let sparse_verify_perm = if rho_sparse_verify_enabled() {
      rho_sparse_verify_perm(num_perm)
    } else {
      0
    };
    let adaptive_probes_enabled =
      read_env_flag("RENSA_RHO_ADAPTIVE_PROBES", false);
    let long_doc_threshold = rho_long_doc_threshold(num_perm);
    let densify_enabled = read_env_flag("RENSA_RHO_DENSIFY", false);

    Self {
      probes,
      default_token_budget,
      has_token_budget_override,
      medium_token_threshold,
      medium_token_budget,
      sparse_occupancy_threshold,
      sparse_verify_perm,
      adaptive_probes_enabled,
      long_doc_threshold,
      densify_enabled,
    }
  }
}

#[inline]
const fn low_u32_from_u64(value: u64) -> u32 {
  let bytes = value.to_le_bytes();
  u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

#[inline]
const fn low_u32_from_usize(value: usize) -> u32 {
  #[cfg(target_pointer_width = "64")]
  {
    let bytes = value.to_le_bytes();
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
  }
  #[cfg(target_pointer_width = "32")]
  {
    u32::from_le_bytes(value.to_le_bytes())
  }
}

#[inline]
const fn u64_to_usize_wrapping(value: u64) -> usize {
  #[cfg(target_pointer_width = "64")]
  {
    usize::from_ne_bytes(value.to_ne_bytes())
  }
  #[cfg(target_pointer_width = "32")]
  {
    let bytes = value.to_ne_bytes();
    let low = u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    usize::from_ne_bytes(low.to_ne_bytes())
  }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Utf8EncodeError {
  SurrogateCodePoint { code_point: u32 },
  CodePointOutOfRange { code_point: u32 },
}

impl std::fmt::Display for Utf8EncodeError {
  fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match *self {
      Self::SurrogateCodePoint { code_point } => write!(
        formatter,
        "unicode token contains surrogate code point: 0x{code_point:04x}"
      ),
      Self::CodePointOutOfRange { code_point } => write!(
        formatter,
        "unicode token contains code point outside Unicode range: 0x{code_point:08x}"
      ),
    }
  }
}

#[allow(clippy::cast_possible_truncation)]
fn push_utf8_from_code_point(
  output: &mut Vec<u8>,
  code_point: u32,
) -> Result<(), Utf8EncodeError> {
  if code_point <= 0x7f {
    output.push(code_point as u8);
    return Ok(());
  }
  if code_point <= 0x7ff {
    output.push((0xc0 | (code_point >> 6)) as u8);
    output.push((0x80 | (code_point & 0x3f)) as u8);
    return Ok(());
  }
  if code_point <= 0xffff {
    if (0xd800..=0xdfff).contains(&code_point) {
      return Err(Utf8EncodeError::SurrogateCodePoint { code_point });
    }
    output.push((0xe0 | (code_point >> 12)) as u8);
    output.push((0x80 | ((code_point >> 6) & 0x3f)) as u8);
    output.push((0x80 | (code_point & 0x3f)) as u8);
    return Ok(());
  }
  if code_point <= 0x0010_ffff {
    output.push((0xf0 | (code_point >> 18)) as u8);
    output.push((0x80 | ((code_point >> 12) & 0x3f)) as u8);
    output.push((0x80 | ((code_point >> 6) & 0x3f)) as u8);
    output.push((0x80 | (code_point & 0x3f)) as u8);
    return Ok(());
  }
  Err(Utf8EncodeError::CodePointOutOfRange { code_point })
}

fn hash_bytes(ptr: *const u8, len: usize) -> u64 {
  let bytes = if len == 0 {
    &[][..]
  } else {
    // SAFETY: callers only provide non-null pointers with a matching length.
    unsafe { std::slice::from_raw_parts(ptr, len) }
  };
  calculate_hash_fast(bytes)
}

fn hash_unicode_token_ref(
  kind: TokenBytesKind,
  ptr: TokenBytesPtr,
  len: usize,
  scratch: &mut Vec<u8>,
) -> Result<u64, Utf8EncodeError> {
  if len == 0 {
    return Ok(calculate_hash_fast(&[]));
  }

  scratch.clear();

  match kind {
    TokenBytesKind::Unicode1 => {
      scratch.reserve(len.saturating_mul(2));
      // SAFETY: pointer refers to `len` bytes of 1-byte unicode data.
      let values = unsafe { std::slice::from_raw_parts(ptr.bytes, len) };
      for &value in values {
        if value <= 0x7f {
          scratch.push(value);
        } else {
          scratch.push(0xc0 | (value >> 6));
          scratch.push(0x80 | (value & 0x3f));
        }
      }
      Ok(calculate_hash_fast(scratch.as_slice()))
    }
    TokenBytesKind::Unicode2 => {
      scratch.reserve(len.saturating_mul(3));
      // SAFETY: pointer refers to `len` code units of 2-byte unicode data.
      let values = unsafe { std::slice::from_raw_parts(ptr.u16, len) };
      for &value in values {
        push_utf8_from_code_point(scratch, u32::from(value))?;
      }
      Ok(calculate_hash_fast(scratch.as_slice()))
    }
    TokenBytesKind::Unicode4 => {
      scratch.reserve(len.saturating_mul(4));
      // SAFETY: pointer refers to `len` code units of 4-byte unicode data.
      let values = unsafe { std::slice::from_raw_parts(ptr.u32, len) };
      for &value in values {
        push_utf8_from_code_point(scratch, value)?;
      }
      Ok(calculate_hash_fast(scratch.as_slice()))
    }
    TokenBytesKind::Bytes => Ok(hash_bytes(unsafe { ptr.bytes }, len)),
  }
}

struct RhoRowScratch {
  token_hashes: Vec<u64>,
  utf8_scratch: Vec<u8>,
}

impl RhoRowScratch {
  fn new() -> Self {
    Self {
      token_hashes: Vec::with_capacity(64),
      utf8_scratch: Vec::with_capacity(256),
    }
  }
}

#[derive(Clone, Copy)]
struct SendPyObjectPtr(*mut ffi::PyObject);

// SAFETY: these pointers are only stored after taking a strong reference under
// the GIL. They are never dereferenced without reacquiring the GIL in Drop.
unsafe impl Send for SendPyObjectPtr {}

struct OwnedPyObjects {
  ptrs: Vec<SendPyObjectPtr>,
}

impl OwnedPyObjects {
  fn with_capacity(capacity: usize) -> Self {
    Self {
      ptrs: Vec::with_capacity(capacity),
    }
  }

  fn push_incref(&mut self, object_ptr: *mut ffi::PyObject) {
    // SAFETY: caller holds the GIL and `object_ptr` is a live Python object.
    unsafe { ffi::Py_INCREF(object_ptr) };
    self.ptrs.push(SendPyObjectPtr(object_ptr));
  }
}

impl Drop for OwnedPyObjects {
  fn drop(&mut self) {
    if self.ptrs.is_empty() {
      return;
    }
    let ptrs = std::mem::take(&mut self.ptrs);
    Python::attach(|_| {
      for ptr in ptrs {
        // SAFETY: every pointer in `ptrs` was previously incref'd.
        unsafe { ffi::Py_DECREF(ptr.0) };
      }
    });
  }
}

struct RhoWorkChunk {
  row_start: usize,
  row_end: usize,
  token_refs: Vec<TokenBytesRef>,
  _row_owners: OwnedPyObjects,
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
  const fn new(total: usize, limit: usize) -> Self {
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
  const fn next(&mut self) -> usize {
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
struct RhoOutputs {
  matrix: SendPtr<u32>,
  non_empty: SendPtr<u16>,
  source_counts: SendPtr<u16>,
  sparse_active: Option<SendPtr<u8>>,
  sparse_signatures: Option<SendPtr<u32>>,
}

#[inline]
unsafe fn tuple_len(tuple_ptr: *mut ffi::PyObject) -> ffi::Py_ssize_t {
  unsafe { ffi::PyTuple_Size(tuple_ptr) }
}

fn collect_token_refs_from_tuple(
  py: Python<'_>,
  tuple_ptr: *mut ffi::PyObject,
  token_len: usize,
  take: usize,
  token_refs: &mut Vec<TokenBytesRef>,
) -> PyResult<bool> {
  if take == 0 {
    return Ok(false);
  }

  // SAFETY: caller validated tuple rows and keeps a strong ref to the tuple.
  unsafe {
    let first_item_ptr = ffi::PyTuple_GetItem(tuple_ptr, 0);
    let (first_is_unicode, first_is_bytes, first_type_ptr) = (
      ffi::PyUnicode_Check(first_item_ptr) != 0,
      ffi::PyBytes_Check(first_item_ptr) != 0,
      ffi::Py_TYPE(first_item_ptr),
    );

    let collect = |index: usize| -> PyResult<Option<TokenBytesRef>> {
      debug_assert!(index <= ffi::Py_ssize_t::MAX as usize);
      #[allow(clippy::cast_possible_wrap)]
      let index_ssize = index as ffi::Py_ssize_t;
      let item_ptr = ffi::PyTuple_GetItem(tuple_ptr, index_ssize);
      if first_is_unicode {
        let item_type_ptr = ffi::Py_TYPE(item_ptr);
        if item_type_ptr == first_type_ptr {
          let token_ref = token_bytes_ref_from_unicode_ptr(py, item_ptr)?;
          return Ok(Some(token_ref));
        }
      } else if first_is_bytes {
        let item_type_ptr = ffi::Py_TYPE(item_ptr);
        if item_type_ptr == first_type_ptr {
          let token_ref = token_bytes_ref_from_bytes_ptr(py, item_ptr)?;
          return Ok(Some(token_ref));
        }
      }
      token_bytes_ref_from_token_ptr(py, item_ptr)
    };

    if take == token_len {
      for index in 0..take {
        let Some(token_ref) = collect(index)? else {
          return Ok(true);
        };
        token_refs.push(token_ref);
      }
      return Ok(false);
    }

    let mut sampler = MidpointSampler::new(token_len, take);
    for _ in 0..take {
      let index = sampler.next();
      let Some(token_ref) = collect(index)? else {
        return Ok(true);
      };
      token_refs.push(token_ref);
    }
    Ok(false)
  }
}

fn append_sparse_sidecar_for_row(
  non_empty_count: usize,
  sparse_occupancy_threshold: usize,
  sparse_verify_perm: usize,
  token_hashes: &[u64],
  seed: u64,
  sparse_verify_active: &mut Vec<u8>,
  sparse_verify_signatures: &mut Vec<u32>,
) {
  if sparse_verify_perm == 0 {
    return;
  }
  let sparse = non_empty_count < sparse_occupancy_threshold;
  sparse_verify_active.push(u8::from(sparse));
  let signature_start = sparse_verify_signatures.len();
  sparse_verify_signatures
    .resize(signature_start + sparse_verify_perm, u32::MAX);
  if sparse {
    RMinHash::compute_sparse_verify_signature_into(
      &mut sparse_verify_signatures
        [signature_start..signature_start + sparse_verify_perm],
      token_hashes,
      seed,
    );
  }
}

impl RMinHash {
  #[inline]
  const fn rho_bucket_index(
    mixed_hash: u64,
    num_perm_u64: u64,
    is_power_of_two: bool,
  ) -> usize {
    let mapped = if is_power_of_two {
      mixed_hash & (num_perm_u64 - 1)
    } else {
      mixed_hash % num_perm_u64
    };
    u64_to_usize_wrapping(mapped)
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
      let bucket = Self::rho_bucket_index(mixed, num_perm_u64, is_power_of_two);
      digest_row[bucket] = digest_row[bucket].min((mixed >> 32) as u32);
      if probe + 1 < probes {
        mixed = splitmix64(mixed ^ RHO_SALTS[(probe + 1) & 3]);
      }
      probe += 1;
    }
  }

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
        let index_mix = low_u32_from_usize(index).wrapping_mul(0x9e37_79b9);
        let probe_mix = low_u32_from_usize(probe).wrapping_mul(0x85eb_ca6b);
        let seed_mix = low_u32_from_u64(seed).wrapping_mul(0xc2b2_ae35);
        digest_row[index] = mix_u32(value ^ index_mix ^ probe_mix ^ seed_mix);
      } else {
        digest_row[index] = mix_u32(
          low_u32_from_u64(seed)
            ^ low_u32_from_usize(index).wrapping_mul(0x27d4_eb2d),
        );
      }
    }
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

  fn compute_rho_digest_from_token_hashes_into(
    digest_row: &mut [u32],
    token_hashes: &[u64],
    seed: u64,
    probes: usize,
    token_budget: Option<usize>,
    densify_enabled: bool,
  ) {
    // Callers pre-initialize rows with EMPTY_BUCKET.
    if digest_row.is_empty() || token_hashes.is_empty() {
      return;
    }
    let num_perm_u64 = digest_row.len() as u64;
    let is_power_of_two = digest_row.len().is_power_of_two();
    if let Some(limit) =
      token_budget.filter(|&budget| budget > 0 && token_hashes.len() > budget)
    {
      let mut sampler = MidpointSampler::new(token_hashes.len(), limit);
      for _ in 0..limit {
        let index = sampler.next();
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
    if densify_enabled {
      Self::densify_rho_row(digest_row, seed);
    }
  }

  fn compute_rho_row_from_token_refs(
    row: &mut [u32],
    scratch: &mut RhoRowScratch,
    token_refs: &[TokenBytesRef],
    seed: u64,
    probes: usize,
    densify_enabled: bool,
  ) -> Result<usize, Utf8EncodeError> {
    scratch.token_hashes.clear();
    let num_perm_u64 = row.len() as u64;
    let is_power_of_two = row.len().is_power_of_two();
    for token_ref in token_refs.iter().copied() {
      let token_hash = hash_unicode_token_ref(
        token_ref.kind,
        token_ref.ptr,
        token_ref.len,
        &mut scratch.utf8_scratch,
      )?;
      scratch.token_hashes.push(token_hash);
      Self::apply_rho_probes_to_row(
        row,
        token_hash,
        seed,
        probes,
        num_perm_u64,
        is_power_of_two,
      );
    }
    if densify_enabled {
      Self::densify_rho_row(row, seed);
    }
    Ok(Self::count_non_empty_buckets(row))
  }

  pub(in crate::rminhash) fn try_build_rho_digest_matrix_from_token_sets_parallel(
    token_sets: &Bound<'_, PyAny>,
    num_perm: usize,
    seed: u64,
    probes: usize,
  ) -> PyResult<Option<RMinHashDigestMatrix>> {
    // Avoid initializing Rayon in the single-thread lane. This keeps the
    // cold-start cost (global pool init) out of the timed path for small
    // workloads like the simple benchmark.
    if let Some(value) = std::env::var("RAYON_NUM_THREADS")
      .ok()
      .and_then(|value| value.parse::<usize>().ok())
    {
      // Treat `0` as "unset" since Rayon ignores invalid/empty values.
      if value == 1 {
        return Ok(None);
      }
    }

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

    let sketch = RhoSketchConfig::from_env(num_perm, probes);
    let probes = sketch.probes;
    let default_token_budget = sketch.default_token_budget;
    let has_token_budget_override = sketch.has_token_budget_override;
    let medium_token_threshold = sketch.medium_token_threshold;
    let medium_token_budget = sketch.medium_token_budget;
    let sparse_occupancy_threshold = sketch.sparse_occupancy_threshold;
    let sparse_verify_perm = sketch.sparse_verify_perm;
    let adaptive_probes_enabled = sketch.adaptive_probes_enabled;
    let long_doc_threshold = sketch.long_doc_threshold;
    let densify_enabled = sketch.densify_enabled;

    let matrix_len = checked_len_mul(rows, num_perm, "rho matrix")?;
    let mut matrix_data = vec![EMPTY_BUCKET; matrix_len];
    let mut non_empty_counts = vec![0u16; rows];
    let mut source_token_counts = vec![0u16; rows];
    let mut sparse_verify_signatures = if sparse_verify_perm > 0 {
      let sparse_sig_len =
        checked_len_mul(rows, sparse_verify_perm, "rho sparse verify")?;
      vec![u32::MAX; sparse_sig_len]
    } else {
      Vec::new()
    };
    let mut sparse_verify_active = if sparse_verify_perm > 0 {
      vec![0u8; rows]
    } else {
      Vec::new()
    };

    let py = token_sets.py();

    let chunk_size = config.doc_chunk_size.max(1);
    let output_ptrs = RhoOutputs {
      matrix: SendPtr::new(matrix_data.as_mut_ptr()),
      non_empty: SendPtr::new(non_empty_counts.as_mut_ptr()),
      source_counts: SendPtr::new(source_token_counts.as_mut_ptr()),
      sparse_active: if sparse_verify_perm > 0 {
        Some(SendPtr::new(sparse_verify_active.as_mut_ptr()))
      } else {
        None
      },
      sparse_signatures: if sparse_verify_perm > 0 {
        Some(SendPtr::new(sparse_verify_signatures.as_mut_ptr()))
      } else {
        None
      },
    };

    let (work_tx, work_rx) =
      mpsc::sync_channel::<RhoWorkChunk>(config.pipeline_queue_cap);
    let (result_tx, result_rx) = mpsc::channel::<Result<(), Utf8EncodeError>>();

    let worker = thread::spawn(move || {
      for chunk in work_rx {
        let row_count = chunk.row_end - chunk.row_start;
        let matrix_offset = chunk.row_start * num_perm;
        let matrix_chunk_len = row_count * num_perm;

        // SAFETY: output pointers refer to preallocated vectors which remain alive
        // until the worker thread is joined. Each chunk writes to a disjoint row range.
        let matrix_chunk = unsafe {
          std::slice::from_raw_parts_mut(
            output_ptrs.matrix.add(matrix_offset),
            matrix_chunk_len,
          )
        };
        let non_empty_chunk = unsafe {
          std::slice::from_raw_parts_mut(
            output_ptrs.non_empty.add(chunk.row_start),
            row_count,
          )
        };
        let source_token_chunk = unsafe {
          std::slice::from_raw_parts_mut(
            output_ptrs.source_counts.add(chunk.row_start),
            row_count,
          )
        };

        let chunk_result: Result<(), Utf8EncodeError> =
          if sparse_verify_perm == 0 {
            matrix_chunk
              .par_chunks_exact_mut(num_perm)
              .zip(non_empty_chunk.par_iter_mut())
              .zip(source_token_chunk.par_iter_mut())
              .enumerate()
              .try_for_each_init(
                RhoRowScratch::new,
                |scratch,
                 (local_row_index, ((row, non_empty_out), source_out))| {
                  let source_token_count =
                    chunk.source_token_counts[local_row_index];
                  *source_out = saturating_u16(source_token_count);

                  let row_probes = effective_rho_probes(
                    probes,
                    source_token_count,
                    adaptive_probes_enabled,
                    long_doc_threshold,
                  );
                  let token_start = chunk.row_token_offsets[local_row_index];
                  let token_end =
                    chunk.row_token_offsets[local_row_index + 1];
                  let non_empty = Self::compute_rho_row_from_token_refs(
                    row,
                    scratch,
                    &chunk.token_refs[token_start..token_end],
                    seed,
                    row_probes,
                    densify_enabled,
                  )?;
                  *non_empty_out = saturating_u16(non_empty);
                  Ok(())
                },
              )
          } else {
            let Some(sparse_active_ptr) = output_ptrs.sparse_active else {
              break;
            };
            let Some(sparse_sig_ptr) = output_ptrs.sparse_signatures else {
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
                sparse_sig_ptr.add(chunk.row_start * sparse_verify_perm),
                row_count * sparse_verify_perm,
              )
            };

            matrix_chunk
              .par_chunks_exact_mut(num_perm)
              .zip(non_empty_chunk.par_iter_mut())
              .zip(source_token_chunk.par_iter_mut())
              .zip(sparse_active_chunk.par_iter_mut())
              .zip(sparse_sig_chunk.par_chunks_exact_mut(sparse_verify_perm))
              .enumerate()
              .try_for_each_init(
                RhoRowScratch::new,
                |scratch,
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

                  let row_probes = effective_rho_probes(
                    probes,
                    source_token_count,
                    adaptive_probes_enabled,
                    long_doc_threshold,
                  );
                  let token_start = chunk.row_token_offsets[local_row_index];
                  let token_end = chunk.row_token_offsets[local_row_index + 1];
                  let non_empty = Self::compute_rho_row_from_token_refs(
                    row,
                    scratch,
                    &chunk.token_refs[token_start..token_end],
                    seed,
                    row_probes,
                    densify_enabled,
                  )?;
                  *non_empty_out = saturating_u16(non_empty);

                  let is_sparse = non_empty < sparse_occupancy_threshold;
                  *sparse_active_out = u8::from(is_sparse);
                  if is_sparse {
                    Self::compute_sparse_verify_signature_into(
                      signature_row,
                      &scratch.token_hashes,
                      seed,
                    );
                  }
                  Ok(())
                },
              )
          };

        match chunk_result {
          Ok(()) => {
            if result_tx.send(Ok(())).is_err() {
              break;
            }
          }
          Err(err) => {
            let _ = result_tx.send(Err(err));
            break;
          }
        }
      }
    });

    let mut chunks_sent = 0usize;
    let mut extraction_error: Option<PyErr> = None;
    let mut should_fallback = false;
    let mut row_start = 0usize;
    while row_start < rows {
      let row_end = (row_start + chunk_size).min(rows);

      let chunk_rows = row_end - row_start;
      let max_take_per_row = default_token_budget
        .unwrap_or(0)
        .max(DEFAULT_RHO_SHORT_FULL_TOKEN_THRESHOLD)
        .max(medium_token_budget);
      let token_ref_capacity = match checked_len_mul(
        chunk_rows,
        max_take_per_row,
        "rho token refs chunk",
      ) {
        Ok(value) => value,
        Err(err) => {
          extraction_error = Some(err);
          break;
        }
      };
      let mut token_refs = Vec::with_capacity(token_ref_capacity);
      let mut owners = OwnedPyObjects::with_capacity(token_ref_capacity);
      let mut row_token_offsets = Vec::with_capacity(chunk_rows + 1);
      row_token_offsets.push(0);
      let mut chunk_source_counts = Vec::with_capacity(chunk_rows);

      for row_index in row_start..row_end {
        // SAFETY: row indices come from a CPython `Py_ssize_t` length.
        debug_assert!(row_index <= ffi::Py_ssize_t::MAX as usize);
        #[allow(clippy::cast_possible_wrap)]
        let row_index_ssize = row_index as ffi::Py_ssize_t;
        // SAFETY: outer_is_list/outer_is_tuple guarantees GET_ITEM indexing.
        let document_ptr = unsafe {
          if outer_is_list {
            ffi::PyList_GET_ITEM(object_ptr, row_index_ssize)
          } else {
            ffi::PyTuple_GetItem(object_ptr, row_index_ssize)
          }
        };

        // SAFETY: type checks under the GIL with a borrowed CPython pointer.
        let (document_is_list, document_is_tuple) = unsafe {
          (
            ffi::PyList_Check(document_ptr) != 0,
            ffi::PyTuple_Check(document_ptr) != 0,
          )
        };
        if !document_is_list && !document_is_tuple {
          should_fallback = true;
          break;
        }
        if document_is_list {
          // Parallel Rho keeps raw token pointers alive without copying token
          // bytes. That is cheap and safe for immutable tuple rows, but for
          // mutable Python lists the ownership bookkeeping costs more than the
          // sketch work in our benchmarked workloads. Fall back to the
          // streaming path instead of paying that tax here.
          should_fallback = true;
          break;
        }
        owners.push_incref(document_ptr);

        // SAFETY: tuple rows stay alive for the whole chunk via `owners`.
        let token_len_ssize = unsafe { tuple_len(document_ptr) };
        debug_assert!(token_len_ssize >= 0);
        #[allow(clippy::cast_sign_loss)]
        let token_len = token_len_ssize as usize;
        chunk_source_counts.push(token_len);

        let row_token_budget = rho_adaptive_token_budget_for_row(
          Some(token_len),
          default_token_budget,
          has_token_budget_override,
          medium_token_threshold,
          medium_token_budget,
        );
        let take =
          row_token_budget.map_or(token_len, |limit| token_len.min(limit));

        match collect_token_refs_from_tuple(
          py,
          document_ptr,
          token_len,
          take,
          &mut token_refs,
        ) {
          Ok(fallback) => {
            if fallback {
              should_fallback = true;
              break;
            }
          }
          Err(err) => {
            extraction_error = Some(err);
            break;
          }
        }

        row_token_offsets.push(token_refs.len());
      }

      if extraction_error.is_some() || should_fallback {
        break;
      }

      if let Err(err) = work_tx.send(RhoWorkChunk {
        row_start,
        row_end,
        token_refs,
        _row_owners: owners,
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
        Ok(Ok(())) => {}
        Ok(Err(err)) => {
          extraction_error = Some(PyValueError::new_err(format!(
            "rho pipeline worker failed while hashing unicode tokens: {err}"
          )));
          break;
        }
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
      seed,
      rows,
      data: matrix_data,
      rho_sidecar: Some(rho_sidecar),
    }))
  }

  pub(in crate::rminhash) fn build_rho_digest_matrix_from_token_sets_streaming(
    token_sets: &Bound<'_, PyAny>,
    num_perm: usize,
    seed: u64,
    probes: usize,
  ) -> PyResult<RMinHashDigestMatrix> {
    let capacity = Self::token_sets_capacity(token_sets);
    let matrix_capacity =
      checked_len_mul(capacity, num_perm, "rho matrix capacity")?;
    let mut matrix_data = Vec::with_capacity(matrix_capacity);
    let mut rows = 0usize;
    let sketch = RhoSketchConfig::from_env(num_perm, probes);
    let probes = sketch.probes;
    let default_token_budget = sketch.default_token_budget;
    let has_token_budget_override = sketch.has_token_budget_override;
    let medium_token_threshold = sketch.medium_token_threshold;
    let medium_token_budget = sketch.medium_token_budget;
    let sparse_occupancy_threshold = sketch.sparse_occupancy_threshold;
    let sparse_verify_perm = sketch.sparse_verify_perm;
    let adaptive_probes_enabled = sketch.adaptive_probes_enabled;
    let long_doc_threshold = sketch.long_doc_threshold;
    let densify_enabled = sketch.densify_enabled;
    let mut token_hashes = Vec::new();
    let mut non_empty_counts = Vec::with_capacity(capacity);
    let mut source_token_counts = Vec::with_capacity(capacity);
    let sparse_verify_capacity = checked_len_mul(
      capacity,
      sparse_verify_perm,
      "rho sparse verify capacity",
    )?;
    let mut sparse_verify_signatures =
      Vec::with_capacity(sparse_verify_capacity);
    let mut sparse_verify_active = Vec::with_capacity(capacity);

    Self::for_each_document(token_sets, |document| {
      let row_start = matrix_data.len();
      matrix_data.resize(row_start + num_perm, EMPTY_BUCKET);
      let row = &mut matrix_data[row_start..row_start + num_perm];
      let source_token_count = fast_sequence_length(&document)?;
      let row_token_budget = rho_adaptive_token_budget_for_row(
        source_token_count,
        default_token_budget,
        has_token_budget_override,
        medium_token_threshold,
        medium_token_budget,
      );
      token_hashes.clear();
      extend_token_hashes_from_document_with_limit(
        &document,
        &mut token_hashes,
        row_token_budget,
      )?;
      let row_source_token_count =
        source_token_count.unwrap_or(token_hashes.len());
      let row_probes = effective_rho_probes(
        probes,
        row_source_token_count,
        adaptive_probes_enabled,
        long_doc_threshold,
      );
      Self::compute_rho_digest_from_token_hashes_into(
        row,
        &token_hashes,
        seed,
        row_probes,
        row_token_budget,
        densify_enabled,
      );
      let non_empty_count = Self::count_non_empty_buckets(row);
      non_empty_counts.push(saturating_u16(non_empty_count));
      source_token_counts.push(saturating_u16(row_source_token_count));
      append_sparse_sidecar_for_row(
        non_empty_count,
        sparse_occupancy_threshold,
        sparse_verify_perm,
        &token_hashes,
        seed,
        &mut sparse_verify_active,
        &mut sparse_verify_signatures,
      );
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
      seed,
      rows,
      data: matrix_data,
      rho_sidecar: Some(rho_sidecar),
    })
  }

  pub(in crate::rminhash) fn build_rho_digest_matrix_from_token_hash_sets(
    token_hash_sets: &Bound<'_, PyAny>,
    num_perm: usize,
    seed: u64,
    probes: usize,
  ) -> PyResult<RMinHashDigestMatrix> {
    let capacity = Self::token_sets_capacity(token_hash_sets);
    let matrix_capacity =
      checked_len_mul(capacity, num_perm, "rho matrix capacity")?;
    let mut matrix_data = Vec::with_capacity(matrix_capacity);
    let mut rows = 0usize;
    let sketch = RhoSketchConfig::from_env(num_perm, probes);
    let probes = sketch.probes;
    let default_token_budget = sketch.default_token_budget;
    let has_token_budget_override = sketch.has_token_budget_override;
    let medium_token_threshold = sketch.medium_token_threshold;
    let medium_token_budget = sketch.medium_token_budget;
    let sparse_occupancy_threshold = sketch.sparse_occupancy_threshold;
    let sparse_verify_perm = sketch.sparse_verify_perm;
    let adaptive_probes_enabled = sketch.adaptive_probes_enabled;
    let long_doc_threshold = sketch.long_doc_threshold;
    let densify_enabled = sketch.densify_enabled;
    let mut token_hashes = Vec::new();
    let mut non_empty_counts = Vec::with_capacity(capacity);
    let mut source_token_counts = Vec::with_capacity(capacity);
    let sparse_verify_capacity = checked_len_mul(
      capacity,
      sparse_verify_perm,
      "rho sparse verify capacity",
    )?;
    let mut sparse_verify_signatures =
      Vec::with_capacity(sparse_verify_capacity);
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
        has_token_budget_override,
        medium_token_threshold,
        medium_token_budget,
      );
      let row_probes = effective_rho_probes(
        probes,
        token_hashes.len(),
        adaptive_probes_enabled,
        long_doc_threshold,
      );
      Self::compute_rho_digest_from_token_hashes_into(
        row,
        &token_hashes,
        seed,
        row_probes,
        row_token_budget,
        densify_enabled,
      );
      let non_empty_count = Self::count_non_empty_buckets(row);
      non_empty_counts.push(saturating_u16(non_empty_count));
      source_token_counts.push(saturating_u16(token_hashes.len()));
      append_sparse_sidecar_for_row(
        non_empty_count,
        sparse_occupancy_threshold,
        sparse_verify_perm,
        &token_hashes,
        seed,
        &mut sparse_verify_active,
        &mut sparse_verify_signatures,
      );
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
      seed,
      rows,
      data: matrix_data,
      rho_sidecar: Some(rho_sidecar),
    })
  }

  pub(in crate::rminhash) fn build_rho_digest_matrix_from_flat_token_hashes(
    token_hashes: &[u64],
    row_offsets: &[usize],
    num_perm: usize,
    seed: u64,
    probes: usize,
  ) -> PyResult<RMinHashDigestMatrix> {
    Self::validate_flat_row_offsets(row_offsets, token_hashes.len())?;
    let rows = row_offsets.len().saturating_sub(1);
    let matrix_len = checked_len_mul(rows, num_perm, "rho matrix")?;
    let mut matrix_data = vec![EMPTY_BUCKET; matrix_len];
    let sketch = RhoSketchConfig::from_env(num_perm, probes);
    let probes = sketch.probes;
    let default_token_budget = sketch.default_token_budget;
    let has_token_budget_override = sketch.has_token_budget_override;
    let medium_token_threshold = sketch.medium_token_threshold;
    let medium_token_budget = sketch.medium_token_budget;
    let sparse_occupancy_threshold = sketch.sparse_occupancy_threshold;
    let sparse_verify_perm = sketch.sparse_verify_perm;
    let adaptive_probes_enabled = sketch.adaptive_probes_enabled;
    let long_doc_threshold = sketch.long_doc_threshold;
    let densify_enabled = sketch.densify_enabled;
    let mut non_empty_counts = Vec::with_capacity(rows);
    let mut source_token_counts = Vec::with_capacity(rows);
    let sparse_verify_capacity =
      checked_len_mul(rows, sparse_verify_perm, "rho sparse verify capacity")?;
    let mut sparse_verify_signatures =
      Vec::with_capacity(sparse_verify_capacity);
    let mut sparse_verify_active = Vec::with_capacity(rows);
    for (row_index, row) in matrix_data.chunks_exact_mut(num_perm).enumerate() {
      let start = row_offsets[row_index];
      let end = row_offsets[row_index + 1];
      let token_count = end.saturating_sub(start);
      let row_token_budget = rho_adaptive_token_budget_for_row(
        Some(token_count),
        default_token_budget,
        has_token_budget_override,
        medium_token_threshold,
        medium_token_budget,
      );
      let row_probes = effective_rho_probes(
        probes,
        token_count,
        adaptive_probes_enabled,
        long_doc_threshold,
      );
      Self::compute_rho_digest_from_token_hashes_into(
        row,
        &token_hashes[start..end],
        seed,
        row_probes,
        row_token_budget,
        densify_enabled,
      );
      let non_empty_count = Self::count_non_empty_buckets(row);
      non_empty_counts.push(saturating_u16(non_empty_count));
      source_token_counts.push(saturating_u16(token_count));
      append_sparse_sidecar_for_row(
        non_empty_count,
        sparse_occupancy_threshold,
        sparse_verify_perm,
        &token_hashes[start..end],
        seed,
        &mut sparse_verify_active,
        &mut sparse_verify_signatures,
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
      seed,
      rows,
      data: matrix_data,
      rho_sidecar: Some(rho_sidecar),
    })
  }
}
