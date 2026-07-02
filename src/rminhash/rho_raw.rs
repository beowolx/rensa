//! Fully parallel rho sketch builder over raw `CPython` reads.
//!
//! The classic parallel builder extracts token references on the GIL thread
//! and ships them to workers, which caps throughput at the speed of that one
//! producer. This builder removes the producer: worker threads read list
//! items, type flags, and compact-ASCII/bytes payloads directly from
//! `CPython` object memory.
//!
//! # Safety model
//!
//! The calling thread holds the GIL for the whole build and never executes
//! Python code, so no other Python thread can run, the GC cannot trigger, and
//! every reachable object is frozen in memory. Under those conditions,
//! `PyList_GET_ITEM`/`PyTuple_GET_ITEM`, `Py_TYPE` flag checks, and compact
//! ASCII/bytes payload reads are plain loads from immutable memory and are
//! safe from any thread. This is the same invariant `TokenBytesRef` already
//! relies on; here it also covers sequence indexing. The module is declared
//! `gil_used = true`, so free-threaded `CPython` re-enables the GIL on import
//! and the invariant holds there as well.
//!
//! Rows containing tokens that cannot be read this way (non-compact-ASCII
//! strings, bytearrays, arbitrary objects) are re-sketched afterwards on the
//! GIL thread through the generic extraction path.

use crate::py_input::{
  compact_ascii_bytes, extend_token_hashes_from_document_with_limit,
  fast_sequence_length,
};
use crate::rminhash::matrix::RhoDigestSidecar;
use crate::rminhash::rho::{
  checked_len_mul, effective_rho_probes, rho_adaptive_token_budget_for_row,
  saturating_u16, MidpointSampler, RhoSketchConfig,
};
use crate::rminhash::{
  DigestBuildConfig, RMinHash, RMinHashDigestMatrix, EMPTY_BUCKET,
};
use crate::utils::calculate_hash_fast;
use pyo3::ffi;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::mem::MaybeUninit;

/// Number of leading documents probed to decide whether the corpus looks
/// raw-readable before committing to this builder.
const RAW_PROBE_ROWS: usize = 8;

fn raw_parallel_enabled() -> bool {
  std::env::var("RENSA_RHO_RAW_PARALLEL")
    .ok()
    .is_none_or(|value| value != "0")
}

/// Borrowed `CPython` object pointer that workers may read (not call) through.
#[derive(Clone, Copy)]
struct RawPyPtr(*mut ffi::PyObject);

// SAFETY: workers only perform plain reads through this pointer while the
// launching thread holds the GIL and keeps the object alive (see module docs).
unsafe impl Send for RawPyPtr {}
unsafe impl Sync for RawPyPtr {}

#[derive(Clone, Copy)]
struct RawRowContext {
  outer: RawPyPtr,
  outer_is_list: bool,
  seed: u64,
  num_perm: usize,
  num_perm_u64: u64,
  is_power_of_two: bool,
  sketch: RhoSketchConfig,
}

#[inline]
#[allow(clippy::cast_possible_wrap)]
unsafe fn seq_get_item(
  sequence_ptr: *mut ffi::PyObject,
  is_list: bool,
  index: usize,
) -> *mut ffi::PyObject {
  debug_assert!(index <= ffi::Py_ssize_t::MAX as usize);
  let index_ssize = index as ffi::Py_ssize_t;
  // SAFETY: caller validated the sequence kind and index bounds; these are
  // plain `ob_item` reads (see module docs).
  unsafe {
    if is_list {
      ffi::PyList_GET_ITEM(sequence_ptr, index_ssize)
    } else {
      ffi::PyTuple_GET_ITEM(sequence_ptr, index_ssize)
    }
  }
}

/// Reads a token's byte payload without calling into `CPython`.
///
/// Returns `None` when the token is not a compact-ASCII `str` or exact-layout
/// `bytes`, meaning the row must be handled on the GIL thread.
#[inline]
#[allow(clippy::cast_sign_loss)]
unsafe fn raw_token_bytes<'bytes>(
  item_ptr: *mut ffi::PyObject,
) -> Option<&'bytes [u8]> {
  // SAFETY: type-flag checks and payload reads are plain loads on a frozen
  // object graph (see module docs). The returned slice borrows CPython-owned
  // memory that outlives the build because the GIL thread keeps the outer
  // sequence alive.
  unsafe {
    if ffi::PyUnicode_Check(item_ptr) != 0 {
      let (ptr, len) = compact_ascii_bytes(item_ptr)?;
      if len == 0 {
        return Some(&[]);
      }
      return Some(std::slice::from_raw_parts(ptr, len));
    }
    if ffi::PyBytes_Check(item_ptr) != 0 {
      let length = ffi::Py_SIZE(item_ptr);
      debug_assert!(length >= 0);
      if length == 0 {
        return Some(&[]);
      }
      let data = ffi::PyBytes_AS_STRING(item_ptr).cast::<u8>();
      return Some(std::slice::from_raw_parts(data, length as usize));
    }
  }
  None
}

/// Initializes one uninitialized matrix row with `EMPTY_BUCKET` and returns
/// it as a normal `u32` slice.
#[inline]
const fn init_row_empty(row: &mut [MaybeUninit<u32>]) -> &mut [u32] {
  let ptr = row.as_mut_ptr().cast::<u32>();
  // SAFETY: `EMPTY_BUCKET` is `u32::MAX` (all bytes 0xFF, asserted in rho.rs),
  // so after this memset every element is initialized.
  unsafe {
    std::ptr::write_bytes(ptr, 0xFF, row.len());
    std::slice::from_raw_parts_mut(ptr, row.len())
  }
}

/// Sketches one row using only raw reads. Returns the row's non-empty bucket
/// count, or `None` when the row needs the GIL-thread fallback (in which case
/// the row is left as all `EMPTY_BUCKET`).
#[allow(clippy::cast_sign_loss)]
unsafe fn sketch_raw_row(
  ctx: &RawRowContext,
  row_index: usize,
  row: &mut [u32],
  mixed_values: &mut Vec<u64>,
  source_out: &mut u16,
) -> Option<usize> {
  mixed_values.clear();
  // SAFETY: outer kind was validated on the GIL thread; row_index < rows.
  let document_ptr =
    unsafe { seq_get_item(ctx.outer.0, ctx.outer_is_list, row_index) };
  // SAFETY: type-flag checks are plain reads (see module docs).
  let (document_is_list, token_len_ssize) = unsafe {
    if ffi::PyList_Check(document_ptr) != 0 {
      (true, ffi::PyList_GET_SIZE(document_ptr))
    } else if ffi::PyTuple_Check(document_ptr) != 0 {
      (false, ffi::PyTuple_GET_SIZE(document_ptr))
    } else {
      return None;
    }
  };
  debug_assert!(token_len_ssize >= 0);
  let token_len = token_len_ssize as usize;
  *source_out = saturating_u16(token_len);

  let row_token_budget = rho_adaptive_token_budget_for_row(
    Some(token_len),
    ctx.sketch.default_token_budget,
    ctx.sketch.has_token_budget_override,
    ctx.sketch.medium_token_threshold,
    ctx.sketch.medium_token_budget,
  );
  let take = row_token_budget.map_or(token_len, |limit| token_len.min(limit));
  let row_probes =
    effective_rho_probes(ctx.sketch.probes, token_len, ctx.num_perm);

  let mut hash_token_at = |index: usize, row: &mut [u32]| -> bool {
    // SAFETY: index < token_len for the validated list/tuple document.
    let item_ptr =
      unsafe { seq_get_item(document_ptr, document_is_list, index) };
    // SAFETY: see `raw_token_bytes`.
    let Some(token_bytes) = (unsafe { raw_token_bytes(item_ptr) }) else {
      return false;
    };
    let token_hash = calculate_hash_fast(token_bytes);
    let mixed = RMinHash::apply_rho_probes_to_row(
      row,
      token_hash,
      ctx.seed,
      row_probes,
      ctx.num_perm_u64,
      ctx.is_power_of_two,
    );
    mixed_values.push(mixed);
    true
  };

  if take == token_len {
    for index in 0..take {
      if !hash_token_at(index, row) {
        row.fill(EMPTY_BUCKET);
        return None;
      }
    }
  } else {
    let mut sampler = MidpointSampler::new(token_len, take);
    for _ in 0..take {
      let index = sampler.next();
      if !hash_token_at(index, row) {
        row.fill(EMPTY_BUCKET);
        return None;
      }
    }
  }

  if ctx.sketch.densify_enabled {
    RMinHash::densify_rho_row(row, ctx.seed);
  }
  Some(RMinHash::count_non_empty_buckets(row))
}

/// Cheap corpus probe: checks the first token of the first few non-empty
/// documents. A miss routes the build to the pipelined GIL-extraction path,
/// which overlaps extraction with hashing for corpora this builder cannot
/// read raw.
unsafe fn corpus_looks_raw_readable(
  outer: *mut ffi::PyObject,
  outer_is_list: bool,
  rows: usize,
) -> bool {
  for row_index in 0..rows.min(RAW_PROBE_ROWS) {
    // SAFETY: kind/index validated by the caller; plain reads throughout.
    unsafe {
      let document_ptr = seq_get_item(outer, outer_is_list, row_index);
      let (document_is_list, token_len) =
        if ffi::PyList_Check(document_ptr) != 0 {
          (true, ffi::PyList_GET_SIZE(document_ptr))
        } else if ffi::PyTuple_Check(document_ptr) != 0 {
          (false, ffi::PyTuple_GET_SIZE(document_ptr))
        } else {
          return false;
        };
      if token_len == 0 {
        continue;
      }
      let first_token = seq_get_item(document_ptr, document_is_list, 0);
      return raw_token_bytes(first_token).is_some();
    }
  }
  true
}

impl RMinHash {
  /// Attempts the fully parallel raw-read rho builder.
  ///
  /// Returns `Ok(None)` when the input shape does not qualify, in which case
  /// the caller falls back to the pipelined or streaming builder.
  pub(in crate::rminhash) fn try_build_rho_digest_matrix_raw_parallel(
    token_sets: &Bound<'_, PyAny>,
    num_perm: usize,
    seed: u64,
    probes: usize,
  ) -> PyResult<Option<RMinHashDigestMatrix>> {
    if rayon::current_num_threads() <= 1 || !raw_parallel_enabled() {
      return Ok(None);
    }
    let Some(rows) = fast_sequence_length(token_sets)? else {
      return Ok(None);
    };
    let config = DigestBuildConfig::from_env();
    if rows < config.doc_par_batch_size || u32::try_from(rows).is_err() {
      return Ok(None);
    }

    let object_ptr = token_sets.as_ptr();
    // SAFETY: type checks under the GIL.
    let outer_is_list = unsafe { ffi::PyList_Check(object_ptr) != 0 };
    let outer_is_tuple = unsafe { ffi::PyTuple_Check(object_ptr) != 0 };
    if !outer_is_list && !outer_is_tuple {
      return Ok(None);
    }
    // SAFETY: outer kind validated above; probing performs only raw reads.
    if !unsafe { corpus_looks_raw_readable(object_ptr, outer_is_list, rows) } {
      return Ok(None);
    }

    let sketch = RhoSketchConfig::from_env(num_perm, probes);
    let sparse_verify_perm = sketch.sparse_verify_perm;
    let sparse_occupancy_threshold = sketch.sparse_occupancy_threshold;
    let sig_pairs =
      Self::sparse_verify_signature_pairs(seed, sparse_verify_perm);

    let matrix_len = checked_len_mul(rows, num_perm, "rho matrix")?;
    let mut matrix_storage: Vec<MaybeUninit<u32>> =
      Vec::with_capacity(matrix_len);
    // SAFETY: `MaybeUninit<u32>` requires no initialization; every row is
    // filled by `init_row_empty` in the parallel pass below.
    unsafe { matrix_storage.set_len(matrix_len) };
    let mut non_empty_counts = vec![0u16; rows];
    let mut source_token_counts = vec![0u16; rows];
    let sparse_sig_len =
      checked_len_mul(rows, sparse_verify_perm, "rho sparse verify")?;
    let mut sparse_verify_signatures = vec![u32::MAX; sparse_sig_len];
    let mut sparse_verify_active = vec![0u8; rows];

    let ctx = RawRowContext {
      outer: RawPyPtr(object_ptr),
      outer_is_list,
      seed,
      num_perm,
      num_perm_u64: num_perm as u64,
      is_power_of_two: num_perm.is_power_of_two(),
      sketch,
    };

    let fallback_rows: Vec<u32> = if sparse_verify_perm == 0 {
      matrix_storage
        .par_chunks_exact_mut(num_perm)
        .zip(non_empty_counts.par_iter_mut())
        .zip(source_token_counts.par_iter_mut())
        .enumerate()
        .map_init(
          || Vec::<u64>::with_capacity(64),
          |mixed_values,
           (row_index, ((row_uninit, non_empty_out), source_out))| {
            let row = init_row_empty(row_uninit);
            // SAFETY: see module docs; the GIL thread inside this rayon scope
            // keeps every object alive and unmutated.
            let non_empty = unsafe {
              sketch_raw_row(&ctx, row_index, row, mixed_values, source_out)
            };
            non_empty.map_or_else(
              || Some(low_u32_from_row_index(row_index)),
              |count| {
                *non_empty_out = saturating_u16(count);
                None
              },
            )
          },
        )
        .flatten_iter()
        .collect()
    } else {
      matrix_storage
        .par_chunks_exact_mut(num_perm)
        .zip(non_empty_counts.par_iter_mut())
        .zip(source_token_counts.par_iter_mut())
        .zip(sparse_verify_active.par_iter_mut())
        .zip(sparse_verify_signatures.par_chunks_exact_mut(sparse_verify_perm))
        .enumerate()
        .map_init(
          || Vec::<u64>::with_capacity(64),
          |mixed_values,
           (
            row_index,
            (
              (((row_uninit, non_empty_out), source_out), sparse_active_out),
              signature_row,
            ),
          )| {
            let row = init_row_empty(row_uninit);
            // SAFETY: see module docs.
            let non_empty = unsafe {
              sketch_raw_row(&ctx, row_index, row, mixed_values, source_out)
            };
            let Some(count) = non_empty else {
              return Some(low_u32_from_row_index(row_index));
            };
            *non_empty_out = saturating_u16(count);
            let is_sparse = count < sparse_occupancy_threshold;
            *sparse_active_out = u8::from(is_sparse);
            if is_sparse {
              Self::compute_sparse_verify_signature_into(
                signature_row,
                mixed_values,
                &sig_pairs,
              );
            }
            None
          },
        )
        .flatten_iter()
        .collect()
    };

    let mut matrix_data = {
      let mut storage = std::mem::ManuallyDrop::new(matrix_storage);
      let ptr = storage.as_mut_ptr().cast::<u32>();
      let (len, capacity) = (storage.len(), storage.capacity());
      // SAFETY: same allocation and layout; every row was initialized by
      // `init_row_empty` (fallback rows stay all EMPTY_BUCKET).
      unsafe { Vec::from_raw_parts(ptr, len, capacity) }
    };

    if !fallback_rows.is_empty() {
      Self::sketch_fallback_rows_on_gil_thread(
        token_sets.py(),
        &ctx,
        &fallback_rows,
        &mut matrix_data,
        &mut non_empty_counts,
        &mut source_token_counts,
        &mut sparse_verify_active,
        &mut sparse_verify_signatures,
        &sig_pairs,
      )?;
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

  /// Re-sketches rows the raw pass could not read, using the generic
  /// GIL-thread extraction path. Matches the streaming builder row-for-row.
  #[allow(clippy::too_many_arguments)]
  fn sketch_fallback_rows_on_gil_thread(
    py: Python<'_>,
    ctx: &RawRowContext,
    fallback_rows: &[u32],
    matrix_data: &mut [u32],
    non_empty_counts: &mut [u16],
    source_token_counts: &mut [u16],
    sparse_verify_active: &mut [u8],
    sparse_verify_signatures: &mut [u32],
    sig_pairs: &[(u64, u64)],
  ) -> PyResult<()> {
    let sketch = &ctx.sketch;
    let mut token_hashes = Vec::new();
    let mut mixed_values = Vec::new();
    for &row_u32 in fallback_rows {
      let row_index = row_u32 as usize;
      // SAFETY: outer kind/index validated during the parallel pass.
      let document_ptr =
        unsafe { seq_get_item(ctx.outer.0, ctx.outer_is_list, row_index) };
      // SAFETY: borrowed pointer is valid under the GIL.
      let document =
        unsafe { Bound::<'_, PyAny>::from_borrowed_ptr(py, document_ptr) };

      let row_start = row_index * ctx.num_perm;
      let row = &mut matrix_data[row_start..row_start + ctx.num_perm];
      let source_token_count = fast_sequence_length(&document)?;
      let row_token_budget = rho_adaptive_token_budget_for_row(
        source_token_count,
        sketch.default_token_budget,
        sketch.has_token_budget_override,
        sketch.medium_token_threshold,
        sketch.medium_token_budget,
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
        sketch.probes,
        row_source_token_count,
        ctx.num_perm,
      );
      Self::compute_rho_digest_from_token_hashes_into(
        row,
        &token_hashes,
        ctx.seed,
        row_probes,
        row_token_budget,
        sketch.densify_enabled,
        &mut mixed_values,
      );
      let non_empty_count = Self::count_non_empty_buckets(row);
      non_empty_counts[row_index] = saturating_u16(non_empty_count);
      source_token_counts[row_index] = saturating_u16(row_source_token_count);
      if sketch.sparse_verify_perm > 0 {
        let is_sparse = non_empty_count < sketch.sparse_occupancy_threshold;
        sparse_verify_active[row_index] = u8::from(is_sparse);
        let signature_start = row_index * sketch.sparse_verify_perm;
        let signature_row = &mut sparse_verify_signatures
          [signature_start..signature_start + sketch.sparse_verify_perm];
        if is_sparse {
          Self::compute_sparse_verify_signature_into(
            signature_row,
            &mixed_values,
            sig_pairs,
          );
        } else {
          signature_row.fill(u32::MAX);
        }
      }
    }
    Ok(())
  }
}

#[inline]
fn low_u32_from_row_index(row_index: usize) -> u32 {
  u32::try_from(row_index).unwrap_or(u32::MAX)
}
