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

const RAW_PROBE_ROWS: usize = 8;

fn raw_parallel_enabled() -> bool {
  std::env::var("RENSA_RHO_RAW_PARALLEL")
    .ok()
    .is_none_or(|value| value != "0")
}

#[derive(Clone, Copy)]
struct RawPyPtr(*mut ffi::PyObject);

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
  unsafe {
    if is_list {
      ffi::PyList_GET_ITEM(sequence_ptr, index_ssize)
    } else {
      ffi::PyTuple_GET_ITEM(sequence_ptr, index_ssize)
    }
  }
}

#[inline]
#[allow(clippy::cast_sign_loss)]
unsafe fn raw_token_bytes<'bytes>(
  item_ptr: *mut ffi::PyObject,
) -> Option<&'bytes [u8]> {
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

#[inline]
const fn init_row_empty(row: &mut [MaybeUninit<u32>]) -> &mut [u32] {
  let ptr = row.as_mut_ptr().cast::<u32>();
  unsafe {
    std::ptr::write_bytes(ptr, 0xFF, row.len());
    std::slice::from_raw_parts_mut(ptr, row.len())
  }
}

#[allow(clippy::cast_sign_loss)]
unsafe fn sketch_raw_row(
  ctx: &RawRowContext,
  row_index: usize,
  row: &mut [u32],
  mixed_values: &mut Vec<u64>,
  source_out: &mut u16,
) -> Option<usize> {
  mixed_values.clear();
  let document_ptr =
    unsafe { seq_get_item(ctx.outer.0, ctx.outer_is_list, row_index) };
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
    let item_ptr =
      unsafe { seq_get_item(document_ptr, document_is_list, index) };
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

unsafe fn corpus_looks_raw_readable(
  outer: *mut ffi::PyObject,
  outer_is_list: bool,
  rows: usize,
) -> bool {
  for row_index in 0..rows.min(RAW_PROBE_ROWS) {
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
    let outer_is_list = unsafe { ffi::PyList_Check(object_ptr) != 0 };
    let outer_is_tuple = unsafe { ffi::PyTuple_Check(object_ptr) != 0 };
    if !outer_is_list && !outer_is_tuple {
      return Ok(None);
    }
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
      let document_ptr =
        unsafe { seq_get_item(ctx.outer.0, ctx.outer_is_list, row_index) };
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
