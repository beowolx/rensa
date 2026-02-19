use crate::py_input::extend_prehashed_token_values_from_document;
use crate::rminhash::cache::digest_cache_key_env;
use crate::rminhash::permutation_cache::AdaptivePermutationCache;
use crate::rminhash::send_ptr::SendPtr;
use crate::rminhash::{
  DigestBuildConfig, RMinHash, RMinHashDigestMatrix, FLAT_ROW_OFFSETS_ERROR,
  FLAT_ROW_OFFSET_TYPE_ERROR,
};
use crate::simd::dispatch::PermutationSoA;
use pyo3::buffer::{Element, PyBuffer};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyIterator, PyList, PyTuple};
use rayon::prelude::*;
use std::sync::mpsc;
use std::thread;

fn new_permutation_cache(
  config: DigestBuildConfig,
) -> Option<AdaptivePermutationCache> {
  if config.max_perm_cache_hashes == 0 {
    return None;
  }
  Some(AdaptivePermutationCache::new(
    config.perm_cache_min_frequency,
    config.max_perm_cache_hashes,
  ))
}

struct DigestChunkJob {
  flat: Vec<u64>,
  ranges: Vec<(usize, usize)>,
  output_ptr: SendPtr<u32>,
  output_len: usize,
}

#[derive(Clone, Copy)]
struct DigestComputeContext<'a> {
  num_perm: usize,
  permutations: &'a [(u64, u64)],
  permutations_soa: &'a PermutationSoA,
  config: DigestBuildConfig,
}

impl RMinHash {
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
    job: &DigestChunkJob,
    num_perm: usize,
    permutations: &[(u64, u64)],
    permutations_soa: &PermutationSoA,
    config: DigestBuildConfig,
    permutation_cache: Option<&mut AdaptivePermutationCache>,
  ) {
    let row_count = job.ranges.len();
    let data = unsafe {
      std::slice::from_raw_parts_mut(job.output_ptr.as_ptr(), job.output_len)
    };
    let use_parallel = row_count >= config.doc_par_batch_size
      && rayon::current_num_threads() > 1;
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
      let mut misses = Vec::new();
      for (row, &(start, end)) in
        data.chunks_exact_mut(num_perm).zip(job.ranges.iter())
      {
        Self::apply_token_hashes_with_cache(
          row,
          &job.flat[start..end],
          permutations,
          permutations_soa,
          cache,
          &mut misses,
        );
      }
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
  }

  fn flush_digest_chunk(
    token_hashes_chunk: &mut Vec<u64>,
    token_hash_ranges: &mut Vec<(usize, usize)>,
    matrix_data: &mut Vec<u32>,
    context: DigestComputeContext<'_>,
    permutation_cache: Option<&mut AdaptivePermutationCache>,
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
    matrix_data.resize(matrix_start + rows * context.num_perm, u32::MAX);

    let job = DigestChunkJob {
      flat,
      ranges,
      output_ptr: SendPtr::new(matrix_data[matrix_start..].as_mut_ptr()),
      output_len: rows * context.num_perm,
    };
    Python::attach(|py| {
      py.detach(|| {
        Self::compute_digest_chunk(
          &job,
          context.num_perm,
          context.permutations,
          context.permutations_soa,
          context.config,
          permutation_cache,
        );
      });
    });

    *token_hashes_chunk = Vec::with_capacity(flat_capacity);
    *token_hash_ranges = Vec::with_capacity(ranges_capacity);
  }

  pub(in crate::rminhash) fn build_token_hash_rows(
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

  pub(in crate::rminhash) fn parse_flat_token_hashes(
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

  pub(in crate::rminhash) fn parse_row_offsets(
    row_offsets: &Bound<'_, PyAny>,
  ) -> PyResult<Vec<usize>> {
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

  pub(in crate::rminhash) fn validate_flat_row_offsets(
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

  pub(in crate::rminhash) fn build_digest_matrix_from_flat_token_hashes(
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
    let mut matrix_data = vec![u32::MAX; rows.saturating_mul(num_perm)];

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
          let mut permutation_cache = new_permutation_cache(config);
          if let Some(cache) = permutation_cache.as_mut() {
            let mut misses = Vec::new();
            for (row_index, row) in
              matrix_data.chunks_exact_mut(num_perm).enumerate()
            {
              let start = row_offsets[row_index];
              let end = row_offsets[row_index + 1];
              Self::apply_token_hashes_with_cache(
                row,
                &token_hashes[start..end],
                &permutations,
                &permutations_soa,
                cache,
                &mut misses,
              );
            }
          } else {
            for (row_index, row) in
              matrix_data.chunks_exact_mut(num_perm).enumerate()
            {
              let start = row_offsets[row_index];
              let end = row_offsets[row_index + 1];
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

    Ok(RMinHashDigestMatrix {
      num_perm,
      rows,
      data: matrix_data,
      rho_sidecar: None,
    })
  }

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
    let mut matrix_data = vec![u32::MAX; rows.saturating_mul(num_perm)];
    let mut token_hashes_chunk = Vec::new();
    let mut token_hash_ranges = Vec::with_capacity(config.doc_chunk_size);
    let mut chunk_row_start = 0usize;

    if config.pipeline_queue_cap == 0 {
      let mut permutation_cache = new_permutation_cache(config);
      for document in documents {
        let start = token_hashes_chunk.len();
        document_hasher(&document, &mut token_hashes_chunk)?;
        let end = token_hashes_chunk.len();
        token_hash_ranges.push((start, end));

        if token_hash_ranges.len() == config.doc_chunk_size {
          let flat = std::mem::take(&mut token_hashes_chunk);
          let flat_capacity = flat.capacity();
          let ranges = std::mem::take(&mut token_hash_ranges);
          let ranges_capacity = ranges.capacity();
          let row_count = ranges.len();
          let output_start = chunk_row_start * num_perm;
          let job = DigestChunkJob {
            flat,
            ranges,
            output_ptr: SendPtr::new(matrix_data[output_start..].as_mut_ptr()),
            output_len: row_count * num_perm,
          };
          Self::compute_digest_chunk(
            &job,
            num_perm,
            permutations,
            permutations_soa,
            config,
            permutation_cache.as_mut(),
          );
          chunk_row_start += row_count;
          token_hashes_chunk = Vec::with_capacity(flat_capacity);
          token_hash_ranges = Vec::with_capacity(ranges_capacity);
        }
      }

      if !token_hash_ranges.is_empty() {
        let flat = std::mem::take(&mut token_hashes_chunk);
        let ranges = std::mem::take(&mut token_hash_ranges);
        let row_count = ranges.len();
        let output_start = chunk_row_start * num_perm;
        let job = DigestChunkJob {
          flat,
          ranges,
          output_ptr: SendPtr::new(matrix_data[output_start..].as_mut_ptr()),
          output_len: row_count * num_perm,
        };
        Self::compute_digest_chunk(
          &job,
          num_perm,
          permutations,
          permutations_soa,
          config,
          permutation_cache.as_mut(),
        );
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
    let permutations_owned = permutations.to_vec();
    let permutations_soa_owned = permutations_soa.clone();
    let worker_config = config;
    let worker = thread::Builder::new()
      .name(String::from("rensa-digest-worker"))
      .spawn(move || {
        let mut permutation_cache = new_permutation_cache(worker_config);
        while let Ok(job) = job_rx.recv() {
          Self::compute_digest_chunk(
            &job,
            num_perm,
            &permutations_owned,
            &permutations_soa_owned,
            worker_config,
            permutation_cache.as_mut(),
          );
        }
      })
      .map_err(|err| {
        PyValueError::new_err(format!("failed to spawn digest worker: {err}"))
      })?;

    let mut extraction_error: Option<PyErr> = None;
    for document in documents {
      let start = token_hashes_chunk.len();
      if let Err(err) = document_hasher(&document, &mut token_hashes_chunk) {
        extraction_error = Some(err);
        break;
      }
      let end = token_hashes_chunk.len();
      token_hash_ranges.push((start, end));

      if token_hash_ranges.len() == config.doc_chunk_size {
        let flat = std::mem::take(&mut token_hashes_chunk);
        let flat_capacity = flat.capacity();
        let ranges = std::mem::take(&mut token_hash_ranges);
        let ranges_capacity = ranges.capacity();
        let row_count = ranges.len();
        let output_start = chunk_row_start * num_perm;
        let job = DigestChunkJob {
          flat,
          ranges,
          output_ptr: SendPtr::new(matrix_data[output_start..].as_mut_ptr()),
          output_len: row_count * num_perm,
        };
        if let Err(err) = job_tx.send(job) {
          extraction_error = Some(PyValueError::new_err(format!(
            "digest worker stopped unexpectedly: {err}"
          )));
          break;
        }
        chunk_row_start += row_count;
        token_hashes_chunk = Vec::with_capacity(flat_capacity);
        token_hash_ranges = Vec::with_capacity(ranges_capacity);
      }
    }

    if extraction_error.is_none() && !token_hash_ranges.is_empty() {
      let flat = std::mem::take(&mut token_hashes_chunk);
      let ranges = std::mem::take(&mut token_hash_ranges);
      let row_count = ranges.len();
      let output_start = chunk_row_start * num_perm;
      let job = DigestChunkJob {
        flat,
        ranges,
        output_ptr: SendPtr::new(matrix_data[output_start..].as_mut_ptr()),
        output_len: row_count * num_perm,
      };
      if let Err(err) = job_tx.send(job) {
        extraction_error = Some(PyValueError::new_err(format!(
          "digest worker stopped unexpectedly: {err}"
        )));
      }
    }

    drop(job_tx);

    if worker.join().is_err() {
      return Err(PyValueError::new_err("digest worker thread panicked"));
    }

    if let Some(err) = extraction_error {
      return Err(err);
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
    let mut permutation_cache = new_permutation_cache(config);
    Self::for_each_document(token_sets, |document| {
      let start = token_hashes_chunk.len();
      document_hasher(&document, &mut token_hashes_chunk)?;
      let end = token_hashes_chunk.len();
      token_hash_ranges.push((start, end));

      if token_hash_ranges.len() == config.doc_chunk_size {
        Self::flush_digest_chunk(
          &mut token_hashes_chunk,
          &mut token_hash_ranges,
          &mut matrix_data,
          context,
          permutation_cache.as_mut(),
        );
      }
      Ok(())
    })?;

    Self::flush_digest_chunk(
      &mut token_hashes_chunk,
      &mut token_hash_ranges,
      &mut matrix_data,
      context,
      permutation_cache.as_mut(),
    );

    let rows = if num_perm == 0 {
      0
    } else {
      matrix_data.len() / num_perm
    };

    Ok(RMinHashDigestMatrix {
      num_perm,
      rows,
      data: matrix_data,
      rho_sidecar: None,
    })
  }

  pub(in crate::rminhash) fn build_digest_matrix_from_token_sets(
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

  pub(in crate::rminhash) fn digest_rows_from_matrix(
    matrix: &RMinHashDigestMatrix,
  ) -> Vec<Vec<u32>> {
    matrix
      .data
      .chunks_exact(matrix.num_perm)
      .map(std::borrow::ToOwned::to_owned)
      .collect()
  }

  pub(in crate::rminhash) fn from_matrix(
    matrix: &RMinHashDigestMatrix,
    seed: u64,
  ) -> Vec<Self> {
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
}
