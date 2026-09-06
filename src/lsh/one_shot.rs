use crate::lsh::{RMinHashLSH, FX_FINISH_ROTATE};
use crate::rminhash::RMinHashDigestMatrix;
use crate::utils::{calculate_band_hash, multiply_mix};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::collections::hash_map::Entry;
use std::sync::atomic::{AtomicU32, Ordering};

#[cfg(target_pointer_width = "64")]
const USIZE_MASK_U64: u64 = u64::MAX;
#[cfg(target_pointer_width = "32")]
const USIZE_MASK_U64: u64 = u32::MAX as u64;

const MIN_PARALLEL_BAND_ROWS: usize = 2048;
const NO_ROW: u32 = u32::MAX;
const RESCUE_COLLIDED_BIT: u64 = 1 << 32;

#[inline]
fn u64_to_usize_lowbits(value: u64) -> usize {
  let masked = value & USIZE_MASK_U64;
  usize::try_from(masked).unwrap_or_default()
}

#[inline]
const fn usize_to_u64_lowbits(value: usize) -> u64 {
  #[cfg(target_pointer_width = "64")]
  {
    value as u64
  }
  #[cfg(target_pointer_width = "32")]
  {
    value as u32 as u64
  }
}

struct SparseVerifyConfig {
  enabled: bool,
  threshold: f64,
  max_candidates: usize,
}

struct RecallRescueConfig {
  enabled: bool,
  min_tokens: usize,
  max_tokens: usize,
  required_band_matches: usize,
}

struct BandFoldingConfig {
  rho_band_fold: usize,
  effective_num_bands: usize,
  effective_band_size: usize,
}

struct PrecomputedBandHashes {
  hashes: Vec<u64>,
  fold_k_pow: usize,
}

struct BandScanContext<'a> {
  digest_matrix: &'a RMinHashDigestMatrix,
  effective_band_size: usize,
  rho_band_fold: usize,
  precomputed: Option<&'a PrecomputedBandHashes>,
  has_existing_entries: bool,
  required_band_matches: &'a [u32],
  sparse_verify: &'a SparseVerifyConfig,
}

struct BandScanScratch {
  tail_row_by_hash: FxHashMap<u64, u32>,
  chain_prev: Vec<u32>,
  collided_hashes: Vec<u64>,
  group_rows: Vec<u32>,
  subgroup_keys: Vec<(u64, u32)>,
  subgroup_rows: Vec<u32>,
}

impl BandScanScratch {
  fn with_row_capacity(rows: usize) -> Self {
    let mut tail_row_by_hash = FxHashMap::default();
    tail_row_by_hash.reserve(rows);
    Self {
      tail_row_by_hash,
      chain_prev: vec![NO_ROW; rows],
      collided_hashes: Vec::new(),
      group_rows: Vec::new(),
      subgroup_keys: Vec::new(),
      subgroup_rows: Vec::new(),
    }
  }

  fn reset(&mut self) {
    self.tail_row_by_hash.clear();
    self.chain_prev.fill(NO_ROW);
    self.collided_hashes.clear();
  }
}

struct GroupVerifyContext<'a> {
  digest_matrix: &'a RMinHashDigestMatrix,
  required_band_matches: &'a [u32],
  sparse_verify: &'a SparseVerifyConfig,
}

struct RawBandHashes {
  num_bands: usize,
  rows: usize,
  hashes: Vec<u64>,
}

impl RawBandHashes {
  #[inline]
  fn get(&self, row: usize, band_idx: usize) -> u64 {
    self.hashes[row * self.num_bands + band_idx]
  }
}

#[inline]
fn rescue_state_first_row(state: u64) -> usize {
  usize::try_from(state & u64::from(u32::MAX)).unwrap_or_default()
}

fn atomic_counters(rows: usize) -> Vec<AtomicU32> {
  let mut counters = Vec::with_capacity(rows);
  counters.resize_with(rows, AtomicU32::default);
  counters
}

fn grouped_scan_enabled() -> bool {
  std::env::var("RENSA_LSH_GROUPED_SCAN")
    .ok()
    .is_none_or(|value| value != "0")
}

fn chain_scan_band(
  raw: &RawBandHashes,
  band_idx: usize,
  scratch: &mut BandScanScratch,
  raw_match_counts: Option<&[AtomicU32]>,
) {
  scratch.reset();
  for row_index in 0..raw.rows {
    let band_hash = raw.get(row_index, band_idx);
    #[allow(clippy::cast_possible_truncation)]
    let row_index_u32 = row_index as u32;
    match scratch.tail_row_by_hash.entry(band_hash) {
      Entry::Vacant(entry) => {
        entry.insert(row_index_u32);
      }
      Entry::Occupied(mut entry) => {
        let prev_tail = *entry.get();
        if scratch.chain_prev[prev_tail as usize] == NO_ROW {
          scratch.collided_hashes.push(band_hash);
          if let Some(counts) = raw_match_counts {
            counts[prev_tail as usize].fetch_add(1, Ordering::Relaxed);
          }
        }
        if let Some(counts) = raw_match_counts {
          counts[row_index].fetch_add(1, Ordering::Relaxed);
        }
        scratch.chain_prev[row_index] = prev_tail;
        *entry.get_mut() = row_index_u32;
      }
    }
  }
}

fn count_scan_band(
  raw: &RawBandHashes,
  band_idx: usize,
  raw_match_counts: &[AtomicU32],
  bucket_state_by_hash: &mut FxHashMap<u64, u64>,
) {
  bucket_state_by_hash.clear();
  for row_index in 0..raw.rows {
    let band_hash = raw.get(row_index, band_idx);
    match bucket_state_by_hash.entry(band_hash) {
      Entry::Vacant(entry) => {
        entry.insert(usize_to_u64_lowbits(row_index));
      }
      Entry::Occupied(mut entry) => {
        let state = entry.get_mut();
        if *state & RESCUE_COLLIDED_BIT == 0 {
          let first_row = rescue_state_first_row(*state);
          *state |= RESCUE_COLLIDED_BIT;
          raw_match_counts[first_row].fetch_add(1, Ordering::Relaxed);
        }
        raw_match_counts[row_index].fetch_add(1, Ordering::Relaxed);
      }
    }
  }
}

#[inline]
fn window_rest_key(
  raw: &RawBandHashes,
  window_first_band: usize,
  fold: usize,
  row: usize,
) -> u64 {
  if fold == 2 {
    return raw.get(row, window_first_band + 1);
  }
  let mut key = 0xa076_1d64_78bd_642f_u64;
  for offset in 1..fold {
    key = multiply_mix(
      key ^ raw.get(row, window_first_band + offset),
      0xe703_7ed1_a0b4_28db,
    );
  }
  key
}

#[inline]
fn use_parallel_bands(band_count: usize, rows: usize) -> bool {
  band_count > 1
    && rows >= MIN_PARALLEL_BAND_ROWS
    && rayon::current_num_threads() > 1
}

impl RMinHashLSH {
  pub(in crate::lsh) fn query_duplicate_flags_matrix_one_shot_inner(
    &mut self,
    digest_matrix: &RMinHashDigestMatrix,
  ) -> PyResult<Vec<bool>> {
    self.ensure_digest_len(digest_matrix.num_perm())?;

    let rows = digest_matrix.rows();
    if u32::try_from(rows).is_err() {
      return Err(PyValueError::new_err(format!(
        "digest matrix has too many rows for one-shot dedup: {rows}"
      )));
    }
    let has_existing_entries =
      self.hash_tables.iter().any(|table| !table.is_empty());
    let rho_sidecar_present =
      digest_matrix.rho_sparse_occupancy_threshold().is_some();

    let folding = Self::band_folding_config(
      self,
      rho_sidecar_present,
      has_existing_entries,
    );

    let sparse_occupancy_threshold = digest_matrix
      .rho_sparse_occupancy_threshold()
      .unwrap_or_else(|| Self::rho_sparse_occupancy_threshold(self.num_perm));
    let sparse_required_band_matches =
      Self::rho_sparse_required_band_matches(folding.effective_num_bands);
    let (required_band_matches, any_sparse_rows) = Self::required_band_matches(
      digest_matrix,
      rows,
      sparse_occupancy_threshold,
      sparse_required_band_matches,
    );

    let sparse_verify = Self::sparse_verify_config(digest_matrix);
    let recall_rescue = Self::recall_rescue_config(
      self,
      rho_sidecar_present,
      folding.rho_band_fold,
      has_existing_entries,
    );

    if !any_sparse_rows && !sparse_verify.enabled && !recall_rescue.enabled {
      let flags = Self::simple_one_shot_flags(
        self,
        digest_matrix,
        rows,
        folding.effective_num_bands,
        folding.effective_band_size,
        folding.rho_band_fold,
        has_existing_entries,
      );
      self.last_one_shot_sparse_verify_checks = 0;
      self.last_one_shot_sparse_verify_passes = 0;
      return Ok(flags);
    }

    if folding.rho_band_fold > 1 && grouped_scan_enabled() {
      if let Some((flags, checks, passes)) = self.grouped_one_shot_flags(
        digest_matrix,
        rows,
        &folding,
        &required_band_matches,
        &sparse_verify,
        &recall_rescue,
      ) {
        self.last_one_shot_sparse_verify_checks = checks;
        self.last_one_shot_sparse_verify_passes = passes;
        return Ok(flags);
      }
    }

    let precomputed = self.precomputed_band_hashes(
      digest_matrix,
      rows,
      recall_rescue.enabled
        && folding.rho_band_fold > 1
        && self.band_size % 4 == 0,
    );

    let band_match_counts = atomic_counters(rows);
    let mut sparse_verify_checks = 0usize;
    let mut sparse_verify_passes = 0usize;
    {
      let ctx = BandScanContext {
        digest_matrix,
        effective_band_size: folding.effective_band_size,
        rho_band_fold: folding.rho_band_fold,
        precomputed: precomputed.as_ref(),
        has_existing_entries,
        required_band_matches: &required_band_matches,
        sparse_verify: &sparse_verify,
      };
      let this = &*self;
      if use_parallel_bands(folding.effective_num_bands, rows) {
        let band_stats: Vec<(usize, usize)> = (0..folding.effective_num_bands)
          .into_par_iter()
          .map(|band_idx| {
            let mut scratch = BandScanScratch::with_row_capacity(rows);
            Self::scan_effective_band(
              this,
              &ctx,
              band_idx,
              &band_match_counts,
              &mut scratch,
            )
          })
          .collect();
        for (checks, passes) in band_stats {
          sparse_verify_checks = sparse_verify_checks.saturating_add(checks);
          sparse_verify_passes = sparse_verify_passes.saturating_add(passes);
        }
      } else {
        let mut scratch = BandScanScratch::with_row_capacity(rows);
        for band_idx in 0..folding.effective_num_bands {
          let (checks, passes) = Self::scan_effective_band(
            this,
            &ctx,
            band_idx,
            &band_match_counts,
            &mut scratch,
          );
          sparse_verify_checks = sparse_verify_checks.saturating_add(checks);
          sparse_verify_passes = sparse_verify_passes.saturating_add(passes);
        }
      }
    }

    if recall_rescue.enabled {
      Self::apply_recall_rescue(
        self,
        digest_matrix,
        rows,
        precomputed.as_ref(),
        &required_band_matches,
        &band_match_counts,
        &recall_rescue,
      );
    }

    let flags = band_match_counts
      .iter()
      .zip(required_band_matches.iter())
      .map(|(matches, required)| matches.load(Ordering::Relaxed) >= *required)
      .collect();

    self.last_one_shot_sparse_verify_checks = sparse_verify_checks;
    self.last_one_shot_sparse_verify_passes = sparse_verify_passes;

    Ok(flags)
  }

  fn band_folding_config(
    &self,
    rho_sidecar_present: bool,
    has_existing_entries: bool,
  ) -> BandFoldingConfig {
    let mut rho_band_fold = if rho_sidecar_present && !has_existing_entries {
      Self::rho_band_fold(self.num_bands)
    } else {
      1
    };
    if self.num_bands % rho_band_fold != 0 {
      rho_band_fold = 1;
    }

    BandFoldingConfig {
      rho_band_fold,
      effective_num_bands: self.num_bands / rho_band_fold,
      effective_band_size: self.band_size * rho_band_fold,
    }
  }

  fn required_band_matches(
    digest_matrix: &RMinHashDigestMatrix,
    rows: usize,
    sparse_occupancy_threshold: usize,
    sparse_required_band_matches: usize,
  ) -> (Vec<u32>, bool) {
    let sparse_required =
      u32::try_from(sparse_required_band_matches).unwrap_or(u32::MAX);
    let mut any_sparse_rows = false;
    let mut required_band_matches = vec![1u32; rows];
    for (row_index, required) in required_band_matches.iter_mut().enumerate() {
      if digest_matrix
        .rho_non_empty_count(row_index)
        .is_some_and(|count| count < sparse_occupancy_threshold)
      {
        *required = sparse_required;
        any_sparse_rows = true;
      }
    }
    (required_band_matches, any_sparse_rows)
  }

  fn sparse_verify_config(
    digest_matrix: &RMinHashDigestMatrix,
  ) -> SparseVerifyConfig {
    let enabled = Self::rho_sparse_verify_enabled()
      && digest_matrix.rho_sparse_verify_perm() > 0;
    SparseVerifyConfig {
      enabled,
      threshold: Self::rho_sparse_verify_threshold(),
      max_candidates: Self::rho_sparse_verify_max_candidates(),
    }
  }

  fn recall_rescue_config(
    &self,
    rho_sidecar_present: bool,
    rho_band_fold: usize,
    has_existing_entries: bool,
  ) -> RecallRescueConfig {
    let enabled = rho_sidecar_present
      && rho_band_fold > 1
      && !has_existing_entries
      && Self::rho_recall_rescue_enabled();
    let min_tokens = Self::rho_recall_rescue_min_tokens();
    let max_tokens = Self::rho_recall_rescue_max_tokens().max(min_tokens);
    RecallRescueConfig {
      enabled,
      min_tokens,
      max_tokens,
      required_band_matches: Self::rho_recall_rescue_required_band_matches(
        self.num_bands,
      ),
    }
  }

  fn grouped_one_shot_flags(
    &self,
    digest_matrix: &RMinHashDigestMatrix,
    rows: usize,
    folding: &BandFoldingConfig,
    required_band_matches: &[u32],
    sparse_verify: &SparseVerifyConfig,
    recall_rescue: &RecallRescueConfig,
  ) -> Option<(Vec<bool>, usize, usize)> {
    if rows == 0 {
      return Some((Vec::new(), 0, 0));
    }
    let fold = folding.rho_band_fold;
    let effective_num_bands = folding.effective_num_bands;
    let raw = self.compute_raw_band_hashes(digest_matrix, rows)?;

    let band_match_counts = atomic_counters(rows);
    let needs_rescue = recall_rescue.enabled
      && (0..rows).any(|row| {
        required_band_matches[row] == 1
          && digest_matrix
            .rho_source_token_count(row)
            .is_some_and(|count| {
              count >= recall_rescue.min_tokens
                && count <= recall_rescue.max_tokens
            })
      });
    let raw_match_counts = if needs_rescue {
      atomic_counters(rows)
    } else {
      Vec::new()
    };
    let raw_counts_ref = needs_rescue.then_some(raw_match_counts.as_slice());

    let rest_bands: Vec<usize> = (0..self.num_bands)
      .filter(|band_idx| band_idx % fold != 0)
      .collect();

    let group_ctx = GroupVerifyContext {
      digest_matrix,
      required_band_matches,
      sparse_verify,
    };
    let mut sparse_verify_checks = 0usize;
    let mut sparse_verify_passes = 0usize;
    let window_stats: Vec<(usize, usize)> =
      if use_parallel_bands(self.num_bands, rows) {
        let (window_stats, ()) = rayon::join(
          || {
            (0..effective_num_bands)
              .into_par_iter()
              .map(|window| {
                let mut scratch = BandScanScratch::with_row_capacity(rows);
                chain_scan_band(
                  &raw,
                  window * fold,
                  &mut scratch,
                  raw_counts_ref,
                );
                Self::refine_window_groups(
                  &group_ctx,
                  &raw,
                  window * fold,
                  fold,
                  &mut scratch,
                  &band_match_counts,
                )
              })
              .collect()
          },
          || {
            if let Some(raw_counts) = raw_counts_ref {
              rest_bands.par_iter().for_each(|&band_idx| {
                let mut bucket_state: FxHashMap<u64, u64> =
                  FxHashMap::default();
                bucket_state.reserve(rows);
                count_scan_band(&raw, band_idx, raw_counts, &mut bucket_state);
              });
            }
          },
        );
        window_stats
      } else {
        let mut scratch = BandScanScratch::with_row_capacity(rows);
        let stats = (0..effective_num_bands)
          .map(|window| {
            chain_scan_band(&raw, window * fold, &mut scratch, raw_counts_ref);
            Self::refine_window_groups(
              &group_ctx,
              &raw,
              window * fold,
              fold,
              &mut scratch,
              &band_match_counts,
            )
          })
          .collect();
        if let Some(raw_counts) = raw_counts_ref {
          let mut bucket_state: FxHashMap<u64, u64> = FxHashMap::default();
          bucket_state.reserve(rows);
          for &band_idx in &rest_bands {
            count_scan_band(&raw, band_idx, raw_counts, &mut bucket_state);
          }
        }
        stats
      };
    for (checks, passes) in window_stats {
      sparse_verify_checks = sparse_verify_checks.saturating_add(checks);
      sparse_verify_passes = sparse_verify_passes.saturating_add(passes);
    }

    if needs_rescue {
      let required_matches =
        u32::try_from(recall_rescue.required_band_matches).unwrap_or(u32::MAX);
      for row_index in 0..rows {
        if band_match_counts[row_index].load(Ordering::Relaxed) != 0
          || required_band_matches[row_index] > 1
        {
          continue;
        }
        if !digest_matrix.rho_source_token_count(row_index).is_some_and(
          |token_count| {
            token_count >= recall_rescue.min_tokens
              && token_count <= recall_rescue.max_tokens
          },
        ) {
          continue;
        }
        if raw_match_counts[row_index].load(Ordering::Relaxed)
          >= required_matches
        {
          band_match_counts[row_index]
            .store(required_band_matches[row_index], Ordering::Relaxed);
        }
      }
    }

    let flags = band_match_counts
      .iter()
      .zip(required_band_matches.iter())
      .map(|(matches, required)| matches.load(Ordering::Relaxed) >= *required)
      .collect();
    Some((flags, sparse_verify_checks, sparse_verify_passes))
  }

  fn compute_raw_band_hashes(
    &self,
    digest_matrix: &RMinHashDigestMatrix,
    rows: usize,
  ) -> Option<RawBandHashes> {
    let num_bands = self.num_bands;
    let total = rows.checked_mul(num_bands)?;
    let band_size = self.band_size;
    let mut hashes = vec![0u64; total];
    let fill_row = |row_index: usize, out: &mut [u64]| {
      let row = digest_matrix.row(row_index);
      for (slot, band) in out.iter_mut().zip(row.chunks_exact(band_size)) {
        *slot = calculate_band_hash(band);
      }
    };
    if use_parallel_bands(num_bands, rows) {
      hashes
        .par_chunks_exact_mut(num_bands)
        .enumerate()
        .for_each(|(row_index, out)| fill_row(row_index, out));
    } else {
      for (row_index, out) in hashes.chunks_exact_mut(num_bands).enumerate() {
        fill_row(row_index, out);
      }
    }
    Some(RawBandHashes {
      num_bands,
      rows,
      hashes,
    })
  }

  fn refine_window_groups(
    group_ctx: &GroupVerifyContext<'_>,
    raw: &RawBandHashes,
    window_first_band: usize,
    fold: usize,
    scratch: &mut BandScanScratch,
    band_match_counts: &[AtomicU32],
  ) -> (usize, usize) {
    let mut sparse_verify_checks = 0usize;
    let mut sparse_verify_passes = 0usize;

    for group_index in 0..scratch.collided_hashes.len() {
      let band_hash = scratch.collided_hashes[group_index];
      let Some(&tail_row) = scratch.tail_row_by_hash.get(&band_hash) else {
        continue;
      };
      scratch.group_rows.clear();
      let mut cursor = tail_row;
      while cursor != NO_ROW {
        scratch.group_rows.push(cursor);
        cursor = scratch.chain_prev[cursor as usize];
      }
      scratch.group_rows.reverse();

      if scratch.group_rows.len() == 2 {
        let left = scratch.group_rows[0] as usize;
        let right = scratch.group_rows[1] as usize;
        if window_rest_key(raw, window_first_band, fold, left)
          == window_rest_key(raw, window_first_band, fold, right)
        {
          let (checks, passes) = Self::process_collision_group(
            group_ctx,
            &scratch.group_rows,
            band_match_counts,
          );
          sparse_verify_checks = sparse_verify_checks.saturating_add(checks);
          sparse_verify_passes = sparse_verify_passes.saturating_add(passes);
        }
        continue;
      }

      scratch.subgroup_keys.clear();
      for (position, &row) in scratch.group_rows.iter().enumerate() {
        #[allow(clippy::cast_possible_truncation)]
        let position_u32 = position as u32;
        scratch.subgroup_keys.push((
          window_rest_key(raw, window_first_band, fold, row as usize),
          position_u32,
        ));
      }
      scratch.subgroup_keys.sort_unstable();

      let mut start = 0usize;
      while start < scratch.subgroup_keys.len() {
        let key = scratch.subgroup_keys[start].0;
        let mut end = start + 1;
        while end < scratch.subgroup_keys.len()
          && scratch.subgroup_keys[end].0 == key
        {
          end += 1;
        }
        if end - start >= 2 {
          scratch.subgroup_rows.clear();
          for &(_, position) in &scratch.subgroup_keys[start..end] {
            scratch
              .subgroup_rows
              .push(scratch.group_rows[position as usize]);
          }
          let (checks, passes) = Self::process_collision_group(
            group_ctx,
            &scratch.subgroup_rows,
            band_match_counts,
          );
          sparse_verify_checks = sparse_verify_checks.saturating_add(checks);
          sparse_verify_passes = sparse_verify_passes.saturating_add(passes);
        }
        start = end;
      }
    }

    (sparse_verify_checks, sparse_verify_passes)
  }

  fn precomputed_band_hashes(
    &self,
    digest_matrix: &RMinHashDigestMatrix,
    rows: usize,
    enabled: bool,
  ) -> Option<PrecomputedBandHashes> {
    if !enabled {
      return None;
    }

    let raw = self.compute_raw_band_hashes(digest_matrix, rows)?;
    let steps = Self::fx_poly_steps(self.band_size);
    Some(PrecomputedBandHashes {
      hashes: raw.hashes,
      fold_k_pow: Self::fx_poly_k_pow(steps),
    })
  }

  fn simple_one_shot_flags(
    &self,
    digest_matrix: &RMinHashDigestMatrix,
    rows: usize,
    effective_num_bands: usize,
    effective_band_size: usize,
    rho_band_fold: usize,
    has_existing_entries: bool,
  ) -> Vec<bool> {
    let mut flags = vec![false; rows];
    let mut first_row_by_hash: FxHashMap<u64, usize> = FxHashMap::default();
    first_row_by_hash.reserve(rows);
    for band_idx in 0..effective_num_bands {
      let table = if rho_band_fold == 1 {
        self.hash_tables.get(band_idx)
      } else {
        None
      };

      first_row_by_hash.clear();
      for row_index in 0..rows {
        let row = digest_matrix.row(row_index);
        let start = band_idx * effective_band_size;
        let end = start + effective_band_size;
        let band_hash = calculate_band_hash(&row[start..end]);
        if has_existing_entries
          && table.is_some_and(|band_table| band_table.contains_key(&band_hash))
        {
          flags[row_index] = true;
        }
        match first_row_by_hash.entry(band_hash) {
          Entry::Vacant(entry) => {
            entry.insert(row_index);
          }
          Entry::Occupied(entry) => {
            flags[row_index] = true;
            flags[*entry.get()] = true;
          }
        }
      }
    }
    flags
  }

  fn scan_effective_band(
    &self,
    ctx: &BandScanContext<'_>,
    band_idx: usize,
    band_match_counts: &[AtomicU32],
    scratch: &mut BandScanScratch,
  ) -> (usize, usize) {
    let table = if ctx.rho_band_fold == 1 {
      self.hash_tables.get(band_idx)
    } else {
      None
    };
    scratch.reset();
    let mut sparse_verify_checks = 0usize;
    let mut sparse_verify_passes = 0usize;

    for (row_index, band_match_count) in band_match_counts.iter().enumerate() {
      let band_hash = Self::effective_band_hash(
        self,
        ctx.digest_matrix,
        row_index,
        band_idx,
        ctx.effective_band_size,
        ctx.rho_band_fold,
        ctx.precomputed,
      );

      if ctx.has_existing_entries
        && table.is_some_and(|band_table| band_table.contains_key(&band_hash))
      {
        band_match_count.fetch_add(1, Ordering::Relaxed);
        continue;
      }

      #[allow(clippy::cast_possible_truncation)]
      let row_index_u32 = row_index as u32;
      match scratch.tail_row_by_hash.entry(band_hash) {
        Entry::Vacant(entry) => {
          entry.insert(row_index_u32);
        }
        Entry::Occupied(mut entry) => {
          let prev_tail = *entry.get();
          if scratch.chain_prev[prev_tail as usize] == NO_ROW {
            scratch.collided_hashes.push(band_hash);
          }
          scratch.chain_prev[row_index] = prev_tail;
          *entry.get_mut() = row_index_u32;
        }
      }
    }

    let group_ctx = GroupVerifyContext {
      digest_matrix: ctx.digest_matrix,
      required_band_matches: ctx.required_band_matches,
      sparse_verify: ctx.sparse_verify,
    };
    for group_index in 0..scratch.collided_hashes.len() {
      let band_hash = scratch.collided_hashes[group_index];
      let Some(&tail_row) = scratch.tail_row_by_hash.get(&band_hash) else {
        continue;
      };
      scratch.group_rows.clear();
      let mut cursor = tail_row;
      while cursor != NO_ROW {
        scratch.group_rows.push(cursor);
        cursor = scratch.chain_prev[cursor as usize];
      }
      scratch.group_rows.reverse();

      let (checks, passes) = Self::process_collision_group(
        &group_ctx,
        &scratch.group_rows,
        band_match_counts,
      );
      sparse_verify_checks = sparse_verify_checks.saturating_add(checks);
      sparse_verify_passes = sparse_verify_passes.saturating_add(passes);
    }

    (sparse_verify_checks, sparse_verify_passes)
  }

  fn process_collision_group(
    ctx: &GroupVerifyContext<'_>,
    members: &[u32],
    band_match_counts: &[AtomicU32],
  ) -> (usize, usize) {
    let mut sparse_verify_checks = 0usize;
    let mut sparse_verify_passes = 0usize;

    if let &[left, right] = members {
      let left = left as usize;
      let right = right as usize;
      let needs_verify = ctx.sparse_verify.enabled
        && (ctx.required_band_matches[left] > 1
          || ctx.required_band_matches[right] > 1);
      let matched = if needs_verify {
        if ctx.sparse_verify.max_candidates == 0 {
          return (0, 0);
        }
        sparse_verify_checks = 2;
        let passed = Self::sparse_verify_pair_passes(
          ctx.digest_matrix,
          left,
          right,
          ctx.sparse_verify.threshold,
        );
        sparse_verify_passes = if passed { 2 } else { 0 };
        passed
      } else {
        true
      };
      if matched {
        band_match_counts[left].fetch_add(1, Ordering::Relaxed);
        band_match_counts[right].fetch_add(1, Ordering::Relaxed);
      }
      return (sparse_verify_checks, sparse_verify_passes);
    }

    for &row_index in members {
      let row_index = row_index as usize;
      let row_sparse = ctx.required_band_matches[row_index] > 1;
      let mut checked_candidates = 0usize;
      let mut matched = false;

      for &other_row in members {
        let other_row = other_row as usize;
        if other_row == row_index {
          continue;
        }
        let other_sparse = ctx.required_band_matches[other_row] > 1;
        let needs_sparse_verify =
          ctx.sparse_verify.enabled && (row_sparse || other_sparse);
        if !needs_sparse_verify {
          matched = true;
          break;
        }

        if checked_candidates >= ctx.sparse_verify.max_candidates {
          break;
        }
        checked_candidates += 1;
        sparse_verify_checks = sparse_verify_checks.saturating_add(1);
        if Self::sparse_verify_pair_passes(
          ctx.digest_matrix,
          row_index,
          other_row,
          ctx.sparse_verify.threshold,
        ) {
          sparse_verify_passes = sparse_verify_passes.saturating_add(1);
          matched = true;
          break;
        }
      }

      if matched {
        band_match_counts[row_index].fetch_add(1, Ordering::Relaxed);
      }
    }

    (sparse_verify_checks, sparse_verify_passes)
  }

  #[inline]
  fn sparse_verify_pair_passes(
    digest_matrix: &RMinHashDigestMatrix,
    left_row: usize,
    right_row: usize,
    threshold: f64,
  ) -> bool {
    match (
      digest_matrix.rho_sparse_verify_signature(left_row),
      digest_matrix.rho_sparse_verify_signature(right_row),
    ) {
      (Some(left), Some(right)) => {
        Self::sparse_verify_similarity(left, right) >= threshold
      }
      _ => true,
    }
  }

  #[inline]
  fn effective_band_hash(
    &self,
    digest_matrix: &RMinHashDigestMatrix,
    row_index: usize,
    band_idx: usize,
    effective_band_size: usize,
    rho_band_fold: usize,
    precomputed: Option<&PrecomputedBandHashes>,
  ) -> u64 {
    if rho_band_fold == 1 {
      let row = digest_matrix.row(row_index);
      let start = band_idx * effective_band_size;
      let end = start + effective_band_size;
      return calculate_band_hash(&row[start..end]);
    }

    if let Some(precomputed) = precomputed {
      let start_band = band_idx * rho_band_fold;
      let base_offset = row_index * self.num_bands + start_band;
      let mut state = u64_to_usize_lowbits(precomputed.hashes[base_offset])
        .rotate_right(FX_FINISH_ROTATE);
      for offset in 1..rho_band_fold {
        let next_state =
          u64_to_usize_lowbits(precomputed.hashes[base_offset + offset])
            .rotate_right(FX_FINISH_ROTATE);
        state = state
          .wrapping_mul(precomputed.fold_k_pow)
          .wrapping_add(next_state);
      }
      return usize_to_u64_lowbits(state.rotate_left(FX_FINISH_ROTATE));
    }

    let row = digest_matrix.row(row_index);
    let start = band_idx * effective_band_size;
    let end = start + effective_band_size;
    calculate_band_hash(&row[start..end])
  }

  fn recall_rescue_band_counts(
    &self,
    digest_matrix: &RMinHashDigestMatrix,
    precomputed: Option<&PrecomputedBandHashes>,
    band_idx: usize,
    rescue_candidate_mask: &[u8],
    rescue_band_match_counts: &[AtomicU32],
    bucket_state_by_hash: &mut FxHashMap<u64, u64>,
  ) {
    bucket_state_by_hash.clear();
    for (row_index, rescue_candidate) in
      rescue_candidate_mask.iter().enumerate()
    {
      let band_hash = precomputed.map_or_else(
        || {
          let row = digest_matrix.row(row_index);
          let start = band_idx * self.band_size;
          let end = start + self.band_size;
          calculate_band_hash(&row[start..end])
        },
        |precomputed| precomputed.hashes[row_index * self.num_bands + band_idx],
      );
      match bucket_state_by_hash.entry(band_hash) {
        Entry::Vacant(entry) => {
          entry.insert(usize_to_u64_lowbits(row_index));
        }
        Entry::Occupied(mut entry) => {
          let bucket_state = entry.get_mut();
          if *bucket_state & RESCUE_COLLIDED_BIT == 0 {
            let first_row = rescue_state_first_row(*bucket_state);
            *bucket_state |= RESCUE_COLLIDED_BIT;
            if rescue_candidate_mask[first_row] == 1 {
              rescue_band_match_counts[first_row]
                .fetch_add(1, Ordering::Relaxed);
            }
          }
          if *rescue_candidate == 1 {
            rescue_band_match_counts[row_index].fetch_add(1, Ordering::Relaxed);
          }
        }
      }
    }
  }

  fn apply_recall_rescue(
    &self,
    digest_matrix: &RMinHashDigestMatrix,
    rows: usize,
    precomputed: Option<&PrecomputedBandHashes>,
    required_band_matches: &[u32],
    band_match_counts: &[AtomicU32],
    recall_rescue: &RecallRescueConfig,
  ) {
    let mut rescue_candidate_mask = vec![0u8; rows];
    let mut rescue_candidate_count = 0usize;
    for row_index in 0..rows {
      if band_match_counts[row_index].load(Ordering::Relaxed) != 0
        || required_band_matches[row_index] > 1
      {
        continue;
      }
      if !digest_matrix.rho_source_token_count(row_index).is_some_and(
        |token_count| {
          token_count >= recall_rescue.min_tokens
            && token_count <= recall_rescue.max_tokens
        },
      ) {
        continue;
      }
      rescue_candidate_mask[row_index] = 1;
      rescue_candidate_count = rescue_candidate_count.saturating_add(1);
    }

    if rescue_candidate_count == 0 {
      return;
    }

    let rescue_band_match_counts = atomic_counters(rows);
    if use_parallel_bands(self.num_bands, rows) {
      (0..self.num_bands).into_par_iter().for_each(|band_idx| {
        let mut bucket_state_by_hash: FxHashMap<u64, u64> =
          FxHashMap::default();
        bucket_state_by_hash.reserve(rows);
        self.recall_rescue_band_counts(
          digest_matrix,
          precomputed,
          band_idx,
          &rescue_candidate_mask,
          &rescue_band_match_counts,
          &mut bucket_state_by_hash,
        );
      });
    } else {
      let mut bucket_state_by_hash: FxHashMap<u64, u64> = FxHashMap::default();
      bucket_state_by_hash.reserve(rows);
      for band_idx in 0..self.num_bands {
        self.recall_rescue_band_counts(
          digest_matrix,
          precomputed,
          band_idx,
          &rescue_candidate_mask,
          &rescue_band_match_counts,
          &mut bucket_state_by_hash,
        );
      }
    }

    let required_matches =
      u32::try_from(recall_rescue.required_band_matches).unwrap_or(u32::MAX);
    for row_index in 0..rows {
      if rescue_candidate_mask[row_index] == 1
        && rescue_band_match_counts[row_index].load(Ordering::Relaxed)
          >= required_matches
      {
        band_match_counts[row_index]
          .store(required_band_matches[row_index], Ordering::Relaxed);
      }
    }
  }
}

#[cfg(test)]
mod tests {
  use crate::lsh::one_shot::{
    atomic_counters, count_scan_band, window_rest_key, BandFoldingConfig,
    BandScanContext, BandScanScratch, GroupVerifyContext, RawBandHashes,
    RecallRescueConfig, SparseVerifyConfig,
  };
  use crate::lsh::RMinHashLSH;
  use crate::rminhash::RMinHashDigestMatrix;
  use crate::utils::calculate_band_hash;
  use rustc_hash::FxHashMap;
  use std::sync::atomic::Ordering;

  #[test]
  fn precomputed_hashes_match_direct_bands_and_folded_windows() {
    for rows in [2, 2048] {
      for band_size in [4, 8, 12] {
        let num_perm = band_size * 12;
        let data = (0..rows * num_perm)
          .map(|index| u32::try_from(index).unwrap().wrapping_mul(0x9e37_79b9))
          .collect();
        let matrix = RMinHashDigestMatrix::test_matrix(
          num_perm,
          data,
          Vec::new(),
          Vec::new(),
        );
        let lsh = RMinHashLSH::from_validated(0.5, num_perm, 12);
        assert!(lsh.precomputed_band_hashes(&matrix, rows, false).is_none());
        let precomputed =
          lsh.precomputed_band_hashes(&matrix, rows, true).unwrap();
        for row_index in 0..rows {
          let row = matrix.row(row_index);
          for (band, values) in row.chunks_exact(band_size).enumerate() {
            assert_eq!(
              precomputed.hashes[row_index * 12 + band],
              calculate_band_hash(values)
            );
          }
          for fold in [2, 3, 4, 6, 12] {
            for (band, values) in row.chunks_exact(band_size * fold).enumerate()
            {
              assert_eq!(
                lsh.effective_band_hash(
                  &matrix,
                  row_index,
                  band,
                  band_size * fold,
                  fold,
                  Some(&precomputed)
                ),
                calculate_band_hash(values),
              );
            }
          }
        }
      }
    }
  }

  #[test]
  fn existing_and_batch_hits_count_each_band_once() {
    let matrix = RMinHashDigestMatrix::test_matrix(
      2,
      vec![10, 20, 10, 30, 50, 40, 50, 60],
      vec![7; 4],
      vec![1; 4],
    );
    let mut lsh = RMinHashLSH::from_validated(0.5, 2, 2);
    lsh.insert_digest(7, &[10, 40]).unwrap();
    for enabled in [false, true] {
      let verify = SparseVerifyConfig {
        enabled,
        threshold: 0.75,
        max_candidates: 1,
      };
      let context = BandScanContext {
        digest_matrix: &matrix,
        effective_band_size: 1,
        rho_band_fold: 1,
        precomputed: None,
        has_existing_entries: true,
        required_band_matches: &[2; 4],
        sparse_verify: &verify,
      };
      let counters = atomic_counters(4);
      let mut scratch = BandScanScratch::with_row_capacity(4);
      let mut statistics = (0, 0);
      for band in 0..2 {
        let (checks, passes) =
          lsh.scan_effective_band(&context, band, &counters, &mut scratch);
        statistics.0 += checks;
        statistics.1 += passes;
      }
      let counts: Vec<_> = counters
        .iter()
        .map(|count| count.load(Ordering::Relaxed))
        .collect();
      assert_eq!(counts, vec![1, 1, 2, 1]);
      assert_eq!(statistics, if enabled { (2, 2) } else { (0, 0) });
    }
  }

  #[test]
  fn two_member_verification_preserves_counts_and_statistics() {
    for enabled in [false, true] {
      for max_candidates in [0, 1, 4] {
        for required in [[1, 1], [1, 2], [2, 1], [2, 2]] {
          for active in [vec![1, 1], vec![1, 0], vec![0, 1]] {
            for matching in [false, true] {
              let signatures = if matching {
                vec![7, 8, 7, 8]
              } else {
                vec![7, 8, 9, 10]
              };
              let matrix = RMinHashDigestMatrix::test_matrix(
                1,
                vec![0, 0],
                signatures,
                active.clone(),
              );
              let config = SparseVerifyConfig {
                enabled,
                threshold: 0.75,
                max_candidates,
              };
              let ctx = GroupVerifyContext {
                digest_matrix: &matrix,
                required_band_matches: &required,
                sparse_verify: &config,
              };
              let counters = atomic_counters(2);
              let actual =
                RMinHashLSH::process_collision_group(&ctx, &[0, 1], &counters);
              let needs_verify = enabled && required.contains(&2);
              let checked = needs_verify && max_candidates > 0;
              let passed = matching || active.contains(&0);
              let matched = !needs_verify || (checked && passed);
              assert_eq!(
                actual,
                (usize::from(checked) * 2, usize::from(checked && passed) * 2)
              );
              for counter in &counters {
                assert_eq!(counter.load(Ordering::Relaxed), u32::from(matched));
              }
            }
          }
        }
      }
    }
  }

  #[test]
  fn band_counts_count_each_colliding_row_once_and_reset_scratch() {
    let raw = RawBandHashes {
      num_bands: 2,
      rows: 5,
      hashes: vec![10, 1, 10, 2, 10, 3, 20, 4, 30, 4],
    };
    let counters = atomic_counters(raw.rows);
    let mut scratch = FxHashMap::default();
    count_scan_band(&raw, 0, &counters, &mut scratch);
    count_scan_band(&raw, 1, &counters, &mut scratch);
    let counts: Vec<_> = counters
      .iter()
      .map(|count| count.load(Ordering::Relaxed))
      .collect();
    assert_eq!(counts, vec![1, 1, 1, 1, 1]);
  }

  #[test]
  fn simple_flags_and_rescue_counts_preserve_first_and_later_members() {
    let matrix = RMinHashDigestMatrix::test_matrix(
      1,
      vec![10, 10, 10, 20, 30],
      Vec::new(),
      Vec::new(),
    );
    let lsh = RMinHashLSH::from_validated(0.5, 1, 1);
    assert_eq!(
      lsh.simple_one_shot_flags(&matrix, 5, 1, 1, 1, false),
      vec![true, true, true, false, false],
    );
    for mask in [[1, 0, 1, 1, 1], [0, 1, 1, 1, 1]] {
      let counters = atomic_counters(5);
      lsh.recall_rescue_band_counts(
        &matrix,
        None,
        0,
        &mask,
        &counters,
        &mut FxHashMap::default(),
      );
      let counts: Vec<_> = counters
        .iter()
        .map(|count| count.load(Ordering::Relaxed))
        .collect();
      assert_eq!(
        counts,
        vec![u32::from(mask[0]), u32::from(mask[1]), 1, 0, 0]
      );
    }
  }

  // Independent eager reference: count raw collisions against every other row,
  // and group complete folded keys directly before applying rescue.
  fn eager_grouped_reference(
    lsh: &RMinHashLSH,
    matrix: &RMinHashDigestMatrix,
    folding: &BandFoldingConfig,
    required: &[u32],
    verify: &SparseVerifyConfig,
    rescue: &RecallRescueConfig,
  ) -> (Vec<bool>, usize, usize) {
    let rows = matrix.rows();
    let raw = lsh.compute_raw_band_hashes(matrix, rows).unwrap();
    let counters = atomic_counters(rows);
    let context = GroupVerifyContext {
      digest_matrix: matrix,
      required_band_matches: required,
      sparse_verify: verify,
    };
    let mut stats = (0, 0);
    for window in 0..folding.effective_num_bands {
      let first = window * folding.rho_band_fold;
      let mut groups: FxHashMap<(u64, u64), Vec<u32>> = FxHashMap::default();
      for row in 0..rows {
        groups
          .entry((
            raw.get(row, first),
            window_rest_key(&raw, first, folding.rho_band_fold, row),
          ))
          .or_default()
          .push(u32::try_from(row).unwrap());
      }
      for members in groups.values().filter(|members| members.len() > 1) {
        let (checks, passes) =
          RMinHashLSH::process_collision_group(&context, members, &counters);
        stats.0 += checks;
        stats.1 += passes;
      }
    }
    let flags = (0..rows)
      .map(|row| {
        let matches = counters[row].load(Ordering::Relaxed);
        let eligible = rescue.enabled
          && matches == 0
          && required[row] == 1
          && matrix.rho_source_token_count(row).is_some_and(|count| {
            count >= rescue.min_tokens && count <= rescue.max_tokens
          });
        let raw_matches = (0..lsh.num_bands)
          .filter(|&band| {
            (0..rows).any(|other| {
              other != row && raw.get(row, band) == raw.get(other, band)
            })
          })
          .count();
        matches >= required[row]
          || (eligible && raw_matches >= rescue.required_band_matches)
      })
      .collect();
    (flags, stats.0, stats.1)
  }

  #[test]
  fn rescue_eligibility_matches_eager_flags_and_verification_statistics() {
    let lsh = RMinHashLSH::from_validated(0.5, 4, 4);
    let corpora = [
      // No collisions; eligible candidates require rescue scans.
      vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
      // Every row already has folded matches.
      vec![1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
      // First-band collisions alone meet a rescue threshold of two.
      vec![1, 2, 3, 4, 1, 6, 3, 8, 9, 10, 11, 12, 9, 14, 11, 16],
      // Only second bands collide; sparse rows must remain partners.
      vec![1, 2, 3, 4, 5, 2, 7, 4, 9, 10, 11, 12, 13, 10, 15, 12],
    ];
    for data in corpora {
      let matrix = RMinHashDigestMatrix::test_matrix(
        4,
        data,
        vec![7, 8, 7, 8, 9, 10, 11, 12],
        vec![1, 1, 1, 0],
      );
      for fold in [2, 4] {
        let folding = BandFoldingConfig {
          rho_band_fold: fold,
          effective_num_bands: 4 / fold,
          effective_band_size: fold,
        };
        for required in [[1, 1, 1, 1], [1, 2, 1, 2], [2, 2, 2, 2]] {
          for (enabled, max_candidates) in
            [(false, 1), (true, 0), (true, 1), (true, 4)]
          {
            let verify = SparseVerifyConfig {
              enabled,
              threshold: 0.75,
              max_candidates,
            };
            for (enabled, min_tokens, max_tokens) in
              [(false, 1, 1), (true, 1, 1), (true, 2, 3), (true, 0, 0)]
            {
              for required_band_matches in [1, 2, 4] {
                let rescue = RecallRescueConfig {
                  enabled,
                  min_tokens,
                  max_tokens,
                  required_band_matches,
                };
                assert_eq!(
                  lsh.grouped_one_shot_flags(
                    &matrix,
                    matrix.rows(),
                    &folding,
                    &required,
                    &verify,
                    &rescue
                  ),
                  Some(eager_grouped_reference(
                    &lsh, &matrix, &folding, &required, &verify, &rescue
                  )),
                );
              }
            }
          }
        }
      }
    }
  }

  #[test]
  fn rescue_eligibility_large_batch_keeps_ineligible_collision_partners() {
    let rows = 2048;
    let data = (0..rows)
      .flat_map(|row| {
        let value = u32::try_from(row).unwrap();
        [value, 17, value, 19]
      })
      .collect();
    let matrix =
      RMinHashDigestMatrix::test_matrix(4, data, vec![7; rows], vec![1; rows]);
    let lsh = RMinHashLSH::from_validated(0.5, 4, 4);
    let folding = BandFoldingConfig {
      rho_band_fold: 2,
      effective_num_bands: 2,
      effective_band_size: 2,
    };
    let required: Vec<_> = (0..rows)
      .map(|row| if row % 2 == 0 { 1 } else { 2 })
      .collect();
    let verify = SparseVerifyConfig {
      enabled: true,
      threshold: 0.75,
      max_candidates: 1,
    };
    let rescue = RecallRescueConfig {
      enabled: true,
      min_tokens: 1,
      max_tokens: 1,
      required_band_matches: 2,
    };
    assert_eq!(
      lsh.grouped_one_shot_flags(
        &matrix, rows, &folding, &required, &verify, &rescue
      ),
      Some(eager_grouped_reference(
        &lsh, &matrix, &folding, &required, &verify, &rescue
      )),
    );
  }
}
