use crate::lsh::{RMinHashLSH, FX_FINISH_ROTATE};
use crate::rminhash::RMinHashDigestMatrix;
use crate::utils::calculate_band_hash;
use pyo3::prelude::*;
use rustc_hash::FxHashMap;
use std::collections::hash_map::Entry;

#[cfg(target_pointer_width = "64")]
const USIZE_MASK_U64: u64 = u64::MAX;
#[cfg(target_pointer_width = "32")]
const USIZE_MASK_U64: u64 = u64::from(u32::MAX);

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
    u64::from(value as u32)
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
  rows: usize,
  effective_band_size: usize,
  rho_band_fold: usize,
  precomputed: Option<&'a PrecomputedBandHashes>,
  has_existing_entries: bool,
  required_band_matches: &'a [usize],
  sparse_verify: &'a SparseVerifyConfig,
}

struct BandScanState<'a> {
  band_match_counts: &'a mut [usize],
  sparse_verify_checks: &'a mut usize,
  sparse_verify_passes: &'a mut usize,
}

struct RecallRescueBucketState {
  first_row: usize,
  collided: bool,
}

impl RMinHashLSH {
  pub(in crate::lsh) fn query_duplicate_flags_matrix_one_shot_inner(
    &mut self,
    digest_matrix: &RMinHashDigestMatrix,
  ) -> PyResult<Vec<bool>> {
    self.ensure_digest_len(digest_matrix.num_perm())?;

    let rows = digest_matrix.rows();
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

    let precomputed = self.precomputed_band_hashes(
      digest_matrix,
      rows,
      recall_rescue.enabled
        && folding.rho_band_fold > 1
        && self.band_size % 4 == 0,
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

    let mut band_match_counts = vec![0usize; rows];
    let mut sparse_verify_checks = 0usize;
    let mut sparse_verify_passes = 0usize;
    {
      let ctx = BandScanContext {
        digest_matrix,
        rows,
        effective_band_size: folding.effective_band_size,
        rho_band_fold: folding.rho_band_fold,
        precomputed: precomputed.as_ref(),
        has_existing_entries,
        required_band_matches: &required_band_matches,
        sparse_verify: &sparse_verify,
      };
      let mut state = BandScanState {
        band_match_counts: &mut band_match_counts,
        sparse_verify_checks: &mut sparse_verify_checks,
        sparse_verify_passes: &mut sparse_verify_passes,
      };
      for band_idx in 0..folding.effective_num_bands {
        Self::scan_effective_band(self, &ctx, band_idx, &mut state);
      }
    }

    if recall_rescue.enabled {
      Self::apply_recall_rescue(
        self,
        digest_matrix,
        rows,
        precomputed.as_ref(),
        &required_band_matches,
        &mut band_match_counts,
        &recall_rescue,
      );
    }

    let flags = band_match_counts
      .iter()
      .zip(required_band_matches.iter())
      .map(|(matches, required)| *matches >= *required)
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
  ) -> (Vec<usize>, bool) {
    let mut any_sparse_rows = false;
    let mut required_band_matches = vec![1usize; rows];
    for (row_index, required) in required_band_matches.iter_mut().enumerate() {
      if digest_matrix
        .rho_non_empty_count(row_index)
        .is_some_and(|count| count < sparse_occupancy_threshold)
      {
        *required = sparse_required_band_matches;
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

  fn precomputed_band_hashes(
    &self,
    digest_matrix: &RMinHashDigestMatrix,
    rows: usize,
    enabled: bool,
  ) -> Option<PrecomputedBandHashes> {
    if !enabled {
      return None;
    }

    let num_bands = self.num_bands;
    let band_size = self.band_size;

    let total_hashes = rows.checked_mul(num_bands)?;
    let mut hashes = vec![0u64; total_hashes];
    for row_index in 0..rows {
      let row = digest_matrix.row(row_index);
      for band_idx in 0..num_bands {
        let start = band_idx * band_size;
        let end = start + band_size;
        hashes[row_index * num_bands + band_idx] =
          calculate_band_hash(&row[start..end]);
      }
    }

    let steps = Self::fx_poly_steps(band_size);
    Some(PrecomputedBandHashes {
      hashes,
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
    for band_idx in 0..effective_num_bands {
      let table = if rho_band_fold == 1 {
        self.hash_tables.get(band_idx)
      } else {
        None
      };

      let mut first_row_by_hash: FxHashMap<u64, usize> = FxHashMap::default();
      first_row_by_hash.reserve(rows);
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
        if let Some(&first_row) = first_row_by_hash.get(&band_hash) {
          flags[row_index] = true;
          flags[first_row] = true;
        } else {
          first_row_by_hash.insert(band_hash, row_index);
        }
      }
    }
    flags
  }

  fn scan_effective_band(
    &self,
    ctx: &BandScanContext<'_>,
    band_idx: usize,
    state: &mut BandScanState<'_>,
  ) {
    let table = if ctx.rho_band_fold == 1 {
      self.hash_tables.get(band_idx)
    } else {
      None
    };
    let mut first_row_by_hash: FxHashMap<u64, usize> = FxHashMap::default();
    first_row_by_hash.reserve(ctx.rows);
    let mut collisions_by_hash: FxHashMap<u64, Vec<usize>> =
      FxHashMap::default();

    for (row_index, band_match_count) in
      state.band_match_counts.iter_mut().enumerate()
    {
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
        *band_match_count = band_match_count.saturating_add(1);
      }

      match first_row_by_hash.entry(band_hash) {
        Entry::Vacant(entry) => {
          entry.insert(row_index);
        }
        Entry::Occupied(entry) => {
          let first_row = *entry.get();
          collisions_by_hash
            .entry(band_hash)
            .or_insert_with(|| {
              let mut rows = Vec::with_capacity(2);
              rows.push(first_row);
              rows
            })
            .push(row_index);
        }
      }
    }

    for row_indices in collisions_by_hash.values() {
      if row_indices.len() < 2 {
        continue;
      }
      for &row_index in row_indices {
        let row_sparse = ctx.required_band_matches[row_index] > 1;
        let mut checked_candidates = 0usize;
        let mut matched = false;

        for &other_row in row_indices {
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
          *state.sparse_verify_checks =
            (*state.sparse_verify_checks).saturating_add(1);
          if Self::sparse_verify_pair_passes(
            ctx.digest_matrix,
            row_index,
            other_row,
            ctx.sparse_verify.threshold,
          ) {
            *state.sparse_verify_passes =
              (*state.sparse_verify_passes).saturating_add(1);
            matched = true;
            break;
          }
        }

        if matched {
          state.band_match_counts[row_index] =
            state.band_match_counts[row_index].saturating_add(1);
        }
      }
    }
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

  fn apply_recall_rescue(
    &self,
    digest_matrix: &RMinHashDigestMatrix,
    rows: usize,
    precomputed: Option<&PrecomputedBandHashes>,
    required_band_matches: &[usize],
    band_match_counts: &mut [usize],
    recall_rescue: &RecallRescueConfig,
  ) {
    let mut rescue_candidate_mask = vec![0u8; rows];
    let mut rescue_candidate_count = 0usize;
    for row_index in 0..rows {
      if band_match_counts[row_index] != 0
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

    let mut rescue_band_match_counts = vec![0u8; rows];
    let mut bucket_state_by_hash: FxHashMap<u64, RecallRescueBucketState> =
      FxHashMap::default();
    bucket_state_by_hash.reserve(rows);
    for band_idx in 0..self.num_bands {
      bucket_state_by_hash.clear();
      for row_index in 0..rows {
        let band_hash = precomputed.map_or_else(
          || {
            let row = digest_matrix.row(row_index);
            let start = band_idx * self.band_size;
            let end = start + self.band_size;
            calculate_band_hash(&row[start..end])
          },
          |precomputed| {
            precomputed.hashes[row_index * self.num_bands + band_idx]
          },
        );
        if let Some(bucket_state) = bucket_state_by_hash.get_mut(&band_hash) {
          if !bucket_state.collided {
            bucket_state.collided = true;
            let first_row = bucket_state.first_row;
            if rescue_candidate_mask[first_row] == 1 {
              rescue_band_match_counts[first_row] =
                rescue_band_match_counts[first_row].saturating_add(1);
            }
          }
          if rescue_candidate_mask[row_index] == 1 {
            rescue_band_match_counts[row_index] =
              rescue_band_match_counts[row_index].saturating_add(1);
          }
        } else {
          bucket_state_by_hash.insert(
            band_hash,
            RecallRescueBucketState {
              first_row: row_index,
              collided: false,
            },
          );
        }
      }
    }

    let required_matches_u8 =
      u8::try_from(recall_rescue.required_band_matches).unwrap_or(u8::MAX);
    for row_index in 0..rows {
      if rescue_candidate_mask[row_index] == 1
        && rescue_band_match_counts[row_index] >= required_matches_u8
      {
        band_match_counts[row_index] = required_band_matches[row_index];
      }
    }
  }
}
