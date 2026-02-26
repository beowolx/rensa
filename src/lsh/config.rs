use crate::lsh::RMinHashLSH;
use crate::utils::ratio_usize;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

const DEFAULT_RHO_SPARSE_OCCUPANCY_THRESHOLD_BASE: usize = 56;
const MIN_RHO_SPARSE_OCCUPANCY_THRESHOLD_BASE: usize = 1;
const MAX_RHO_SPARSE_OCCUPANCY_THRESHOLD_BASE: usize = 512;
const DEFAULT_RHO_SPARSE_REQUIRED_BAND_MATCHES: usize = 2;
const MIN_RHO_SPARSE_REQUIRED_BAND_MATCHES: usize = 1;
const DEFAULT_RHO_SPARSE_VERIFY_THRESHOLD: f64 = 0.75;
const MIN_RHO_SPARSE_VERIFY_THRESHOLD: f64 = 0.0;
const MAX_RHO_SPARSE_VERIFY_THRESHOLD: f64 = 1.0;
const DEFAULT_RHO_SPARSE_VERIFY_MAX_CANDIDATES: usize = 16;
const MIN_RHO_SPARSE_VERIFY_MAX_CANDIDATES: usize = 1;
const MAX_RHO_SPARSE_VERIFY_MAX_CANDIDATES: usize = 512;
const DEFAULT_RHO_BAND_FOLD: usize = 2;
const MIN_RHO_BAND_FOLD: usize = 1;
const DEFAULT_RHO_RECALL_RESCUE_MIN_TOKENS: usize = 17;
const MIN_RHO_RECALL_RESCUE_MIN_TOKENS: usize = 1;
const MAX_RHO_RECALL_RESCUE_MIN_TOKENS: usize = 65_536;
const DEFAULT_RHO_RECALL_RESCUE_MAX_TOKENS: usize = 96;
const MIN_RHO_RECALL_RESCUE_MAX_TOKENS: usize = 1;
const MAX_RHO_RECALL_RESCUE_MAX_TOKENS: usize = 65_536;
const DEFAULT_RHO_RECALL_RESCUE_REQUIRED_BAND_MATCHES: usize = 2;
const MIN_RHO_RECALL_RESCUE_REQUIRED_BAND_MATCHES: usize = 1;

impl RMinHashLSH {
  pub(in crate::lsh) fn rho_sparse_occupancy_threshold(
    num_perm: usize,
  ) -> usize {
    let base = crate::env::read_env_usize_clamped(
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

  pub(in crate::lsh) fn rho_sparse_required_band_matches(
    num_bands: usize,
  ) -> usize {
    crate::env::read_env_usize_clamped(
      "RENSA_RHO_SPARSE_REQUIRED_BAND_MATCHES",
      DEFAULT_RHO_SPARSE_REQUIRED_BAND_MATCHES,
      MIN_RHO_SPARSE_REQUIRED_BAND_MATCHES,
      num_bands.max(1),
    )
  }

  pub(in crate::lsh) fn rho_sparse_verify_enabled() -> bool {
    std::env::var("RENSA_RHO_SPARSE_VERIFY_ENABLE")
      .ok()
      .is_none_or(|value| value != "0")
  }

  pub(in crate::lsh) fn rho_sparse_verify_threshold() -> f64 {
    crate::env::read_env_f64_clamped(
      "RENSA_RHO_SPARSE_VERIFY_THRESHOLD",
      DEFAULT_RHO_SPARSE_VERIFY_THRESHOLD,
      MIN_RHO_SPARSE_VERIFY_THRESHOLD,
      MAX_RHO_SPARSE_VERIFY_THRESHOLD,
    )
  }

  pub(in crate::lsh) fn rho_sparse_verify_max_candidates() -> usize {
    crate::env::read_env_usize_clamped(
      "RENSA_RHO_SPARSE_VERIFY_MAX_CANDIDATES",
      DEFAULT_RHO_SPARSE_VERIFY_MAX_CANDIDATES,
      MIN_RHO_SPARSE_VERIFY_MAX_CANDIDATES,
      MAX_RHO_SPARSE_VERIFY_MAX_CANDIDATES,
    )
  }

  pub(in crate::lsh) fn rho_band_fold(num_bands: usize) -> usize {
    let max_fold = num_bands.max(1);
    crate::env::read_env_usize_clamped(
      "RENSA_RHO_BAND_FOLD",
      DEFAULT_RHO_BAND_FOLD,
      MIN_RHO_BAND_FOLD,
      max_fold,
    )
    .max(1)
  }

  pub(in crate::lsh) fn rho_recall_rescue_enabled() -> bool {
    std::env::var("RENSA_RHO_RECALL_RESCUE_ENABLE")
      .ok()
      .is_none_or(|value| value != "0")
  }

  pub(in crate::lsh) fn rho_recall_rescue_min_tokens() -> usize {
    crate::env::read_env_usize_clamped(
      "RENSA_RHO_RECALL_RESCUE_MIN_TOKENS",
      DEFAULT_RHO_RECALL_RESCUE_MIN_TOKENS,
      MIN_RHO_RECALL_RESCUE_MIN_TOKENS,
      MAX_RHO_RECALL_RESCUE_MIN_TOKENS,
    )
  }

  pub(in crate::lsh) fn rho_recall_rescue_max_tokens() -> usize {
    crate::env::read_env_usize_clamped(
      "RENSA_RHO_RECALL_RESCUE_MAX_TOKENS",
      DEFAULT_RHO_RECALL_RESCUE_MAX_TOKENS,
      MIN_RHO_RECALL_RESCUE_MAX_TOKENS,
      MAX_RHO_RECALL_RESCUE_MAX_TOKENS,
    )
  }

  pub(in crate::lsh) fn rho_recall_rescue_required_band_matches(
    num_bands: usize,
  ) -> usize {
    crate::env::read_env_usize_clamped(
      "RENSA_RHO_RECALL_RESCUE_REQUIRED_BAND_MATCHES",
      DEFAULT_RHO_RECALL_RESCUE_REQUIRED_BAND_MATCHES,
      MIN_RHO_RECALL_RESCUE_REQUIRED_BAND_MATCHES,
      num_bands.max(1),
    )
  }

  pub(in crate::lsh) fn sparse_verify_similarity(
    signature_a: &[u32],
    signature_b: &[u32],
  ) -> f64 {
    if signature_a.is_empty() || signature_a.len() != signature_b.len() {
      return 0.0;
    }
    let equal = signature_a
      .iter()
      .zip(signature_b.iter())
      .filter(|(left, right)| left == right)
      .count();
    ratio_usize(equal, signature_a.len())
  }

  pub(in crate::lsh) fn validate_threshold(threshold: f64) -> PyResult<()> {
    if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
      return Err(PyValueError::new_err(
        "threshold must be a finite value between 0.0 and 1.0",
      ));
    }
    Ok(())
  }

  pub(in crate::lsh) fn validate_params(
    threshold: f64,
    num_perm: usize,
    num_bands: usize,
  ) -> PyResult<usize> {
    Self::validate_threshold(threshold)?;
    if num_perm == 0 {
      return Err(PyValueError::new_err("num_perm must be greater than 0"));
    }
    if num_bands == 0 {
      return Err(PyValueError::new_err("num_bands must be greater than 0"));
    }
    if num_bands > num_perm {
      return Err(PyValueError::new_err(format!(
        "num_bands ({num_bands}) must be less than or equal to num_perm ({num_perm})"
      )));
    }
    if num_perm % num_bands != 0 {
      return Err(PyValueError::new_err(format!(
        "num_perm ({num_perm}) must be divisible by num_bands ({num_bands})"
      )));
    }

    Ok(num_perm / num_bands)
  }
}
