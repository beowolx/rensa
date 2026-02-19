use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub(in crate::inline_dedup) const PAIR_ENTRY_ERROR: &str =
  "each entry must be a tuple of (str, minhash_or_tokens)";

pub(in crate::inline_dedup) fn validate_threshold(
  threshold: f64,
) -> PyResult<()> {
  if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
    return Err(PyValueError::new_err(
      "threshold must be a finite value between 0.0 and 1.0",
    ));
  }
  Ok(())
}
