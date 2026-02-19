use crate::cminhash::{CMinHash, ReduceResult, HASH_BATCH_SIZE};
use crate::py_input::extend_token_hashes_from_document;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyType};
use rand_core::{RngCore, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

#[pymethods]
impl CMinHash {
  /// Creates a new `CMinHash` instance.
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

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);

    let sigma_a = rng.next_u64() | 1;
    let sigma_b = rng.next_u64();
    let pi_c = rng.next_u64() | 1;
    let pi_d = rng.next_u64();

    Ok(Self {
      num_perm,
      seed,
      hash_values: vec![u64::MAX; num_perm],
      sigma_a,
      sigma_b,
      pi_c,
      pi_d,
      pi_precomputed: Self::build_pi_precomputed(num_perm, pi_c, pi_d),
    })
  }

  /// Creates `CMinHash` objects from an iterable of token iterables.
  ///
  /// # Errors
  ///
  /// Returns an error if `num_perm` is zero, the outer input is not iterable,
  /// or any token has an unsupported type.
  #[classmethod]
  #[pyo3(signature = (token_sets, num_perm, seed))]
  pub fn from_token_sets(
    _cls: &Bound<'_, PyType>,
    token_sets: &Bound<'_, PyAny>,
    num_perm: usize,
    seed: u64,
  ) -> PyResult<Vec<Self>> {
    Self::from_token_sets_inner(token_sets, num_perm, seed)
  }

  /// Computes `CMinHash` 32-bit digests from an iterable of token iterables.
  ///
  /// # Errors
  ///
  /// Returns an error if `num_perm` is zero, the outer input is not iterable,
  /// or any token has an unsupported type.
  #[classmethod]
  #[pyo3(signature = (token_sets, num_perm, seed))]
  pub fn digests_from_token_sets(
    _cls: &Bound<'_, PyType>,
    token_sets: &Bound<'_, PyAny>,
    num_perm: usize,
    seed: u64,
  ) -> PyResult<Vec<Vec<u32>>> {
    Self::digests_from_token_sets_inner(token_sets, num_perm, seed)
  }

  /// Computes `CMinHash` 64-bit digests from an iterable of token iterables.
  ///
  /// # Errors
  ///
  /// Returns an error if `num_perm` is zero, the outer input is not iterable,
  /// or any token has an unsupported type.
  #[classmethod]
  #[pyo3(signature = (token_sets, num_perm, seed))]
  pub fn digests64_from_token_sets(
    _cls: &Bound<'_, PyType>,
    token_sets: &Bound<'_, PyAny>,
    num_perm: usize,
    seed: u64,
  ) -> PyResult<Vec<Vec<u64>>> {
    Self::digests64_from_token_sets_inner(token_sets, num_perm, seed)
  }

  /// Computes `CMinHash` 64-bit digests from pre-hashed token iterables.
  ///
  /// # Errors
  ///
  /// Returns an error if `num_perm` is zero, the outer input is not iterable,
  /// or any token hash is not an unsigned 64-bit integer.
  #[classmethod]
  #[pyo3(signature = (token_hash_sets, num_perm, seed))]
  pub fn digests64_from_token_hash_sets(
    _cls: &Bound<'_, PyType>,
    token_hash_sets: &Bound<'_, PyAny>,
    num_perm: usize,
    seed: u64,
  ) -> PyResult<Vec<Vec<u64>>> {
    Self::digests64_from_token_hash_sets_inner(token_hash_sets, num_perm, seed)
  }

  /// Updates the `CMinHash` with a new set of items.
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
  pub fn update(&mut self, items: &Bound<'_, PyAny>) -> PyResult<()> {
    let mut token_hashes = Vec::with_capacity(HASH_BATCH_SIZE);
    extend_token_hashes_from_document(items, &mut token_hashes)?;
    self.update_hashed_tokens(&token_hashes);
    Ok(())
  }

  /// Returns the current `MinHash` digest as u32 values for compatibility.
  #[inline]
  #[must_use]
  pub fn digest(&self) -> Vec<u32> {
    self.hash_values.iter().map(|&v| (v >> 32) as u32).collect()
  }

  /// Returns the current `MinHash` digest as u64 values.
  #[inline]
  #[must_use]
  pub fn digest_u64(&self) -> Vec<u64> {
    self.hash_values.clone()
  }

  /// Calculates the Jaccard similarity between this `CMinHash` and another.
  ///
  /// # Errors
  ///
  /// Returns an error when instances have incompatible parameters.
  #[inline]
  pub fn jaccard(&self, other: &Self) -> PyResult<f64> {
    self.ensure_compatible_for_jaccard(other)?;
    Ok(self.jaccard_unchecked(other))
  }

  fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> {
    let decoded: Self =
      postcard::from_bytes(state.as_bytes()).map_err(|err| {
        PyValueError::new_err(format!(
          "failed to deserialize CMinHash state: {err}"
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
        "failed to serialize CMinHash state: {err}"
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
