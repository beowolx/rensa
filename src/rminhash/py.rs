use crate::py_input::{
  extend_byte_token_hashes_from_document,
  extend_prehashed_token_values_from_document,
  extend_token_hashes_from_document,
};
use crate::rminhash::{
  RMinHash, RMinHashDigestMatrix, ReduceResult, DEFAULT_RHO_PROBES,
  HASH_BATCH_SIZE,
};
use crate::utils::ratio_usize;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyType};

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
    Some(ratio_usize(sparse_rows, sidecar.non_empty_counts.len()))
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
    let permutations_soa =
      crate::simd::dispatch::PermutationSoA::from_permutations(&permutations);

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
    token_sets: &Bound<'_, PyAny>,
    num_perm: usize,
    seed: u64,
  ) -> PyResult<Vec<Self>> {
    Self::validate_num_perm(num_perm)?;
    let matrix = Self::build_digest_matrix_from_token_sets(
      token_sets,
      num_perm,
      seed,
      "tokens",
      extend_token_hashes_from_document,
    )?;
    Ok(Self::from_matrix(&matrix, seed))
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
    token_sets: &Bound<'_, PyAny>,
    num_perm: usize,
    seed: u64,
  ) -> PyResult<Vec<Vec<u32>>> {
    Self::validate_num_perm(num_perm)?;
    let matrix = Self::build_digest_matrix_from_token_sets(
      token_sets,
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
    token_sets: &Bound<'_, PyAny>,
  ) -> PyResult<Vec<Vec<u64>>> {
    Self::build_token_hash_rows(token_sets, extend_token_hashes_from_document)
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
    token_sets: &Bound<'_, PyAny>,
    num_perm: usize,
    seed: u64,
  ) -> PyResult<RMinHashDigestMatrix> {
    Self::validate_num_perm(num_perm)?;
    Self::build_digest_matrix_from_token_sets(
      token_sets,
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
    token_sets: &Bound<'_, PyAny>,
    num_perm: usize,
    seed: u64,
    probes: usize,
  ) -> PyResult<RMinHashDigestMatrix> {
    Self::validate_num_perm(num_perm)?;
    let matrix = if let Some(matrix) =
      Self::try_build_rho_digest_matrix_from_token_sets_parallel(
        token_sets, num_perm, seed, probes,
      )? {
      matrix
    } else {
      Self::build_rho_digest_matrix_from_token_sets_streaming(
        token_sets, num_perm, seed, probes,
      )?
    };

    Ok(matrix)
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
    token_hash_sets: &Bound<'_, PyAny>,
    num_perm: usize,
    seed: u64,
  ) -> PyResult<RMinHashDigestMatrix> {
    Self::validate_num_perm(num_perm)?;
    Self::build_digest_matrix_from_token_sets(
      token_hash_sets,
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
    token_hash_sets: &Bound<'_, PyAny>,
    num_perm: usize,
    seed: u64,
    probes: usize,
  ) -> PyResult<RMinHashDigestMatrix> {
    Self::validate_num_perm(num_perm)?;
    Self::build_rho_digest_matrix_from_token_hash_sets(
      token_hash_sets,
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
    token_hashes: &Bound<'_, PyAny>,
    row_offsets: &Bound<'_, PyAny>,
    num_perm: usize,
    seed: u64,
  ) -> PyResult<RMinHashDigestMatrix> {
    Self::validate_num_perm(num_perm)?;
    let flat_hashes = Self::parse_flat_token_hashes(token_hashes)?;
    let offsets = Self::parse_row_offsets(row_offsets)?;
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
    token_hashes: &Bound<'_, PyAny>,
    row_offsets: &Bound<'_, PyAny>,
    num_perm: usize,
    seed: u64,
    probes: usize,
  ) -> PyResult<RMinHashDigestMatrix> {
    Self::validate_num_perm(num_perm)?;
    let flat_hashes = Self::parse_flat_token_hashes(token_hashes)?;
    let offsets = Self::parse_row_offsets(row_offsets)?;
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
    token_byte_sets: &Bound<'_, PyAny>,
    num_perm: usize,
    seed: u64,
  ) -> PyResult<RMinHashDigestMatrix> {
    Self::validate_num_perm(num_perm)?;
    Self::build_digest_matrix_from_token_sets(
      token_byte_sets,
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
  pub fn update(&mut self, items: &Bound<'_, PyAny>) -> PyResult<()> {
    self.ensure_permutations();
    let mut hash_batch = Vec::with_capacity(HASH_BATCH_SIZE);
    Self::apply_document_to_values(
      items,
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
