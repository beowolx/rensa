//! Implementation of C-MinHash, an optimized `MinHash` variant.
//! This algorithm provides an efficient way to estimate Jaccard similarity
//! between sets, using a technique that rigorously reduces K permutations to two.
//!
//! - C-MinHash: Rigorously Reducing K Permutations to Two.
//!   Ping Li, Arnd Christian König.
//!   [arXiv:2109.03337](https://arxiv.org/abs/2109.03337)
//!
//! The implementation focuses on high single-threaded performance through
//! optimized memory access patterns and batch processing. It uses two main
//! hash transformations:
//!   - An initial permutation σ applied to item hashes.
//!   - A second set of parameters π used to generate the `num_perm` signature values.
//!
//! The `update` method processes items in batches to improve cache utilization,
//! and the Jaccard calculation is optimized using chunked operations.

use crate::py_input::{
  extend_prehashed_token_values_from_document,
  extend_token_hashes_from_document,
};
use crate::utils::calculate_hash_fast;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyIterator, PyList, PyTuple, PyType};
use rand_core::{RngCore, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use serde::{Deserialize, Serialize};

const HASH_BATCH_SIZE: usize = 32;
type ReduceResult = (Py<PyAny>, (usize, u64), Py<PyAny>);

/// `CMinHash` implements an optimized version of `C-MinHash` with better memory access patterns
/// and aggressive optimizations for maximum single-threaded performance.
#[derive(Serialize, Deserialize, Clone)]
#[pyclass(module = "rensa", skip_from_py_object)]
pub struct CMinHash {
  num_perm: usize,
  seed: u64,
  hash_values: Vec<u64>,
  // Permutation σ parameters (a, b)
  sigma_a: u64,
  sigma_b: u64,
  // Permutation π parameters (c, d)
  pi_c: u64,
  pi_d: u64,
  // Precomputed pi_c * k + pi_d for k in 0..num_perm
  #[serde(skip, default)]
  pi_precomputed: Vec<u64>,
}

impl CMinHash {
  fn build_pi_precomputed(num_perm: usize, pi_c: u64, pi_d: u64) -> Vec<u64> {
    (0..num_perm)
      .map(|k| pi_c.wrapping_mul(k as u64).wrapping_add(pi_d))
      .collect()
  }

  #[inline]
  pub(crate) const fn num_perm(&self) -> usize {
    self.num_perm
  }

  fn validate_num_perm(num_perm: usize) -> PyResult<()> {
    if num_perm == 0 {
      return Err(PyValueError::new_err("num_perm must be greater than 0"));
    }
    Ok(())
  }

  fn token_sets_capacity(token_sets: &Bound<'_, PyAny>) -> usize {
    if let Ok(py_list) = token_sets.cast::<PyList>() {
      return py_list.len();
    }
    if let Ok(py_tuple) = token_sets.cast::<PyTuple>() {
      return py_tuple.len();
    }
    token_sets.len().unwrap_or_default()
  }

  fn for_each_document<F>(
    token_sets: &Bound<'_, PyAny>,
    mut visitor: F,
  ) -> PyResult<()>
  where
    F: FnMut(Bound<'_, PyAny>) -> PyResult<()>,
  {
    if let Ok(py_list) = token_sets.cast::<PyList>() {
      for document in py_list.iter() {
        visitor(document)?;
      }
      return Ok(());
    }

    if let Ok(py_tuple) = token_sets.cast::<PyTuple>() {
      for document in py_tuple.iter() {
        visitor(document)?;
      }
      return Ok(());
    }

    let iterator = PyIterator::from_object(token_sets)?;
    for document in iterator {
      visitor(document?)?;
    }
    Ok(())
  }

  fn validate_state(&self) -> PyResult<()> {
    Self::validate_num_perm(self.num_perm)?;
    if self.hash_values.len() != self.num_perm {
      return Err(PyValueError::new_err(format!(
        "invalid CMinHash state: hash_values length {} does not match num_perm {}",
        self.hash_values.len(),
        self.num_perm
      )));
    }
    if !self.pi_precomputed.is_empty()
      && self.pi_precomputed.len() != self.num_perm
    {
      return Err(PyValueError::new_err(format!(
        "invalid CMinHash state: pi_precomputed length {} does not match num_perm {} (or be compacted to 0)",
        self.pi_precomputed.len(),
        self.num_perm
      )));
    }
    Ok(())
  }

  pub(crate) fn ensure_compatible_for_jaccard(
    &self,
    other: &Self,
  ) -> PyResult<()> {
    self.validate_state()?;
    other.validate_state()?;
    if self.num_perm != other.num_perm {
      return Err(PyValueError::new_err(format!(
        "num_perm mismatch: left is {}, right is {}",
        self.num_perm, other.num_perm
      )));
    }
    Ok(())
  }

  #[inline]
  const fn sigma_transform(&self, hash: u64) -> u64 {
    self.sigma_a.wrapping_mul(hash).wrapping_add(self.sigma_b)
  }

  fn ensure_pi_precomputed(&mut self) {
    if self.pi_precomputed.len() != self.num_perm {
      self.pi_precomputed =
        Self::build_pi_precomputed(self.num_perm, self.pi_c, self.pi_d);
    }
  }

  #[inline]
  pub(crate) fn jaccard_unchecked(&self, other: &Self) -> f64 {
    let mut equal_count = 0usize;

    let chunks_a = self.hash_values.chunks_exact(8);
    let chunks_b = other.hash_values.chunks_exact(8);

    for (chunk_a, chunk_b) in chunks_a.zip(chunks_b) {
      equal_count += usize::from(chunk_a[0] == chunk_b[0]);
      equal_count += usize::from(chunk_a[1] == chunk_b[1]);
      equal_count += usize::from(chunk_a[2] == chunk_b[2]);
      equal_count += usize::from(chunk_a[3] == chunk_b[3]);
      equal_count += usize::from(chunk_a[4] == chunk_b[4]);
      equal_count += usize::from(chunk_a[5] == chunk_b[5]);
      equal_count += usize::from(chunk_a[6] == chunk_b[6]);
      equal_count += usize::from(chunk_a[7] == chunk_b[7]);
    }

    let remainder_start = (self.num_perm / 8) * 8;
    if remainder_start < self.num_perm {
      equal_count += self.hash_values[remainder_start..]
        .iter()
        .zip(&other.hash_values[remainder_start..])
        .filter(|&(&a, &b)| a == b)
        .count();
    }

    equal_count as f64 / self.num_perm as f64
  }

  fn apply_sigma_batch_to_values(
    hash_values: &mut [u64],
    pi_precomputed: &[u64],
    pi_c: u64,
    sigma_batch: &[u64],
  ) {
    let chunks_iter = hash_values.chunks_exact_mut(16);
    let pi_chunks_iter = pi_precomputed.chunks_exact(16);

    for (hash_chunk, pi_chunk) in chunks_iter.zip(pi_chunks_iter) {
      let mut current = [0u64; 16];
      current.copy_from_slice(hash_chunk);

      for &sigma_h in sigma_batch {
        let base = pi_c.wrapping_mul(sigma_h);

        for i in 0..16 {
          let pi_value = base.wrapping_add(pi_chunk[i]);
          current[i] = current[i].min(pi_value);
        }
      }

      hash_chunk.copy_from_slice(&current);
    }

    let num_perm = hash_values.len();
    let remainder_start = (num_perm / 16) * 16;
    if remainder_start < num_perm {
      let hash_remainder = &mut hash_values[remainder_start..];
      let pi_remainder = &pi_precomputed[remainder_start..];

      for &sigma_h in sigma_batch {
        let base = pi_c.wrapping_mul(sigma_h);

        for (hash_val, &pi_val) in
          hash_remainder.iter_mut().zip(pi_remainder.iter())
        {
          let pi_value = base.wrapping_add(pi_val);
          *hash_val = (*hash_val).min(pi_value);
        }
      }
    }
  }

  fn apply_sigma_batch(&mut self, sigma_batch: &[u64]) {
    Self::apply_sigma_batch_to_values(
      &mut self.hash_values,
      &self.pi_precomputed,
      self.pi_c,
      sigma_batch,
    );
  }

  fn apply_token_hashes_to_values(
    hash_values: &mut [u64],
    token_hashes: &[u64],
    sigma_a: u64,
    sigma_b: u64,
    pi_c: u64,
    pi_precomputed: &[u64],
  ) {
    let mut sigma_batch = Vec::with_capacity(HASH_BATCH_SIZE);
    for &token_hash in token_hashes {
      sigma_batch.push(sigma_a.wrapping_mul(token_hash).wrapping_add(sigma_b));
      if sigma_batch.len() == HASH_BATCH_SIZE {
        Self::apply_sigma_batch_to_values(
          hash_values,
          pi_precomputed,
          pi_c,
          &sigma_batch,
        );
        sigma_batch.clear();
      }
    }
    if !sigma_batch.is_empty() {
      Self::apply_sigma_batch_to_values(
        hash_values,
        pi_precomputed,
        pi_c,
        &sigma_batch,
      );
    }
  }

  pub(crate) fn compact_from_template(template: &Self) -> Self {
    Self {
      num_perm: template.num_perm,
      seed: template.seed,
      hash_values: vec![u64::MAX; template.num_perm],
      sigma_a: template.sigma_a,
      sigma_b: template.sigma_b,
      pi_c: template.pi_c,
      pi_d: template.pi_d,
      pi_precomputed: Vec::new(),
    }
  }

  pub(crate) fn reset_from_token_hashes_with_template(
    &mut self,
    token_hashes: &[u64],
    template: &Self,
  ) {
    self.hash_values.fill(u64::MAX);
    Self::apply_token_hashes_to_values(
      &mut self.hash_values,
      token_hashes,
      template.sigma_a,
      template.sigma_b,
      template.pi_c,
      &template.pi_precomputed,
    );
  }

  fn update_internal<I, S>(&mut self, items: I)
  where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
  {
    self.ensure_pi_precomputed();
    let mut sigma_batch = Vec::with_capacity(HASH_BATCH_SIZE);

    for item in items {
      let h = calculate_hash_fast(item.as_ref().as_bytes());
      sigma_batch.push(self.sigma_transform(h));
      if sigma_batch.len() == HASH_BATCH_SIZE {
        self.apply_sigma_batch(&sigma_batch);
        sigma_batch.clear();
      }
    }

    if !sigma_batch.is_empty() {
      self.apply_sigma_batch(&sigma_batch);
    }
  }

  fn update_hashed_tokens(&mut self, token_hashes: &[u64]) {
    self.ensure_pi_precomputed();
    Self::apply_token_hashes_to_values(
      &mut self.hash_values,
      token_hashes,
      self.sigma_a,
      self.sigma_b,
      self.pi_c,
      &self.pi_precomputed,
    );
  }

  /// Updates the `CMinHash` with items from any iterable of string-like values.
  pub fn update_iter<I, S>(&mut self, items: I)
  where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
  {
    self.update_internal(items);
  }

  /// Updates the `CMinHash` with a new set of items from a vector of strings.
  pub fn update_vec(&mut self, items: Vec<String>) {
    self.update_iter(items);
  }
}

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
    token_sets: Bound<'_, PyAny>,
    num_perm: usize,
    seed: u64,
  ) -> PyResult<Vec<Self>> {
    Self::validate_num_perm(num_perm)?;

    let capacity = Self::token_sets_capacity(&token_sets);
    let mut minhashes = Vec::with_capacity(capacity);
    let mut token_hashes = Vec::with_capacity(HASH_BATCH_SIZE);
    let template = Self::new(num_perm, seed)?;
    let mut hash_values = vec![u64::MAX; num_perm];

    Self::for_each_document(&token_sets, |document| {
      token_hashes.clear();
      extend_token_hashes_from_document(&document, &mut token_hashes)?;
      hash_values.fill(u64::MAX);
      Self::apply_token_hashes_to_values(
        &mut hash_values,
        &token_hashes,
        template.sigma_a,
        template.sigma_b,
        template.pi_c,
        &template.pi_precomputed,
      );
      minhashes.push(Self {
        num_perm,
        seed,
        hash_values: hash_values.clone(),
        sigma_a: template.sigma_a,
        sigma_b: template.sigma_b,
        pi_c: template.pi_c,
        pi_d: template.pi_d,
        pi_precomputed: Vec::new(),
      });
      Ok(())
    })?;

    Ok(minhashes)
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
    token_sets: Bound<'_, PyAny>,
    num_perm: usize,
    seed: u64,
  ) -> PyResult<Vec<Vec<u32>>> {
    Self::validate_num_perm(num_perm)?;

    let capacity = Self::token_sets_capacity(&token_sets);
    let mut digests = Vec::with_capacity(capacity);
    let mut token_hashes = Vec::with_capacity(HASH_BATCH_SIZE);
    let template = Self::new(num_perm, seed)?;
    let mut hash_values = vec![u64::MAX; num_perm];

    Self::for_each_document(&token_sets, |document| {
      token_hashes.clear();
      extend_token_hashes_from_document(&document, &mut token_hashes)?;
      hash_values.fill(u64::MAX);
      Self::apply_token_hashes_to_values(
        &mut hash_values,
        &token_hashes,
        template.sigma_a,
        template.sigma_b,
        template.pi_c,
        &template.pi_precomputed,
      );
      digests.push(hash_values.iter().map(|&v| (v >> 32) as u32).collect());
      Ok(())
    })?;

    Ok(digests)
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
    token_sets: Bound<'_, PyAny>,
    num_perm: usize,
    seed: u64,
  ) -> PyResult<Vec<Vec<u64>>> {
    Self::validate_num_perm(num_perm)?;

    let capacity = Self::token_sets_capacity(&token_sets);
    let mut digests = Vec::with_capacity(capacity);
    let mut token_hashes = Vec::with_capacity(HASH_BATCH_SIZE);
    let template = Self::new(num_perm, seed)?;
    let mut hash_values = vec![u64::MAX; num_perm];

    Self::for_each_document(&token_sets, |document| {
      token_hashes.clear();
      extend_token_hashes_from_document(&document, &mut token_hashes)?;
      hash_values.fill(u64::MAX);
      Self::apply_token_hashes_to_values(
        &mut hash_values,
        &token_hashes,
        template.sigma_a,
        template.sigma_b,
        template.pi_c,
        &template.pi_precomputed,
      );
      digests.push(hash_values.clone());
      Ok(())
    })?;

    Ok(digests)
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
    token_hash_sets: Bound<'_, PyAny>,
    num_perm: usize,
    seed: u64,
  ) -> PyResult<Vec<Vec<u64>>> {
    Self::validate_num_perm(num_perm)?;

    let capacity = Self::token_sets_capacity(&token_hash_sets);
    let mut digests = Vec::with_capacity(capacity);
    let mut token_hashes = Vec::with_capacity(HASH_BATCH_SIZE);
    let template = Self::new(num_perm, seed)?;
    let mut hash_values = vec![u64::MAX; num_perm];

    Self::for_each_document(&token_hash_sets, |document| {
      token_hashes.clear();
      extend_prehashed_token_values_from_document(
        &document,
        &mut token_hashes,
      )?;
      hash_values.fill(u64::MAX);
      Self::apply_token_hashes_to_values(
        &mut hash_values,
        &token_hashes,
        template.sigma_a,
        template.sigma_b,
        template.pi_c,
        &template.pi_precomputed,
      );
      digests.push(hash_values.clone());
      Ok(())
    })?;

    Ok(digests)
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
  pub fn update(&mut self, items: Bound<'_, PyAny>) -> PyResult<()> {
    let mut token_hashes = Vec::with_capacity(HASH_BATCH_SIZE);
    extend_token_hashes_from_document(&items, &mut token_hashes)?;
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
