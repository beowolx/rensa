use crate::cminhash::{
  CMinHash, CMinHashParams, HASH_BATCH_SIZE, STATE_VERSION,
};
use crate::py_input::{
  extend_prehashed_token_values_from_document,
  extend_token_hashes_from_document,
};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyIterator, PyList, PyTuple};

impl CMinHash {
  fn map_token_sets_with_reused_signature<T, F>(
    token_sets: &Bound<'_, PyAny>,
    num_perm: usize,
    seed: u64,
    extend_token_hashes: fn(&Bound<'_, PyAny>, &mut Vec<u64>) -> PyResult<()>,
    mut map: F,
  ) -> PyResult<Vec<T>>
  where
    F: FnMut(&[u64]) -> T,
  {
    Self::validate_num_perm(num_perm)?;

    let capacity = Self::token_sets_capacity(token_sets);
    let mut outputs = Vec::with_capacity(capacity);
    let mut token_hashes = Vec::with_capacity(HASH_BATCH_SIZE);
    let params = CMinHashParams::new(seed);
    let mut hash_values = vec![u64::MAX; num_perm];

    Self::for_each_document(token_sets, |document| {
      token_hashes.clear();
      extend_token_hashes(&document, &mut token_hashes)?;

      hash_values.fill(u64::MAX);
      Self::apply_token_hashes_to_values(
        &mut hash_values,
        &token_hashes,
        params.sigma_key,
        params.pi_key,
      );

      outputs.push(map(&hash_values));
      Ok(())
    })?;

    Ok(outputs)
  }

  fn map_token_sets_with_owned_signature<T, F>(
    token_sets: &Bound<'_, PyAny>,
    num_perm: usize,
    seed: u64,
    extend_token_hashes: fn(&Bound<'_, PyAny>, &mut Vec<u64>) -> PyResult<()>,
    mut map: F,
  ) -> PyResult<Vec<T>>
  where
    F: FnMut(Vec<u64>, &CMinHashParams) -> T,
  {
    Self::validate_num_perm(num_perm)?;

    let capacity = Self::token_sets_capacity(token_sets);
    let mut outputs = Vec::with_capacity(capacity);
    let mut token_hashes = Vec::with_capacity(HASH_BATCH_SIZE);
    let params = CMinHashParams::new(seed);

    Self::for_each_document(token_sets, |document| {
      token_hashes.clear();
      extend_token_hashes(&document, &mut token_hashes)?;

      let mut hash_values = vec![u64::MAX; num_perm];
      Self::apply_token_hashes_to_values(
        &mut hash_values,
        &token_hashes,
        params.sigma_key,
        params.pi_key,
      );

      outputs.push(map(hash_values, &params));
      Ok(())
    })?;

    Ok(outputs)
  }

  pub(in crate::cminhash) fn token_sets_capacity(
    token_sets: &Bound<'_, PyAny>,
  ) -> usize {
    if let Ok(py_list) = token_sets.cast::<PyList>() {
      return py_list.len();
    }
    if let Ok(py_tuple) = token_sets.cast::<PyTuple>() {
      return py_tuple.len();
    }
    token_sets.len().unwrap_or_default()
  }

  pub(in crate::cminhash) fn for_each_document<F>(
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

  pub(in crate::cminhash) fn from_token_sets_inner(
    token_sets: &Bound<'_, PyAny>,
    num_perm: usize,
    seed: u64,
  ) -> PyResult<Vec<Self>> {
    Self::map_token_sets_with_owned_signature(
      token_sets,
      num_perm,
      seed,
      extend_token_hashes_from_document,
      |hash_values, params| Self {
        version: STATE_VERSION,
        num_perm,
        seed,
        hash_values,
        sigma_key: params.sigma_key,
        pi_key: params.pi_key,
      },
    )
  }

  pub(in crate::cminhash) fn digests_from_token_sets_inner(
    token_sets: &Bound<'_, PyAny>,
    num_perm: usize,
    seed: u64,
  ) -> PyResult<Vec<Vec<u32>>> {
    Self::map_token_sets_with_reused_signature(
      token_sets,
      num_perm,
      seed,
      extend_token_hashes_from_document,
      |hash_values| hash_values.iter().map(|&v| (v >> 32) as u32).collect(),
    )
  }

  pub(in crate::cminhash) fn digests64_from_token_sets_inner(
    token_sets: &Bound<'_, PyAny>,
    num_perm: usize,
    seed: u64,
  ) -> PyResult<Vec<Vec<u64>>> {
    Self::map_token_sets_with_owned_signature(
      token_sets,
      num_perm,
      seed,
      extend_token_hashes_from_document,
      |hash_values, _params| hash_values,
    )
  }

  pub(in crate::cminhash) fn digests64_from_token_hash_sets_inner(
    token_hash_sets: &Bound<'_, PyAny>,
    num_perm: usize,
    seed: u64,
  ) -> PyResult<Vec<Vec<u64>>> {
    Self::map_token_sets_with_owned_signature(
      token_hash_sets,
      num_perm,
      seed,
      extend_prehashed_token_values_from_document,
      |hash_values, _params| hash_values,
    )
  }
}
