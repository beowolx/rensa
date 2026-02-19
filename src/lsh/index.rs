use crate::lsh::RMinHashLSH;
use crate::utils::calculate_band_hash;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rustc_hash::{FxHashSet, FxHasher};
use std::collections::HashMap;
use std::hash::BuildHasherDefault;

impl RMinHashLSH {
  pub(crate) fn from_validated(
    threshold: f64,
    num_perm: usize,
    num_bands: usize,
  ) -> Self {
    let hasher = BuildHasherDefault::<FxHasher>::default();
    let hash_tables = (0..num_bands)
      .map(|_| HashMap::with_hasher(hasher.clone()))
      .collect();

    Self {
      threshold,
      num_perm,
      num_bands,
      band_size: num_perm / num_bands,
      hash_tables,
      key_bands: HashMap::with_hasher(hasher),
      last_one_shot_sparse_verify_checks: 0,
      last_one_shot_sparse_verify_passes: 0,
    }
  }

  pub(in crate::lsh) fn validate_state(&self) -> PyResult<()> {
    let expected_band_size =
      Self::validate_params(self.threshold, self.num_perm, self.num_bands)?;
    if self.band_size != expected_band_size {
      return Err(PyValueError::new_err(format!(
        "invalid RMinHashLSH state: band_size {} does not match expected {}",
        self.band_size, expected_band_size
      )));
    }
    if self.hash_tables.len() != self.num_bands {
      return Err(PyValueError::new_err(format!(
        "invalid RMinHashLSH state: hash_tables length {} does not match num_bands {}",
        self.hash_tables.len(),
        self.num_bands
      )));
    }
    for (key, band_hashes) in &self.key_bands {
      if band_hashes.len() != self.num_bands {
        return Err(PyValueError::new_err(format!(
          "invalid RMinHashLSH state: key {key} stores {} band hashes, expected {}",
          band_hashes.len(),
          self.num_bands
        )));
      }
    }
    Ok(())
  }

  pub(in crate::lsh) fn ensure_digest_len(
    &self,
    digest_len: usize,
  ) -> PyResult<()> {
    if digest_len != self.num_perm {
      return Err(PyValueError::new_err(format!(
        "MinHash has {digest_len} permutations but LSH expects {}",
        self.num_perm
      )));
    }
    Ok(())
  }

  pub(in crate::lsh) fn digest_band_hashes(&self, digest: &[u32]) -> Vec<u64> {
    let mut band_hashes = Vec::with_capacity(self.num_bands);
    for i in 0..self.num_bands {
      band_hashes.push(calculate_band_hash(
        &digest[i * self.band_size..(i + 1) * self.band_size],
      ));
    }
    band_hashes
  }

  pub(in crate::lsh) fn remove_key_from_bands(
    &mut self,
    key: usize,
    band_hashes: &[u64],
  ) {
    for (table, &band_hash) in
      self.hash_tables.iter_mut().zip(band_hashes.iter())
    {
      if let Some(keys) = table.get_mut(&band_hash) {
        keys.retain(|&stored_key| stored_key != key);
        if keys.is_empty() {
          table.remove(&band_hash);
        }
      }
    }
  }

  pub(in crate::lsh) fn collect_unique_candidates(
    &self,
    digest: &[u32],
    seen: &mut FxHashSet<usize>,
    candidates: &mut Vec<usize>,
  ) {
    seen.clear();
    candidates.clear();

    for (i, table) in self.hash_tables.iter().enumerate() {
      let band_hash = calculate_band_hash(
        &digest[i * self.band_size..(i + 1) * self.band_size],
      );
      if let Some(keys) = table.get(&band_hash) {
        for &key in keys {
          if seen.insert(key) {
            candidates.push(key);
          }
        }
      }
    }
  }

  pub(in crate::lsh) fn has_multiple_candidates(
    &self,
    digest: &[u32],
    seen: &mut FxHashSet<usize>,
  ) -> bool {
    seen.clear();

    for (i, table) in self.hash_tables.iter().enumerate() {
      let band_hash = calculate_band_hash(
        &digest[i * self.band_size..(i + 1) * self.band_size],
      );
      if let Some(keys) = table.get(&band_hash) {
        for &key in keys {
          if seen.insert(key) && seen.len() > 1 {
            return true;
          }
        }
      }
    }
    false
  }

  pub(in crate::lsh) fn insert_digest(
    &mut self,
    key: usize,
    digest: &[u32],
  ) -> PyResult<()> {
    self.ensure_digest_len(digest.len())?;

    if let Some(previous_band_hashes) = self.key_bands.remove(&key) {
      self.remove_key_from_bands(key, &previous_band_hashes);
    }

    let band_hashes = self.digest_band_hashes(digest);
    for (table, &band_hash) in
      self.hash_tables.iter_mut().zip(band_hashes.iter())
    {
      table.entry(band_hash).or_default().push(key);
    }

    self.key_bands.insert(key, band_hashes);
    Ok(())
  }
}
