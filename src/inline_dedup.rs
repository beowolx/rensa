//! Inline deduplication support for continuous duplicate detection.
//!
//! This module provides functionality to check new records against an existing
//! dataset for duplicates in real-time, supporting all `MinHash` variants.

use crate::cminhash::CMinHash;
use crate::lsh::RMinHashLSH;
use crate::rminhash::RMinHash;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rustc_hash::FxHashMap;

#[pyclass(module = "rensa")]
pub struct RMinHashDeduplicator {
  threshold: f64,
  existing_signatures: FxHashMap<String, RMinHash>,
  lsh_index: Option<RMinHashLSH>,
  next_id: usize,
  id_to_key: FxHashMap<usize, String>,
  key_to_id: FxHashMap<String, usize>,
}

#[pymethods]
impl RMinHashDeduplicator {
  #[new]
  #[pyo3(signature = (threshold, num_perm, use_lsh, num_bands=None))]
  ///
  /// # Errors
  ///
  /// Returns an error if thresholds or permutation parameters are invalid.
  pub fn new(
    threshold: f64,
    num_perm: usize,
    use_lsh: bool,
    num_bands: Option<usize>,
  ) -> PyResult<Self> {
    if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
      return Err(PyValueError::new_err(
        "threshold must be a finite value between 0.0 and 1.0",
      ));
    }
    if num_perm == 0 {
      return Err(PyValueError::new_err("num_perm must be greater than 0"));
    }

    let lsh_index = if use_lsh {
      let default_bands = if threshold >= 0.9 {
        4
      } else if threshold >= 0.8 {
        8
      } else if threshold >= 0.7 {
        16
      } else if threshold >= 0.5 {
        32
      } else {
        (num_perm / 2).max(1)
      };
      let bands = if let Some(requested_bands) = num_bands {
        if requested_bands == 0 {
          return Err(PyValueError::new_err(
            "num_bands must be greater than 0",
          ));
        }
        if requested_bands > num_perm {
          return Err(PyValueError::new_err(format!(
            "num_bands ({requested_bands}) must be less than or equal to num_perm ({num_perm})",
          )));
        }
        if !num_perm.is_multiple_of(requested_bands) {
          return Err(PyValueError::new_err(format!(
            "num_perm ({num_perm}) must be divisible by num_bands ({requested_bands})",
          )));
        }
        requested_bands
      } else {
        let upper = default_bands.min(num_perm).max(1);
        (1..=upper)
          .rev()
          .find(|&candidate| num_perm.is_multiple_of(candidate))
          .unwrap_or(1)
      };

      Some(RMinHashLSH::new(threshold, num_perm, bands)?)
    } else {
      None
    };

    Ok(Self {
      threshold,
      existing_signatures: FxHashMap::default(),
      lsh_index,
      next_id: 0,
      id_to_key: FxHashMap::default(),
      key_to_id: FxHashMap::default(),
    })
  }

  /// Add a new item to the deduplicator. Returns true if added (not a duplicate), false if duplicate.
  ///
  /// # Errors
  ///
  /// Returns an error if `minhash` is incompatible with the configured signature size.
  pub fn add(&mut self, key: String, minhash: &RMinHash) -> PyResult<bool> {
    if self.is_duplicate(&key, minhash)? {
      return Ok(false);
    }

    self
      .existing_signatures
      .insert(key.clone(), minhash.clone());

    if let Some(ref mut lsh) = self.lsh_index {
      let next_id = self.next_id;
      lsh.insert(next_id, minhash)?;
      self.id_to_key.insert(next_id, key.clone());
      self.key_to_id.insert(key, next_id);
      self.next_id += 1;
    }

    Ok(true)
  }

  /// Check if an item is a duplicate without adding it
  ///
  /// # Errors
  ///
  /// Returns an error if `minhash` is incompatible with stored signatures.
  pub fn is_duplicate(&self, key: &str, minhash: &RMinHash) -> PyResult<bool> {
    if self.existing_signatures.contains_key(key) {
      return Ok(true);
    }

    if let Some(ref lsh) = self.lsh_index {
      let candidates = lsh.query(minhash)?;
      for candidate_id in candidates {
        if let Some(candidate_key) = self.id_to_key.get(&candidate_id) {
          if let Some(candidate_minhash) =
            self.existing_signatures.get(candidate_key)
          {
            if minhash.jaccard(candidate_minhash)? >= self.threshold {
              return Ok(true);
            }
          }
        }
      }
    } else {
      for existing_minhash in self.existing_signatures.values() {
        if minhash.jaccard(existing_minhash)? >= self.threshold {
          return Ok(true);
        }
      }
    }

    Ok(false)
  }

  /// Get duplicate candidates for a given `MinHash`
  ///
  /// # Errors
  ///
  /// Returns an error if `minhash` is incompatible with stored signatures.
  pub fn get_duplicates(&self, minhash: &RMinHash) -> PyResult<Vec<String>> {
    let mut duplicates = Vec::new();

    if let Some(ref lsh) = self.lsh_index {
      let candidates = lsh.query(minhash)?;
      for candidate_id in candidates {
        if let Some(candidate_key) = self.id_to_key.get(&candidate_id) {
          if let Some(candidate_minhash) =
            self.existing_signatures.get(candidate_key)
          {
            if minhash.jaccard(candidate_minhash)? >= self.threshold {
              duplicates.push(candidate_key.clone());
            }
          }
        }
      }
    } else {
      for (key, existing_minhash) in &self.existing_signatures {
        if minhash.jaccard(existing_minhash)? >= self.threshold {
          duplicates.push(key.clone());
        }
      }
    }

    Ok(duplicates)
  }

  /// Remove an item from the deduplicator
  pub fn remove(&mut self, key: &str) -> bool {
    if let Some(id) = self.key_to_id.remove(key) {
      self.id_to_key.remove(&id);
      if let Some(ref mut lsh) = self.lsh_index {
        let _ = lsh.remove(id);
      }
    }

    self.existing_signatures.remove(key).is_some()
  }

  /// Get the number of items in the deduplicator
  #[must_use]
  pub fn len(&self) -> usize {
    self.existing_signatures.len()
  }

  /// Check if the deduplicator is empty
  #[must_use]
  pub fn is_empty(&self) -> bool {
    self.existing_signatures.is_empty()
  }

  /// Clear all items from the deduplicator
  ///
  /// # Errors
  ///
  /// Returns an error if the internal LSH index cannot be recreated.
  pub fn clear(&mut self) -> PyResult<()> {
    self.existing_signatures.clear();
    self.id_to_key.clear();
    self.key_to_id.clear();
    self.next_id = 0;
    if let Some((num_perm, num_bands)) = self
      .lsh_index
      .as_ref()
      .map(|lsh| (lsh.get_num_perm(), lsh.get_num_bands()))
    {
      self.lsh_index =
        Some(RMinHashLSH::new(self.threshold, num_perm, num_bands)?);
    }
    Ok(())
  }
}

/// `InlineDeduplicator` for `CMinHash`
#[pyclass(module = "rensa")]
pub struct CMinHashDeduplicator {
  threshold: f64,
  existing_signatures: FxHashMap<String, CMinHash>,
}

#[pymethods]
impl CMinHashDeduplicator {
  #[new]
  ///
  /// # Errors
  ///
  /// Returns an error if `threshold` is invalid.
  pub fn new(threshold: f64) -> PyResult<Self> {
    if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
      return Err(PyValueError::new_err(
        "threshold must be a finite value between 0.0 and 1.0",
      ));
    }

    Ok(Self {
      threshold,
      existing_signatures: FxHashMap::default(),
    })
  }

  /// # Errors
  ///
  /// Returns an error if `minhash` is incompatible with stored signatures.
  pub fn add(&mut self, key: String, minhash: &CMinHash) -> PyResult<bool> {
    if self.is_duplicate(&key, minhash)? {
      return Ok(false);
    }

    self.existing_signatures.insert(key, minhash.clone());
    Ok(true)
  }

  /// # Errors
  ///
  /// Returns an error if `minhash` is incompatible with stored signatures.
  pub fn is_duplicate(&self, key: &str, minhash: &CMinHash) -> PyResult<bool> {
    if self.existing_signatures.contains_key(key) {
      return Ok(true);
    }

    for existing_minhash in self.existing_signatures.values() {
      if minhash.jaccard(existing_minhash)? >= self.threshold {
        return Ok(true);
      }
    }

    Ok(false)
  }

  /// # Errors
  ///
  /// Returns an error if `minhash` is incompatible with stored signatures.
  pub fn get_duplicates(&self, minhash: &CMinHash) -> PyResult<Vec<String>> {
    let mut duplicates = Vec::new();

    for (key, existing_minhash) in &self.existing_signatures {
      if minhash.jaccard(existing_minhash)? >= self.threshold {
        duplicates.push(key.clone());
      }
    }

    Ok(duplicates)
  }

  pub fn remove(&mut self, key: &str) -> bool {
    self.existing_signatures.remove(key).is_some()
  }

  #[must_use]
  pub fn len(&self) -> usize {
    self.existing_signatures.len()
  }

  #[must_use]
  pub fn is_empty(&self) -> bool {
    self.existing_signatures.is_empty()
  }

  pub fn clear(&mut self) {
    self.existing_signatures.clear();
  }
}
