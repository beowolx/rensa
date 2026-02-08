//! Inline deduplication support for continuous duplicate detection.
//!
//! This module provides functionality to check new records against an existing
//! dataset for duplicates in real-time, supporting all `MinHash` variants.

use crate::cminhash::CMinHash;
use crate::lsh::RMinHashLSH;
use crate::rminhash::RMinHash;
use pyo3::prelude::*;
use rustc_hash::FxHashMap;

#[pyclass(module = "rensa")]
pub struct RMinHashDeduplicator {
  threshold: f64,
  entries_by_id: FxHashMap<usize, (String, RMinHash)>,
  lsh_index: Option<RMinHashLSH>,
  next_id: usize,
  key_to_id: FxHashMap<String, usize>,
}

#[pymethods]
impl RMinHashDeduplicator {
  #[new]
  #[pyo3(signature = (threshold, num_perm, use_lsh, num_bands=None))]
  #[must_use]
  pub fn new(
    threshold: f64,
    num_perm: usize,
    use_lsh: bool,
    num_bands: Option<usize>,
  ) -> Self {
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
        num_perm / 2
      };
      let bands = num_bands.unwrap_or(default_bands);

      let bands = if num_perm.is_multiple_of(bands) {
        bands
      } else {
        (1..=num_perm)
          .rev()
          .find(|&b| num_perm.is_multiple_of(b) && b <= bands)
          .unwrap_or(1)
      };

      Some(RMinHashLSH::new(threshold, num_perm, bands))
    } else {
      None
    };

    Self {
      threshold,
      entries_by_id: FxHashMap::default(),
      lsh_index,
      next_id: 0,
      key_to_id: FxHashMap::default(),
    }
  }

  /// Add a new item to the deduplicator. Returns true if added (not a duplicate), false if duplicate.
  pub fn add(&mut self, key: String, minhash: &RMinHash) -> bool {
    if self.is_duplicate(&key, minhash) {
      return false;
    }

    let next_id = self.next_id;
    self
      .entries_by_id
      .insert(next_id, (key.clone(), minhash.clone()));
    self.key_to_id.insert(key, next_id);

    if let Some(ref mut lsh) = self.lsh_index {
      lsh.insert(next_id, minhash);
    }

    self.next_id += 1;
    true
  }

  /// Check if an item is a duplicate without adding it
  #[must_use]
  pub fn is_duplicate(&self, key: &str, minhash: &RMinHash) -> bool {
    if self.key_to_id.contains_key(key) {
      return true;
    }

    if let Some(ref lsh) = self.lsh_index {
      let candidates = lsh.query(minhash);
      for candidate_id in candidates {
        if let Some((_, candidate_minhash)) =
          self.entries_by_id.get(&candidate_id)
        {
          if minhash.jaccard(candidate_minhash) >= self.threshold {
            return true;
          }
        }
      }
    } else {
      for (_, existing_minhash) in self.entries_by_id.values() {
        if minhash.jaccard(existing_minhash) >= self.threshold {
          return true;
        }
      }
    }

    false
  }

  /// Get duplicate candidates for a given `MinHash`
  #[must_use]
  pub fn get_duplicates(&self, minhash: &RMinHash) -> Vec<String> {
    let mut duplicates = Vec::new();

    if let Some(ref lsh) = self.lsh_index {
      let candidates = lsh.query(minhash);
      for candidate_id in candidates {
        if let Some((candidate_key, candidate_minhash)) =
          self.entries_by_id.get(&candidate_id)
        {
          if minhash.jaccard(candidate_minhash) >= self.threshold {
            duplicates.push(candidate_key.clone());
          }
        }
      }
    } else {
      for (key, existing_minhash) in self.entries_by_id.values() {
        if minhash.jaccard(existing_minhash) >= self.threshold {
          duplicates.push(key.clone());
        }
      }
    }

    duplicates
  }

  /// Remove an item from the deduplicator
  pub fn remove(&mut self, key: &str) -> bool {
    let Some(id) = self.key_to_id.remove(key) else {
      return false;
    };

    if let Some(ref mut lsh) = self.lsh_index {
      let _ = lsh.remove(id);
    }

    self.entries_by_id.remove(&id).is_some()
  }

  /// Get the number of items in the deduplicator
  #[must_use]
  pub fn len(&self) -> usize {
    self.entries_by_id.len()
  }

  /// Check if the deduplicator is empty
  #[must_use]
  pub fn is_empty(&self) -> bool {
    self.entries_by_id.is_empty()
  }

  /// Clear all items from the deduplicator
  pub fn clear(&mut self) {
    self.entries_by_id.clear();
    self.key_to_id.clear();
    self.next_id = 0;
    if let Some(ref mut lsh) = self.lsh_index {
      // Recreate LSH index
      let threshold = self.threshold;
      let num_perm = lsh.get_num_perm();
      let num_bands = lsh.get_num_bands();
      self.lsh_index = Some(RMinHashLSH::new(threshold, num_perm, num_bands));
    }
  }
}

/// `InlineDeduplicator` for `CMinHash`
#[pyclass(module = "rensa")]
pub struct CMinHashDeduplicator {
  threshold: f64,
  existing_signatures: FxHashMap<String, CMinHash>,
  signature_cache: FxHashMap<String, Vec<u64>>,
}

#[pymethods]
impl CMinHashDeduplicator {
  #[new]
  #[must_use]
  pub fn new(threshold: f64) -> Self {
    Self {
      threshold,
      existing_signatures: FxHashMap::default(),
      signature_cache: FxHashMap::default(),
    }
  }

  pub fn add(&mut self, key: String, minhash: &CMinHash) -> bool {
    if self.is_duplicate(&key, minhash) {
      return false;
    }

    let signature = minhash.digest_u64();
    self.signature_cache.insert(key.clone(), signature);
    self.existing_signatures.insert(key, minhash.clone());
    true
  }

  #[must_use]
  pub fn is_duplicate(&self, key: &str, minhash: &CMinHash) -> bool {
    if self.existing_signatures.contains_key(key) {
      return true;
    }

    for existing_minhash in self.existing_signatures.values() {
      if minhash.jaccard(existing_minhash) >= self.threshold {
        return true;
      }
    }

    false
  }

  #[must_use]
  pub fn get_duplicates(&self, minhash: &CMinHash) -> Vec<String> {
    let mut duplicates = Vec::new();

    for (key, existing_minhash) in &self.existing_signatures {
      if minhash.jaccard(existing_minhash) >= self.threshold {
        duplicates.push(key.clone());
      }
    }

    duplicates
  }

  pub fn remove(&mut self, key: &str) -> bool {
    self.signature_cache.remove(key);
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
    self.signature_cache.clear();
  }
}
