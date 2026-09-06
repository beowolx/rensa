//! Locality-Sensitive Hashing (LSH) for `MinHash`.
//!
//! This module implements `RMinHashLSH`, a Locality-Sensitive Hashing scheme
//! that uses `RMinHash` signatures to efficiently find
//! approximate nearest neighbors in large datasets. It's designed for identifying
//! items with high Jaccard similarity.
//!
//! The core idea of LSH is to hash input items such that similar items are mapped
//! to the same "buckets" with high probability, while dissimilar items are not.
//! This implementation achieves this by:
//! 1. Generating `MinHash` signatures for items using `RMinHash`.
//! 2. Dividing these signatures into several "bands".
//! 3. For each band, hashing its portion of the signature.
//! 4. Items are considered candidates for similarity if they share the same hash
//!    value in at least one band.
//!
//! This approach allows for querying similar items much faster than pairwise
//! comparisons, especially for large numbers of items.
//!
//! ## Usage:
//!
//! An `RMinHashLSH` index is initialized with a Jaccard similarity threshold, the number of
//! permutations for the `MinHash` signatures, and the number of bands to use for LSH.
//! `RMinHash` objects (representing items) are inserted into the index. Queries with an
//! `RMinHash` object will return a set of keys of potentially similar items.
//!
//! Key methods include:
//! - `new(threshold, num_perm, num_bands)`: Initializes a new LSH index.
//! - `insert(key, minhash)`: Inserts an item's `MinHash` signature into the index.
//! - `remove(key)`: Removes a previously inserted key from the index.
//! - `query(minhash)`: Retrieves candidate keys that are potentially similar to the query `MinHash`.
//! - `is_similar(minhash1, minhash2)`: Directly checks if two `MinHashes` meet the similarity threshold.
//!
//! This LSH implementation is particularly useful for tasks such as near-duplicate detection,
//! document clustering, etc.

use pyo3::prelude::*;
use rustc_hash::FxHasher;
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::HashMap;
use std::hash::BuildHasherDefault;

mod config;
mod index;
mod one_shot;
mod py;

#[cfg(target_pointer_width = "64")]
const FX_POLY_K: usize = 0xf135_7aea_2e62_a9c5;
#[cfg(target_pointer_width = "32")]
const FX_POLY_K: usize = 0x93d7_65dd;

#[cfg(target_pointer_width = "64")]
const FX_FINISH_ROTATE: u32 = 26;
#[cfg(target_pointer_width = "32")]
const FX_FINISH_ROTATE: u32 = 15;

/// `RMinHashLSH` implements Locality-Sensitive Hashing using `MinHash` for efficient similarity search.
#[pyclass(module = "rensa")]
pub struct RMinHashLSH {
  threshold: f64,
  num_perm: usize,
  num_bands: usize,
  band_size: usize,
  // These maps are keyed by internal band hashes / numeric IDs, not
  // attacker-controlled strings, so FxHasher is chosen for speed.
  hash_tables: Vec<HashMap<u64, Vec<usize>, BuildHasherDefault<FxHasher>>>,
  key_bands: HashMap<usize, Vec<u64>, BuildHasherDefault<FxHasher>>,
  last_one_shot_sparse_verify_checks: usize,
  last_one_shot_sparse_verify_passes: usize,
}

#[derive(Deserialize)]
struct RMinHashLSHState {
  #[serde(
    rename = "version",
    deserialize_with = "crate::rminhash::deserialize_state_version"
  )]
  _version: (),
  threshold: f64,
  num_perm: usize,
  num_bands: usize,
  band_size: usize,
  hash_tables: Vec<HashMap<u64, Vec<usize>, BuildHasherDefault<FxHasher>>>,
  #[serde(default)]
  key_bands: HashMap<usize, Vec<u64>, BuildHasherDefault<FxHasher>>,
}

impl Serialize for RMinHashLSH {
  fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
  where
    S: Serializer,
  {
    let mut state = serializer.serialize_struct("RMinHashLSH", 7)?;
    state.serialize_field("version", &crate::rminhash::STATE_VERSION)?;
    state.serialize_field("threshold", &self.threshold)?;
    state.serialize_field("num_perm", &self.num_perm)?;
    state.serialize_field("num_bands", &self.num_bands)?;
    state.serialize_field("band_size", &self.band_size)?;
    state.serialize_field("hash_tables", &self.hash_tables)?;
    state.serialize_field("key_bands", &self.key_bands)?;
    state.end()
  }
}

impl<'de> Deserialize<'de> for RMinHashLSH {
  fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
  where
    D: Deserializer<'de>,
  {
    let state = RMinHashLSHState::deserialize(deserializer)?;
    let decoded = Self {
      threshold: state.threshold,
      num_perm: state.num_perm,
      num_bands: state.num_bands,
      band_size: state.band_size,
      hash_tables: state.hash_tables,
      key_bands: state.key_bands,
      last_one_shot_sparse_verify_checks: 0,
      last_one_shot_sparse_verify_passes: 0,
    };
    decoded
      .validate_state_inner()
      .map_err(serde::de::Error::custom)?;
    Ok(decoded)
  }
}

impl RMinHashLSH {
  fn validate_params_inner(
    threshold: f64,
    num_perm: usize,
    num_bands: usize,
  ) -> Result<usize, String> {
    if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
      return Err(
        "threshold must be a finite value between 0.0 and 1.0".to_owned(),
      );
    }
    if num_perm == 0 {
      return Err("num_perm must be greater than 0".to_owned());
    }
    if num_bands == 0 {
      return Err("num_bands must be greater than 0".to_owned());
    }
    if num_bands > num_perm {
      return Err(format!(
        "num_bands ({num_bands}) must be less than or equal to num_perm ({num_perm})"
      ));
    }
    if num_perm % num_bands != 0 {
      return Err(format!(
        "num_perm ({num_perm}) must be divisible by num_bands ({num_bands})"
      ));
    }
    Ok(num_perm / num_bands)
  }

  fn validate_state_inner(&self) -> Result<(), String> {
    let expected_band_size = Self::validate_params_inner(
      self.threshold,
      self.num_perm,
      self.num_bands,
    )?;
    if self.band_size != expected_band_size {
      return Err(format!(
        "invalid RMinHashLSH state: band_size {} does not match expected {}",
        self.band_size, expected_band_size
      ));
    }
    if self.hash_tables.len() != self.num_bands {
      return Err(format!(
        "invalid RMinHashLSH state: hash_tables length {} does not match num_bands {}",
        self.hash_tables.len(), self.num_bands
      ));
    }
    for (key, band_hashes) in &self.key_bands {
      if band_hashes.len() != self.num_bands {
        return Err(format!(
          "invalid RMinHashLSH state: key {key} stores {} band hashes, expected {}",
          band_hashes.len(), self.num_bands
        ));
      }
    }
    Ok(())
  }

  #[inline]
  const fn fx_poly_steps(len_u32: usize) -> usize {
    // `calculate_band_hash` packs 4x u32 into 2x u64 writes, then writes any
    // remainder u32 values. The polynomial state multiplies by K per write.
    (len_u32 / 4) * 2 + (len_u32 % 4)
  }

  #[inline]
  fn fx_poly_k_pow(steps: usize) -> usize {
    let mut result = 1_usize;
    for _ in 0..steps {
      result = result.wrapping_mul(FX_POLY_K);
    }
    result
  }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod deserialization_tests {
  use crate::lsh::RMinHashLSH;

  #[test]
  fn native_deserialization_rejects_invalid_index_dimensions() {
    for invalid_field in 0..9 {
      let mut index = RMinHashLSH::from_validated(0.8, 128, 8);
      match invalid_field {
        0 => index.threshold = f64::NAN,
        1 => index.threshold = 1.1,
        2 => index.num_perm = 0,
        3 => index.num_bands = 0,
        4 => index.num_bands = 129,
        5 => index.num_perm = 129,
        6 => index.band_size = 129,
        7 => {
          index.hash_tables.pop();
        }
        _ => {
          index.key_bands.insert(0, vec![0; 7]);
        }
      }
      let bytes = postcard::to_allocvec(&index).unwrap();
      assert!(
        postcard::from_bytes::<RMinHashLSH>(&bytes).is_err(),
        "accepted invalid field {invalid_field}"
      );
    }
  }
}
