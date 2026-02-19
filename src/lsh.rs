//! Locality-Sensitive Hashing (LSH) for `MinHash`.
//!
//! This module implements `RMinHashLSH`, a Locality-Sensitive Hashing scheme
//! that uses `RMinHash` (Rensa's novel `MinHash` variant) to efficiently find
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
use serde::{Deserialize, Deserializer, Serialize};
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
#[derive(Serialize)]
#[pyclass(module = "rensa")]
pub struct RMinHashLSH {
  threshold: f64,
  num_perm: usize,
  num_bands: usize,
  band_size: usize,
  hash_tables: Vec<HashMap<u64, Vec<usize>, BuildHasherDefault<FxHasher>>>,
  #[serde(default)]
  key_bands: HashMap<usize, Vec<u64>, BuildHasherDefault<FxHasher>>,
  #[serde(skip, default)]
  last_one_shot_sparse_verify_checks: usize,
  #[serde(skip, default)]
  last_one_shot_sparse_verify_passes: usize,
}

#[derive(Deserialize)]
struct RMinHashLSHState {
  threshold: f64,
  num_perm: usize,
  num_bands: usize,
  band_size: usize,
  hash_tables: Vec<HashMap<u64, Vec<usize>, BuildHasherDefault<FxHasher>>>,
  #[serde(default)]
  key_bands: HashMap<usize, Vec<u64>, BuildHasherDefault<FxHasher>>,
}

impl<'de> Deserialize<'de> for RMinHashLSH {
  fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
  where
    D: Deserializer<'de>,
  {
    let state = RMinHashLSHState::deserialize(deserializer)?;
    Ok(Self {
      threshold: state.threshold,
      num_perm: state.num_perm,
      num_bands: state.num_bands,
      band_size: state.band_size,
      hash_tables: state.hash_tables,
      key_bands: state.key_bands,
      last_one_shot_sparse_verify_checks: 0,
      last_one_shot_sparse_verify_passes: 0,
    })
  }
}

impl RMinHashLSH {
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
