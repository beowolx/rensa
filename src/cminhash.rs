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

use pyo3::prelude::*;
use serde::{Deserialize, Deserializer, Serialize};

mod batch;
mod core;
mod py;

const HASH_BATCH_SIZE: usize = 32;
type ReduceResult = (Py<PyAny>, (usize, u64), Py<PyAny>);

/// `CMinHash` implements an optimized version of `C-MinHash` with better memory access patterns
/// and aggressive optimizations for maximum single-threaded performance.
#[derive(Serialize, Clone)]
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

#[derive(Deserialize)]
struct CMinHashState {
  num_perm: usize,
  seed: u64,
  hash_values: Vec<u64>,
  sigma_a: u64,
  sigma_b: u64,
  pi_c: u64,
  pi_d: u64,
}

impl<'de> Deserialize<'de> for CMinHash {
  fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
  where
    D: Deserializer<'de>,
  {
    let state = CMinHashState::deserialize(deserializer)?;
    Ok(Self {
      num_perm: state.num_perm,
      seed: state.seed,
      hash_values: state.hash_values,
      sigma_a: state.sigma_a,
      sigma_b: state.sigma_b,
      pi_c: state.pi_c,
      pi_d: state.pi_d,
      pi_precomputed: Vec::new(),
    })
  }
}
