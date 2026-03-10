//! Inline deduplication support for continuous duplicate detection.
//!
//! This module provides functionality to check new records against an existing
//! dataset for duplicates in real-time, supporting all `MinHash` variants.

use crate::cminhash::CMinHash;
use crate::lsh::RMinHashLSH;
use crate::rminhash::RMinHash;
use pyo3::prelude::*;
use rustc_hash::FxHashMap;

mod cminhash;
mod common;
mod rminhash;

#[pyclass(module = "rensa")]
pub struct RMinHashDeduplicator {
  threshold: f64,
  num_perm: usize,
  seed: u64,
  entries_by_id: FxHashMap<usize, (String, RMinHash)>,
  lsh_index: Option<RMinHashLSH>,
  next_id: usize,
  key_to_id: FxHashMap<String, usize>,
}

/// `InlineDeduplicator` for `CMinHash`
#[pyclass(module = "rensa")]
pub struct CMinHashDeduplicator {
  threshold: f64,
  existing_signatures: FxHashMap<String, CMinHash>,
  num_perm: Option<usize>,
  seed: u64,
}
