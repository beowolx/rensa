use crate::{CMinHash, OptDensMinHash, RMinHash};
use pyo3::prelude::*;
use std::collections::HashSet;

/// Inline deduplicator using `RMinHash`.
#[pyclass(module = "rensa")]
pub struct RMinHashDeduper {
    num_perm: usize,
    seed: u64,
    #[pyo3(get)]
    seen: HashSet<Vec<u32>>, 
    rminhash: RMinHash,
}

#[pymethods]
impl RMinHashDeduper {
    #[new]
    #[must_use]
    pub fn new(num_perm: usize, seed: u64) -> Self {
        Self {
            num_perm,
            seed,
            seen: HashSet::new(),
            rminhash: RMinHash::new(num_perm, seed),
        }
    }

    /// Adds a new record. Returns `True` if it is unique.
    pub fn add(&mut self, items: Vec<String>) -> bool {
        self.rminhash.update(items);
        let digest = self.rminhash.digest();
        self.rminhash.clear();
        self.seen.insert(digest)
    }
}

/// Inline deduplicator using `CMinHash`.
#[pyclass(module = "rensa")]
pub struct CMinHashDeduper {
    num_perm: usize,
    seed: u64,
    #[pyo3(get)]
    seen: HashSet<Vec<u32>>, 
    cminhash: CMinHash,
}

#[pymethods]
impl CMinHashDeduper {
    #[new]
    #[must_use]
    pub fn new(num_perm: usize, seed: u64) -> Self {
        Self {
            num_perm,
            seed,
            seen: HashSet::new(),
            cminhash: CMinHash::new(num_perm, seed),
        }
    }

    /// Adds a new record. Returns `True` if it is unique.
    pub fn add(&mut self, items: Vec<String>) -> bool {
        self.cminhash.update(items);
        let digest = self.cminhash.digest();
        self.cminhash.clear();
        self.seen.insert(digest)
    }
}

/// Inline deduplicator using `OptDensMinHash`.
#[pyclass(module = "rensa")]
pub struct OptDensDeduper {
    num_perm: usize,
    seed: u64,
    #[pyo3(get)]
    seen: HashSet<Vec<u32>>, 
    minhash: OptDensMinHash,
}

#[pymethods]
impl OptDensDeduper {
    #[new]
    #[must_use]
    pub fn new(num_perm: usize, seed: u64) -> Self {
        Self {
            num_perm,
            seed,
            seen: HashSet::new(),
            minhash: OptDensMinHash::new(num_perm, seed),
        }
    }

    /// Adds a new record. Returns `True` if it is unique.
    pub fn add(&mut self, items: Vec<String>) -> bool {
        self.minhash.update(items);
        let digest = self.minhash.digest();
        self.minhash.clear();
        self.seen.insert(digest)
    }
}

