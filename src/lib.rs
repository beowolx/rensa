#![deny(clippy::all)]
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]
#![allow(clippy::unsafe_derive_deserialize)]
#![allow(clippy::cast_precision_loss)]

use bincode::{deserialize, serialize};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rand::prelude::*;
use rustc_hash::FxHasher;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// RMinHash implements the MinHash algorithm for efficient similarity estimation.
#[derive(Serialize, Deserialize)]
#[pyclass(module = "rensa")]
struct RMinHash {
  num_perm: usize,
  seed: u64,
  hash_values: Vec<u32>,
  permutations: Vec<(u64, u64)>,
}

#[pymethods]
impl RMinHash {
  /// Creates a new RMinHash instance.
  ///
  /// # Arguments
  ///
  /// * `num_perm` - The number of permutations to use in the MinHash algorithm.
  /// * `seed` - A seed value for the random number generator.
  #[new]
  fn new(num_perm: usize, seed: u64) -> Self {
    let mut rng = StdRng::seed_from_u64(seed);
    let permutations: Vec<(u64, u64)> = (0..num_perm)
      .map(|_| (rng.random(), rng.random()))
      .collect();

    Self {
      num_perm,
      seed,
      hash_values: vec![u32::MAX; num_perm],
      permutations,
    }
  }

  /// Updates the MinHash with a new set of items.
  ///
  /// # Arguments
  ///
  /// * `items` - A vector of strings to be hashed and incorporated into the MinHash.
  fn update(&mut self, items: Vec<String>) {
    for item in items {
      let item_hash = calculate_hash(&item);
      for (i, &(a, b)) in self.permutations.iter().enumerate() {
        let hash = permute_hash(item_hash, a, b);
        self.hash_values[i] = self.hash_values[i].min(hash);
      }
    }
  }

  /// Returns the current MinHash digest.
  ///
  /// # Returns
  ///
  /// A vector of u32 values representing the MinHash signature.
  fn digest(&self) -> Vec<u32> {
    self.hash_values.clone()
  }

  /// Calculates the Jaccard similarity between this MinHash and another.
  ///
  /// # Arguments
  ///
  /// * `other` - Another RMinHash instance to compare with.
  ///
  /// # Returns
  ///
  /// A float value representing the estimated Jaccard similarity.
  fn jaccard(&self, other: &Self) -> f64 {
    let equal_count = self
      .hash_values
      .iter()
      .zip(&other.hash_values)
      .filter(|&(&a, &b)| a == b)
      .count();
    // Safe because self.num_perm is expected to be << 2^53.
    equal_count as f64 / self.num_perm as f64
  }

  fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) {
    *self = deserialize(state.as_bytes()).unwrap();
  }

  fn __getstate__<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
    PyBytes::new(py, &serialize(&self).unwrap())
  }

  const fn __getnewargs__(&self) -> (usize, u64) {
    (self.num_perm, self.seed)
  }

  fn __reduce__(&self) -> PyResult<(PyObject, (usize, u64), PyObject)> {
    Python::with_gil(|py| {
      let type_obj = py.get_type::<Self>().into();
      let state = self.__getstate__(py).into();
      Ok((type_obj, (self.num_perm, self.seed), state))
    })
  }
}

/// RMinHashLSH implements Locality-Sensitive Hashing using MinHash for efficient similarity search.
#[derive(Serialize, Deserialize)]
#[pyclass(module = "rensa")]
struct RMinHashLSH {
  threshold: f64,
  num_perm: usize,
  num_bands: usize,
  band_size: usize,
  hash_tables: Vec<HashMap<u64, Vec<usize>>>,
}

#[pymethods]
impl RMinHashLSH {
  /// Creates a new RMinHashLSH instance.
  ///
  /// # Arguments
  ///
  /// * `threshold` - The similarity threshold for considering items as similar.
  /// * `num_perm` - The number of permutations used in the MinHash algorithm.
  /// * `num_bands` - The number of bands for the LSH algorithm.
  #[new]
  fn new(threshold: f64, num_perm: usize, num_bands: usize) -> Self {
    Self {
      threshold,
      num_perm,
      num_bands,
      band_size: num_perm / num_bands,
      hash_tables: vec![HashMap::new(); num_bands],
    }
  }

  /// Inserts a MinHash into the LSH index.
  ///
  /// # Arguments
  ///
  /// * `key` - A unique identifier for the MinHash.
  /// * `minhash` - The RMinHash instance to be inserted.
  fn insert(&mut self, key: usize, minhash: &RMinHash) {
    let digest = minhash.digest();
    for (i, table) in self.hash_tables.iter_mut().enumerate() {
      let start = i * self.band_size;
      let end = start + self.band_size;
      let band_hash = calculate_band_hash(&digest[start..end]);
      table.entry(band_hash).or_insert_with(Vec::new).push(key);
    }
  }

  /// Queries the LSH index for similar items.
  ///
  /// # Arguments
  ///
  /// * `minhash` - The RMinHash instance to query for.
  ///
  /// # Returns
  ///
  /// A vector of keys (usize) of potentially similar items.
  fn query(&self, minhash: &RMinHash) -> Vec<usize> {
    let digest = minhash.digest();
    let mut candidates = Vec::new();
    for (i, table) in self.hash_tables.iter().enumerate() {
      let start = i * self.band_size;
      let end = start + self.band_size;
      let band_hash = calculate_band_hash(&digest[start..end]);
      if let Some(keys) = table.get(&band_hash) {
        candidates.extend(keys);
      }
    }
    candidates.sort_unstable();
    candidates.dedup();
    candidates
  }

  /// Checks if two MinHashes are similar based on the LSH threshold.
  ///
  /// # Arguments
  ///
  /// * `minhash1` - The first RMinHash instance.
  /// * `minhash2` - The second RMinHash instance.
  ///
  /// # Returns
  ///
  /// A boolean indicating whether the MinHashes are considered similar.
  fn is_similar(&self, minhash1: &RMinHash, minhash2: &RMinHash) -> bool {
    minhash1.jaccard(minhash2) >= self.threshold
  }

  /// Returns the number of permutations used in the LSH index.
  const fn get_num_perm(&self) -> usize {
    self.num_perm
  }

  /// Returns the number of bands used in the LSH index.
  const fn get_num_bands(&self) -> usize {
    self.num_bands
  }

  fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) {
    *self = deserialize(state.as_bytes()).unwrap();
  }

  fn __getstate__<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
    PyBytes::new(py, &serialize(&self).unwrap())
  }

  const fn __getnewargs__(&self) -> (f64, usize, usize) {
    (self.threshold, self.num_perm, self.num_bands)
  }
}

/// Calculates a hash value for a given item.
#[inline]
fn calculate_hash<T: Hash>(t: &T) -> u64 {
  let mut s = FxHasher::default();
  t.hash(&mut s);
  s.finish()
}

/// Applies a permutation to a hash value.
#[inline]
const fn permute_hash(hash: u64, a: u64, b: u64) -> u32 {
  ((a.wrapping_mul(hash).wrapping_add(b)) >> 32) as u32
}

/// Calculates a hash value for a band of `MinHash` values.
#[inline]
fn calculate_band_hash(band: &[u32]) -> u64 {
  let mut hasher = FxHasher::default();
  for &value in band {
    hasher.write_u32(value);
  }
  hasher.finish()
}

/// Python module for MinHash and LSH implementations
///
/// # Errors
/// Returns an error if the module initialization fails or classes cannot be added
#[pymodule]
pub fn rensa(m: &Bound<'_, PyModule>) -> PyResult<()> {
  m.add_class::<RMinHash>()?;
  m.add_class::<RMinHashLSH>()?;
  Ok(())
}
