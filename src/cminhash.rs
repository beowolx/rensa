use bincode::{deserialize, serialize};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rand::prelude::*;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

use crate::utils::{calculate_hash, permute_hash};

/// CMinHash implementation using circular permutation
///
/// This implementation uses a more efficient permutation scheme that requires
/// only two random permutations (sigma and pi) to generate all k hash functions.
#[pyclass(module = "rensa")]
#[derive(Serialize, Deserialize)]
pub struct CMinHash {
  num_perm: usize,
  seed: u64,
  hash_values: Vec<u32>,
  sigma: (u64, u64),
  pi: (u64, u64),
}

#[pymethods]
impl CMinHash {
  /// Creates a new CMinHash instance
  ///
  /// # Arguments
  /// * `num_perm` - Number of hash functions to generate
  /// * `seed` - Random seed for reproducibility
  ///
  /// # Returns
  /// A new CMinHash instance initialized with the specified parameters
  #[new]
  pub fn new(num_perm: usize, seed: u64) -> Self {
    let mut rng = StdRng::seed_from_u64(seed);
    let sigma = (rng.random(), rng.random());
    let pi = (rng.random(), rng.random());
    Self {
      num_perm,
      seed,
      hash_values: vec![u32::MAX; num_perm],
      sigma,
      pi,
    }
  }

  /// Updates the MinHash signature with new items
  ///
  /// Uses circular permutation to generate hash values more efficiently
  /// than the traditional MinHash approach.
  ///
  /// # Arguments
  /// * `items` - Vector of strings to be hashed and incorporated into the signature
  pub fn update(&mut self, items: Vec<String>) {
    const DELTA: u32 = 0x9e37_79b9; // 32-bit golden ratio
    for item in items {
      let item_hash = calculate_hash(&item);
      let sigma_hash = permute_hash(item_hash, self.sigma.0, self.sigma.1);
      let base = permute_hash(u64::from(sigma_hash), self.pi.0, self.pi.1);
      for k in 0..self.num_perm {
        let shift = u32::try_from(k).unwrap().wrapping_mul(DELTA);
        let candidate = base.wrapping_add(shift);
        self.hash_values[k] = self.hash_values[k].min(candidate);
      }
    }
  }

  /// Returns the current MinHash signature
  ///
  /// # Returns
  /// A vector of 32-bit hash values representing the MinHash signature
  pub fn digest(&self) -> Vec<u32> {
    self.hash_values.clone()
  }

  /// Estimates Jaccard similarity with another MinHash instance
  ///
  /// # Arguments
  /// * `other` - Another CMinHash instance to compare with
  ///
  /// # Returns
  pub fn jaccard(&self, other: &Self) -> f64 {
    let equal = self
      .hash_values
      .iter()
      .zip(other.hash_values.iter())
      .filter(|(&a, &b)| a == b)
      .count();
    equal as f64 / self.num_perm as f64
  }

  pub fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) {
    *self = deserialize(state.as_bytes()).unwrap();
  }

  pub fn __getstate__<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
    PyBytes::new(py, &serialize(&self).unwrap())
  }

  pub const fn __getnewargs__(&self) -> (usize, u64) {
    (self.num_perm, self.seed)
  }

  pub fn __reduce__(&self) -> PyResult<(PyObject, (usize, u64), PyObject)> {
    Python::with_gil(|py| {
      let type_obj = py.get_type::<Self>().into();
      let state = self.__getstate__(py);
      Ok((type_obj, (self.num_perm, self.seed), state.into()))
    })
  }
}
