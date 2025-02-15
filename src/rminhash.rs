use bincode::{deserialize, serialize};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rand::prelude::*;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

use crate::utils::{calculate_hash, permute_hash};

/// RMinHash implementation using random permutations
///
/// This implementation uses k independent hash functions to simulate k permutations
/// of the input set, where k is specified by `num_perm`.
#[pyclass(module = "rensa")]
#[derive(Serialize, Deserialize)]
pub struct RMinHash {
  num_perm: usize,
  seed: u64,
  hash_values: Vec<u32>,
  permutations: Vec<(u64, u64)>,
}

#[pymethods]
impl RMinHash {
  /// Creates a new RMinHash instance
  ///
  /// # Arguments
  /// * `num_perm` - Number of permutations to use
  /// * `seed` - Random seed for reproducibility
  ///
  /// # Returns
  /// A new RMinHash instance initialized with the specified parameters
  #[new]
  pub fn new(num_perm: usize, seed: u64) -> Self {
    let mut rng = StdRng::seed_from_u64(seed);
    let permutations = (0..num_perm)
      .map(|_| (rng.random(), rng.random()))
      .collect();

    Self {
      num_perm,
      seed,
      hash_values: vec![u32::MAX; num_perm],
      permutations,
    }
  }

  /// Updates the MinHash signature with new items
  ///
  /// # Arguments
  /// * `items` - Vector of strings to be hashed and incorporated into the signature
  pub fn update(&mut self, items: Vec<String>) {
    for item in items {
      let item_hash = calculate_hash(&item);
      for (i, &(a, b)) in self.permutations.iter().enumerate() {
        let h = permute_hash(item_hash, a, b);
        self.hash_values[i] = self.hash_values[i].min(h);
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
  /// * `other` - Another RMinHash instance to compare with
  ///
  /// # Returns
  /// Estimated Jaccard similarity as a float between 0 and 1
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

  #[allow(non_snake_case)]
  pub const fn __getnewargs__(&self) -> (usize, u64) {
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
