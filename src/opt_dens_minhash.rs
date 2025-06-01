//! Implementation of `MinHash` with Optimal Densification.
//!
//! This module provides `OptDensMinHash`, a data sketch algorithm for
//! estimating Jaccard similarity between sets. It uses the `MinHash` technique,
//! significantly enhanced by an optimal densification strategy to ensure that
//! the minhashes are always fully populated, improving accuracy, especially for
//! sparse datasets or smaller numbers of permutations.
//!
//! The core ideas are drawn from research on densified `MinHash` algorithms:
//! - Shrivastava, A. (2017). Optimal Densification for Fast and Accurate Minwise Hashing. *PMLR*.
//!   [Link](https://proceedings.mlr.press/v70/shrivastava17a.html)
//! - Mai, T., Rao, A., Kapilevitch, L., Rossi, R., Abbasi-Yadkori, Y., & Sinha, K. (2020).
//!   On densification for `MinWise` Hashing. *PMLR*.
//!   [Link](http://proceedings.mlr.press/v115/mai20a/mai20a.pdf)
//!
//! `OptDensMinHash` is designed for unweighted data. Items are added to the sketch
//! using the `update(items: Vec<String>)` method. The resulting `MinHash` signature,
//! which can be a `Vec<u32>` (via `digest()`) or `Vec<u64>` (via `digest_u64()`),
//! is used for Jaccard index estimation or other similarity comparisons.
//!
//! **Important**: The densification process, which ensures all hash slots are filled,
//! is automatically triggered by an internal `end_sketch()` method when `digest()`,
//! `digest_u64()`, or `jaccard()` are called. This guarantees that the sketch is
//! properly finalized before use, even if some hash slots were not initially filled
//! by incoming data.

use crate::utils::calculate_hash;
use murmur3::murmur3_32;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rand::distr::Uniform;
use rand::prelude::*;
use rand_chacha::ChaCha12Rng;
use rand_xoshiro::Xoshiro256PlusPlus;
use serde::{Deserialize, Serialize};
use std::io::Cursor;

/// OptDensMinHash implements a MinHash variant using optimal densification.
#[derive(Serialize, Deserialize)]
#[pyclass(module = "rensa")]
pub struct OptDensMinHash {
  num_perm: usize,
  seed: u64,
  hsketch: Vec<f64>,
  values: Vec<u64>,
  init: Vec<bool>,
  nb_empty: i64,
}

#[pymethods]
impl OptDensMinHash {
  #[new]
  #[must_use]
  pub fn new(num_perm: usize, seed: u64) -> Self {
    Self {
      num_perm,
      seed,
      hsketch: vec![f64::MAX; num_perm],
      values: vec![u64::MAX; num_perm],
      init: vec![false; num_perm],
      nb_empty: i64::try_from(num_perm).unwrap_or(i64::MAX),
    }
  }

  pub fn update(&mut self, items: Vec<String>) {
    for item in items {
      let h = calculate_hash(&item);
      self.sketch_hash(h);
    }
  }

  /// Returns the current MinHash digest as u32 values.
  ///
  /// # Panics
  ///
  /// Panics if the murmur3_32 hash function fails.
  pub fn digest(&mut self) -> Vec<u32> {
    self.end_sketch();
    self
      .values
      .iter()
      .map(|v| murmur3_32(&mut Cursor::new(v.to_ne_bytes()), 127).unwrap())
      .collect()
  }

  pub fn digest_u64(&mut self) -> Vec<u64> {
    self.end_sketch();
    self.values.clone()
  }

  pub fn jaccard(&mut self, other: &mut Self) -> f64 {
    self.end_sketch();
    other.end_sketch();
    let equal_count = self
      .values
      .iter()
      .zip(other.values.iter())
      .filter(|(&a, &b)| a == b)
      .count();
    equal_count as f64 / self.num_perm as f64
  }

  fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) {
    *self = bincode::serde::decode_from_slice(
      state.as_bytes(),
      bincode::config::standard(),
    )
    .unwrap()
    .0;
  }

  fn __getstate__<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
    PyBytes::new(
      py,
      &bincode::serde::encode_to_vec(self, bincode::config::standard())
        .unwrap(),
    )
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

impl OptDensMinHash {
  fn sketch_hash(&mut self, hash: u64) {
    let m = self.num_perm;
    let unit_range = Uniform::<f64>::new(0.0, 1.0).unwrap();
    let mut rand_generator = Xoshiro256PlusPlus::seed_from_u64(hash);
    let r: f64 = unit_range.sample(&mut rand_generator);
    let k: usize = Uniform::<usize>::new(0, m)
      .unwrap()
      .sample(&mut rand_generator);
    if r <= self.hsketch[k] {
      self.hsketch[k] = r;
      self.values[k] = hash;
      if !self.init[k] {
        self.init[k] = true;
        self.nb_empty -= 1;
      }
    }
  }

  fn end_sketch(&mut self) {
    if self.nb_empty > 0
      && self.nb_empty < i64::try_from(self.num_perm).unwrap_or(i64::MAX)
    {
      self.densify();
    }
  }

  fn densify(&mut self) {
    if self.nb_empty == i64::try_from(self.num_perm).unwrap_or(i64::MAX) {
      return;
    }

    let m = self.num_perm;
    let uniform = Uniform::new(0, m).unwrap();
    for k in 0..m {
      if !self.init[k] {
        let mut rng = ChaCha12Rng::seed_from_u64(k as u64 + 123_743);
        loop {
          let j: usize = uniform.sample(&mut rng);
          if self.init[j] {
            self.values[k] = self.values[j];
            self.hsketch[k] = self.hsketch[j];
            self.init[k] = true;
            self.nb_empty -= 1;
            break;
          }
        }
      }
    }
  }
}
