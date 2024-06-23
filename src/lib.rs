use fxhash::FxHasher;
use pyo3::prelude::*;
use rand::prelude::*;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// CMinHash implements the MinHash algorithm for efficient similarity estimation.
#[pyclass]
struct CMinHash {
  num_perm: usize,
  hash_values: Vec<u32>,
  permutations: Vec<(u64, u64)>,
}

#[pymethods]
impl CMinHash {
  /// Creates a new CMinHash instance.
  ///
  /// # Arguments
  ///
  /// * `num_perm` - The number of permutations to use in the MinHash algorithm.
  /// * `seed` - A seed value for the random number generator.
  #[new]
  fn new(num_perm: usize, seed: u64) -> Self {
    let mut rng = StdRng::seed_from_u64(seed);
    let permutations: Vec<(u64, u64)> =
      (0..num_perm).map(|_| (rng.gen(), rng.gen())).collect();

    CMinHash {
      num_perm,
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
  /// * `other` - Another CMinHash instance to compare with.
  ///
  /// # Returns
  ///
  /// A float value representing the estimated Jaccard similarity.
  fn jaccard(&self, other: &CMinHash) -> f64 {
    let equal_count = self
      .hash_values
      .iter()
      .zip(&other.hash_values)
      .filter(|&(&a, &b)| a == b)
      .count();
    equal_count as f64 / self.num_perm as f64
  }
}

/// CMinHashLSH implements Locality-Sensitive Hashing using MinHash for efficient similarity search.
#[pyclass]
struct CMinHashLSH {
  threshold: f64,
  num_perm: usize,
  num_bands: usize,
  band_size: usize,
  hash_tables: Vec<HashMap<u64, Vec<usize>>>,
}

#[pymethods]
impl CMinHashLSH {
  /// Creates a new CMinHashLSH instance.
  ///
  /// # Arguments
  ///
  /// * `threshold` - The similarity threshold for considering items as similar.
  /// * `num_perm` - The number of permutations used in the MinHash algorithm.
  /// * `num_bands` - The number of bands for the LSH algorithm.
  #[new]
  fn new(threshold: f64, num_perm: usize, num_bands: usize) -> Self {
    CMinHashLSH {
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
  /// * `minhash` - The CMinHash instance to be inserted.
  fn insert(&mut self, key: usize, minhash: &CMinHash) {
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
  /// * `minhash` - The CMinHash instance to query for.
  ///
  /// # Returns
  ///
  /// A vector of keys (usize) of potentially similar items.
  fn query(&self, minhash: &CMinHash) -> Vec<usize> {
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
  /// * `minhash1` - The first CMinHash instance.
  /// * `minhash2` - The second CMinHash instance.
  ///
  /// # Returns
  ///
  /// A boolean indicating whether the MinHashes are considered similar.
  fn is_similar(&self, minhash1: &CMinHash, minhash2: &CMinHash) -> bool {
    minhash1.jaccard(minhash2) >= self.threshold
  }

  /// Returns the number of permutations used in the LSH index.
  fn get_num_perm(&self) -> usize {
    self.num_perm
  }

  /// Returns the number of bands used in the LSH index.
  fn get_num_bands(&self) -> usize {
    self.num_bands
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
fn permute_hash(hash: u64, a: u64, b: u64) -> u32 {
  ((a.wrapping_mul(hash).wrapping_add(b)) >> 32) as u32
}

/// Calculates a hash value for a band of MinHash values.
#[inline]
fn calculate_band_hash(band: &[u32]) -> u64 {
  let mut hasher = FxHasher::default();
  for &value in band {
    hasher.write_u32(value);
  }
  hasher.finish()
}

#[pymodule]
fn rensa(m: &Bound<'_, PyModule>) -> PyResult<()> {
  m.add_class::<CMinHash>()?;
  m.add_class::<CMinHashLSH>()?;
  Ok(())
}
