use crate::cminhash::{CMinHash, HASH_BATCH_SIZE};
use crate::utils::{calculate_hash_fast, ratio_usize};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[derive(Default)]
struct SigmaBatch {
  values: [u64; HASH_BATCH_SIZE],
  len: usize,
}

impl SigmaBatch {
  #[inline]
  fn push(&mut self, value: u64) -> bool {
    debug_assert!(self.len < HASH_BATCH_SIZE);
    self.values[self.len] = value;
    self.len += 1;
    self.len == HASH_BATCH_SIZE
  }

  #[inline]
  const fn clear(&mut self) {
    self.len = 0;
  }

  #[inline]
  fn as_slice(&self) -> &[u64] {
    &self.values[..self.len]
  }

  #[inline]
  const fn is_empty(&self) -> bool {
    self.len == 0
  }
}

impl CMinHash {
  pub(in crate::cminhash) fn build_pi_precomputed(
    num_perm: usize,
    pi_c: u64,
    pi_d: u64,
  ) -> Vec<u64> {
    (0..num_perm)
      .map(|k| pi_c.wrapping_mul(k as u64).wrapping_add(pi_d))
      .collect()
  }

  #[inline]
  pub(crate) const fn num_perm(&self) -> usize {
    self.num_perm
  }

  #[inline]
  pub(crate) const fn seed(&self) -> u64 {
    self.seed
  }

  pub(in crate::cminhash) fn validate_num_perm(
    num_perm: usize,
  ) -> PyResult<()> {
    if num_perm == 0 {
      return Err(PyValueError::new_err("num_perm must be greater than 0"));
    }
    Ok(())
  }

  pub(in crate::cminhash) fn validate_state(&self) -> PyResult<()> {
    Self::validate_num_perm(self.num_perm)?;
    if self.hash_values.len() != self.num_perm {
      return Err(PyValueError::new_err(format!(
        "invalid CMinHash state: hash_values length {} does not match num_perm {}",
        self.hash_values.len(),
        self.num_perm
      )));
    }
    if !self.pi_precomputed.is_empty()
      && self.pi_precomputed.len() != self.num_perm
    {
      return Err(PyValueError::new_err(format!(
        "invalid CMinHash state: pi_precomputed length {} does not match num_perm {} (or be compacted to 0)",
        self.pi_precomputed.len(),
        self.num_perm
      )));
    }
    Ok(())
  }

  pub(crate) fn ensure_compatible_for_jaccard(
    &self,
    other: &Self,
  ) -> PyResult<()> {
    self.validate_state()?;
    other.validate_state()?;
    if self.num_perm != other.num_perm {
      return Err(PyValueError::new_err(format!(
        "num_perm mismatch: left is {}, right is {}",
        self.num_perm, other.num_perm
      )));
    }
    Ok(())
  }

  #[inline]
  const fn sigma_transform(&self, hash: u64) -> u64 {
    self.sigma_a.wrapping_mul(hash).wrapping_add(self.sigma_b)
  }

  fn ensure_pi_precomputed(&mut self) {
    if self.pi_precomputed.len() != self.num_perm {
      self.pi_precomputed =
        Self::build_pi_precomputed(self.num_perm, self.pi_c, self.pi_d);
    }
  }

  #[inline]
  pub(crate) fn jaccard_unchecked(&self, other: &Self) -> f64 {
    let mut equal_count = 0usize;

    let mut chunks_a = self.hash_values.chunks_exact(8);
    let mut chunks_b = other.hash_values.chunks_exact(8);

    for (chunk_a, chunk_b) in chunks_a.by_ref().zip(chunks_b.by_ref()) {
      equal_count += usize::from(chunk_a[0] == chunk_b[0]);
      equal_count += usize::from(chunk_a[1] == chunk_b[1]);
      equal_count += usize::from(chunk_a[2] == chunk_b[2]);
      equal_count += usize::from(chunk_a[3] == chunk_b[3]);
      equal_count += usize::from(chunk_a[4] == chunk_b[4]);
      equal_count += usize::from(chunk_a[5] == chunk_b[5]);
      equal_count += usize::from(chunk_a[6] == chunk_b[6]);
      equal_count += usize::from(chunk_a[7] == chunk_b[7]);
    }

    equal_count += chunks_a
      .remainder()
      .iter()
      .zip(chunks_b.remainder())
      .filter(|(&a, &b)| a == b)
      .count();

    ratio_usize(equal_count, self.num_perm)
  }

  fn apply_sigma_batch_to_values(
    hash_values: &mut [u64],
    pi_precomputed: &[u64],
    pi_c: u64,
    sigma_batch: &[u64],
  ) {
    debug_assert_eq!(hash_values.len(), pi_precomputed.len());

    let mut hash_chunks = hash_values.chunks_exact_mut(16);
    let mut pi_chunks = pi_precomputed.chunks_exact(16);

    for (hash_chunk, pi_chunk) in hash_chunks.by_ref().zip(pi_chunks.by_ref()) {
      let mut current = [0u64; 16];
      current.copy_from_slice(hash_chunk);

      for &sigma_h in sigma_batch {
        let base = pi_c.wrapping_mul(sigma_h);

        for i in 0..16 {
          let pi_value = base.wrapping_add(pi_chunk[i]);
          current[i] = current[i].min(pi_value);
        }
      }

      hash_chunk.copy_from_slice(&current);
    }

    let hash_remainder = hash_chunks.into_remainder();
    let pi_remainder = pi_chunks.remainder();
    debug_assert_eq!(hash_remainder.len(), pi_remainder.len());

    for &sigma_h in sigma_batch {
      let base = pi_c.wrapping_mul(sigma_h);

      for (hash_val, &pi_val) in hash_remainder.iter_mut().zip(pi_remainder) {
        let pi_value = base.wrapping_add(pi_val);
        *hash_val = (*hash_val).min(pi_value);
      }
    }
  }

  fn apply_sigma_batch(&mut self, sigma_batch: &[u64]) {
    Self::apply_sigma_batch_to_values(
      &mut self.hash_values,
      &self.pi_precomputed,
      self.pi_c,
      sigma_batch,
    );
  }

  pub(in crate::cminhash) fn apply_token_hashes_to_values(
    hash_values: &mut [u64],
    token_hashes: &[u64],
    sigma_a: u64,
    sigma_b: u64,
    pi_c: u64,
    pi_precomputed: &[u64],
  ) {
    let mut sigma_batch = SigmaBatch::default();
    for &token_hash in token_hashes {
      let sigma_h = sigma_a.wrapping_mul(token_hash).wrapping_add(sigma_b);
      if sigma_batch.push(sigma_h) {
        Self::apply_sigma_batch_to_values(
          hash_values,
          pi_precomputed,
          pi_c,
          sigma_batch.as_slice(),
        );
        sigma_batch.clear();
      }
    }
    if !sigma_batch.is_empty() {
      Self::apply_sigma_batch_to_values(
        hash_values,
        pi_precomputed,
        pi_c,
        sigma_batch.as_slice(),
      );
    }
  }

  pub(crate) fn compact_from_template(template: &Self) -> Self {
    Self {
      num_perm: template.num_perm,
      seed: template.seed,
      hash_values: vec![u64::MAX; template.num_perm],
      sigma_a: template.sigma_a,
      sigma_b: template.sigma_b,
      pi_c: template.pi_c,
      pi_d: template.pi_d,
      pi_precomputed: Vec::new(),
    }
  }

  pub(crate) fn reset_from_token_hashes_with_template(
    &mut self,
    token_hashes: &[u64],
    template: &Self,
  ) {
    self.hash_values.fill(u64::MAX);
    Self::apply_token_hashes_to_values(
      &mut self.hash_values,
      token_hashes,
      template.sigma_a,
      template.sigma_b,
      template.pi_c,
      &template.pi_precomputed,
    );
  }

  fn update_internal<I, S>(&mut self, items: I)
  where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
  {
    self.ensure_pi_precomputed();
    let mut sigma_batch = SigmaBatch::default();

    for item in items {
      let h = calculate_hash_fast(item.as_ref().as_bytes());
      if sigma_batch.push(self.sigma_transform(h)) {
        self.apply_sigma_batch(sigma_batch.as_slice());
        sigma_batch.clear();
      }
    }

    if !sigma_batch.is_empty() {
      self.apply_sigma_batch(sigma_batch.as_slice());
    }
  }

  pub(in crate::cminhash) fn update_hashed_tokens(
    &mut self,
    token_hashes: &[u64],
  ) {
    self.ensure_pi_precomputed();
    Self::apply_token_hashes_to_values(
      &mut self.hash_values,
      token_hashes,
      self.sigma_a,
      self.sigma_b,
      self.pi_c,
      &self.pi_precomputed,
    );
  }

  /// Updates the `CMinHash` with items from any iterable of string-like values.
  pub fn update_iter<I, S>(&mut self, items: I)
  where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
  {
    self.update_internal(items);
  }

  /// Updates the `CMinHash` with a new set of items from a vector of strings.
  pub fn update_vec(&mut self, items: Vec<String>) {
    self.update_iter(items);
  }
}
