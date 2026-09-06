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
  #[inline]
  pub(crate) const fn num_perm(&self) -> usize {
    self.num_perm
  }

  #[inline]
  pub(crate) fn signature_values(&self) -> &[u64] {
    &self.hash_values
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
    if self.seed != other.seed {
      return Err(PyValueError::new_err(format!(
        "seed mismatch: left is {}, right is {}",
        self.seed, other.seed
      )));
    }
    Ok(())
  }

  // SplitMix64's bijective finalizer, by Sebastiano Vigna:
  // https://prng.di.unimi.it/splitmix64.c
  #[inline]
  const fn mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
  }

  #[inline]
  const fn sigma_transform(&self, hash: u64) -> u64 {
    Self::mix64(hash.wrapping_add(self.sigma_key))
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

  #[inline]
  pub(crate) fn jaccard_at_least_unchecked(
    &self,
    other: &Self,
    threshold: f64,
  ) -> bool {
    let mut mismatches = 0;
    for (left, right) in
      self.hash_values.chunks(8).zip(other.hash_values.chunks(8))
    {
      mismatches += left.iter().zip(right).filter(|(a, b)| a != b).count();
      // Even if all remaining lanes match, this pair cannot reach threshold.
      if ratio_usize(self.num_perm - mismatches, self.num_perm) < threshold {
        return false;
      }
    }
    true
  }

  fn apply_sigma_batch_to_values(
    hash_values: &mut [u64],
    pi_key: u64,
    sigma_batch: &[u64],
  ) {
    let remainder_start = hash_values.len() / 16 * 16;
    let mut hash_chunks = hash_values.chunks_exact_mut(16);
    for (block, hash_chunk) in hash_chunks.by_ref().enumerate() {
      let mut current = [0u64; 16];
      current.copy_from_slice(hash_chunk);
      let shift = (block as u64).wrapping_mul(16).wrapping_add(1);
      for &sigma in sigma_batch {
        let base = sigma.wrapping_add(pi_key).wrapping_sub(shift);
        for (lane, value) in current.iter_mut().enumerate() {
          *value = (*value).min(Self::mix64(base.wrapping_sub(lane as u64)));
        }
      }
      hash_chunk.copy_from_slice(&current);
    }

    let remainder = hash_chunks.into_remainder();
    let shift = (remainder_start as u64).wrapping_add(1);
    for &sigma in sigma_batch {
      let base = sigma.wrapping_add(pi_key).wrapping_sub(shift);
      for (lane, value) in remainder.iter_mut().enumerate() {
        *value = (*value).min(Self::mix64(base.wrapping_sub(lane as u64)));
      }
    }
  }

  fn apply_sigma_batch(&mut self, sigma_batch: &[u64]) {
    Self::apply_sigma_batch_to_values(
      &mut self.hash_values,
      self.pi_key,
      sigma_batch,
    );
  }

  pub(in crate::cminhash) fn apply_token_hashes_to_values(
    hash_values: &mut [u64],
    token_hashes: &[u64],
    sigma_key: u64,
    pi_key: u64,
  ) {
    let mut sigma_batch = SigmaBatch::default();
    for &token_hash in token_hashes {
      let sigma = Self::mix64(token_hash.wrapping_add(sigma_key));
      if sigma_batch.push(sigma) {
        Self::apply_sigma_batch_to_values(
          hash_values,
          pi_key,
          sigma_batch.as_slice(),
        );
        sigma_batch.clear();
      }
    }
    if !sigma_batch.is_empty() {
      Self::apply_sigma_batch_to_values(
        hash_values,
        pi_key,
        sigma_batch.as_slice(),
      );
    }
  }

  pub(crate) fn reset_from_token_hashes(&mut self, token_hashes: &[u64]) {
    self.hash_values.fill(u64::MAX);
    self.update_hashed_tokens(token_hashes);
  }

  /// Updates the `CMinHash` with items from any iterable of string-like values.
  pub fn update_iter<I, S>(&mut self, items: I)
  where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
  {
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
    Self::apply_token_hashes_to_values(
      &mut self.hash_values,
      token_hashes,
      self.sigma_key,
      self.pi_key,
    );
  }

  /// Updates the `CMinHash` with a new set of items from a vector of strings.
  pub fn update_vec(&mut self, items: Vec<String>) {
    self.update_iter(items);
  }
}

#[cfg(test)]
mod tests {
  use crate::cminhash::{CMinHash, CMinHashParams};
  use rand_core::{RngCore, SeedableRng};
  use rand_xoshiro::Xoshiro256PlusPlus;

  fn reference_permutation(value: u64, key: u64) -> u64 {
    let mut value = value.wrapping_add(key);
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
  }

  #[test]
  fn token_hash_updates_match_scalar_circulant_reference() {
    for seed in [0, 1, 42, u64::MAX] {
      let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
      for num_perm in [1, 16, 128, 129, 512] {
        let params = CMinHashParams::new(seed);
        for token_len in [0, 1, 31, 32, 127, 128, 1024] {
          let mut hashes: Vec<u64> =
            (0..token_len).map(|_| rng.next_u64()).collect();
          for (hash, boundary) in hashes.iter_mut().zip([
            0,
            1,
            u64::MAX - 1,
            u64::MAX,
            1 << 63,
            u64::from(u32::MAX),
            1 << 32,
            params.sigma_key.wrapping_neg(),
          ]) {
            *hash = boundary;
          }
          for populated in [false, true] {
            let initial: Vec<u64> = (0..num_perm)
              .map(|_| if populated { rng.next_u64() } else { u64::MAX })
              .collect();
            let mut expected = initial.clone();
            for &hash in &hashes {
              let sigma = reference_permutation(hash, params.sigma_key);
              for (k, value) in expected.iter_mut().enumerate() {
                let shifted = sigma.wrapping_sub(k as u64 + 1);
                let permuted = reference_permutation(shifted, params.pi_key);
                *value = (*value).min(permuted);
              }
            }
            let mut actual = initial.clone();
            CMinHash::apply_token_hashes_to_values(
              &mut actual,
              &hashes,
              params.sigma_key,
              params.pi_key,
            );
            assert_eq!(actual, expected);

            let mut incremental = initial;
            for chunk in hashes.chunks(129) {
              CMinHash::apply_token_hashes_to_values(
                &mut incremental,
                chunk,
                params.sigma_key,
                params.pi_key,
              );
            }
            assert_eq!(incremental, expected);
          }
        }
      }
    }
  }

  #[test]
  fn threshold_predicate_matches_full_jaccard_at_float_boundaries(
  ) -> pyo3::PyResult<()> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(78);
    for num_perm in [1, 7, 8, 17, 63, 129] {
      let mut left = CMinHash::new(num_perm, 42)?;
      let mut right = CMinHash::new(num_perm, 42)?;
      for matching in 0..=num_perm {
        for (index, (a, b)) in left
          .hash_values
          .iter_mut()
          .zip(&mut right.hash_values)
          .enumerate()
        {
          *a = rng.next_u64();
          *b = if index < matching { *a } else { !*a };
        }
        // Shuffle both signatures together to distribute matches throughout
        // the chunks, while retaining exactly the chosen number of matches.
        for index in 1..num_perm {
          let other =
            usize::try_from(rng.next_u64() % (index as u64 + 1)).unwrap();
          left.hash_values.swap(index, other);
          right.hash_values.swap(index, other);
        }
        let similarity = left.jaccard_unchecked(&right);
        let below = if similarity == 0.0 {
          0.0
        } else {
          f64::from_bits(similarity.to_bits() - 1)
        };
        let above = f64::from_bits(similarity.to_bits() + 1);
        for threshold in [0.0, 1.0, 0.8, similarity, below, above] {
          assert_eq!(
            left.jaccard_at_least_unchecked(&right, threshold),
            similarity >= threshold,
            "num_perm={num_perm}, matching={matching}, threshold={threshold}"
          );
        }
      }
    }
    Ok(())
  }
}
