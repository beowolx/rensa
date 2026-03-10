use crate::simd::dispatch::{apply_hash_batch_to_values, PermutationSoA};
use rand_core::{RngCore, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoreError {
  NumPermZero,
  OutputLenMismatch { expected: usize, got: usize },
}

impl std::fmt::Display for CoreError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::NumPermZero => write!(f, "num_perm must be greater than 0"),
      Self::OutputLenMismatch { expected, got } => {
        write!(f, "output length mismatch: expected {expected}, got {got}")
      }
    }
  }
}

impl std::error::Error for CoreError {}

#[derive(Clone)]
pub struct RMinHashContext {
  num_perm: usize,
  seed: u64,
  permutations: Vec<(u64, u64)>,
  permutations_soa: PermutationSoA,
}

#[must_use]
pub fn build_permutations(num_perm: usize, seed: u64) -> Vec<(u64, u64)> {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
  (0..num_perm)
    .map(|_| {
      let a = rng.next_u64() | 1;
      let b = rng.next_u64();
      (a, b)
    })
    .collect()
}

impl RMinHashContext {
  /// Build a reusable R-MinHash context with cached permutations.
  ///
  /// # Errors
  ///
  /// Returns an error when `num_perm` is zero.
  pub fn new(num_perm: usize, seed: u64) -> Result<Self, CoreError> {
    if num_perm == 0 {
      return Err(CoreError::NumPermZero);
    }

    let permutations = build_permutations(num_perm, seed);
    let permutations_soa = PermutationSoA::from_permutations(&permutations);
    Ok(Self {
      num_perm,
      seed,
      permutations,
      permutations_soa,
    })
  }

  #[inline]
  #[must_use]
  pub const fn num_perm(&self) -> usize {
    self.num_perm
  }

  #[inline]
  #[must_use]
  pub const fn seed(&self) -> u64 {
    self.seed
  }

  #[inline]
  #[must_use]
  pub fn permutations(&self) -> &[(u64, u64)] {
    &self.permutations
  }

  #[inline]
  #[must_use]
  pub const fn permutations_soa(&self) -> &PermutationSoA {
    &self.permutations_soa
  }

  /// Compute an R-MinHash digest from pre-hashed tokens using cached permutations.
  ///
  /// # Errors
  ///
  /// Returns an error if `out_digest.len()` does not match `self.num_perm`.
  pub fn digest_prehashed_into(
    &self,
    token_hashes: &[u64],
    out_digest: &mut [u32],
  ) -> Result<(), CoreError> {
    if out_digest.len() != self.num_perm {
      return Err(CoreError::OutputLenMismatch {
        expected: self.num_perm,
        got: out_digest.len(),
      });
    }

    out_digest.fill(u32::MAX);
    apply_token_hashes_to_values(
      out_digest,
      &self.permutations,
      &self.permutations_soa,
      token_hashes,
    );
    Ok(())
  }
}

pub fn apply_token_hashes_to_values(
  hash_values: &mut [u32],
  permutations: &[(u64, u64)],
  permutations_soa: &PermutationSoA,
  token_hashes: &[u64],
) {
  apply_hash_batch_to_values(
    hash_values,
    permutations,
    permutations_soa,
    token_hashes,
  );
}

/// Compute an R-MinHash digest from pre-hashed tokens (`u64` feature hashes).
///
/// # Errors
///
/// Returns an error if `num_perm` is zero, or if `out_digest.len()` does not
/// match `num_perm`.
pub fn digest_prehashed(
  num_perm: usize,
  seed: u64,
  token_hashes: &[u64],
  out_digest: &mut [u32],
) -> Result<(), CoreError> {
  let context = RMinHashContext::new(num_perm, seed)?;
  context.digest_prehashed_into(token_hashes, out_digest)
}

#[cfg(test)]
mod tests {
  use crate::rminhash::{digest_prehashed, RMinHashContext};

  #[test]
  fn digest_prehashed_matches_golden_vector() {
    let num_perm = 16;
    let seed = 42;
    let token_hashes = [1_u64, 2, 3, 10, 20, 30, 999];

    let mut digest = vec![0_u32; num_perm];
    digest_prehashed(num_perm, seed, &token_hashes, &mut digest).unwrap();

    let expected = [
      571_772_611,
      936_133_835,
      752_080_753,
      481_772_021,
      45_178_047,
      175_317_472,
      470_513_164,
      928_706_636,
      766_355_169,
      186_861_125,
      104_841_669,
      219_275_888,
      723_396_917,
      903_182_598,
      455_784_955,
      153_681_280,
    ];
    assert_eq!(&digest, &expected);
  }

  #[test]
  fn reusable_context_matches_stateless_digest() {
    let num_perm = 16;
    let seed = 42;
    let token_hashes = [1_u64, 2, 3, 10, 20, 30, 999];

    let context = RMinHashContext::new(num_perm, seed).unwrap();
    assert_eq!(context.num_perm(), num_perm);
    assert_eq!(context.seed(), seed);

    let mut from_context = vec![0_u32; num_perm];
    context
      .digest_prehashed_into(&token_hashes, &mut from_context)
      .unwrap();

    let mut stateless = vec![0_u32; num_perm];
    digest_prehashed(num_perm, seed, &token_hashes, &mut stateless).unwrap();

    assert_eq!(from_context, stateless);
  }
}
