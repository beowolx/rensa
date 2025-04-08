use rustc_hash::FxHasher;
use std::hash::{Hash, Hasher};

/// Calculates a 64-bit hash value for any hashable type using `FxHasher`
///
/// This function is used as the base hash function for both `MinHash` implementations.
/// `FxHasher` is chosen for its speed and good distribution properties.
///
/// # Arguments
/// * `t` - Reference to any type that implements the `Hash` trait
///
/// # Returns
/// A 64-bit hash value
#[inline]
pub fn calculate_hash<T: Hash>(t: &T) -> u64 {
  let mut s = FxHasher::default();
  t.hash(&mut s);
  s.finish()
}

/// Applies a permutation to a 64-bit hash value
///
/// Uses the (a * x + b) mod m method to create a permutation of the input hash.
/// The result is truncated to 32 bits by taking the upper half of the 64-bit result.
///
/// # Arguments
/// * `hash` - The input hash value to permute
/// * `a` - The multiplication coefficient
/// * `b` - The addition coefficient
///
/// # Returns
/// A 32-bit permuted hash value
#[inline]
pub const fn permute_hash(hash: u64, a: u64, b: u64) -> u32 {
  ((a.wrapping_mul(hash).wrapping_add(b)) >> 32) as u32
}

/// Calculates a hash value for a band of `MinHash` values
///
/// Used by the LSH implementation to hash bands of `MinHash` signatures
/// for bucket assignment.
///
/// # Arguments
/// * `band` - Slice of 32-bit `MinHash` values representing a band
///
/// # Returns
/// A 64-bit hash value for the entire band
#[inline]
pub fn calculate_band_hash(band: &[u32]) -> u64 {
  let mut hasher = FxHasher::default();
  for &value in band {
    hasher.write_u32(value);
  }
  hasher.finish()
}
