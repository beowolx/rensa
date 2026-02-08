use rustc_hash::FxHasher;
use std::hash::Hasher;

/// Fast hash function for byte arrays
#[inline]
pub fn calculate_hash_fast(data: &[u8]) -> u64 {
  // Use a simplified version of FxHash for byte arrays
  let mut hash = 0xcbf2_9ce4_8422_2325_u64;

  // Process 8 bytes at a time
  let chunks = data.chunks_exact(8);
  let remainder = chunks.remainder();

  for chunk in chunks {
    let mut bytes = [0_u8; 8];
    bytes.copy_from_slice(chunk);
    let val = u64::from_le_bytes(bytes);
    hash = hash.wrapping_mul(0x0100_0000_01b3).wrapping_add(val);
  }

  // Handle remainder bytes
  for &byte in remainder {
    hash = hash
      .wrapping_mul(0x0100_0000_01b3)
      .wrapping_add(u64::from(byte));
  }

  hash
}

/// Applies a permutation to a hash value.
#[inline]
pub const fn permute_hash(hash: u64, a: u64, b: u64) -> u32 {
  ((a.wrapping_mul(hash).wrapping_add(b)) >> 32) as u32
}

/// Calculates a hash value for a band of `MinHash` values.
#[inline]
pub fn calculate_band_hash(band: &[u32]) -> u64 {
  let mut hasher = FxHasher::default();

  // Process 4 u32s at a time for better throughput
  let chunks = band.chunks_exact(4);
  let remainder = chunks.remainder();

  for chunk in chunks {
    // Process as two u64s for better performance
    let val1 = u64::from(chunk[0]) | (u64::from(chunk[1]) << 32);
    let val2 = u64::from(chunk[2]) | (u64::from(chunk[3]) << 32);
    hasher.write_u64(val1);
    hasher.write_u64(val2);
  }

  // Handle remainder
  for &value in remainder {
    hasher.write_u32(value);
  }

  hasher.finish()
}

#[cfg(test)]
mod tests {
  use crate::utils::{calculate_band_hash, calculate_hash_fast, permute_hash};
  use rustc_hash::FxHasher;
  use std::hash::Hasher;

  fn reference_hash_fast(data: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    let mut index = 0_usize;

    while index + 8 <= data.len() {
      let mut bytes = [0_u8; 8];
      bytes.copy_from_slice(&data[index..index + 8]);
      hash = hash
        .wrapping_mul(0x0100_0000_01b3)
        .wrapping_add(u64::from_le_bytes(bytes));
      index += 8;
    }

    for &byte in &data[index..] {
      hash = hash
        .wrapping_mul(0x0100_0000_01b3)
        .wrapping_add(u64::from(byte));
    }

    hash
  }

  fn reference_band_hash(band: &[u32]) -> u64 {
    let mut hasher = FxHasher::default();
    let mut index = 0_usize;

    while index + 4 <= band.len() {
      let val1 = u64::from(band[index]) | (u64::from(band[index + 1]) << 32);
      let val2 =
        u64::from(band[index + 2]) | (u64::from(band[index + 3]) << 32);
      hasher.write_u64(val1);
      hasher.write_u64(val2);
      index += 4;
    }

    for &value in &band[index..] {
      hasher.write_u32(value);
    }

    hasher.finish()
  }

  #[test]
  fn calculate_hash_fast_matches_reference_across_chunk_boundaries() {
    let data = [
      b"".as_slice(),
      b"a".as_slice(),
      b"abcdefgh".as_slice(),
      b"abcdefghi".as_slice(),
      b"abcdefghijklmno".as_slice(),
      b"abcdefghijklmnop".as_slice(),
      b"abcdefghijklmnopqrstuvwxyz0123456789".as_slice(),
    ];

    for bytes in data {
      assert_eq!(calculate_hash_fast(bytes), reference_hash_fast(bytes));
    }
  }

  #[test]
  fn permute_hash_matches_expected_math() {
    let hash = 0x0123_4567_89ab_cdef_u64;
    let a = 0x9e37_79b9_7f4a_7c15_u64;
    let b = 0xbf58_476d_1ce4_e5b9_u64;
    let expected = ((a.wrapping_mul(hash).wrapping_add(b)) >> 32) as u32;

    assert_eq!(permute_hash(hash, a, b), expected);
  }

  #[test]
  fn calculate_band_hash_matches_reference_and_handles_remainders() {
    let cases: [&[u32]; 6] = [
      &[],
      &[1],
      &[1, 2, 3],
      &[1, 2, 3, 4],
      &[1, 2, 3, 4, 5],
      &[1, 2, 3, 4, 5, 6, 7, 8, 9],
    ];

    for band in cases {
      assert_eq!(calculate_band_hash(band), reference_band_hash(band));
    }
  }
}
