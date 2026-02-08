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
