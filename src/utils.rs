#[cfg(target_pointer_width = "64")]
const K_USIZE: usize = 0xf135_7aea_2e62_a9c5;
#[cfg(target_pointer_width = "64")]
const K_U64: u64 = 0xf135_7aea_2e62_a9c5;
#[cfg(target_pointer_width = "32")]
const K_USIZE: usize = 0x93d7_65dd;
#[cfg(target_pointer_width = "32")]
const K_U64: u64 = 0x93d7_65dd;

#[cfg(target_pointer_width = "64")]
const ROTATE: u32 = 26;
#[cfg(target_pointer_width = "32")]
const ROTATE: u32 = 15;

const SEED1: u64 = 0x243f_6a88_85a3_08d3;
const SEED2: u64 = 0x1319_8a2e_0370_7344;
const PREVENT_TRIVIAL_ZERO_COLLAPSE: u64 = 0xa409_3822_299f_31d0;

#[inline]
const fn read_u64_le(bytes: &[u8], offset: usize) -> u64 {
  u64::from_le_bytes([
    bytes[offset],
    bytes[offset + 1],
    bytes[offset + 2],
    bytes[offset + 3],
    bytes[offset + 4],
    bytes[offset + 5],
    bytes[offset + 6],
    bytes[offset + 7],
  ])
}

#[inline]
const fn read_u32_le(bytes: &[u8], offset: usize) -> u32 {
  u32::from_le_bytes([
    bytes[offset],
    bytes[offset + 1],
    bytes[offset + 2],
    bytes[offset + 3],
  ])
}

#[cfg(target_pointer_width = "32")]
#[inline]
const fn low_u32(value: u64) -> u32 {
  let bytes = value.to_le_bytes();
  u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

#[cfg(target_pointer_width = "32")]
#[inline]
const fn high_u32(value: u64) -> u32 {
  let bytes = value.to_le_bytes();
  u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]])
}

#[inline]
fn multiply_mix(x: u64, y: u64) -> u64 {
  #[cfg(target_pointer_width = "64")]
  {
    let full = u128::from(x) * u128::from(y);
    let bytes = full.to_le_bytes();
    let lo = u64::from_le_bytes([
      bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6],
      bytes[7],
    ]);
    let hi = u64::from_le_bytes([
      bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13],
      bytes[14], bytes[15],
    ]);
    lo ^ hi
  }

  #[cfg(target_pointer_width = "32")]
  {
    let lx = low_u32(x);
    let ly = low_u32(y);
    let hx = high_u32(x);
    let hy = high_u32(y);
    let afull = (lx as u64) * (hy as u64);
    let bfull = (hx as u64) * (ly as u64);
    afull ^ bfull.rotate_right(32)
  }
}

#[inline]
const fn hash_add_u64(mut hash: usize, value: u64) -> usize {
  #[cfg(target_pointer_width = "64")]
  {
    hash = hash
      .wrapping_add(usize::from_ne_bytes(value.to_ne_bytes()))
      .wrapping_mul(K_USIZE);
  }

  #[cfg(target_pointer_width = "32")]
  {
    let bytes = value.to_ne_bytes();
    let low = u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let high = u32::from_ne_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    hash = hash
      .wrapping_add(usize::try_from(low).unwrap_or(usize::MAX))
      .wrapping_mul(K_USIZE);
    hash = hash
      .wrapping_add(usize::try_from(high).unwrap_or(usize::MAX))
      .wrapping_mul(K_USIZE);
  }
  hash
}

#[inline]
fn hash_add_u32(hash: usize, value: u32) -> usize {
  hash
    .wrapping_add(usize::try_from(value).unwrap_or(usize::MAX))
    .wrapping_mul(K_USIZE)
}

#[inline]
pub fn usize_to_f64(value: usize) -> f64 {
  #[cfg(target_pointer_width = "64")]
  {
    let bytes = value.to_be_bytes();
    let high = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let low = u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    f64::from(high).mul_add(4_294_967_296.0, f64::from(low))
  }

  #[cfg(target_pointer_width = "32")]
  {
    f64::from(u32::from_ne_bytes(value.to_ne_bytes()))
  }
}

#[inline]
pub fn ratio_usize(numerator: usize, denominator: usize) -> f64 {
  if denominator == 0 {
    0.0
  } else {
    usize_to_f64(numerator) / usize_to_f64(denominator)
  }
}

#[inline]
fn hash_bytes(bytes: &[u8]) -> u64 {
  let len = bytes.len();
  let mut s0 = SEED1;
  let mut s1 = SEED2;

  if len <= 16 {
    if len >= 8 {
      s0 ^= read_u64_le(bytes, 0);
      s1 ^= read_u64_le(bytes, len - 8);
    } else if len >= 4 {
      s0 ^= u64::from(read_u32_le(bytes, 0));
      s1 ^= u64::from(read_u32_le(bytes, len - 4));
    } else if len > 0 {
      let lo = bytes[0];
      let mid = bytes[len / 2];
      let hi = bytes[len - 1];
      s0 ^= u64::from(lo);
      s1 ^= (u64::from(hi) << 8) | u64::from(mid);
    }
  } else {
    let mut off = 0usize;
    while off < len - 16 {
      let x = read_u64_le(bytes, off);
      let y = read_u64_le(bytes, off + 8);
      let t = multiply_mix(s0 ^ x, PREVENT_TRIVIAL_ZERO_COLLAPSE ^ y);
      s0 = s1;
      s1 = t;
      off += 16;
    }

    let suffix_off = len - 16;
    s0 ^= read_u64_le(bytes, suffix_off);
    s1 ^= read_u64_le(bytes, suffix_off + 8);
  }

  multiply_mix(s0, s1) ^ (len as u64)
}

/// Fast hash function for byte arrays
#[inline]
pub fn calculate_hash_fast(data: &[u8]) -> u64 {
  let compressed = hash_bytes(data);
  #[cfg(target_pointer_width = "64")]
  {
    let hash = compressed.wrapping_mul(K_U64);
    hash.rotate_left(ROTATE)
  }

  #[cfg(target_pointer_width = "32")]
  {
    let mut hash = usize::try_from(low_u32(compressed))
      .unwrap_or(usize::MAX)
      .wrapping_mul(K_USIZE);
    hash = hash
      .wrapping_add(usize::try_from(high_u32(compressed)).unwrap_or(usize::MAX))
      .wrapping_mul(K_USIZE);
    u64::from(u32::from_ne_bytes(hash.rotate_left(ROTATE).to_ne_bytes()))
  }
}

/// Applies a permutation to a hash value.
#[inline]
pub const fn permute_hash(hash: u64, a: u64, b: u64) -> u32 {
  ((a.wrapping_mul(hash).wrapping_add(b)) >> 32) as u32
}

/// Calculates a hash value for a band of `MinHash` values.
#[inline]
pub fn calculate_band_hash(band: &[u32]) -> u64 {
  let mut hash = 0_usize;
  let mut index = 0_usize;

  while index + 4 <= band.len() {
    let val1 = u64::from(band[index]) | (u64::from(band[index + 1]) << 32);
    let val2 = u64::from(band[index + 2]) | (u64::from(band[index + 3]) << 32);

    // Mirror `rustc_hash::FxHasher`'s specialized integer hashing, but without
    // the trait dispatch overhead.
    hash = hash_add_u64(hash, val1);
    hash = hash_add_u64(hash, val2);

    index += 4;
  }

  for &value in &band[index..] {
    hash = hash_add_u32(hash, value);
  }

  #[cfg(target_pointer_width = "64")]
  {
    u64::from_ne_bytes(hash.rotate_left(ROTATE).to_ne_bytes())
  }
  #[cfg(target_pointer_width = "32")]
  {
    u64::from(u32::from_ne_bytes(hash.rotate_left(ROTATE).to_ne_bytes()))
  }
}

#[cfg(test)]
mod tests {
  use crate::utils::{calculate_band_hash, calculate_hash_fast, permute_hash};
  use rustc_hash::FxHasher;
  use std::hash::Hasher;

  fn reference_hash_fast(data: &[u8]) -> u64 {
    let mut hasher = FxHasher::default();
    hasher.write(data);
    hasher.finish()
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
