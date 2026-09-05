use crate::utils::permute_hash;
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KernelKind {
  Scalar,
  #[cfg(target_arch = "aarch64")]
  Neon,
  #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
  Avx2,
}

#[derive(Clone, Default)]
pub struct PermutationSoA {
  len: usize,
  #[cfg(target_arch = "aarch64")]
  a_hi: Vec<u32>,
  #[cfg(target_arch = "aarch64")]
  a_lo: Vec<u32>,
  #[cfg(target_arch = "aarch64")]
  b_hi: Vec<u32>,
  #[cfg(target_arch = "aarch64")]
  b_lo: Vec<u32>,
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[allow(clippy::cast_possible_truncation)]
pub(super) const fn split_u64_words(value: u64) -> (u32, u32) {
  (value as u32, (value >> 32) as u32)
}

impl PermutationSoA {
  #[must_use]
  #[cfg(not(target_arch = "aarch64"))]
  pub const fn from_permutations(permutations: &[(u64, u64)]) -> Self {
    Self {
      len: permutations.len(),
    }
  }

  #[must_use]
  #[cfg(target_arch = "aarch64")]
  pub fn from_permutations(permutations: &[(u64, u64)]) -> Self {
    let len = permutations.len();
    let mut a_hi = Vec::with_capacity(len);
    let mut a_lo = Vec::with_capacity(len);
    let mut b_hi = Vec::with_capacity(len);
    let mut b_lo = Vec::with_capacity(len);

    for &(a, b) in permutations {
      let (a_low, a_high) = split_u64_words(a);
      let (b_low, b_high) = split_u64_words(b);
      a_hi.push(a_high);
      a_lo.push(a_low);
      b_hi.push(b_high);
      b_lo.push(b_low);
    }

    Self {
      len,
      a_hi,
      a_lo,
      b_hi,
      b_lo,
    }
  }

  #[inline]
  #[must_use]
  pub const fn len(&self) -> usize {
    self.len
  }

  #[inline]
  #[must_use]
  pub const fn is_empty(&self) -> bool {
    self.len == 0
  }

  #[cfg(target_arch = "aarch64")]
  #[inline]
  #[must_use]
  pub(super) fn a_hi(&self) -> &[u32] {
    &self.a_hi
  }

  #[cfg(target_arch = "aarch64")]
  #[inline]
  #[must_use]
  pub(super) fn a_lo(&self) -> &[u32] {
    &self.a_lo
  }

  #[cfg(target_arch = "aarch64")]
  #[inline]
  #[must_use]
  pub(super) fn b_hi(&self) -> &[u32] {
    &self.b_hi
  }

  #[cfg(target_arch = "aarch64")]
  #[inline]
  #[must_use]
  pub(super) fn b_lo(&self) -> &[u32] {
    &self.b_lo
  }
}

static KERNEL_KIND: OnceLock<KernelKind> = OnceLock::new();

#[must_use]
fn kernel_kind() -> KernelKind {
  *KERNEL_KIND.get_or_init(detect_kernel_kind)
}

#[cfg(test)]
#[must_use]
const fn kernel_kind_name(kind: KernelKind) -> &'static str {
  match kind {
    KernelKind::Scalar => "scalar",
    #[cfg(target_arch = "aarch64")]
    KernelKind::Neon => "neon",
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    KernelKind::Avx2 => "avx2",
  }
}

fn detect_kernel_kind() -> KernelKind {
  if let Some(value) = std::env::var_os("RENSA_FORCE_KERNEL") {
    if let Some(kind) = parse_forced_kernel(&value.to_string_lossy()) {
      if forced_kernel_supported(kind) {
        return kind;
      }
    }
  }
  default_kernel_kind()
}

#[inline]
fn forced_kernel_supported(kind: KernelKind) -> bool {
  match kind {
    KernelKind::Scalar => true,
    #[cfg(target_arch = "aarch64")]
    KernelKind::Neon => std::arch::is_aarch64_feature_detected!("neon"),
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    KernelKind::Avx2 => std::arch::is_x86_feature_detected!("avx2"),
  }
}

fn parse_forced_kernel(value: &str) -> Option<KernelKind> {
  let value = value.trim();
  if value.eq_ignore_ascii_case("scalar") {
    return Some(KernelKind::Scalar);
  }

  #[cfg(target_arch = "aarch64")]
  if value.eq_ignore_ascii_case("neon") {
    return Some(KernelKind::Neon);
  }

  #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
  if value.eq_ignore_ascii_case("avx2") {
    return Some(KernelKind::Avx2);
  }

  None
}

#[cfg(target_arch = "aarch64")]
fn default_kernel_kind() -> KernelKind {
  if std::arch::is_aarch64_feature_detected!("neon") {
    KernelKind::Neon
  } else {
    KernelKind::Scalar
  }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn default_kernel_kind() -> KernelKind {
  if std::arch::is_x86_feature_detected!("avx2") {
    KernelKind::Avx2
  } else {
    KernelKind::Scalar
  }
}

#[cfg(not(any(
  target_arch = "aarch64",
  target_arch = "x86",
  target_arch = "x86_64"
)))]
fn default_kernel_kind() -> KernelKind {
  KernelKind::Scalar
}

pub fn apply_hash_batch_to_values(
  hash_values: &mut [u32],
  permutations: &[(u64, u64)],
  permutations_soa: &PermutationSoA,
  hash_batch: &[u64],
) {
  if hash_batch.is_empty() || hash_values.is_empty() {
    return;
  }
  debug_assert!(
    permutations_soa.is_empty() || permutations_soa.len() == permutations.len(),
    "permutations_soa length must match permutations length",
  );

  match kernel_kind() {
    KernelKind::Scalar => {
      let _ = permutations_soa;
      scalar_apply_hash_batch_to_values(hash_values, permutations, hash_batch);
    }
    #[cfg(target_arch = "aarch64")]
    KernelKind::Neon => {
      crate::simd::arm64_neon::apply_hash_batch_to_values_neon(
        hash_values,
        permutations_soa,
        hash_batch,
      );
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    KernelKind::Avx2 => {
      crate::simd::x86::apply_hash_batch_to_values_avx2(
        hash_values,
        permutations,
        hash_batch,
      );
    }
  }
}

pub fn apply_bucket_fallback(
  hash_values: &mut [u32],
  permutations: &[(u64, u64)],
  hashes: &[u64],
  rank_bits: u32,
) {
  const PACKED_LANES: usize = 128;
  debug_assert!((1..32).contains(&rank_bits));
  let fallback = u32::MAX << (32 - rank_bits);
  let discarded = (1 << rank_bits) - 1;
  let len = hash_values.len().min(permutations.len());
  let kind = kernel_kind();
  let mut cursor = 0;
  while cursor < len {
    let mut indices = [0; PACKED_LANES];
    let mut pairs = [(0, 0); PACKED_LANES];
    let mut values = [u32::MAX; PACKED_LANES];
    let mut count = 0;
    while cursor < len && count < PACKED_LANES {
      if hash_values[cursor] >= fallback {
        indices[count] = cursor;
        pairs[count] = permutations[cursor];
        // Preserve every raw rank represented by the truncated fraction.
        values[count] = (hash_values[cursor] << rank_bits) | discarded;
        count += 1;
      }
      cursor += 1;
    }
    if count == 0 {
      break;
    }
    if count < 4 {
      for (value, &(a, b)) in values[..count].iter_mut().zip(&pairs) {
        for &hash in hashes {
          *value = (*value).min(permute_hash(hash, a, b));
        }
      }
    } else {
      match kind {
        KernelKind::Scalar => {
          scalar_apply_hash_batch_to_values(
            &mut values[..count],
            &pairs[..count],
            hashes,
          );
        }
        #[cfg(target_arch = "aarch64")]
        KernelKind::Neon => {
          let lanes = if count == 4 {
            4
          } else {
            count.next_multiple_of(8)
          };
          crate::simd::arm64_neon::apply_hash_batch_to_values_neon_compact(
            &mut values[..lanes],
            &pairs,
            hashes,
          );
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        KernelKind::Avx2 => {
          let lanes = count.next_multiple_of(8);
          crate::simd::x86::apply_hash_batch_to_values_avx2(
            &mut values[..lanes],
            &pairs[..lanes],
            hashes,
          );
        }
      }
    }
    for (&index, &value) in indices[..count].iter().zip(&values) {
      hash_values[index] = fallback | (value >> rank_bits);
    }
  }
}

fn scalar_apply_hash_batch_to_values(
  hash_values: &mut [u32],
  permutations: &[(u64, u64)],
  hash_batch: &[u64],
) {
  let perm_len = hash_values.len().min(permutations.len());
  debug_assert!(perm_len <= permutations.len());
  debug_assert!(perm_len <= hash_values.len());
  let permutations_ptr = permutations.as_ptr();
  let values_ptr = hash_values.as_mut_ptr();
  let hash_ptr = hash_batch.as_ptr();
  let hash_len = hash_batch.len();
  let mut index = 0usize;

  while index + 8 <= perm_len {
    // SAFETY: `index + 7 < perm_len <= permutations.len()` in this loop.
    let (a0, b0) = unsafe { *permutations_ptr.add(index) };
    let (a1, b1) = unsafe { *permutations_ptr.add(index + 1) };
    let (a2, b2) = unsafe { *permutations_ptr.add(index + 2) };
    let (a3, b3) = unsafe { *permutations_ptr.add(index + 3) };
    let (a4, b4) = unsafe { *permutations_ptr.add(index + 4) };
    let (a5, b5) = unsafe { *permutations_ptr.add(index + 5) };
    let (a6, b6) = unsafe { *permutations_ptr.add(index + 6) };
    let (a7, b7) = unsafe { *permutations_ptr.add(index + 7) };

    let mut min0 = unsafe { *values_ptr.add(index) };
    let mut min1 = unsafe { *values_ptr.add(index + 1) };
    let mut min2 = unsafe { *values_ptr.add(index + 2) };
    let mut min3 = unsafe { *values_ptr.add(index + 3) };
    let mut min4 = unsafe { *values_ptr.add(index + 4) };
    let mut min5 = unsafe { *values_ptr.add(index + 5) };
    let mut min6 = unsafe { *values_ptr.add(index + 6) };
    let mut min7 = unsafe { *values_ptr.add(index + 7) };

    let mut hash_index = 0usize;
    while hash_index + 1 < hash_len {
      // SAFETY: `hash_index + 1 < hash_len == hash_batch.len()` in this loop.
      let item_hash0 = unsafe { *hash_ptr.add(hash_index) };
      let item_hash1 = unsafe { *hash_ptr.add(hash_index + 1) };

      min0 = min0.min(permute_hash(item_hash0, a0, b0));
      min1 = min1.min(permute_hash(item_hash0, a1, b1));
      min2 = min2.min(permute_hash(item_hash0, a2, b2));
      min3 = min3.min(permute_hash(item_hash0, a3, b3));
      min4 = min4.min(permute_hash(item_hash0, a4, b4));
      min5 = min5.min(permute_hash(item_hash0, a5, b5));
      min6 = min6.min(permute_hash(item_hash0, a6, b6));
      min7 = min7.min(permute_hash(item_hash0, a7, b7));

      min0 = min0.min(permute_hash(item_hash1, a0, b0));
      min1 = min1.min(permute_hash(item_hash1, a1, b1));
      min2 = min2.min(permute_hash(item_hash1, a2, b2));
      min3 = min3.min(permute_hash(item_hash1, a3, b3));
      min4 = min4.min(permute_hash(item_hash1, a4, b4));
      min5 = min5.min(permute_hash(item_hash1, a5, b5));
      min6 = min6.min(permute_hash(item_hash1, a6, b6));
      min7 = min7.min(permute_hash(item_hash1, a7, b7));
      hash_index += 2;
    }

    if hash_index < hash_len {
      // SAFETY: `hash_index < hash_len == hash_batch.len()`.
      let item_hash = unsafe { *hash_ptr.add(hash_index) };
      min0 = min0.min(permute_hash(item_hash, a0, b0));
      min1 = min1.min(permute_hash(item_hash, a1, b1));
      min2 = min2.min(permute_hash(item_hash, a2, b2));
      min3 = min3.min(permute_hash(item_hash, a3, b3));
      min4 = min4.min(permute_hash(item_hash, a4, b4));
      min5 = min5.min(permute_hash(item_hash, a5, b5));
      min6 = min6.min(permute_hash(item_hash, a6, b6));
      min7 = min7.min(permute_hash(item_hash, a7, b7));
    }

    unsafe {
      *values_ptr.add(index) = min0;
      *values_ptr.add(index + 1) = min1;
      *values_ptr.add(index + 2) = min2;
      *values_ptr.add(index + 3) = min3;
      *values_ptr.add(index + 4) = min4;
      *values_ptr.add(index + 5) = min5;
      *values_ptr.add(index + 6) = min6;
      *values_ptr.add(index + 7) = min7;
    }
    index += 8;
  }

  while index < perm_len {
    // SAFETY: `index < perm_len <= permutations.len()`.
    let (a, b) = unsafe { *permutations_ptr.add(index) };
    // SAFETY: `index < perm_len <= hash_values.len()`.
    let mut min_value = unsafe { *values_ptr.add(index) };
    let mut hash_index = 0usize;
    while hash_index + 1 < hash_len {
      // SAFETY: `hash_index + 1 < hash_len == hash_batch.len()`.
      let item_hash0 = unsafe { *hash_ptr.add(hash_index) };
      let item_hash1 = unsafe { *hash_ptr.add(hash_index + 1) };
      min_value = min_value.min(permute_hash(item_hash0, a, b));
      min_value = min_value.min(permute_hash(item_hash1, a, b));
      hash_index += 2;
    }
    if hash_index < hash_len {
      // SAFETY: `hash_index < hash_len == hash_batch.len()`.
      let item_hash = unsafe { *hash_ptr.add(hash_index) };
      min_value = min_value.min(permute_hash(item_hash, a, b));
    }
    unsafe {
      *values_ptr.add(index) = min_value;
    }
    index += 1;
  }
}

#[cfg(test)]
mod tests {
  use crate::simd::dispatch::{
    forced_kernel_supported, kernel_kind_name, parse_forced_kernel,
    scalar_apply_hash_batch_to_values, KernelKind, PermutationSoA,
  };
  use crate::utils::permute_hash;
  use rand_core::{RngCore, SeedableRng};
  use rand_xoshiro::Xoshiro256PlusPlus;

  #[test]
  fn permutation_soa_keeps_lane_values() {
    let permutations = vec![
      (0x1234_5678_90ab_cdef, 0x0fed_cba9_8765_4321),
      (0xffff_0000_abcd_0001, 0x0102_0304_0506_0708),
    ];
    let soa = PermutationSoA::from_permutations(&permutations);
    assert_eq!(soa.len(), 2);
    #[cfg(target_arch = "aarch64")]
    {
      assert_eq!(soa.a_hi[0], 0x1234_5678);
      assert_eq!(soa.a_lo[0], 0x90ab_cdef);
      assert_eq!(soa.b_hi[1], 0x0102_0304);
      assert_eq!(soa.b_lo[1], 0x0506_0708);
    }
  }

  #[test]
  fn kernel_kind_name_is_stable() {
    assert_eq!(kernel_kind_name(KernelKind::Scalar), "scalar");
    #[cfg(target_arch = "aarch64")]
    assert_eq!(kernel_kind_name(KernelKind::Neon), "neon");
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    assert_eq!(kernel_kind_name(KernelKind::Avx2), "avx2");
  }

  #[test]
  fn parse_forced_kernel_understands_supported_values() {
    assert_eq!(parse_forced_kernel("scalar"), Some(KernelKind::Scalar));
    assert_eq!(parse_forced_kernel("SCALAR"), Some(KernelKind::Scalar));
    #[cfg(target_arch = "aarch64")]
    assert_eq!(parse_forced_kernel("neon"), Some(KernelKind::Neon));
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    assert_eq!(parse_forced_kernel("avx2"), Some(KernelKind::Avx2));
    assert_eq!(parse_forced_kernel("unknown"), None);
  }

  #[test]
  fn forced_kernel_support_mapping_matches_runtime() {
    assert!(forced_kernel_supported(KernelKind::Scalar));
    #[cfg(target_arch = "aarch64")]
    assert_eq!(
      forced_kernel_supported(KernelKind::Neon),
      std::arch::is_aarch64_feature_detected!("neon")
    );
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    assert_eq!(
      forced_kernel_supported(KernelKind::Avx2),
      std::arch::is_x86_feature_detected!("avx2")
    );
  }

  #[test]
  fn scalar_soa_path_matches_reference() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x9f0a_12bc);
    for _ in 0..64 {
      let num_perm = 1 + (rng.next_u64() % 96) as usize;
      let hashes_len = 1 + (rng.next_u64() % 128) as usize;
      let mut permutations = Vec::with_capacity(num_perm);
      for _ in 0..num_perm {
        let a = rng.next_u64() | 1;
        let b = rng.next_u64();
        permutations.push((a, b));
      }
      let mut hash_batch = Vec::with_capacity(hashes_len);
      for _ in 0..hashes_len {
        hash_batch.push(rng.next_u64());
      }

      let mut soa_values = vec![u32::MAX; num_perm];
      let mut reference_values = vec![u32::MAX; num_perm];
      scalar_apply_hash_batch_to_values(
        &mut soa_values,
        &permutations,
        &hash_batch,
      );
      for (value, &(a, b)) in reference_values.iter_mut().zip(&permutations) {
        for &hash in &hash_batch {
          *value = (*value).min(permute_hash(hash, a, b));
        }
      }
      assert_eq!(soa_values, reference_values);
    }
  }

  fn assert_kernels_match_naive(
    initial: &[u32],
    permutations: &[(u64, u64)],
    hashes: &[u64],
  ) {
    let mut expected = initial.to_vec();
    for (value, &(a, b)) in expected.iter_mut().zip(permutations) {
      for &hash in hashes {
        // Independent wide-integer oracle: truncation modulo 2^64, then high word.
        let result = (u128::from(a) * u128::from(hash) + u128::from(b)) >> 32;
        *value =
          (*value).min(u32::try_from(result & u128::from(u32::MAX)).unwrap());
      }
    }
    let mut scalar = initial.to_vec();
    scalar_apply_hash_batch_to_values(&mut scalar, permutations, hashes);
    assert_eq!(scalar, expected);

    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
      let mut actual = initial.to_vec();
      crate::simd::arm64_neon::apply_hash_batch_to_values_neon(
        &mut actual,
        &PermutationSoA::from_permutations(permutations),
        hashes,
      );
      assert_eq!(actual, expected, "NEON mismatch");
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::arch::is_x86_feature_detected!("avx2") {
      let mut actual = initial.to_vec();
      crate::simd::x86::apply_hash_batch_to_values_avx2(
        &mut actual,
        permutations,
        hashes,
      );
      assert_eq!(actual, expected, "AVX2 mismatch");
    }
  }

  #[test]
  fn kernels_match_naive_for_all_small_sizes_and_tails() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5441_494c);
    for num_perm in 0..=33 {
      let permutations: Vec<_> = (0..num_perm)
        .map(|_| (rng.next_u64(), rng.next_u64()))
        .collect();
      for num_hashes in 0..=17 {
        let hashes: Vec<_> = (0..num_hashes).map(|_| rng.next_u64()).collect();
        for num_values in [num_perm / 2, num_perm, num_perm + 3] {
          let initial: Vec<_> =
            (0..num_values).map(|_| rng.next_u32()).collect();
          assert_kernels_match_naive(&initial, &permutations, &hashes);
        }
      }
    }
  }

  #[test]
  fn kernels_match_naive_for_full_width_carries_and_overflow() {
    let words = [0, 1, 0x7fff_ffff, 0x8000_0000, 0xffff_ffff];
    let mut boundaries = Vec::new();
    for &high in &words {
      for &low in &words {
        boundaries.push((high << 32) | low);
      }
    }
    let permutations: Vec<_> = boundaries
      .iter()
      .flat_map(|&a| boundaries.iter().map(move |&b| (a, b)))
      .collect();
    for hash in boundaries {
      // Single hashes ensure a smaller minimum cannot hide an incorrect lane.
      assert_kernels_match_naive(
        &vec![u32::MAX; permutations.len()],
        &permutations,
        &[hash],
      );
    }
  }
}
