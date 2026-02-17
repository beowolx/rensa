use crate::utils::permute_hash;
#[cfg(target_arch = "aarch64")]
use std::hint::black_box;
use std::sync::OnceLock;
#[cfg(target_arch = "aarch64")]
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelKind {
  Scalar,
  #[cfg(target_arch = "aarch64")]
  Neon,
  #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
  Avx2,
}

#[derive(Clone, Default)]
pub struct PermutationSoA {
  pub a_hi: Vec<u32>,
  #[cfg(target_arch = "aarch64")]
  pub a_lo: Vec<u32>,
  #[cfg(target_arch = "aarch64")]
  pub b_hi: Vec<u32>,
  #[cfg(target_arch = "aarch64")]
  pub b_lo: Vec<u32>,
}

#[inline]
const fn split_u64_words(value: u64) -> (u32, u32) {
  let bytes = value.to_le_bytes();
  let low = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
  let high = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
  (low, high)
}

impl PermutationSoA {
  #[must_use]
  pub fn from_permutations(permutations: &[(u64, u64)]) -> Self {
    let mut a_hi = Vec::with_capacity(permutations.len());
    #[cfg(target_arch = "aarch64")]
    let mut a_lo = Vec::with_capacity(permutations.len());
    #[cfg(target_arch = "aarch64")]
    let mut b_hi = Vec::with_capacity(permutations.len());
    #[cfg(target_arch = "aarch64")]
    let mut b_lo = Vec::with_capacity(permutations.len());

    for &(a, b) in permutations {
      #[cfg(target_arch = "aarch64")]
      {
        let (a_low, a_high) = split_u64_words(a);
        let (b_low, b_high) = split_u64_words(b);
        a_hi.push(a_high);
        a_lo.push(a_low);
        b_hi.push(b_high);
        b_lo.push(b_low);
      }
      #[cfg(not(target_arch = "aarch64"))]
      {
        let _ = b;
        let (_, a_high) = split_u64_words(a);
        a_hi.push(a_high);
      }
    }

    Self {
      a_hi,
      #[cfg(target_arch = "aarch64")]
      a_lo,
      #[cfg(target_arch = "aarch64")]
      b_hi,
      #[cfg(target_arch = "aarch64")]
      b_lo,
    }
  }

  #[inline]
  #[must_use]
  pub fn len(&self) -> usize {
    self.a_hi.len()
  }

  #[inline]
  #[must_use]
  pub fn is_empty(&self) -> bool {
    self.a_hi.is_empty()
  }
}

static KERNEL_KIND: OnceLock<KernelKind> = OnceLock::new();

#[cfg(target_arch = "aarch64")]
const AUTO_CAL_NUM_PERM: usize = 128;
#[cfg(target_arch = "aarch64")]
const AUTO_CAL_HASH_BATCH_SIZES: [usize; 2] = [256, 768];
#[cfg(target_arch = "aarch64")]
const AUTO_CAL_WARMUP_ITERS: usize = 3;
#[cfg(target_arch = "aarch64")]
const AUTO_CAL_TIMED_ITERS: usize = 5;
#[cfg(target_arch = "aarch64")]
const AUTO_CAL_SCALAR_PREFERENCE_EPSILON: f64 = 0.02;
#[cfg(target_arch = "aarch64")]
const AUTO_CAL_PER_SAMPLE_NEON_ADVANTAGE: f64 = 0.08;

#[must_use]
pub fn kernel_kind() -> KernelKind {
  *KERNEL_KIND.get_or_init(detect_kernel_kind)
}

#[must_use]
pub const fn kernel_kind_name(kind: KernelKind) -> &'static str {
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
      return kind;
    }
  }

  #[cfg(target_arch = "aarch64")]
  {
    detect_aarch64_kernel_kind()
  }

  #[cfg(not(target_arch = "aarch64"))]
  {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
      if std::arch::is_x86_feature_detected!("avx2") {
        return KernelKind::Avx2;
      }
    }
    KernelKind::Scalar
  }
}

#[cfg(target_arch = "aarch64")]
fn detect_aarch64_kernel_kind() -> KernelKind {
  if !std::arch::is_aarch64_feature_detected!("neon") {
    return KernelKind::Scalar;
  }
  calibrate_aarch64_kernel()
}

#[cfg(target_arch = "aarch64")]
fn calibrate_aarch64_kernel() -> KernelKind {
  let permutations =
    calibration_permutations(AUTO_CAL_NUM_PERM, 0x59f9_5c8a_1d2b_4f67);
  let permutations_soa = PermutationSoA::from_permutations(&permutations);
  let mut scalar_samples = Vec::with_capacity(AUTO_CAL_HASH_BATCH_SIZES.len());
  let mut neon_samples = Vec::with_capacity(AUTO_CAL_HASH_BATCH_SIZES.len());

  for hash_count in AUTO_CAL_HASH_BATCH_SIZES {
    let hash_batch = calibration_hash_batch(hash_count, hash_count as u64);
    scalar_samples.push(benchmark_kernel(
      KernelKind::Scalar,
      &permutations,
      &permutations_soa,
      &hash_batch,
    ));
    neon_samples.push(benchmark_kernel(
      KernelKind::Neon,
      &permutations,
      &permutations_soa,
      &hash_batch,
    ));
  }

  if !neon_wins_all_samples(&scalar_samples, &neon_samples) {
    return KernelKind::Scalar;
  }

  select_aarch64_kernel(
    geometric_mean(&scalar_samples),
    geometric_mean(&neon_samples),
  )
}

#[cfg(target_arch = "aarch64")]
fn calibration_permutations(num_perm: usize, seed: u64) -> Vec<(u64, u64)> {
  let mut state = seed;
  let mut permutations = Vec::with_capacity(num_perm);
  for _ in 0..num_perm {
    let a = lcg_next(&mut state) | 1;
    let b = lcg_next(&mut state);
    permutations.push((a, b));
  }
  permutations
}

#[cfg(target_arch = "aarch64")]
fn calibration_hash_batch(size: usize, seed: u64) -> Vec<u64> {
  let mut state = seed ^ 0x9e37_79b9_7f4a_7c15;
  let mut batch = Vec::with_capacity(size);
  for _ in 0..size {
    batch.push(lcg_next(&mut state));
  }
  batch
}

#[cfg(target_arch = "aarch64")]
#[inline]
const fn lcg_next(state: &mut u64) -> u64 {
  *state = state
    .wrapping_mul(6_364_136_223_846_793_005)
    .wrapping_add(1_442_695_040_888_963_407);
  *state
}

#[cfg(target_arch = "aarch64")]
fn benchmark_kernel(
  kind: KernelKind,
  permutations: &[(u64, u64)],
  permutations_soa: &PermutationSoA,
  hash_batch: &[u64],
) -> f64 {
  let mut samples = Vec::with_capacity(AUTO_CAL_TIMED_ITERS);
  for iter in 0..(AUTO_CAL_WARMUP_ITERS + AUTO_CAL_TIMED_ITERS) {
    let mut hash_values = vec![u32::MAX; permutations.len()];
    let start = Instant::now();
    match kind {
      KernelKind::Scalar => scalar_apply_hash_batch_to_values(
        &mut hash_values,
        permutations,
        hash_batch,
      ),
      KernelKind::Neon => {
        crate::simd::arm64_neon::apply_hash_batch_to_values_neon(
          &mut hash_values,
          permutations_soa,
          hash_batch,
        );
      }
      #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
      KernelKind::Avx2 => crate::simd::x86::apply_hash_batch_to_values_avx2(
        &mut hash_values,
        permutations,
        hash_batch,
      ),
    }
    black_box(&hash_values);
    if iter >= AUTO_CAL_WARMUP_ITERS {
      samples.push(start.elapsed().as_secs_f64());
    }
  }
  geometric_mean(&samples)
}

#[cfg(target_arch = "aarch64")]
fn geometric_mean(samples: &[f64]) -> f64 {
  if samples.is_empty() {
    return f64::INFINITY;
  }
  let log_sum: f64 = samples.iter().map(|value| value.ln()).sum();
  (log_sum / samples.len() as f64).exp()
}

#[cfg(target_arch = "aarch64")]
fn neon_wins_all_samples(scalar_samples: &[f64], neon_samples: &[f64]) -> bool {
  scalar_samples.len() == neon_samples.len()
    && scalar_samples
      .iter()
      .zip(neon_samples)
      .all(|(&scalar, &neon)| {
        neon < scalar * (1.0 - AUTO_CAL_PER_SAMPLE_NEON_ADVANTAGE)
      })
}

#[cfg(target_arch = "aarch64")]
fn select_aarch64_kernel(scalar_time: f64, neon_time: f64) -> KernelKind {
  if neon_time < scalar_time * (1.0 - AUTO_CAL_SCALAR_PREFERENCE_EPSILON) {
    KernelKind::Neon
  } else {
    KernelKind::Scalar
  }
}

fn parse_forced_kernel(value: &str) -> Option<KernelKind> {
  match value.to_ascii_lowercase().as_str() {
    "scalar" => Some(KernelKind::Scalar),
    #[cfg(target_arch = "aarch64")]
    "neon" => Some(KernelKind::Neon),
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    "avx2" => Some(KernelKind::Avx2),
    _ => None,
  }
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

fn scalar_apply_hash_batch_to_values(
  hash_values: &mut [u32],
  permutations: &[(u64, u64)],
  hash_batch: &[u64],
) {
  let perm_len = hash_values.len().min(permutations.len());
  let permutations_ptr = permutations.as_ptr();
  let values_ptr = hash_values.as_mut_ptr();
  let mut index = 0usize;

  while index + 8 <= perm_len {
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

    let hash_ptr = hash_batch.as_ptr();
    let hash_len = hash_batch.len();
    let mut hash_index = 0usize;
    while hash_index + 1 < hash_len {
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
    let (a, b) = unsafe { *permutations_ptr.add(index) };
    let mut min_value = unsafe { *values_ptr.add(index) };
    let hash_ptr = hash_batch.as_ptr();
    let hash_len = hash_batch.len();
    let mut hash_index = 0usize;
    while hash_index + 1 < hash_len {
      let item_hash0 = unsafe { *hash_ptr.add(hash_index) };
      let item_hash1 = unsafe { *hash_ptr.add(hash_index + 1) };
      min_value = min_value.min(permute_hash(item_hash0, a, b));
      min_value = min_value.min(permute_hash(item_hash1, a, b));
      hash_index += 2;
    }
    if hash_index < hash_len {
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
    kernel_kind_name, parse_forced_kernel, scalar_apply_hash_batch_to_values,
    KernelKind, PermutationSoA,
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
    assert_eq!(soa.a_hi[0], 0x1234_5678);
    #[cfg(target_arch = "aarch64")]
    assert_eq!(soa.a_lo[0], 0x90ab_cdef);
    #[cfg(target_arch = "aarch64")]
    assert_eq!(soa.b_hi[1], 0x0102_0304);
    #[cfg(target_arch = "aarch64")]
    assert_eq!(soa.b_lo[1], 0x0506_0708);
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

  #[cfg(target_arch = "aarch64")]
  #[test]
  fn aarch64_tie_break_prefers_scalar() {
    use crate::simd::dispatch::{
      select_aarch64_kernel, AUTO_CAL_SCALAR_PREFERENCE_EPSILON,
    };
    let scalar = 1.0;
    let barely_faster_neon =
      scalar * (1.0 - AUTO_CAL_SCALAR_PREFERENCE_EPSILON / 2.0);
    assert_eq!(
      select_aarch64_kernel(scalar, barely_faster_neon),
      KernelKind::Scalar
    );
    let clearly_faster_neon =
      scalar * (1.0 - AUTO_CAL_SCALAR_PREFERENCE_EPSILON - 0.01);
    assert_eq!(
      select_aarch64_kernel(scalar, clearly_faster_neon),
      KernelKind::Neon
    );
  }

  #[cfg(target_arch = "aarch64")]
  #[test]
  fn aarch64_requires_neon_to_win_every_sample() {
    use crate::simd::dispatch::neon_wins_all_samples;
    let scalar = [1.0, 1.0];
    let mixed_neon = [0.89, 0.95];
    assert!(!neon_wins_all_samples(&scalar, &mixed_neon));
    let clear_neon = [0.89, 0.90];
    assert!(neon_wins_all_samples(&scalar, &clear_neon));
  }
}
