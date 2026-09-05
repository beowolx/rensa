use crate::utils::permute_hash;

/// Mixes eight independent token values; coordinate updates remain scalar.
#[inline]
pub(super) unsafe fn splitmix64x8_avx512(
  values: *const u64,
  seed: u64,
) -> [u64; 8] {
  const CONSTANTS: [u64; 3] = [
    0x9e37_79b9_7f4a_7c15,
    0xbf58_476d_1ce4_e5b9,
    0x94d0_49bb_1331_11eb,
  ];
  let mut output = std::mem::MaybeUninit::<[u64; 8]>::uninit();
  // SAFETY: the caller proves AVX-512F/DQ support and that values points to
  // eight readable u64 values. The unaligned store initializes exactly the
  // eight output values; broadcasts read one seed and three constants. All
  // used vector registers are declared clobbered. Assembly retains Rust 1.83
  // compatibility, before AVX-512 intrinsics and target_feature stabilized.
  // Every normal return follows the full 64-byte store, so assume_init is
  // valid and avoids zeroing output that the assembly immediately replaces.
  unsafe {
    core::arch::asm!(
      "vmovdqu64 zmm0, [{values}]",
      "vpbroadcastq zmm1, [{seed}]",
      "vpxorq zmm0, zmm0, zmm1",
      "vpbroadcastq zmm1, [{constants}]",
      "vpaddq zmm0, zmm0, zmm1",
      "vpsrlq zmm2, zmm0, 30",
      "vpxorq zmm0, zmm0, zmm2",
      "vpbroadcastq zmm1, [{constants} + 8]",
      "vpmullq zmm0, zmm0, zmm1",
      "vpsrlq zmm2, zmm0, 27",
      "vpxorq zmm0, zmm0, zmm2",
      "vpbroadcastq zmm1, [{constants} + 16]",
      "vpmullq zmm0, zmm0, zmm1",
      "vpsrlq zmm2, zmm0, 31",
      "vpxorq zmm0, zmm0, zmm2",
      "vmovdqu64 [{output}], zmm0",
      values = in(reg) values,
      output = in(reg) output.as_mut_ptr(),
      seed = in(reg) &raw const seed,
      constants = in(reg) CONSTANTS.as_ptr(),
      out("zmm0") _,
      out("zmm1") _,
      out("zmm2") _,
      options(nostack, preserves_flags),
    );
    output.assume_init()
  }
}

/// Returns surviving lane positions, writing all mixed values to output only
/// when at least one rank is below the supplied bound.
#[inline]
#[cfg(target_arch = "x86")]
pub(super) unsafe fn splitmix64x8_below_avx512<const RANK_SHIFT: u32>(
  values: *const u64,
  seed: u64,
  bound: u32,
  output: &mut [u64; 8],
) -> u8 {
  const CONSTANTS: [u64; 3] = [
    0x9e37_79b9_7f4a_7c15,
    0xbf58_476d_1ce4_e5b9,
    0x94d0_49bb_1331_11eb,
  ];
  const {
    assert!(RANK_SHIFT >= 33 && RANK_SHIFT < 64);
  }
  let state = [seed, u64::from(bound)];
  let mask: u32;
  // SAFETY: the caller proves AVX-512F/DQ support and eight readable input
  // values. Every broadcast reads one initialized u64. The final xor only
  // changes bits 0..=32, so ranks beginning at bit 33 can be compared first.
  // A nonzero mask always reaches the full 64-byte output store; a zero mask
  // leaves output unchanged. All vector, mask and GPR changes are declared.
  unsafe {
    core::arch::asm!(
      "vmovdqu64 zmm0, [{values}]",
      "vpbroadcastq zmm1, [{state}]",
      "vpxorq zmm0, zmm0, zmm1",
      "vpbroadcastq zmm1, [{constants}]",
      "vpaddq zmm0, zmm0, zmm1",
      "vpsrlq zmm2, zmm0, 30",
      "vpxorq zmm0, zmm0, zmm2",
      "vpbroadcastq zmm1, [{constants} + 8]",
      "vpmullq zmm0, zmm0, zmm1",
      "vpsrlq zmm2, zmm0, 27",
      "vpxorq zmm0, zmm0, zmm2",
      "vpbroadcastq zmm1, [{constants} + 16]",
      "vpmullq zmm0, zmm0, zmm1",
      "vpsrlq zmm2, zmm0, {rank_shift}",
      "vpbroadcastq zmm1, [{state} + 8]",
      "vpcmpuq k1, zmm2, zmm1, 1",
      "kmovw {mask:e}, k1",
      "test {mask:e}, {mask:e}",
      "jz 2f",
      "vpsrlq zmm2, zmm0, 31",
      "vpxorq zmm0, zmm0, zmm2",
      "vmovdqu64 [{output}], zmm0",
      "2:",
      values = in(reg) values,
      state = in(reg) state.as_ptr(),
      constants = in(reg) CONSTANTS.as_ptr(),
      output = inout(reg) output.as_mut_ptr() => _,
      mask = lateout(reg) mask,
      rank_shift = const RANK_SHIFT,
      out("zmm0") _,
      out("zmm1") _,
      out("zmm2") _,
      out("k1") _,
      options(nostack),
    );
  }
  // vpcmpuq writes exactly eight mask bits for eight u64 lanes.
  #[allow(clippy::cast_possible_truncation)]
  let mask = mask as u8;
  mask
}

/// Updates complete blocks of eight tokens below a fixed rank bound.
/// Returns the consumed prefix length; the caller processes the tail in Rust.
///
/// # Safety
/// AVX-512F/DQ must be available, and `input` must address `input_len` readable `u64` values.
#[cfg(target_arch = "x86_64")]
pub(super) unsafe fn apply_buckets_below_avx512(
  input: *const u64,
  input_len: usize,
  seed: u64,
  bound: u32,
  hash_values: &mut [u32],
) -> usize {
  const CONSTANTS: [u64; 3] = [
    0x9e37_79b9_7f4a_7c15,
    0xbf58_476d_1ce4_e5b9,
    0x94d0_49bb_1331_11eb,
  ];
  let Ok(width) = u32::try_from(hash_values.len()) else {
    return 0;
  };
  let blocks = input_len / 8;
  if !width.is_power_of_two() || blocks == 0 {
    return 0;
  }
  let shift = 32 - width.trailing_zeros();
  let state = [seed, u64::from(bound)];
  let mut scratch = [0u64; 8];
  // SAFETY: input supplies blocks * 8 tokens. The checked width is 2^p,
  // p in 0..=31; zero-extending the low word and shifting by 32-p gives
  // a valid coordinate, including width one. Every scratch read follows
  // its complete 64-byte store, with a lane in 0..8 from the nonzero mask.
  // Updates occur in lane order, so duplicate coordinates preserve minima.
  // All modified registers and flags are declared. Constants remain live
  // across the whole span, with no per-block stack state or broadcasts.
  unsafe {
    core::arch::asm!(
      "vpbroadcastq zmm2, [{state}]",
      "vpbroadcastq zmm3, [{constants}]",
      "vpbroadcastq zmm4, [{constants} + 8]",
      "vpbroadcastq zmm5, [{constants} + 16]",
      "vpbroadcastq zmm6, [{state} + 8]",
      "2:",
      "vmovdqu64 zmm0, [{cursor}]",
      "vpxorq zmm0, zmm0, zmm2",
      "vpaddq zmm0, zmm0, zmm3",
      "vpsrlq zmm1, zmm0, 30",
      "vpxorq zmm0, zmm0, zmm1",
      "vpmullq zmm0, zmm0, zmm4",
      "vpsrlq zmm1, zmm0, 27",
      "vpxorq zmm0, zmm0, zmm1",
      "vpmullq zmm0, zmm0, zmm5",
      // The final xor changes only bits 0..=32, leaving these ranks intact.
      "vpsrlq zmm1, zmm0, 34",
      "vpcmpuq k1, zmm1, zmm6, 1",
      "kortestw k1, k1",
      "jz 3f",
      "vpsrlq zmm1, zmm0, 31",
      "vpxorq zmm0, zmm0, zmm1",
      "vmovdqu64 [{scratch}], zmm0",
      "kmovw {mask:e}, k1",
      "4:",
      "bsf {lane:e}, {mask:e}",
      "mov {mixed}, [{scratch} + {lane} * 8]",
      "mov {bucket:e}, {mixed:e}",
      "shr {bucket}, cl",
      "shr {mixed}, 34",
      "cmp dword ptr [{values} + {bucket} * 4], {mixed:e}",
      "jbe 5f",
      "mov dword ptr [{values} + {bucket} * 4], {mixed:e}",
      "5:",
      "lea {lane:e}, [{mask} - 1]",
      "and {mask:e}, {lane:e}",
      "jnz 4b",
      "3:",
      "add {cursor}, 64",
      "dec {count}",
      "jnz 2b",
      state = in(reg) state.as_ptr(),
      constants = in(reg) CONSTANTS.as_ptr(),
      scratch = inout(reg) scratch.as_mut_ptr() => _,
      values = inout(reg) hash_values.as_mut_ptr() => _,
      cursor = inout(reg) input => _,
      count = inout(reg) blocks => _,
      inout("rcx") u64::from(shift) => _,
      mask = lateout(reg) _,
      lane = lateout(reg) _,
      mixed = lateout(reg) _,
      bucket = lateout(reg) _,
      out("zmm0") _,
      out("zmm1") _,
      out("zmm2") _,
      out("zmm3") _,
      out("zmm4") _,
      out("zmm5") _,
      out("zmm6") _,
      out("k1") _,
      options(nostack),
    );
  }
  blocks * 8
}

#[cfg(target_arch = "x86")]
use core::arch::x86::{
  __m256i, _mm256_loadu_si256, _mm256_min_epu32, _mm256_storeu_si256,
};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
  __m256i, _mm256_loadu_si256, _mm256_min_epu32, _mm256_storeu_si256,
};

pub(super) fn apply_hash_batch_to_values_avx2(
  hash_values: &mut [u32],
  permutations: &[(u64, u64)],
  hash_batch: &[u64],
) {
  if hash_values.is_empty() || hash_batch.is_empty() {
    return;
  }

  let perm_len = hash_values.len().min(permutations.len());
  if perm_len == 0 {
    return;
  }
  let hash_values = &mut hash_values[..perm_len];
  let permutations = &permutations[..perm_len];

  // SAFETY: dispatch only calls this function when AVX2 is available.
  unsafe {
    apply_hash_batch_to_values_avx2_impl(hash_values, permutations, hash_batch);
  }
}

#[target_feature(enable = "avx2")]
unsafe fn apply_hash_batch_to_values_avx2_impl(
  hash_values: &mut [u32],
  permutations: &[(u64, u64)],
  hash_batch: &[u64],
) {
  let mut value_chunks = hash_values.chunks_exact_mut(8);
  let mut perm_chunks = permutations.chunks_exact(8);

  for (values, perms) in value_chunks.by_ref().zip(perm_chunks.by_ref()) {
    // SAFETY: `values` comes from `chunks_exact_mut(8)`, so it has exactly 8 lanes.
    let mut current = unsafe { load_u32x8(values.as_ptr()) };
    for &item_hash in hash_batch {
      let permuted = [
        permute_hash(item_hash, perms[0].0, perms[0].1),
        permute_hash(item_hash, perms[1].0, perms[1].1),
        permute_hash(item_hash, perms[2].0, perms[2].1),
        permute_hash(item_hash, perms[3].0, perms[3].1),
        permute_hash(item_hash, perms[4].0, perms[4].1),
        permute_hash(item_hash, perms[5].0, perms[5].1),
        permute_hash(item_hash, perms[6].0, perms[6].1),
        permute_hash(item_hash, perms[7].0, perms[7].1),
      ];
      // SAFETY: `permuted` is a local `[u32; 8]`, valid for an 8-lane load.
      let permuted_vec = unsafe { load_u32x8(permuted.as_ptr()) };
      current = _mm256_min_epu32(current, permuted_vec);
    }
    // SAFETY: `values` is still a valid mutable 8-lane chunk here.
    unsafe { store_u32x8(values.as_mut_ptr(), current) };
  }

  for (value, &(a, b)) in value_chunks
    .into_remainder()
    .iter_mut()
    .zip(perm_chunks.remainder().iter())
  {
    let mut min_value = *value;
    for &item_hash in hash_batch {
      min_value = min_value.min(permute_hash(item_hash, a, b));
    }
    *value = min_value;
  }
}

#[inline]
#[allow(clippy::cast_ptr_alignment)] // loadu accepts unaligned addresses.
unsafe fn load_u32x8(ptr: *const u32) -> __m256i {
  // SAFETY: caller guarantees `ptr` is valid for 8 contiguous `u32` values.
  unsafe { _mm256_loadu_si256(ptr.cast::<__m256i>()) }
}

#[inline]
#[allow(clippy::cast_ptr_alignment)] // storeu accepts unaligned addresses.
unsafe fn store_u32x8(ptr: *mut u32, value: __m256i) {
  // SAFETY: caller guarantees `ptr` is valid for 8 contiguous mutable `u32` values.
  unsafe { _mm256_storeu_si256(ptr.cast::<__m256i>(), value) };
}

#[cfg(test)]
mod tests {
  use crate::simd::x86::splitmix64x8_avx512;
  use rand_core::{RngCore, SeedableRng};
  use rand_xoshiro::Xoshiro256PlusPlus;

  fn scalar(mut value: u64, seed: u64) -> u64 {
    value = (value ^ seed).wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
  }

  #[test]
  fn avx512_mixer_matches_wrapping_scalar_for_every_lane() {
    if !std::arch::is_x86_feature_detected!("avx512f")
      || !std::arch::is_x86_feature_detected!("avx512dq")
    {
      return;
    }
    let edges = [
      0,
      1,
      u64::MAX,
      u64::MAX - 1,
      u64::from(u32::MAX),
      1 << 32,
      1 << 63,
      0xaaaa_aaaa_5555_5555,
    ];
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x4156_5835_3132);
    for seed in [0, 1, 42, u64::MAX, rng.next_u64()] {
      for input in std::iter::once(edges)
        .chain(edges.map(|value| [value; 8]))
        .chain((0..128).map(|_| std::array::from_fn(|_| rng.next_u64())))
      {
        let expected = input.map(|value| scalar(value, seed));
        // SAFETY: the test checked both required CPU/OS features above.
        let actual = unsafe { splitmix64x8_avx512(input.as_ptr(), seed) };
        assert_eq!(actual, expected);
      }
    }
  }

  #[test]
  #[cfg(target_arch = "x86")]
  fn avx512_rank_mask_matches_scalar_at_exact_boundaries() {
    if !std::arch::is_x86_feature_detected!("avx512f")
      || !std::arch::is_x86_feature_detected!("avx512dq")
    {
      return;
    }
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5241_4e4b);
    let mut output = [u64::MAX; 8];
    for seed in [0, 42, u64::MAX] {
      for batch in 0..128 {
        let input = if batch == 0 {
          [0, 1, u64::MAX, u64::MAX - 1, 1 << 31, 1 << 32, 1 << 63, 42]
        } else {
          std::array::from_fn(|_| rng.next_u64())
        };
        let expected = input.map(|value| scalar(value, seed));
        #[allow(clippy::cast_possible_truncation)]
        let boundary = (expected[batch % 8] >> 34) as u32;
        for bound in [0, 1, 1 << 25, 1 << 30, u32::MAX, boundary, boundary + 1]
        {
          let mask =
            expected
              .iter()
              .enumerate()
              .fold(0u8, |mask, (lane, &value)| {
                if value >> 34 < u64::from(bound) {
                  mask | (1 << lane)
                } else {
                  mask
                }
              });
          let before = output;
          // SAFETY: CPU/OS support and the complete eight-value input were established above.
          let actual = unsafe {
            crate::simd::x86::splitmix64x8_below_avx512::<34>(
              input.as_ptr(),
              seed,
              bound,
              &mut output,
            )
          };
          assert_eq!(actual, mask);
          assert_eq!(output, if mask == 0 { before } else { expected });
        }
      }
    }
  }
  #[test]
  #[cfg(target_arch = "x86_64")]
  fn avx512_bucket_span_matches_scalar_with_bounds_tails_and_collisions() {
    if !std::arch::is_x86_feature_detected!("avx512f")
      || !std::arch::is_x86_feature_detected!("avx512dq")
    {
      return;
    }
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5350_414e);
    let mut inputs = [0; 129];
    for value in &mut inputs {
      *value = rng.next_u64();
    }
    inputs[..8].copy_from_slice(&[
      0,
      1,
      u64::MAX,
      u64::MAX - 1,
      1 << 31,
      1 << 32,
      1 << 63,
      42,
    ]);
    for seed in [0, 42, u64::MAX] {
      // Repeated tokens force colliding coordinates even with wide sketches.
      for input in [&inputs, &[u64::MAX; 129]] {
        let boundary = u32::try_from(scalar(input[0], seed) >> 34).unwrap();
        for len in (0..=17).chain([31, 32, 33, 63, 64, 65, 128, 129]) {
          for width in [0_usize, 1, 2, 3, 4, 8, 128, 129, 512] {
            for bound in
              [0, 1, 1 << 25, 1 << 30, u32::MAX, boundary, boundary + 1]
            {
              let mut actual: Vec<u32> = (0..width + 2)
                .map(|index| match (index + len) % 4 {
                  0 => u32::MAX,
                  1 => 0,
                  2 => 1,
                  _ => rng.next_u32(),
                })
                .collect();
              let mut expected = actual.clone();
              let consumed = if width.is_power_of_two() {
                len / 8 * 8
              } else {
                0
              };
              for &value in &input[..consumed] {
                let mixed = scalar(value, seed);
                let rank = u32::try_from(mixed >> 34).unwrap();
                if rank < bound {
                  // Product mapping is independent of the kernel's bit shift.
                  let product = (mixed & u64::from(u32::MAX))
                    * u64::try_from(width).unwrap();
                  let bucket = usize::try_from(product >> 32).unwrap();
                  expected[bucket + 1] = expected[bucket + 1].min(rank);
                }
              }
              // SAFETY: features were checked, input covers len values and the
              // output sub-slice includes exactly width initialized buckets.
              let processed = unsafe {
                crate::simd::x86::apply_buckets_below_avx512(
                  input.as_ptr(),
                  len,
                  seed,
                  bound,
                  &mut actual[1..][..width],
                )
              };
              assert_eq!(processed, consumed);
              assert_eq!(
                actual, expected,
                "len={len}, width={width}, bound={bound}"
              );
            }
          }
        }
      }
    }
  }
}
