use crate::utils::permute_hash;

// The direct loads below consume the tuple's actual in-memory field order.
const _: () = assert!(std::mem::size_of::<(u64, u64)>() == 16);
const _: () = assert!(std::mem::offset_of!((u64, u64), 0) == 0);
const _: () = assert!(std::mem::offset_of!((u64, u64), 1) == 8);

const COEFFICIENT_INDICES: [[u64; 8]; 2] =
  [[0, 2, 4, 6, 8, 10, 12, 14], [1, 3, 5, 7, 9, 11, 13, 15]];

/// Applies eight affine permutations at a time, retaining their coefficients
/// and minima in registers across the token batch.
///
/// # Safety
/// The caller must establish AVX-512F and AVX-512DQ CPU/OS support.
pub(super) unsafe fn apply_hash_batch_to_values_avx512(
  hash_values: &mut [u32],
  permutations: &[(u64, u64)],
  hash_batch: &[u64],
) {
  let perm_len = hash_values.len().min(permutations.len());
  if perm_len == 0 || hash_batch.is_empty() {
    return;
  }

  let mut value_chunks = hash_values[..perm_len].chunks_exact_mut(8);
  let mut perm_chunks = permutations[..perm_len].chunks_exact(8);
  for (values, perms) in value_chunks.by_ref().zip(perm_chunks.by_ref()) {
    // SAFETY: the caller establishes AVX-512F/DQ support. The
    // coefficient chunk supplies 128 readable bytes with the tuple layout
    // checked above; indices supply 128 readable bytes. The values chunk
    // supplies 32 readable/writable bytes. The nonempty token slice supplies
    // exactly count reads of 8 bytes. All modified registers are declared.
    // Inline assembly keeps this kernel compatible with Rust 1.83, before
    // AVX-512 intrinsics and target_feature were stabilized.
    unsafe {
      core::arch::asm!(
        "vmovdqu64 zmm2, [{perms}]",
        "vmovdqu64 zmm3, [{perms} + 64]",
        "vmovdqu64 zmm0, [{indices}]",
        "vmovdqu64 zmm1, [{indices} + 64]",
        "vpermi2q zmm0, zmm2, zmm3",
        "vpermi2q zmm1, zmm2, zmm3",
        "vpmovzxdq zmm2, ymmword ptr [{values}]",
        "2:",
        "vpbroadcastq zmm3, [{cursor}]",
        "vpmullq zmm3, zmm3, zmm0",
        "vpaddq zmm3, zmm3, zmm1",
        "vpsrlq zmm3, zmm3, 32",
        "vpminuq zmm2, zmm2, zmm3",
        "add {cursor}, 8",
        "dec {count}",
        "jnz 2b",
        "vpmovqd ymmword ptr [{values}], zmm2",
        perms = in(reg) perms.as_ptr(),
        indices = in(reg) COEFFICIENT_INDICES.as_ptr(),
        values = in(reg) values.as_mut_ptr(),
        cursor = inout(reg) hash_batch.as_ptr() => _,
        count = inout(reg) hash_batch.len() => _,
        out("zmm0") _,
        out("zmm1") _,
        out("zmm2") _,
        out("zmm3") _,
        options(nostack),
      );
    }
  }

  for (value, &(a, b)) in value_chunks
    .into_remainder()
    .iter_mut()
    .zip(perm_chunks.remainder())
  {
    for &hash in hash_batch {
      *value = (*value).min(permute_hash(hash, a, b));
    }
  }
}

#[cfg(test)]
mod tests {
  use crate::simd::x86_avx512::apply_hash_batch_to_values_avx512;
  use rand_core::{RngCore, SeedableRng};
  use rand_xoshiro::Xoshiro256PlusPlus;

  fn supported() -> bool {
    std::arch::is_x86_feature_detected!("avx512f")
      && std::arch::is_x86_feature_detected!("avx512dq")
  }

  fn reference(hash: u64, a: u64, b: u64) -> u32 {
    let product = u128::from(hash) * u128::from(a) + u128::from(b);
    u32::try_from((product >> 32) & u128::from(u32::MAX)).unwrap()
  }

  #[test]
  fn avx512_affine_preserves_full_width_carries_and_overflow() {
    if !supported() {
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
    for &hash in &edges {
      for &a in &edges {
        let permutations = edges.map(|b| (a, b));
        let mut values = [u32::MAX; 8];
        let expected = edges.map(|b| reference(hash, a, b));
        // SAFETY: both required CPU/OS features were detected above.
        unsafe {
          apply_hash_batch_to_values_avx512(
            &mut values,
            &permutations,
            &[hash],
          );
        }
        assert_eq!(values, expected, "hash={hash}, a={a}");
      }
    }
  }

  #[test]
  fn avx512_affine_matches_scalar_for_tails_empty_and_existing_minima() {
    if !supported() {
      return;
    }
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x4156_5835_3132);
    for width in (0..=33).chain([63, 64, 65, 127, 128, 129, 511, 512]) {
      let permutations: Vec<_> = (0..width)
        .map(|_| (rng.next_u64(), rng.next_u64()))
        .collect();
      for hash_len in [0, 1, 2, 7, 8, 9, 31, 32, 129] {
        let hashes: Vec<_> = (0..hash_len).map(|_| rng.next_u64()).collect();
        for populated in [false, true] {
          let mut actual: Vec<_> = (0..width + 3)
            .map(|_| if populated { rng.next_u32() } else { u32::MAX })
            .collect();
          let mut expected = actual.clone();
          // Offset the mutable view and leave extra lanes without coefficients.
          // The kernel must preserve both those lanes and the outer sentinels.
          for (value, &(a, b)) in expected[1..].iter_mut().zip(&permutations) {
            for &hash in &hashes {
              *value = (*value).min(reference(hash, a, b));
            }
          }
          // SAFETY: both required CPU/OS features were detected above.
          unsafe {
            apply_hash_batch_to_values_avx512(
              &mut actual[1..],
              &permutations,
              &hashes,
            );
          }
          assert_eq!(actual, expected, "width={width}, hashes={hash_len}");
        }
      }
    }
  }
}
