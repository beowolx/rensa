use crate::simd::dispatch::{split_u64_words, PermutationSoA};
use crate::utils::permute_hash;

#[cfg(target_arch = "x86")]
use core::arch::x86::{
  __m256i, _mm256_add_epi64, _mm256_and_si256, _mm256_loadu_si256,
  _mm256_min_epu32, _mm256_mul_epu32, _mm256_or_si256, _mm256_set1_epi32,
  _mm256_set1_epi64x, _mm256_slli_epi64, _mm256_srli_epi64,
  _mm256_storeu_si256,
};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
  __m256i, _mm256_add_epi64, _mm256_and_si256, _mm256_loadu_si256,
  _mm256_min_epu32, _mm256_mul_epu32, _mm256_or_si256, _mm256_set1_epi32,
  _mm256_set1_epi64x, _mm256_slli_epi64, _mm256_srli_epi64,
  _mm256_storeu_si256,
};

#[inline]
fn combine_u32_words(low: u32, high: u32) -> u64 {
  u64::from(low) | (u64::from(high) << 32)
}

#[inline]
unsafe fn permute_four_lanes(
  a_lo: __m256i,
  a_hi: __m256i,
  b_lo: __m256i,
  b_hi: __m256i,
  h_lo: __m256i,
  h_hi: __m256i,
  mask32: __m256i,
) -> __m256i {
  let ll = _mm256_mul_epu32(a_lo, h_lo);
  let mid_1 = _mm256_mul_epu32(a_hi, h_lo);
  let mid_2 = _mm256_mul_epu32(a_lo, h_hi);
  let mid = _mm256_add_epi64(mid_1, mid_2);

  let ll_hi = _mm256_srli_epi64(ll, 32);
  let mid_lo = _mm256_and_si256(mid, mask32);

  let hi = _mm256_add_epi64(_mm256_add_epi64(ll_hi, mid_lo), b_hi);
  let low = _mm256_add_epi64(_mm256_and_si256(ll, mask32), b_lo);
  let carry = _mm256_srli_epi64(low, 32);
  _mm256_add_epi64(hi, carry)
}

pub(super) fn apply_hash_batch_to_values_avx2(
  hash_values: &mut [u32],
  permutations_soa: &PermutationSoA,
  hash_batch: &[u64],
) {
  if hash_values.is_empty()
    || hash_batch.is_empty()
    || permutations_soa.is_empty()
  {
    return;
  }

  // SAFETY: dispatch only calls this function when AVX2 is available.
  unsafe {
    apply_hash_batch_to_values_avx2_impl(
      hash_values,
      permutations_soa,
      hash_batch,
    );
  }
}

#[target_feature(enable = "avx2")]
unsafe fn apply_hash_batch_to_values_avx2_impl(
  hash_values: &mut [u32],
  permutations_soa: &PermutationSoA,
  hash_batch: &[u64],
) {
  let perm_len = permutations_soa.len().min(hash_values.len());
  let a_hi = permutations_soa.a_hi();
  let a_lo = permutations_soa.a_lo();
  let b_hi = permutations_soa.b_hi();
  let b_lo = permutations_soa.b_lo();
  let mask32 = _mm256_set1_epi64x(0xffff_ffff);

  let mut index = 0usize;
  while index + 8 <= perm_len {
    // SAFETY: `index + 7 < perm_len`, and all SoA slices plus `hash_values`
    // have length >= `perm_len`, so these loads are in-bounds.
    let a_lo_vec = unsafe { load_u32x8(a_lo.as_ptr().add(index)) };
    let a_hi_vec = unsafe { load_u32x8(a_hi.as_ptr().add(index)) };
    let b_lo_vec = unsafe { load_u32x8(b_lo.as_ptr().add(index)) };
    let b_hi_vec = unsafe { load_u32x8(b_hi.as_ptr().add(index)) };
    let mut current = unsafe { load_u32x8(hash_values.as_ptr().add(index)) };

    for &item_hash in hash_batch {
      let (h_lo_word, h_hi_word) = split_u64_words(item_hash);
      let h_lo_vec = _mm256_set1_epi32(h_lo_word as i32);
      let h_hi_vec = _mm256_set1_epi32(h_hi_word as i32);

      // `_mm256_mul_epu32` only consumes the low 32 bits of each 64-bit lane.
      // Shifting by 32 lets us reuse the same math for the odd u32 lanes.
      let a_lo_odd = _mm256_srli_epi64(a_lo_vec, 32);
      let a_hi_odd = _mm256_srli_epi64(a_hi_vec, 32);
      let b_lo_odd = _mm256_and_si256(_mm256_srli_epi64(b_lo_vec, 32), mask32);
      let b_hi_odd = _mm256_and_si256(_mm256_srli_epi64(b_hi_vec, 32), mask32);
      let h_lo_odd = _mm256_srli_epi64(h_lo_vec, 32);
      let h_hi_odd = _mm256_srli_epi64(h_hi_vec, 32);

      // SAFETY: AVX2 is enabled for this function by `#[target_feature(enable = "avx2")]`.
      let even_perm = unsafe {
        permute_four_lanes(
          a_lo_vec, a_hi_vec, b_lo_vec, b_hi_vec, h_lo_vec, h_hi_vec, mask32,
        )
      };
      // SAFETY: same feature invariant as above.
      let odd_perm = unsafe {
        permute_four_lanes(
          a_lo_odd, a_hi_odd, b_lo_odd, b_hi_odd, h_lo_odd, h_hi_odd, mask32,
        )
      };
      let permuted = _mm256_or_si256(
        _mm256_and_si256(even_perm, mask32),
        _mm256_slli_epi64(_mm256_and_si256(odd_perm, mask32), 32),
      );
      current = _mm256_min_epu32(current, permuted);
    }

    // SAFETY: same bounds argument as the corresponding loads above.
    unsafe {
      store_u32x8(hash_values.as_mut_ptr().add(index), current);
    }
    index += 8;
  }

  for lane_index in index..perm_len {
    let a = combine_u32_words(a_lo[lane_index], a_hi[lane_index]);
    let b = combine_u32_words(b_lo[lane_index], b_hi[lane_index]);
    let mut min_value = hash_values[lane_index];
    for &item_hash in hash_batch {
      min_value = min_value.min(permute_hash(item_hash, a, b));
    }
    hash_values[lane_index] = min_value;
  }
}

#[inline]
unsafe fn load_u32x8(ptr: *const u32) -> __m256i {
  // SAFETY: caller guarantees `ptr` is valid for 8 contiguous `u32` values.
  unsafe { _mm256_loadu_si256(ptr.cast::<__m256i>()) }
}

#[inline]
unsafe fn store_u32x8(ptr: *mut u32, value: __m256i) {
  // SAFETY: caller guarantees `ptr` is valid for 8 contiguous mutable `u32` values.
  unsafe { _mm256_storeu_si256(ptr.cast::<__m256i>(), value) };
}

#[cfg(test)]
mod tests {
  use crate::simd::dispatch::PermutationSoA;
  use crate::simd::x86::apply_hash_batch_to_values_avx2;
  use crate::utils::permute_hash;
  use rand_core::{RngCore, SeedableRng};
  use rand_xoshiro::Xoshiro256PlusPlus;

  fn scalar_reference(
    values: &mut [u32],
    permutations: &[(u64, u64)],
    hash_batch: &[u64],
  ) {
    for (value, &(a, b)) in values.iter_mut().zip(permutations.iter()) {
      let mut min_value = *value;
      for &hash in hash_batch {
        min_value = min_value.min(permute_hash(hash, a, b));
      }
      *value = min_value;
    }
  }

  #[test]
  fn avx2_kernel_matches_scalar_reference() {
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
      return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if !std::arch::is_x86_feature_detected!("avx2") {
      return;
    }

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(11);
    for _ in 0..128 {
      let num_perm = 1 + (rng.next_u64() % 96) as usize;
      let hashes_len = 1 + (rng.next_u64() % 128) as usize;
      let mut permutations = Vec::with_capacity(num_perm);
      for _ in 0..num_perm {
        let a = rng.next_u64() | 1;
        let b = rng.next_u64();
        permutations.push((a, b));
      }
      let permutations_soa = PermutationSoA::from_permutations(&permutations);

      let mut hash_batch = Vec::with_capacity(hashes_len);
      for _ in 0..hashes_len {
        hash_batch.push(rng.next_u64());
      }

      let mut avx_values = vec![u32::MAX; num_perm];
      let mut scalar_values = vec![u32::MAX; num_perm];
      apply_hash_batch_to_values_avx2(
        &mut avx_values,
        &permutations_soa,
        &hash_batch,
      );
      scalar_reference(&mut scalar_values, &permutations, &hash_batch);
      assert_eq!(avx_values, scalar_values);
    }
  }
}
