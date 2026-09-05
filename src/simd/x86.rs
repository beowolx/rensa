use crate::utils::permute_hash;

#[cfg(target_arch = "x86")]
use core::arch::x86::{
  __m256i, _mm256_add_epi32, _mm256_add_epi64, _mm256_blend_epi32,
  _mm256_loadu_si256, _mm256_min_epu32, _mm256_mul_epu32, _mm256_mullo_epi32,
  _mm256_set1_epi32, _mm256_srli_epi64, _mm256_storeu_si256,
};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
  __m256i, _mm256_add_epi32, _mm256_add_epi64, _mm256_blend_epi32,
  _mm256_loadu_si256, _mm256_min_epu32, _mm256_mul_epu32, _mm256_mullo_epi32,
  _mm256_set1_epi32, _mm256_srli_epi64, _mm256_storeu_si256,
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

#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
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
    // Short batches do not amortize the vector coefficient setup.
    if hash_batch.len() < 16 {
      for &item_hash in hash_batch {
        let permuted: [u32; 8] = std::array::from_fn(|lane| {
          permute_hash(item_hash, perms[lane].0, perms[lane].1)
        });
        // SAFETY: the local array contains eight readable lanes.
        let permuted = unsafe { load_u32x8(permuted.as_ptr()) };
        current = _mm256_min_epu32(current, permuted);
      }
      // SAFETY: values contains exactly eight writable lanes.
      unsafe { store_u32x8(values.as_mut_ptr(), current) };
      continue;
    }
    // Split coefficients once per chunk, outside the hash loop. Keep the
    // existing AoS representation so construction and scalar dispatch are unchanged.
    let a_lo: [u32; 8] = std::array::from_fn(|lane| perms[lane].0 as u32);
    let a_hi: [u32; 8] =
      std::array::from_fn(|lane| (perms[lane].0 >> 32) as u32);
    let b_even: [u64; 4] = std::array::from_fn(|lane| perms[lane * 2].1);
    let b_odd: [u64; 4] = std::array::from_fn(|lane| perms[lane * 2 + 1].1);
    // SAFETY: each local array contains exactly 32 readable bytes.
    let a_lo = unsafe { load_u32x8(a_lo.as_ptr()) };
    let a_hi = unsafe { load_u32x8(a_hi.as_ptr()) };
    let b_even = unsafe { load_u32x8(b_even.as_ptr().cast()) };
    let b_odd = unsafe { load_u32x8(b_odd.as_ptr().cast()) };
    let a_lo_odd = _mm256_srli_epi64(a_lo, 32);
    for &item_hash in hash_batch {
      let h_lo = _mm256_set1_epi32(item_hash as i32);
      let h_hi = _mm256_set1_epi32((item_hash >> 32) as i32);
      // high32(a*h+b) = high32(a_lo*h_lo+b) + a_hi*h_lo + a_lo*h_hi
      // modulo 2^32. Full-width b additions include the low-word carry.
      let even = _mm256_add_epi64(_mm256_mul_epu32(a_lo, h_lo), b_even);
      let odd = _mm256_add_epi64(_mm256_mul_epu32(a_lo_odd, h_lo), b_odd);
      let upper = _mm256_blend_epi32(_mm256_srli_epi64(even, 32), odd, 0xaa);
      let cross = _mm256_add_epi32(
        _mm256_mullo_epi32(a_hi, h_lo),
        _mm256_mullo_epi32(a_lo, h_hi),
      );
      let permuted_vec = _mm256_add_epi32(upper, cross);
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
