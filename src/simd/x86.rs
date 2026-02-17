use crate::utils::permute_hash;

#[cfg(target_arch = "x86")]
use core::arch::x86::{
  __m256i, _mm256_loadu_si256, _mm256_min_epu32, _mm256_storeu_si256,
};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
  __m256i, _mm256_loadu_si256, _mm256_min_epu32, _mm256_storeu_si256,
};

pub(crate) fn apply_hash_batch_to_values_avx2(
  hash_values: &mut [u32],
  permutations: &[(u64, u64)],
  hash_batch: &[u64],
) {
  if hash_values.is_empty() || hash_batch.is_empty() {
    return;
  }
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
      let permuted_vec = unsafe { load_u32x8(permuted.as_ptr()) };
      current = _mm256_min_epu32(current, permuted_vec);
    }
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
unsafe fn load_u32x8(ptr: *const u32) -> __m256i {
  unsafe { _mm256_loadu_si256(ptr.cast::<__m256i>()) }
}

#[inline]
unsafe fn store_u32x8(ptr: *mut u32, value: __m256i) {
  unsafe { _mm256_storeu_si256(ptr.cast::<__m256i>(), value) };
}
