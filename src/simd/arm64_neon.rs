use crate::simd::dispatch::PermutationSoA;
use crate::utils::permute_hash;
use core::arch::aarch64::{
  uint32x2_t, uint64x2_t, vaddq_u64, vandq_u64, vcombine_u32, vdup_n_u32,
  vdupq_n_u64, vget_high_u32, vget_low_u32, vld1q_u32, vminq_u32, vmovl_u32,
  vmovn_u64, vmull_u32, vshrq_n_u64, vst1q_u32,
};

#[inline]
const fn split_u64_words(value: u64) -> (u32, u32) {
  let bytes = value.to_le_bytes();
  let low = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
  let high = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
  (low, high)
}

#[inline]
fn combine_u32_words(low: u32, high: u32) -> u64 {
  u64::from(low) | (u64::from(high) << 32)
}

#[inline]
unsafe fn permute_two_lanes(
  a_lo: uint32x2_t,
  a_hi: uint32x2_t,
  b_lo64: uint64x2_t,
  b_hi64: uint64x2_t,
  h_lo_vec: uint32x2_t,
  h_hi_vec: uint32x2_t,
  mask32: uint64x2_t,
) -> uint32x2_t {
  let ll = unsafe { vmull_u32(a_lo, h_lo_vec) };
  let mid_1 = unsafe { vmull_u32(a_hi, h_lo_vec) };
  let mid_2 = unsafe { vmull_u32(a_lo, h_hi_vec) };
  let mid = unsafe { vaddq_u64(mid_1, mid_2) };

  let ll_hi = unsafe { vshrq_n_u64(ll, 32) };
  let mid_lo = unsafe { vandq_u64(mid, mask32) };

  let hi = unsafe { vaddq_u64(vaddq_u64(ll_hi, mid_lo), b_hi64) };
  let low = unsafe { vaddq_u64(vandq_u64(ll, mask32), b_lo64) };
  let carry = unsafe { vshrq_n_u64(low, 32) };
  let hi_with_carry = unsafe { vaddq_u64(hi, carry) };
  unsafe { vmovn_u64(hi_with_carry) }
}

pub fn apply_hash_batch_to_values_neon(
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

  // SAFETY: dispatch guarantees NEON support before calling this function.
  unsafe {
    apply_hash_batch_to_values_neon_impl(
      hash_values,
      permutations_soa,
      hash_batch,
    );
  }
}

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_lines)]
unsafe fn apply_hash_batch_to_values_neon_impl(
  hash_values: &mut [u32],
  permutations_soa: &PermutationSoA,
  hash_batch: &[u64],
) {
  let perm_len = permutations_soa.len().min(hash_values.len());
  let a_hi = &permutations_soa.a_hi;
  let a_lo = &permutations_soa.a_lo;
  let b_hi = &permutations_soa.b_hi;
  let b_lo = &permutations_soa.b_lo;
  let mask32 = vdupq_n_u64(0xffff_ffff);

  let mut index = 0usize;
  while index + 8 <= perm_len {
    let a_lo_0 = unsafe { vld1q_u32(a_lo.as_ptr().add(index)) };
    let a_hi_0 = unsafe { vld1q_u32(a_hi.as_ptr().add(index)) };
    let b_lo_0 = unsafe { vld1q_u32(b_lo.as_ptr().add(index)) };
    let b_hi_0 = unsafe { vld1q_u32(b_hi.as_ptr().add(index)) };

    let a_lo_1 = unsafe { vld1q_u32(a_lo.as_ptr().add(index + 4)) };
    let a_hi_1 = unsafe { vld1q_u32(a_hi.as_ptr().add(index + 4)) };
    let b_lo_1 = unsafe { vld1q_u32(b_lo.as_ptr().add(index + 4)) };
    let b_hi_1 = unsafe { vld1q_u32(b_hi.as_ptr().add(index + 4)) };
    let a_lo_0_lo = vget_low_u32(a_lo_0);
    let a_lo_0_hi = vget_high_u32(a_lo_0);
    let a_hi_0_lo = vget_low_u32(a_hi_0);
    let a_hi_0_hi = vget_high_u32(a_hi_0);
    let b_lo_0_lo64 = vmovl_u32(vget_low_u32(b_lo_0));
    let b_lo_0_hi64 = vmovl_u32(vget_high_u32(b_lo_0));
    let b_hi_0_lo64 = vmovl_u32(vget_low_u32(b_hi_0));
    let b_hi_0_hi64 = vmovl_u32(vget_high_u32(b_hi_0));
    let a_lo_1_lo = vget_low_u32(a_lo_1);
    let a_lo_1_hi = vget_high_u32(a_lo_1);
    let a_hi_1_lo = vget_low_u32(a_hi_1);
    let a_hi_1_hi = vget_high_u32(a_hi_1);
    let b_lo_1_lo64 = vmovl_u32(vget_low_u32(b_lo_1));
    let b_lo_1_hi64 = vmovl_u32(vget_high_u32(b_lo_1));
    let b_hi_1_lo64 = vmovl_u32(vget_low_u32(b_hi_1));
    let b_hi_1_hi64 = vmovl_u32(vget_high_u32(b_hi_1));

    let mut current_0 = unsafe { vld1q_u32(hash_values.as_ptr().add(index)) };
    let mut current_1 =
      unsafe { vld1q_u32(hash_values.as_ptr().add(index + 4)) };

    for &item_hash in hash_batch {
      let (h_lo, h_hi) = split_u64_words(item_hash);
      let h_lo_vec = vdup_n_u32(h_lo);
      let h_hi_vec = vdup_n_u32(h_hi);

      let low_perm_0 = unsafe {
        permute_two_lanes(
          a_lo_0_lo,
          a_hi_0_lo,
          b_lo_0_lo64,
          b_hi_0_lo64,
          h_lo_vec,
          h_hi_vec,
          mask32,
        )
      };
      let high_perm_0 = unsafe {
        permute_two_lanes(
          a_lo_0_hi,
          a_hi_0_hi,
          b_lo_0_hi64,
          b_hi_0_hi64,
          h_lo_vec,
          h_hi_vec,
          mask32,
        )
      };
      let permuted_0 = vcombine_u32(low_perm_0, high_perm_0);
      let low_perm_1 = unsafe {
        permute_two_lanes(
          a_lo_1_lo,
          a_hi_1_lo,
          b_lo_1_lo64,
          b_hi_1_lo64,
          h_lo_vec,
          h_hi_vec,
          mask32,
        )
      };
      let high_perm_1 = unsafe {
        permute_two_lanes(
          a_lo_1_hi,
          a_hi_1_hi,
          b_lo_1_hi64,
          b_hi_1_hi64,
          h_lo_vec,
          h_hi_vec,
          mask32,
        )
      };
      let permuted_1 = vcombine_u32(low_perm_1, high_perm_1);
      current_0 = vminq_u32(current_0, permuted_0);
      current_1 = vminq_u32(current_1, permuted_1);
    }

    unsafe {
      vst1q_u32(hash_values.as_mut_ptr().add(index), current_0);
      vst1q_u32(hash_values.as_mut_ptr().add(index + 4), current_1);
    }
    index += 8;
  }

  while index + 4 <= perm_len {
    let a_lo_chunk = unsafe { vld1q_u32(a_lo.as_ptr().add(index)) };
    let a_hi_chunk = unsafe { vld1q_u32(a_hi.as_ptr().add(index)) };
    let b_lo_chunk = unsafe { vld1q_u32(b_lo.as_ptr().add(index)) };
    let b_hi_chunk = unsafe { vld1q_u32(b_hi.as_ptr().add(index)) };
    let a_lo_chunk_lo = vget_low_u32(a_lo_chunk);
    let a_lo_chunk_hi = vget_high_u32(a_lo_chunk);
    let a_hi_chunk_lo = vget_low_u32(a_hi_chunk);
    let a_hi_chunk_hi = vget_high_u32(a_hi_chunk);
    let b_lo_chunk_lo64 = vmovl_u32(vget_low_u32(b_lo_chunk));
    let b_lo_chunk_hi64 = vmovl_u32(vget_high_u32(b_lo_chunk));
    let b_hi_chunk_lo64 = vmovl_u32(vget_low_u32(b_hi_chunk));
    let b_hi_chunk_hi64 = vmovl_u32(vget_high_u32(b_hi_chunk));
    let mut current = unsafe { vld1q_u32(hash_values.as_ptr().add(index)) };

    for &item_hash in hash_batch {
      let (h_lo, h_hi) = split_u64_words(item_hash);
      let h_lo_vec = vdup_n_u32(h_lo);
      let h_hi_vec = vdup_n_u32(h_hi);
      let low_perm = unsafe {
        permute_two_lanes(
          a_lo_chunk_lo,
          a_hi_chunk_lo,
          b_lo_chunk_lo64,
          b_hi_chunk_lo64,
          h_lo_vec,
          h_hi_vec,
          mask32,
        )
      };
      let high_perm = unsafe {
        permute_two_lanes(
          a_lo_chunk_hi,
          a_hi_chunk_hi,
          b_lo_chunk_hi64,
          b_hi_chunk_hi64,
          h_lo_vec,
          h_hi_vec,
          mask32,
        )
      };
      let permuted = vcombine_u32(low_perm, high_perm);
      current = vminq_u32(current, permuted);
    }
    unsafe {
      vst1q_u32(hash_values.as_mut_ptr().add(index), current);
    }
    index += 4;
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

#[cfg(test)]
mod tests {
  use crate::simd::arm64_neon::apply_hash_batch_to_values_neon;
  use crate::simd::dispatch::PermutationSoA;
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
  fn neon_kernel_matches_scalar_reference() {
    #[cfg(not(target_arch = "aarch64"))]
    {
      return;
    }

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
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

      let mut neon_values = vec![u32::MAX; num_perm];
      let mut scalar_values = vec![u32::MAX; num_perm];
      apply_hash_batch_to_values_neon(
        &mut neon_values,
        &permutations_soa,
        &hash_batch,
      );
      scalar_reference(&mut scalar_values, &permutations, &hash_batch);
      assert_eq!(neon_values, scalar_values);
    }
  }
}
