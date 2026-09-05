use crate::simd::dispatch::{split_u64_words, PermutationSoA};
use crate::utils::permute_hash;
use core::arch::aarch64::{
  uint32x4_t, uint64x2_t, vaddq_u64, vcombine_u32, vdupq_n_u32, vget_high_u32,
  vget_low_u32, vld1q_u32, vminq_u32, vmlaq_u32, vmovl_u32, vmovn_u64,
  vmull_u32, vshrq_n_u64, vsliq_n_u64, vst1q_u32,
};

#[inline]
fn combine_u32_words(low: u32, high: u32) -> u64 {
  u64::from(low) | (u64::from(high) << 32)
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn permute_four_lanes(
  a_lo: uint32x4_t,
  a_hi: uint32x4_t,
  offset01: uint64x2_t,
  offset23: uint64x2_t,
  h_lo: uint32x4_t,
  h_hi: uint32x4_t,
) -> uint32x4_t {
  // In arithmetic modulo 2^64, a*h+b = a_lo*h_lo+b +
  // ((a_hi*h_lo + a_lo*h_hi) << 32). Adding full b includes the carry.
  let low =
    vaddq_u64(vmull_u32(vget_low_u32(a_lo), vget_low_u32(h_lo)), offset01);
  let high = vaddq_u64(
    vmull_u32(vget_high_u32(a_lo), vget_high_u32(h_lo)),
    offset23,
  );
  let upper = vcombine_u32(
    vmovn_u64(vshrq_n_u64(low, 32)),
    vmovn_u64(vshrq_n_u64(high, 32)),
  );
  vmlaq_u32(vmlaq_u32(upper, a_hi, h_lo), a_lo, h_hi)
}

pub(super) fn apply_hash_batch_to_values_neon(
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
unsafe fn apply_hash_batch_to_values_neon_impl(
  hash_values: &mut [u32],
  permutations_soa: &PermutationSoA,
  hash_batch: &[u64],
) {
  let perm_len = permutations_soa.len().min(hash_values.len());
  let a_hi = permutations_soa.a_hi();
  let a_lo = permutations_soa.a_lo();
  let b_hi = permutations_soa.b_hi();
  let b_lo = permutations_soa.b_lo();
  let mut index = 0usize;
  while index + 8 <= perm_len {
    // SAFETY: this chunk is within perm_len, bounded by all slice lengths.
    let a_lo_0 = unsafe { vld1q_u32(a_lo.as_ptr().add(index)) };
    let a_hi_0 = unsafe { vld1q_u32(a_hi.as_ptr().add(index)) };
    let b_lo_0 = unsafe { vld1q_u32(b_lo.as_ptr().add(index)) };
    let b_hi_0 = unsafe { vld1q_u32(b_hi.as_ptr().add(index)) };
    let offset01_0 = vsliq_n_u64(
      vmovl_u32(vget_low_u32(b_lo_0)),
      vmovl_u32(vget_low_u32(b_hi_0)),
      32,
    );
    let offset23_0 = vsliq_n_u64(
      vmovl_u32(vget_high_u32(b_lo_0)),
      vmovl_u32(vget_high_u32(b_hi_0)),
      32,
    );
    let mut current_0 = unsafe { vld1q_u32(hash_values.as_ptr().add(index)) };
    // SAFETY: this chunk is within perm_len, bounded by all slice lengths.
    let a_lo_1 = unsafe { vld1q_u32(a_lo.as_ptr().add(index + 4)) };
    let a_hi_1 = unsafe { vld1q_u32(a_hi.as_ptr().add(index + 4)) };
    let b_lo_1 = unsafe { vld1q_u32(b_lo.as_ptr().add(index + 4)) };
    let b_hi_1 = unsafe { vld1q_u32(b_hi.as_ptr().add(index + 4)) };
    let offset01_1 = vsliq_n_u64(
      vmovl_u32(vget_low_u32(b_lo_1)),
      vmovl_u32(vget_low_u32(b_hi_1)),
      32,
    );
    let offset23_1 = vsliq_n_u64(
      vmovl_u32(vget_high_u32(b_lo_1)),
      vmovl_u32(vget_high_u32(b_hi_1)),
      32,
    );
    let mut current_1 =
      unsafe { vld1q_u32(hash_values.as_ptr().add(index + 4)) };
    for &item_hash in hash_batch {
      let (h_lo, h_hi) = split_u64_words(item_hash);
      let h_lo = vdupq_n_u32(h_lo);
      let h_hi = vdupq_n_u32(h_hi);
      // SAFETY: NEON is enabled for this function.
      let permuted_0 = unsafe {
        permute_four_lanes(a_lo_0, a_hi_0, offset01_0, offset23_0, h_lo, h_hi)
      };
      current_0 = vminq_u32(current_0, permuted_0);
      // SAFETY: NEON is enabled for this function.
      let permuted_1 = unsafe {
        permute_four_lanes(a_lo_1, a_hi_1, offset01_1, offset23_1, h_lo, h_hi)
      };
      current_1 = vminq_u32(current_1, permuted_1);
    }
    // SAFETY: same bounds as the corresponding loads.
    unsafe {
      vst1q_u32(hash_values.as_mut_ptr().add(index), current_0);
      vst1q_u32(hash_values.as_mut_ptr().add(index + 4), current_1);
    }
    index += 8;
  }
  while index + 4 <= perm_len {
    // SAFETY: this chunk is within perm_len, bounded by all slice lengths.
    let a_lo_0 = unsafe { vld1q_u32(a_lo.as_ptr().add(index)) };
    let a_hi_0 = unsafe { vld1q_u32(a_hi.as_ptr().add(index)) };
    let b_lo_0 = unsafe { vld1q_u32(b_lo.as_ptr().add(index)) };
    let b_hi_0 = unsafe { vld1q_u32(b_hi.as_ptr().add(index)) };
    let offset01_0 = vsliq_n_u64(
      vmovl_u32(vget_low_u32(b_lo_0)),
      vmovl_u32(vget_low_u32(b_hi_0)),
      32,
    );
    let offset23_0 = vsliq_n_u64(
      vmovl_u32(vget_high_u32(b_lo_0)),
      vmovl_u32(vget_high_u32(b_hi_0)),
      32,
    );
    let mut current_0 = unsafe { vld1q_u32(hash_values.as_ptr().add(index)) };
    for &item_hash in hash_batch {
      let (h_lo, h_hi) = split_u64_words(item_hash);
      let h_lo = vdupq_n_u32(h_lo);
      let h_hi = vdupq_n_u32(h_hi);
      // SAFETY: NEON is enabled for this function.
      let permuted_0 = unsafe {
        permute_four_lanes(a_lo_0, a_hi_0, offset01_0, offset23_0, h_lo, h_hi)
      };
      current_0 = vminq_u32(current_0, permuted_0);
    }
    // SAFETY: same bounds as the corresponding loads.
    unsafe {
      vst1q_u32(hash_values.as_mut_ptr().add(index), current_0);
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
