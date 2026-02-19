use crate::lsh::{RMinHashLSH, FX_FINISH_ROTATE};
use crate::utils::calculate_band_hash;

#[test]
fn folded_band_hash_matches_direct_hashing() {
  // Deterministic pseudo-random generator (splitmix64-ish).
  fn next_u32(state: &mut u64) -> u32 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    let mixed = z ^ (z >> 31);
    let bytes = mixed.to_le_bytes();
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
  }

  fn u64_to_usize(value: u64) -> usize {
    #[cfg(target_pointer_width = "64")]
    {
      usize::from_ne_bytes(value.to_ne_bytes())
    }
    #[cfg(target_pointer_width = "32")]
    {
      let bytes = value.to_ne_bytes();
      let low = u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
      usize::try_from(low).unwrap_or(usize::MAX)
    }
  }

  for band_size in [4_usize, 8, 12, 16, 20, 28, 32, 64] {
    let steps = RMinHashLSH::fx_poly_steps(band_size);
    let k_pow = RMinHashLSH::fx_poly_k_pow(steps);
    let mut rng = 0x1234_5678_9abc_def0_u64 ^ (band_size as u64);

    for _ in 0..50 {
      let mut values = vec![0u32; band_size * 2];
      for slot in &mut values {
        *slot = next_u32(&mut rng);
      }

      let direct = calculate_band_hash(&values);
      let left = calculate_band_hash(&values[..band_size]);
      let right = calculate_band_hash(&values[band_size..]);
      let left_state = u64_to_usize(left).rotate_right(FX_FINISH_ROTATE);
      let right_state = u64_to_usize(right).rotate_right(FX_FINISH_ROTATE);

      let combined_state =
        left_state.wrapping_mul(k_pow).wrapping_add(right_state);
      let combined = (combined_state.rotate_left(FX_FINISH_ROTATE)) as u64;

      assert_eq!(combined, direct);
    }
  }
}
