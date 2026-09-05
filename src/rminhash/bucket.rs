//! Full-set sketch with three bucket rounds and per-coordinate fallback minima.
//!
//! Inspired by Fast Similarity Sketching (arXiv:1704.04370). The round tag is
//! part of each stored priority, so every path computes the same coordinatewise
//! minimum, including incremental updates across different occupancy levels.

use crate::rminhash::rho::splitmix64;
use crate::simd::dispatch::{
  apply_bucket_fallback, apply_hash_batch_to_values, PermutationSoA,
};
use rustc_hash::{FxBuildHasher, FxHashSet};

const RANK_BITS: u32 = 2;
const FRACTION_BITS: u32 = 32 - RANK_BITS;
const BUCKET_ROUNDS: u32 = (1 << RANK_BITS) - 1;
const FALLBACK: u32 = BUCKET_ROUNDS << FRACTION_BITS;
const BUCKET_DOMAIN: u64 = 0x6a09_e667_f3bc_c909;
const ROUND_DOMAIN: u64 = 0x3c6e_f372_fe94_f82b;
const FALLBACK_DOMAIN: u64 = 0xbb67_ae85_84ca_a73b;

#[inline]
#[allow(clippy::cast_possible_truncation)]
fn bucket_index(mut mixed: u64, count: u32) -> usize {
  let threshold = count.wrapping_neg() % count;
  loop {
    let product = u64::from(mixed as u32) * u64::from(count);
    if product as u32 >= threshold {
      return (product >> 32) as usize;
    }
    mixed = splitmix64(mixed);
  }
}

#[inline]
#[allow(clippy::cast_possible_truncation)]
fn wide_bucket_index(mut mixed: u64, count: usize) -> usize {
  let count = count as u64;
  let threshold = count.wrapping_neg() % count;
  loop {
    let product = u128::from(mixed) * u128::from(count);
    if product as u64 >= threshold {
      return (product >> 64) as usize;
    }
    mixed = splitmix64(mixed);
  }
}

pub(super) fn apply(
  hash_values: &mut [u32],
  permutations: &[(u64, u64)],
  soa: &PermutationSoA,
  hashes: &[u64],
) {
  let len = hash_values.len().min(permutations.len());
  if len == 0 || hashes.is_empty() {
    return;
  }
  let hash_values = &mut hash_values[..len];
  let fresh_short = hashes.len() <= 32
    && len >= 128
    && hash_values[0] == u32::MAX
    && hash_values
      .iter()
      .fold(u32::MAX, |combined, &value| combined & value)
      == u32::MAX;
  // Three rounds cannot fill these rows, so write their fallback in place.
  if fresh_short {
    let mut tiny;
    let mut small;
    let mixed = if hashes.len() <= 8 {
      tiny = [0; 8];
      &mut tiny[..hashes.len()]
    } else {
      small = [0; 32];
      &mut small[..hashes.len()]
    };
    let fallback_seed = permutations[0].0 ^ FALLBACK_DOMAIN;
    for (value, &hash) in mixed.iter_mut().zip(hashes) {
      *value = splitmix64(hash ^ fallback_seed);
    }
    apply_hash_batch_to_values(hash_values, permutations, soa, mixed);
    for value in hash_values.iter_mut() {
      *value = FALLBACK | (*value >> RANK_BITS);
    }
  }
  for round in 0..BUCKET_ROUNDS {
    let stage_seed = permutations[0].1
      ^ BUCKET_DOMAIN
      ^ u64::from(round).wrapping_mul(ROUND_DOMAIN);
    let prefix = round << FRACTION_BITS;
    if let Ok(count) = u32::try_from(len) {
      for &hash in hashes {
        let mixed = splitmix64(hash ^ stage_seed);
        let bucket = bucket_index(mixed, count);
        #[allow(clippy::cast_possible_truncation)]
        let rank = prefix | (mixed >> (32 + RANK_BITS)) as u32;
        hash_values[bucket] = hash_values[bucket].min(rank);
      }
    } else {
      for &hash in hashes {
        let mixed = splitmix64(hash ^ stage_seed);
        let bucket = wide_bucket_index(splitmix64(mixed), len);
        #[allow(clippy::cast_possible_truncation)]
        let rank = prefix | (mixed >> (32 + RANK_BITS)) as u32;
        hash_values[bucket] = hash_values[bucket].min(rank);
      }
    }
    let next_prefix = (round + 1) << FRACTION_BITS;
    if !fresh_short && hash_values.iter().all(|&value| value < next_prefix) {
      return;
    }
  }
  if fresh_short {
    return;
  }

  let missing = hash_values
    .iter()
    .filter(|&&value| value >= FALLBACK)
    .count();
  if missing == 0 {
    return;
  }
  let mut tiny;
  let mut small;
  let mut stack;
  let mut heap;
  let fallback_seed = permutations[0].0 ^ FALLBACK_DOMAIN;
  let mixed = if hashes.len() / 4 > len {
    let capacity = len.min(512);
    let mut seen = FxHashSet::with_capacity_and_hasher(capacity, FxBuildHasher);
    heap = Vec::with_capacity(capacity);
    for &hash in hashes {
      if seen.insert(hash) {
        heap.push(splitmix64(hash ^ fallback_seed));
      }
    }
    &mut heap
  } else {
    let mixed = if hashes.len() <= 8 {
      tiny = [0; 8];
      &mut tiny[..hashes.len()]
    } else if hashes.len() <= 32 {
      small = [0; 32];
      &mut small[..hashes.len()]
    } else if hashes.len() <= 128 {
      stack = [0; 128];
      &mut stack[..hashes.len()]
    } else {
      heap = vec![0; hashes.len()];
      &mut heap
    };
    for (value, &hash) in mixed.iter_mut().zip(hashes) {
      *value = splitmix64(hash ^ fallback_seed);
    }
    mixed
  };
  if missing > 3 && (hashes.len() <= 32 || missing >= len - len / 4) {
    let mut small;
    let mut large;
    let mut heap;
    let raw = if len <= 128 {
      small = [u32::MAX; 128];
      &mut small[..len]
    } else if len <= 512 {
      large = [u32::MAX; 512];
      &mut large[..len]
    } else {
      heap = vec![u32::MAX; len];
      &mut heap
    };
    apply_hash_batch_to_values(raw, permutations, soa, mixed);
    for (value, &rank) in hash_values.iter_mut().zip(raw.iter()) {
      *value = (*value).min(FALLBACK | (rank >> RANK_BITS));
    }
  } else {
    apply_bucket_fallback(hash_values, permutations, mixed, RANK_BITS);
  }
}

#[cfg(test)]
mod tests {
  use crate::rminhash::bucket::{
    apply, bucket_index, BUCKET_DOMAIN, BUCKET_ROUNDS, FALLBACK,
    FALLBACK_DOMAIN, FRACTION_BITS, RANK_BITS, ROUND_DOMAIN,
  };
  use crate::rminhash::rho::splitmix64;
  use crate::simd::dispatch::PermutationSoA;
  use rand_core::{RngCore, SeedableRng};
  use rand_xoshiro::Xoshiro256PlusPlus;

  fn reference(
    values: &mut [u32],
    permutations: &[(u64, u64)],
    hashes: &[u64],
  ) {
    if values.is_empty() {
      return;
    }
    let count = u64::try_from(values.len()).unwrap();
    for &hash in hashes {
      let fallback_hash =
        splitmix64(hash ^ permutations[0].0 ^ FALLBACK_DOMAIN);
      for (value, &(a, b)) in values.iter_mut().zip(permutations) {
        let affine = u128::from(a) * u128::from(fallback_hash) + u128::from(b);
        let rank = FALLBACK
          | u32::try_from(
            (affine >> (32 + RANK_BITS))
              & u128::from((1_u32 << FRACTION_BITS) - 1),
          )
          .unwrap();
        *value = (*value).min(rank);
      }
      for round in 0..BUCKET_ROUNDS {
        let mixed = splitmix64(
          hash
            ^ permutations[0].1
            ^ BUCKET_DOMAIN
            ^ u64::from(round).wrapping_mul(ROUND_DOMAIN),
        );
        let mut candidate = mixed;
        let threshold = (1_u64 << 32) % count;
        let bucket = loop {
          let product = (candidate & u64::from(u32::MAX)) * count;
          if product & u64::from(u32::MAX) >= threshold {
            break usize::try_from(product >> 32).unwrap();
          }
          candidate = splitmix64(candidate);
        };
        let rank = (round << FRACTION_BITS)
          | u32::try_from(mixed >> (32 + RANK_BITS)).unwrap();
        values[bucket] = values[bucket].min(rank);
      }
    }
  }

  #[test]
  fn bucket_mapping_rejects_incomplete_intervals() {
    for count in [1, 3, 8, 17, 128, 512, 0x8000_0001, u32::MAX] {
      for mixed in [0, 1, 2, u64::MAX, 0xffff_ffff_0000_0000] {
        let mut candidate = mixed;
        let expected = loop {
          let product = (candidate & u64::from(u32::MAX)) * u64::from(count);
          if product % (1_u64 << 32) >= (1_u64 << 32) % u64::from(count) {
            break usize::try_from(product / (1_u64 << 32)).unwrap();
          }
          candidate = splitmix64(candidate);
        };
        assert_eq!(bucket_index(mixed, count), expected);
        assert!(expected < count as usize);
      }
    }
  }

  #[test]
  fn bucket_kernel_matches_canonical_family_for_subsets_and_state() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x4255_434b);
    let universe = [0, 1, 2, u64::MAX, 1 << 32, rng.next_u64()];
    for count in [0, 1, 3, 8, 17, 128, 512] {
      let permutations: Vec<_> = (0..count)
        .map(|_| (rng.next_u64() | 1, rng.next_u64()))
        .collect();
      let soa = PermutationSoA::from_permutations(&permutations);
      for subset in 0..(1 << universe.len()) {
        let hashes: Vec<_> = universe
          .iter()
          .enumerate()
          .filter_map(|(index, &hash)| {
            ((subset >> index) & 1 == 1).then_some(hash)
          })
          .collect();
        for initial in [
          vec![u32::MAX; count],
          (0..count).map(|_| rng.next_u32()).collect(),
        ] {
          let mut expected = initial.clone();
          reference(&mut expected, &permutations, &hashes);
          let mut actual = initial.clone();
          apply(&mut actual, &permutations, &soa, &hashes);
          assert_eq!(actual, expected);
          apply(&mut actual, &permutations, &soa, &hashes);
          assert_eq!(actual, expected, "duplicate update changed the minimum");

          for split in 0..=hashes.len() {
            let mut chunked = initial.clone();
            apply(&mut chunked, &permutations, &soa, &hashes[..split]);
            apply(&mut chunked, &permutations, &soa, &hashes[split..]);
            assert_eq!(
              chunked, expected,
              "chunk boundary changed the signature"
            );
          }
          let mut reversed = hashes.clone();
          reversed.reverse();
          let mut actual = initial;
          apply(&mut actual, &permutations, &soa, &reversed);
          assert_eq!(actual, expected, "token order changed the signature");
        }
      }
    }
  }

  #[test]
  fn bucket_kernel_matches_canonical_family_for_dense_inputs_and_merges() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(u64::MAX);
    for count in [1, 3, 8, 17, 128, 512] {
      let permutations: Vec<_> = (0..count)
        .map(|_| (rng.next_u64() | 1, rng.next_u64()))
        .collect();
      let soa = PermutationSoA::from_permutations(&permutations);
      for size in [31, 32, 33, count, count * 2, count * 8] {
        let hashes: Vec<_> = (0..size).map(|_| rng.next_u64()).collect();
        let mut expected = vec![u32::MAX; count];
        reference(&mut expected, &permutations, &hashes);
        let mut actual = vec![u32::MAX; count];
        apply(&mut actual, &permutations, &soa, &hashes);
        assert_eq!(actual, expected);

        let mut left = vec![u32::MAX; count];
        let mut right = vec![u32::MAX; count];
        apply(&mut left, &permutations, &soa, &hashes[..size / 2]);
        apply(&mut right, &permutations, &soa, &hashes[size / 2..]);
        for (value, other) in left.iter_mut().zip(right) {
          *value = (*value).min(other);
        }
        assert_eq!(left, expected, "merging changed the signature");
      }
    }
  }

  #[test]
  fn repeated_tokens_keep_unique_signature_across_fallback_boundaries() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5245_5045_4154);
    for count in [128, 129, 512] {
      let permutations: Vec<_> = (0..count)
        .map(|_| (rng.next_u64() | 1, rng.next_u64()))
        .collect();
      let soa = PermutationSoA::from_permutations(&permutations);
      for unique in [&[0_u64][..], &[0, 1, u64::MAX][..]] {
        let mut hashes: Vec<_> =
          unique.iter().copied().cycle().take(8192).collect();
        let mut expected = vec![u32::MAX; count];
        reference(&mut expected, &permutations, unique);
        let mut actual = vec![u32::MAX; count];
        apply(&mut actual, &permutations, &soa, &hashes);
        assert_eq!(actual, expected);

        hashes.reverse();
        for split in [1, count * 4, count * 4 + 4, hashes.len() - 1] {
          let mut actual = vec![u32::MAX; count];
          apply(&mut actual, &permutations, &soa, &hashes[..split]);
          apply(&mut actual, &permutations, &soa, &hashes[split..]);
          assert_eq!(
            actual, expected,
            "fallback deduplication changed the signature"
          );
        }
      }
    }
  }
}
