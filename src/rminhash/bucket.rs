//! Full-set sketch with three bucket rounds and per-coordinate fallback minima.
//!
//! Inspired by Fast Similarity Sketching (arXiv:1704.04370). The round tag is
//! part of each stored priority, so every path computes the same coordinatewise
//! minimum, including incremental updates across different occupancy levels.

use crate::rminhash::rho::splitmix64;
use crate::simd::dispatch::{
  apply_bucket_fallback, apply_hash_batch_to_values, PermutationSoA,
};
use pyo3::buffer::ReadOnlyCell;
use rustc_hash::{FxBuildHasher, FxHashSet};

const RANK_BITS: u32 = 2;
const FRACTION_BITS: u32 = 32 - RANK_BITS;
const BUCKET_ROUNDS: u32 = (1 << RANK_BITS) - 1;
const FALLBACK: u32 = BUCKET_ROUNDS << FRACTION_BITS;
const BUCKET_DOMAIN: u64 = 0x6a09_e667_f3bc_c909;
const ROUND_DOMAIN: u64 = 0x3c6e_f372_fe94_f82b;
const FALLBACK_DOMAIN: u64 = 0xbb67_ae85_84ca_a73b;

pub(super) trait HashValue {
  fn value(&self) -> u64;
}

impl HashValue for u64 {
  #[inline]
  fn value(&self) -> u64 {
    *self
  }
}

impl HashValue for ReadOnlyCell<u64> {
  #[inline]
  fn value(&self) -> u64 {
    self.get()
  }
}

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

pub(super) fn apply<H: HashValue>(
  hash_values: &mut [u32],
  permutations: &[(u64, u64)],
  soa: &PermutationSoA,
  hashes: &[H],
) {
  let len = hash_values.len().min(permutations.len());
  if hashes.len() >= 1 << 20 && len.is_power_of_two() {
    if let Ok(count) = u32::try_from(len) {
      let end = len.saturating_mul(512).max(65_536).min(hashes.len());
      let (initial, remaining) = hashes.split_at(end);
      apply_unfiltered(hash_values, permutations, soa, initial);
      let hash_values = &mut hash_values[..len];
      let upper_bound = hash_values.iter().copied().max().unwrap_or(u32::MAX);
      if upper_bound <= (1 << FRACTION_BITS) / 32 {
        // The bound is below 2^30, so every coordinate has a round-zero
        // value and later rounds cannot improve it.
        let seed = permutations[0].1 ^ BUCKET_DOMAIN;
        let shift = 32 - count.trailing_zeros();
        let mut update = |hash| {
          let mixed = splitmix64(hash ^ seed);
          #[allow(clippy::cast_possible_truncation)]
          let rank = (mixed >> (32 + RANK_BITS)) as u32;
          // Minima only decrease, so this stale maximum remains an upper
          // bound on every coordinate throughout the remaining input.
          if rank >= upper_bound {
            return;
          }
          #[allow(clippy::cast_possible_truncation)]
          let bucket = (u64::from(mixed as u32) >> shift) as usize;
          // SAFETY: count is this slice's power-of-two length, 2^p. Shifting
          // a 32-bit word by 32-p produces an index strictly below count.
          let value = unsafe { hash_values.get_unchecked_mut(bucket) };
          if rank < *value {
            *value = rank;
          }
        };
        let mut chunks = remaining.chunks_exact(8);
        for chunk in chunks.by_ref() {
          let values: [u64; 8] = std::array::from_fn(|i| chunk[i].value());
          for hash in values {
            update(hash);
          }
        }
        for hash in chunks.remainder() {
          update(hash.value());
        }
        return;
      }
      apply_unfiltered(hash_values, permutations, soa, remaining);
      return;
    }
  }
  apply_unfiltered(hash_values, permutations, soa, hashes);
}

fn apply_unfiltered<H: HashValue>(
  hash_values: &mut [u32],
  permutations: &[(u64, u64)],
  soa: &PermutationSoA,
  hashes: &[H],
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
    for (value, hash) in mixed.iter_mut().zip(hashes) {
      *value = splitmix64(hash.value() ^ fallback_seed);
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
      if count.is_power_of_two() {
        let shift = 32 - count.trailing_zeros();
        let mut update = |hash, conditional| {
          let mixed = splitmix64(hash ^ stage_seed);
          // Lemire's product selects these bits without rejection for powers
          // of two. A u64 shift also handles one bucket (shift 32).
          #[allow(clippy::cast_possible_truncation)]
          let bucket = (u64::from(mixed as u32) >> shift) as usize;
          #[allow(clippy::cast_possible_truncation)]
          let rank = prefix | (mixed >> (32 + RANK_BITS)) as u32;
          // SAFETY: count equals this slice's length and is 2^p, with p in
          // 0..=31. The low word is < 2^32, so shifting it by 32-p yields
          // bucket < 2^p = count, including p=0 (a u64 shift by 32).
          let value = unsafe { hash_values.get_unchecked_mut(bucket) };
          if conditional {
            if rank < *value {
              *value = rank;
            }
          } else {
            *value = (*value).min(rank);
          }
        };
        let prefix_len = hashes.len().min(len.saturating_mul(8).max(1024));
        let (initial, remaining) = hashes.split_at(prefix_len);
        let mut chunks = initial.chunks_exact(8);
        for chunk in chunks.by_ref() {
          let values: [u64; 8] = std::array::from_fn(|i| chunk[i].value());
          for hash in values {
            update(hash, false);
          }
        }
        for hash in chunks.remainder() {
          update(hash.value(), false);
        }
        let mut chunks = remaining.chunks_exact(8);
        for chunk in chunks.by_ref() {
          let values: [u64; 8] = std::array::from_fn(|i| chunk[i].value());
          for hash in values {
            update(hash, true);
          }
        }
        for hash in chunks.remainder() {
          update(hash.value(), true);
        }
      } else {
        for hash in hashes {
          let mixed = splitmix64(hash.value() ^ stage_seed);
          let bucket = bucket_index(mixed, count);
          #[allow(clippy::cast_possible_truncation)]
          let rank = prefix | (mixed >> (32 + RANK_BITS)) as u32;
          hash_values[bucket] = hash_values[bucket].min(rank);
        }
      }
    } else {
      for hash in hashes {
        let mixed = splitmix64(hash.value() ^ stage_seed);
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
    for hash in hashes {
      let hash = hash.value();
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
    for (value, hash) in mixed.iter_mut().zip(hashes) {
      *value = splitmix64(hash.value() ^ fallback_seed);
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
    apply, apply_unfiltered, bucket_index, BUCKET_DOMAIN, BUCKET_ROUNDS,
    FALLBACK, FALLBACK_DOMAIN, FRACTION_BITS, RANK_BITS, ROUND_DOMAIN,
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

  #[test]
  fn long_rows_keep_late_improvements_across_incremental_updates() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x0042_4f55_4e44);
    for count in [8, 128, 512] {
      let permutations: Vec<_> = (0..count)
        .map(|_| (rng.next_u64() | 1, rng.next_u64()))
        .collect();
      let soa = PermutationSoA::from_permutations(&permutations);
      let count_u32 = u32::try_from(count).unwrap();
      let block_len = (count * 8).max(1024);
      let stage_seed = permutations[0].1 ^ BUCKET_DOMAIN;
      let mut representatives = vec![None; count];
      let mut missing = count;
      while missing != 0 {
        let hash = rng.next_u64();
        let bucket = bucket_index(splitmix64(hash ^ stage_seed), count_u32);
        if representatives[bucket].replace(hash).is_none() {
          missing -= 1;
        }
      }
      let mut hashes: Vec<_> = representatives.into_iter().flatten().collect();
      hashes.extend((count..block_len).map(|_| rng.next_u64()));
      let prefix_len = hashes.len();
      let mut initial = vec![u32::MAX; count];
      reference(&mut initial, &permutations, &hashes);
      assert!(initial.iter().all(|&value| value < 1 << FRACTION_BITS));

      hashes.extend((0..block_len * 3 + 3).map(|_| rng.next_u64()));
      hashes.extend_from_within(..count);
      let mut expected = vec![u32::MAX; count];
      reference(&mut expected, &permutations, &hashes);
      assert_ne!(expected, initial, "fixture must contain later improvements");

      let mut actual = vec![u32::MAX; count];
      apply(&mut actual, &permutations, &soa, &hashes);
      assert_eq!(actual, expected);
      hashes.reverse();
      for split in [1, prefix_len - 1, prefix_len, prefix_len + 1] {
        let mut actual = vec![u32::MAX; count];
        apply(&mut actual, &permutations, &soa, &hashes[..split]);
        apply(&mut actual, &permutations, &soa, &hashes[split..]);
        assert_eq!(
          actual, expected,
          "chunk boundaries changed an incremental minimum"
        );
      }
    }
  }

  #[test]
  fn very_late_rank_filter_preserves_updates_and_repeated_rows() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x4c41_5445);
    for count in [128, 512] {
      let permutations: Vec<_> = (0..count)
        .map(|_| (rng.next_u64() | 1, rng.next_u64()))
        .collect();
      let soa = PermutationSoA::from_permutations(&permutations);
      let count_u32 = u32::try_from(count).unwrap();
      let end = (count * 512).max(65_536);
      for enabled in [true, false] {
        let base = if enabled {
          (0..end).map(|_| rng.next_u64()).collect::<Vec<_>>()
        } else {
          let mut representatives = vec![None; count];
          let mut missing = count;
          while missing != 0 {
            let hash = rng.next_u64();
            let mixed = splitmix64(hash ^ permutations[0].1 ^ BUCKET_DOMAIN);
            if mixed >> (32 + RANK_BITS) < (1 << FRACTION_BITS) / 2 {
              continue;
            }
            let bucket = bucket_index(mixed, count_u32);
            if representatives[bucket].replace(hash).is_none() {
              missing -= 1;
            }
          }
          representatives.into_iter().flatten().collect()
        };
        let mut expected = vec![u32::MAX; count];
        apply_unfiltered(&mut expected, &permutations, &soa, &base);
        let initial = expected.clone();
        assert_eq!(
          expected.iter().copied().max().unwrap() <= (1 << FRACTION_BITS) / 32,
          enabled
        );
        let late: Vec<_> = (0..4099).map(|_| rng.next_u64()).collect();
        reference(&mut expected, &permutations, &late);
        assert_ne!(expected, initial, "fixture needs a late improvement");
        let mut hashes: Vec<_> =
          base.iter().copied().cycle().take(1 << 20).collect();
        hashes.extend_from_slice(&late);
        let mut actual = vec![u32::MAX; count];
        apply(&mut actual, &permutations, &soa, &hashes);
        assert_eq!(actual, expected);
        hashes.reverse();
        let mut actual = vec![u32::MAX; count];
        for chunk in hashes.chunks(end - 1) {
          apply(&mut actual, &permutations, &soa, chunk);
        }
        assert_eq!(actual, expected);
      }
    }
  }
}
