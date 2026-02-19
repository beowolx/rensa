use crate::simd::dispatch::PermutationSoA;
use rustc_hash::FxHashMap;

pub(in crate::rminhash) struct AdaptivePermutationCache {
  pub(in crate::rminhash) digests: FxHashMap<u64, Box<[u32]>>,
  pub(in crate::rminhash) seen_counts: FxHashMap<u64, u8>,
  pub(in crate::rminhash) min_frequency: usize,
  pub(in crate::rminhash) max_hashes: usize,
  pub(in crate::rminhash) max_tracked_seen_hashes: usize,
}

impl AdaptivePermutationCache {
  pub(in crate::rminhash) fn new(
    min_frequency: usize,
    max_hashes: usize,
  ) -> Self {
    Self {
      digests: FxHashMap::default(),
      seen_counts: FxHashMap::default(),
      min_frequency,
      max_hashes,
      max_tracked_seen_hashes: max_hashes.saturating_mul(4).max(8_192),
    }
  }
}

impl crate::rminhash::RMinHash {
  fn build_cached_digest_for_token(
    num_perm: usize,
    permutations: &[(u64, u64)],
    permutations_soa: &PermutationSoA,
    token_hash: u64,
  ) -> Box<[u32]> {
    let mut digest_row = vec![u32::MAX; num_perm];
    Self::apply_token_hashes_to_values(
      &mut digest_row,
      permutations,
      permutations_soa,
      std::slice::from_ref(&token_hash),
    );
    digest_row.into_boxed_slice()
  }

  pub(in crate::rminhash) fn apply_token_hashes_with_cache(
    digest_row: &mut [u32],
    token_hashes: &[u64],
    permutations: &[(u64, u64)],
    permutations_soa: &PermutationSoA,
    cache: &mut AdaptivePermutationCache,
    miss_hashes: &mut Vec<u64>,
  ) {
    digest_row.fill(u32::MAX);
    if cache.max_hashes == 0 {
      Self::apply_token_hashes_to_values(
        digest_row,
        permutations,
        permutations_soa,
        token_hashes,
      );
      return;
    }

    miss_hashes.clear();
    for &token_hash in token_hashes {
      if let Some(cached_digest) = cache.digests.get(&token_hash) {
        for (value, &cached_value) in
          digest_row.iter_mut().zip(cached_digest.iter())
        {
          *value = (*value).min(cached_value);
        }
        continue;
      }

      miss_hashes.push(token_hash);
      if cache.digests.len() >= cache.max_hashes {
        continue;
      }

      if let Some(count) = cache.seen_counts.get_mut(&token_hash) {
        *count = count.saturating_add(1);
        if usize::from(*count) >= cache.min_frequency {
          let cached_digest = Self::build_cached_digest_for_token(
            digest_row.len(),
            permutations,
            permutations_soa,
            token_hash,
          );
          cache.digests.insert(token_hash, cached_digest);
          cache.seen_counts.remove(&token_hash);
        }
        continue;
      }

      if cache.seen_counts.len() < cache.max_tracked_seen_hashes {
        cache.seen_counts.insert(token_hash, 1);
      }
    }
    if !miss_hashes.is_empty() {
      Self::apply_token_hashes_to_values(
        digest_row,
        permutations,
        permutations_soa,
        miss_hashes,
      );
    }
  }
}
