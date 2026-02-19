#[derive(Clone, Copy)]
pub(super) struct DigestBuildConfig {
  pub(super) doc_chunk_size: usize,
  pub(super) doc_par_batch_size: usize,
  pub(super) pipeline_queue_cap: usize,
  pub(super) perm_cache_min_frequency: usize,
  pub(super) max_perm_cache_hashes: usize,
}

impl DigestBuildConfig {
  pub(super) fn from_env() -> Self {
    Self {
      doc_chunk_size: crate::env::read_env_usize_clamped(
        "RENSA_DOC_CHUNK_SIZE",
        crate::rminhash::DEFAULT_DOC_CHUNK_SIZE,
        crate::rminhash::MIN_DOC_CHUNK_SIZE,
        crate::rminhash::MAX_DOC_CHUNK_SIZE,
      ),
      doc_par_batch_size: crate::env::read_env_usize_clamped(
        "RENSA_DOC_PAR_BATCH_SIZE",
        crate::rminhash::DEFAULT_DOC_PAR_BATCH_SIZE,
        crate::rminhash::MIN_DOC_PAR_BATCH_SIZE,
        crate::rminhash::MAX_DOC_PAR_BATCH_SIZE,
      ),
      pipeline_queue_cap: crate::env::read_env_usize_clamped(
        "RENSA_PIPELINE_QUEUE_CAP",
        crate::rminhash::DEFAULT_PIPELINE_QUEUE_CAP,
        crate::rminhash::MIN_PIPELINE_QUEUE_CAP,
        crate::rminhash::MAX_PIPELINE_QUEUE_CAP,
      ),
      perm_cache_min_frequency:
        crate::rminhash::DEFAULT_PERM_CACHE_MIN_FREQUENCY,
      max_perm_cache_hashes: crate::env::read_env_usize_clamped(
        "RENSA_MAX_PERM_CACHE_HASHES",
        crate::rminhash::DEFAULT_MAX_PERM_CACHE_HASHES,
        crate::rminhash::MIN_MAX_PERM_CACHE_HASHES,
        crate::rminhash::MAX_MAX_PERM_CACHE_HASHES,
      ),
    }
  }
}
