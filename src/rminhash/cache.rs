use crate::rminhash::{RMinHash, RMinHashDigestMatrix};
use std::fs;

const DIGEST_CACHE_MAGIC: [u8; 4] = *b"RDC1";

fn digest_cache_dir() -> std::path::PathBuf {
  if let Some(value) = std::env::var_os("RENSA_DIGEST_MATRIX_CACHE_DIR") {
    if !value.is_empty() {
      return std::path::PathBuf::from(value);
    }
  }
  std::env::temp_dir().join("rensa_digest_matrix_cache")
}

pub(in crate::rminhash) fn digest_cache_key_env() -> Option<String> {
  let value = std::env::var("RENSA_DIGEST_MATRIX_CACHE_KEY").ok()?;
  let trimmed = value.trim();
  if trimmed.is_empty() {
    return None;
  }
  Some(trimmed.to_string())
}

fn sanitize_cache_key_component(input: &str) -> String {
  let mut out = String::with_capacity(input.len());
  for ch in input.chars() {
    if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.') {
      out.push(ch);
    } else {
      out.push('_');
    }
  }
  if out.is_empty() {
    out.push('_');
  }
  out
}

impl RMinHash {
  fn digest_cache_path(
    cache_key: &str,
    cache_domain: &str,
    num_perm: usize,
    seed: u64,
  ) -> std::path::PathBuf {
    let sanitized_key = sanitize_cache_key_component(cache_key);
    digest_cache_dir().join(format!(
      "{cache_domain}_{sanitized_key}_np{num_perm}_s{seed}.bin"
    ))
  }

  pub(in crate::rminhash) fn try_load_cached_digest_matrix(
    cache_key: &str,
    cache_domain: &str,
    num_perm: usize,
    seed: u64,
  ) -> Option<RMinHashDigestMatrix> {
    let path = Self::digest_cache_path(cache_key, cache_domain, num_perm, seed);
    let bytes = fs::read(path).ok()?;
    let header_len = 4 + (8 * 4);
    if bytes.len() < header_len {
      return None;
    }
    if bytes[0..4] != DIGEST_CACHE_MAGIC {
      return None;
    }
    let mut offset = 4usize;
    let read_u64 = |payload: &[u8], cursor: &mut usize| -> Option<u64> {
      let end = cursor.checked_add(8)?;
      let mut array = [0u8; 8];
      array.copy_from_slice(payload.get(*cursor..end)?);
      *cursor = end;
      Some(u64::from_le_bytes(array))
    };

    let cached_num_perm =
      usize::try_from(read_u64(&bytes, &mut offset)?).ok()?;
    let cached_rows = usize::try_from(read_u64(&bytes, &mut offset)?).ok()?;
    let cached_seed = read_u64(&bytes, &mut offset)?;
    let cached_len = usize::try_from(read_u64(&bytes, &mut offset)?).ok()?;
    if cached_num_perm != num_perm || cached_seed != seed {
      return None;
    }
    if cached_rows.checked_mul(cached_num_perm)? != cached_len {
      return None;
    }
    let payload_len = cached_len.checked_mul(std::mem::size_of::<u32>())?;
    if bytes.len() != offset.checked_add(payload_len)? {
      return None;
    }

    let mut data = vec![0u32; cached_len];
    #[cfg(target_endian = "little")]
    unsafe {
      std::ptr::copy_nonoverlapping(
        bytes.as_ptr().add(offset),
        data.as_mut_ptr().cast::<u8>(),
        payload_len,
      );
    }
    #[cfg(not(target_endian = "little"))]
    {
      for (chunk, value) in bytes[offset..].chunks_exact(4).zip(data.iter_mut())
      {
        let mut array = [0u8; 4];
        array.copy_from_slice(chunk);
        *value = u32::from_le_bytes(array);
      }
    }

    Some(RMinHashDigestMatrix {
      num_perm: cached_num_perm,
      rows: cached_rows,
      data,
      rho_sidecar: None,
    })
  }

  pub(in crate::rminhash) fn store_cached_digest_matrix(
    cache_key: &str,
    cache_domain: &str,
    num_perm: usize,
    seed: u64,
    matrix: &RMinHashDigestMatrix,
  ) {
    if matrix.num_perm != num_perm {
      return;
    }
    let cache_dir = digest_cache_dir();
    if fs::create_dir_all(&cache_dir).is_err() {
      return;
    }
    let path = Self::digest_cache_path(cache_key, cache_domain, num_perm, seed);
    let data_len = matrix.data.len();
    let Some(payload_len) = data_len.checked_mul(std::mem::size_of::<u32>())
    else {
      return;
    };
    let mut bytes = Vec::with_capacity(4 + (8 * 4) + payload_len);
    bytes.extend_from_slice(&DIGEST_CACHE_MAGIC);
    bytes.extend_from_slice(&(num_perm as u64).to_le_bytes());
    bytes.extend_from_slice(&(matrix.rows as u64).to_le_bytes());
    bytes.extend_from_slice(&seed.to_le_bytes());
    bytes.extend_from_slice(&(data_len as u64).to_le_bytes());

    #[cfg(target_endian = "little")]
    unsafe {
      let start = bytes.len();
      bytes.resize(start + payload_len, 0);
      std::ptr::copy_nonoverlapping(
        matrix.data.as_ptr().cast::<u8>(),
        bytes.as_mut_ptr().add(start),
        payload_len,
      );
    }
    #[cfg(not(target_endian = "little"))]
    for &value in &matrix.data {
      bytes.extend_from_slice(&value.to_le_bytes());
    }

    let tmp_path = path.with_extension(format!("tmp-{}", std::process::id()));
    if fs::write(&tmp_path, &bytes).is_ok()
      && fs::rename(&tmp_path, &path).is_err()
    {
      let _ = fs::remove_file(&tmp_path);
    }
  }
}
