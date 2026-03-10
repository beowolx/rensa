#![deny(clippy::all)]
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]

use rensa_core::rminhash::{CoreError, RMinHashContext};

const ABI_VERSION: u32 = 1;

const ERR_NULL_PTR: i32 = 1;
const ERR_LEN_MISMATCH: i32 = 2;
const ERR_INVALID_CTX: i32 = 3;

const CTX_MAGIC: u64 = 0x5f46_4649_414e_5341; // "__FFIANSA" (arbitrary, stable)

pub struct RensaRMinHashCtx;

struct RensaRMinHashCtxInner {
  magic: u64,
  inner: RMinHashContext,
}

#[no_mangle]
pub const extern "C" fn rensa_ffi_abi_version() -> u32 {
  ABI_VERSION
}

#[no_mangle]
pub extern "C" fn rensa_rminhash_ctx_new(
  num_perm: usize,
  seed: u64,
) -> *mut RensaRMinHashCtx {
  let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
    RMinHashContext::new(num_perm, seed).ok().map(|inner| {
      let ctx = RensaRMinHashCtxInner {
        magic: CTX_MAGIC,
        inner,
      };
      Box::into_raw(Box::new(ctx)).cast::<RensaRMinHashCtx>()
    })
  }));

  result.map_or(std::ptr::null_mut(), |ptr| {
    ptr.unwrap_or(std::ptr::null_mut())
  })
}

#[no_mangle]
/// Free a context created by [`rensa_rminhash_ctx_new`].
///
/// # Safety
///
/// `ctx` must be either null, or a pointer returned by
/// [`rensa_rminhash_ctx_new`]. It must be freed at most once, and must not be
/// used after being freed.
pub unsafe extern "C" fn rensa_rminhash_ctx_free(ctx: *mut RensaRMinHashCtx) {
  if ctx.is_null() {
    return;
  }
  // SAFETY: caller guarantees `ctx` is a pointer returned by `ctx_new` and is
  // freed at most once.
  drop(Box::from_raw(ctx.cast::<RensaRMinHashCtxInner>()));
}

#[no_mangle]
/// Compute an R-MinHash digest from pre-hashed tokens (`u64` feature hashes).
///
/// The digest is written into `out_digest` (a `u32` array). The output buffer
/// length must match the context's `num_perm`.
///
/// Returns 0 on success, or a small error code:
/// - 1: null pointer
/// - 2: output length mismatch
/// - 3: invalid live context or internal failure
///
/// # Safety
///
/// - `ctx` must be either null, or a pointer returned by
///   [`rensa_rminhash_ctx_new`] that has not been freed.
/// - Passing a freed, foreign, or otherwise invalid pointer is undefined
///   behavior. The defensive magic-tag check only applies to still-live memory.
/// - `out_digest` must be either null (in which case an error is returned), or
///   a pointer to a writable buffer of `out_digest_len` `u32` values.
/// - If `token_hashes_len != 0`, `token_hashes` must be a pointer to a readable
///   buffer of `token_hashes_len` `u64` values.
pub unsafe extern "C" fn rensa_rminhash_digest_prehashed(
  ctx: *const RensaRMinHashCtx,
  token_hashes: *const u64,
  token_hashes_len: usize,
  out_digest: *mut u32,
  out_digest_len: usize,
) -> i32 {
  if ctx.is_null() {
    return ERR_NULL_PTR;
  }
  if out_digest.is_null() {
    return ERR_NULL_PTR;
  }
  if token_hashes_len != 0 && token_hashes.is_null() {
    return ERR_NULL_PTR;
  }

  // SAFETY: caller passes a valid `ctx` pointer.
  let ctx_ref = unsafe { &*ctx.cast::<RensaRMinHashCtxInner>() };
  if ctx_ref.magic != CTX_MAGIC {
    return ERR_INVALID_CTX;
  }
  if out_digest_len != ctx_ref.inner.num_perm() {
    return ERR_LEN_MISMATCH;
  }

  // SAFETY: caller owns buffers and guarantees they are valid for the given
  // lengths.
  let token_hashes_slice = if token_hashes_len == 0 {
    &[]
  } else {
    unsafe { std::slice::from_raw_parts(token_hashes, token_hashes_len) }
  };
  let out_digest_slice =
    unsafe { std::slice::from_raw_parts_mut(out_digest, out_digest_len) };

  let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
    ctx_ref
      .inner
      .digest_prehashed_into(token_hashes_slice, out_digest_slice)
  }));
  match result {
    Ok(Ok(())) => 0,
    Ok(Err(CoreError::OutputLenMismatch { .. })) => ERR_LEN_MISMATCH,
    Ok(Err(CoreError::NumPermZero)) | Err(_) => ERR_INVALID_CTX,
  }
}

#[cfg(test)]
mod tests {
  use super::rensa_rminhash_ctx_free;
  use super::rensa_rminhash_ctx_new;
  use super::rensa_rminhash_digest_prehashed;

  #[test]
  fn digest_errors_on_null_ctx() {
    let tokens = [1_u64, 2, 3];
    let mut out = [0_u32; 16];
    let code = unsafe {
      rensa_rminhash_digest_prehashed(
        std::ptr::null(),
        tokens.as_ptr(),
        tokens.len(),
        out.as_mut_ptr(),
        out.len(),
      )
    };
    assert_eq!(code, 1);
  }

  #[test]
  fn digest_matches_core_for_simple_input() {
    let num_perm = 16;
    let seed = 42;
    let tokens = [1_u64, 2, 3, 10, 20, 30, 999];
    let ctx = rensa_rminhash_ctx_new(num_perm, seed);
    assert!(!ctx.is_null());

    let mut out = [0_u32; 16];
    let code = unsafe {
      rensa_rminhash_digest_prehashed(
        ctx,
        tokens.as_ptr(),
        tokens.len(),
        out.as_mut_ptr(),
        out.len(),
      )
    };
    assert_eq!(code, 0);

    let mut expected = [0_u32; 16];
    rensa_core::rminhash::digest_prehashed(
      num_perm,
      seed,
      &tokens,
      &mut expected,
    )
    .unwrap();
    assert_eq!(out, expected);

    unsafe {
      rensa_rminhash_ctx_free(ctx);
    }
  }
}
