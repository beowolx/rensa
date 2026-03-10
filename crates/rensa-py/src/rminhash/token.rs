use pyo3::exceptions::PyValueError;
use pyo3::ffi;
use pyo3::prelude::*;
use std::os::raw::{c_char, c_uint};

#[repr(C)]
#[derive(Clone, Copy)]
pub union TokenBytesPtr {
  pub bytes: *const u8,
  pub u16: *const u16,
  pub u32: *const u32,
}

#[derive(Clone, Copy)]
pub struct TokenBytesRef {
  pub ptr: TokenBytesPtr,
  pub len: usize,
  pub kind: TokenBytesKind,
}

// SAFETY: `TokenBytesRef` is only constructed from immutable Python `str`/`bytes`
// buffers. Parallel Rho workers keep strong references to tuple rows while
// these pointers are in flight, so the backing token objects cannot be dropped
// before hashing completes. List-backed rows use the streaming fallback path.
unsafe impl Send for TokenBytesRef {}
unsafe impl Sync for TokenBytesRef {}

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TokenBytesKind {
  /// `ptr` points to a byte buffer, `len` is the byte length.
  Bytes = 0,
  /// `ptr` points to `CPython` unicode storage (PEP 393) as `Py_UCS1`,
  /// `len` is the character length.
  Unicode1 = 1,
  /// `ptr` points to `CPython` unicode storage (PEP 393) as `Py_UCS2`,
  /// `len` is the character length.
  Unicode2 = 2,
  /// `ptr` points to `CPython` unicode storage (PEP 393) as `Py_UCS4`,
  /// `len` is the character length.
  Unicode4 = 4,
}

impl TokenBytesKind {
  #[inline]
  #[must_use]
  pub const fn from_py_unicode_kind(kind: c_uint) -> Option<Self> {
    match kind {
      ffi::PyUnicode_1BYTE_KIND => Some(Self::Unicode1),
      ffi::PyUnicode_2BYTE_KIND => Some(Self::Unicode2),
      ffi::PyUnicode_4BYTE_KIND => Some(Self::Unicode4),
      _ => None,
    }
  }
}

#[allow(clippy::cast_sign_loss)]
#[allow(clippy::inline_always)]
#[inline(always)]
pub unsafe fn token_bytes_ref_from_unicode_ptr(
  py: Python<'_>,
  object_ptr: *mut ffi::PyObject,
) -> PyResult<TokenBytesRef> {
  // Ensure unicode internal representation is ready. On Py3.12 this is a no-op,
  // but it keeps us correct on older CPython targets.
  if unsafe { ffi::PyUnicode_READY(object_ptr) } == -1 {
    return Err(PyErr::fetch(py));
  }

  let length = unsafe { ffi::PyUnicode_GET_LENGTH(object_ptr) };
  debug_assert!(length >= 0);
  let char_len = length as usize;

  // Fast path: ASCII unicode uses the same bytes as UTF-8.
  if unsafe { ffi::PyUnicode_IS_ASCII(object_ptr) } != 0 {
    // SAFETY: ASCII storage is a 1-byte array of length `char_len`.
    let data_ptr = unsafe { ffi::PyUnicode_1BYTE_DATA(object_ptr) };
    let ptr = TokenBytesPtr {
      bytes: if char_len == 0 {
        std::ptr::NonNull::<u8>::dangling().as_ptr()
      } else {
        data_ptr.cast::<u8>().cast_const()
      },
    };
    return Ok(TokenBytesRef {
      ptr,
      len: char_len,
      kind: TokenBytesKind::Bytes,
    });
  }

  let kind = unsafe { ffi::PyUnicode_KIND(object_ptr) };
  let Some(kind) = TokenBytesKind::from_py_unicode_kind(kind) else {
    return Err(PyValueError::new_err(format!(
      "unsupported unicode storage kind: {kind}"
    )));
  };
  let ptr = match kind {
    TokenBytesKind::Unicode1 => TokenBytesPtr {
      bytes: if char_len == 0 {
        std::ptr::NonNull::<u8>::dangling().as_ptr()
      } else {
        // SAFETY: 1-byte unicode data is valid for `char_len` code units.
        unsafe { ffi::PyUnicode_1BYTE_DATA(object_ptr) }
          .cast::<u8>()
          .cast_const()
      },
    },
    TokenBytesKind::Unicode2 => TokenBytesPtr {
      u16: if char_len == 0 {
        std::ptr::NonNull::<u16>::dangling().as_ptr()
      } else {
        // SAFETY: 2-byte unicode data is valid for `char_len` code units.
        unsafe { ffi::PyUnicode_2BYTE_DATA(object_ptr) }
          .cast::<u16>()
          .cast_const()
      },
    },
    TokenBytesKind::Unicode4 => TokenBytesPtr {
      u32: if char_len == 0 {
        std::ptr::NonNull::<u32>::dangling().as_ptr()
      } else {
        // SAFETY: 4-byte unicode data is valid for `char_len` code units.
        unsafe { ffi::PyUnicode_4BYTE_DATA(object_ptr) }
          .cast::<u32>()
          .cast_const()
      },
    },
    TokenBytesKind::Bytes => {
      return Err(PyValueError::new_err(
        "unexpected bytes kind for unicode token",
      ))
    }
  };
  Ok(TokenBytesRef {
    ptr,
    len: char_len,
    kind,
  })
}

#[allow(clippy::cast_sign_loss)]
#[allow(clippy::inline_always)]
#[inline(always)]
pub unsafe fn token_bytes_ref_from_bytes_ptr(
  py: Python<'_>,
  object_ptr: *mut ffi::PyObject,
) -> PyResult<TokenBytesRef> {
  let mut data_ptr: *mut c_char = std::ptr::null_mut();
  let mut length: ffi::Py_ssize_t = 0;
  // SAFETY: caller ensures object_ptr is bytes.
  if unsafe {
    ffi::PyBytes_AsStringAndSize(object_ptr, &raw mut data_ptr, &raw mut length)
  } == -1
  {
    return Err(PyErr::fetch(py));
  }
  debug_assert!(length >= 0);
  let byte_len = length as usize;
  let ptr = TokenBytesPtr {
    bytes: if byte_len == 0 {
      std::ptr::NonNull::<u8>::dangling().as_ptr()
    } else {
      data_ptr.cast::<u8>().cast_const()
    },
  };
  Ok(TokenBytesRef {
    ptr,
    len: byte_len,
    kind: TokenBytesKind::Bytes,
  })
}

pub fn token_bytes_ref_from_token_ptr(
  py: Python<'_>,
  object_ptr: *mut ffi::PyObject,
) -> PyResult<Option<TokenBytesRef>> {
  // SAFETY: object_ptr is a borrowed Python pointer under the GIL.
  unsafe {
    if ffi::PyUnicode_Check(object_ptr) != 0 {
      return token_bytes_ref_from_unicode_ptr(py, object_ptr).map(Some);
    }
    if ffi::PyBytes_Check(object_ptr) != 0 {
      return token_bytes_ref_from_bytes_ptr(py, object_ptr).map(Some);
    }
  }
  Ok(None)
}
