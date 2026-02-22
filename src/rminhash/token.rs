use pyo3::ffi;
use pyo3::prelude::*;
use std::os::raw::c_char;

#[derive(Clone, Copy)]
pub struct TokenBytesRef {
  pub ptr: *const u8,
  pub len: usize,
}

// SAFETY: `TokenBytesRef` is only constructed from immutable Python `str`/`bytes`
// buffers. Pointers are only read while the GIL is held (even though work
// runs on Rayon threads), which prevents other Python threads from mutating or
// dropping the backing objects.
unsafe impl Send for TokenBytesRef {}
unsafe impl Sync for TokenBytesRef {}

#[allow(clippy::cast_sign_loss)]
#[allow(clippy::inline_always)]
#[inline(always)]
pub unsafe fn token_bytes_ref_from_unicode_ptr(
  py: Python<'_>,
  object_ptr: *mut ffi::PyObject,
) -> PyResult<TokenBytesRef> {
  let mut length: ffi::Py_ssize_t = 0;
  // SAFETY: caller ensures `object_ptr` points to a unicode object.
  let utf8_ptr =
    unsafe { ffi::PyUnicode_AsUTF8AndSize(object_ptr, &raw mut length) };
  if utf8_ptr.is_null() {
    return Err(PyErr::fetch(py));
  }
  debug_assert!(length >= 0);
  let utf8_len = length as usize;
  let ptr = if utf8_len == 0 {
    std::ptr::NonNull::<u8>::dangling().as_ptr()
  } else {
    utf8_ptr.cast::<u8>()
  };
  Ok(TokenBytesRef { ptr, len: utf8_len })
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
  let ptr = if byte_len == 0 {
    std::ptr::NonNull::<u8>::dangling().as_ptr()
  } else {
    data_ptr.cast::<u8>()
  };
  Ok(TokenBytesRef { ptr, len: byte_len })
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
