use crate::py_input::buffer::hash_buffer_like;
use crate::py_input::convert::py_err_to_type_error;
use crate::utils::calculate_hash_fast;
use pyo3::ffi;
use pyo3::prelude::*;
use std::os::raw::c_char;

#[allow(clippy::cast_sign_loss)]
#[allow(clippy::inline_always)]
#[inline(always)]
pub unsafe fn hash_unicode_ptr(
  py: Python<'_>,
  object_ptr: *mut ffi::PyObject,
) -> PyResult<u64> {
  // SAFETY: caller ensures `object_ptr` points to a unicode object.
  if unsafe { ffi::PyUnicode_IS_READY(object_ptr) } == 0
    && unsafe { ffi::PyUnicode_READY(object_ptr) } == -1
  {
    return Err(PyErr::fetch(py));
  }

  // SAFETY: unicode object is ready.
  if unsafe { ffi::PyUnicode_IS_ASCII(object_ptr) } != 0 {
    // SAFETY: pointer/length are valid for ready unicode object.
    let length_ssize = unsafe { ffi::PyUnicode_GET_LENGTH(object_ptr) };
    debug_assert!(length_ssize >= 0);
    let length = length_ssize as usize;
    let data = unsafe { ffi::PyUnicode_1BYTE_DATA(object_ptr) }.cast::<u8>();
    // SAFETY: pointer returned by CPython stays valid while object is alive.
    let bytes = unsafe { std::slice::from_raw_parts(data, length) };
    return Ok(calculate_hash_fast(bytes));
  }

  let mut length: ffi::Py_ssize_t = 0;
  // SAFETY: unicode object is valid.
  let utf8_ptr =
    unsafe { ffi::PyUnicode_AsUTF8AndSize(object_ptr, &raw mut length) };
  if utf8_ptr.is_null() {
    return Err(PyErr::fetch(py));
  }
  debug_assert!(length >= 0);
  let utf8_len = length as usize;
  // SAFETY: pointer returned by CPython stays valid while object is alive.
  let bytes =
    unsafe { std::slice::from_raw_parts(utf8_ptr.cast::<u8>(), utf8_len) };
  Ok(calculate_hash_fast(bytes))
}

#[allow(clippy::cast_sign_loss)]
#[allow(clippy::inline_always)]
#[inline(always)]
pub unsafe fn hash_bytes_ptr(
  py: Python<'_>,
  object_ptr: *mut ffi::PyObject,
) -> PyResult<u64> {
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
  // SAFETY: pointer/length are valid for bytes object.
  let bytes =
    unsafe { std::slice::from_raw_parts(data_ptr.cast::<u8>(), byte_len) };
  Ok(calculate_hash_fast(bytes))
}

#[allow(clippy::cast_sign_loss)]
#[allow(clippy::inline_always)]
#[inline(always)]
pub unsafe fn hash_bytearray_ptr(
  py: Python<'_>,
  object_ptr: *mut ffi::PyObject,
) -> PyResult<u64> {
  // SAFETY: caller ensures object_ptr is bytearray.
  let data_ptr = unsafe { ffi::PyByteArray_AsString(object_ptr) };
  if data_ptr.is_null() {
    return Err(PyErr::fetch(py));
  }
  // SAFETY: caller ensures object_ptr is bytearray.
  let length_ssize = unsafe { ffi::PyByteArray_Size(object_ptr) };
  debug_assert!(length_ssize >= 0);
  let length = length_ssize as usize;
  // SAFETY: pointer/length are valid while object is alive.
  let bytes =
    unsafe { std::slice::from_raw_parts(data_ptr.cast::<u8>(), length) };
  Ok(calculate_hash_fast(bytes))
}

pub fn hash_token_ptr(
  py: Python<'_>,
  object_ptr: *mut ffi::PyObject,
) -> PyResult<u64> {
  // SAFETY: object_ptr is a borrowed Python object pointer under GIL.
  unsafe {
    if ffi::PyUnicode_Check(object_ptr) != 0 {
      return hash_unicode_ptr(py, object_ptr);
    }
    if ffi::PyBytes_Check(object_ptr) != 0 {
      return hash_bytes_ptr(py, object_ptr);
    }
    if ffi::PyByteArray_Check(object_ptr) != 0 {
      return hash_bytearray_ptr(py, object_ptr);
    }
    if ffi::PyObject_CheckBuffer(object_ptr) != 0 {
      let item = Bound::from_borrowed_ptr(py, object_ptr);
      return hash_buffer_like(&item);
    }
  }
  Err(py_err_to_type_error(crate::py_input::TOKEN_TYPE_ERROR))
}

pub fn hash_byte_token_ptr(
  py: Python<'_>,
  object_ptr: *mut ffi::PyObject,
) -> PyResult<u64> {
  // SAFETY: object_ptr is a borrowed Python object pointer under GIL.
  unsafe {
    if ffi::PyBytes_Check(object_ptr) != 0 {
      return hash_bytes_ptr(py, object_ptr);
    }
    if ffi::PyByteArray_Check(object_ptr) != 0 {
      return hash_bytearray_ptr(py, object_ptr);
    }
    if ffi::PyObject_CheckBuffer(object_ptr) != 0 {
      let item = Bound::from_borrowed_ptr(py, object_ptr);
      return hash_buffer_like(&item);
    }
  }
  Err(py_err_to_type_error(crate::py_input::BYTE_TOKEN_TYPE_ERROR))
}
