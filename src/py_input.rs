use crate::utils::calculate_hash_fast;
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{
  PyByteArray, PyByteArrayMethods, PyBytes, PyBytesMethods, PyMemoryView,
  PyString, PyStringMethods,
};

const TOKEN_TYPE_ERROR: &str =
  "each item must be str, bytes, bytearray, or a C-contiguous u8 buffer";
const BUFFER_TYPE_ERROR: &str =
  "buffer inputs must be C-contiguous and byte-sized (u8)";

#[inline]
fn has_buffer_protocol(item: &Bound<'_, PyAny>) -> bool {
  PyMemoryView::from(item).is_ok()
}

fn hash_buffer_like(item: &Bound<'_, PyAny>) -> PyResult<u64> {
  let buffer = PyBuffer::<u8>::get(item)
    .map_err(|_| PyTypeError::new_err(BUFFER_TYPE_ERROR))?;

  if !buffer.is_c_contiguous() {
    return Err(PyTypeError::new_err(BUFFER_TYPE_ERROR));
  }

  if buffer.len_bytes() == 0 {
    return Ok(calculate_hash_fast(&[]));
  }

  // SAFETY: the buffer is C-contiguous and used immediately for hashing.
  let bytes = unsafe {
    std::slice::from_raw_parts(
      buffer.buf_ptr().cast::<u8>(),
      buffer.len_bytes(),
    )
  };
  Ok(calculate_hash_fast(bytes))
}

pub fn hash_token(item: &Bound<'_, PyAny>) -> PyResult<u64> {
  if let Ok(py_string) = item.cast::<PyString>() {
    return Ok(calculate_hash_fast(py_string.to_str()?.as_bytes()));
  }

  if let Ok(py_bytes) = item.cast::<PyBytes>() {
    return Ok(calculate_hash_fast(py_bytes.as_bytes()));
  }

  if let Ok(py_bytearray) = item.cast::<PyByteArray>() {
    // SAFETY: used only for immediate hashing.
    let bytes = unsafe { py_bytearray.as_bytes() };
    return Ok(calculate_hash_fast(bytes));
  }

  if has_buffer_protocol(item) {
    return hash_buffer_like(item);
  }

  Err(PyTypeError::new_err(TOKEN_TYPE_ERROR))
}

pub fn hash_single_bufferlike(
  items: &Bound<'_, PyAny>,
) -> PyResult<Option<u64>> {
  if items.is_instance_of::<PyString>() {
    return Ok(None);
  }

  if let Ok(py_bytes) = items.cast::<PyBytes>() {
    return Ok(Some(calculate_hash_fast(py_bytes.as_bytes())));
  }

  if let Ok(py_bytearray) = items.cast::<PyByteArray>() {
    // SAFETY: used only for immediate hashing.
    let bytes = unsafe { py_bytearray.as_bytes() };
    return Ok(Some(calculate_hash_fast(bytes)));
  }

  if has_buffer_protocol(items) {
    return hash_buffer_like(items).map(Some);
  }

  Ok(None)
}
