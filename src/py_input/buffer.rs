use crate::utils::calculate_hash_fast;
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyTypeError;
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::PyAny;

#[inline]
pub fn has_buffer_protocol(item: &Bound<'_, PyAny>) -> bool {
  // Probe buffer capability portably (works for CPython and PyPy).
  unsafe {
    let mut view = ffi::Py_buffer::new();
    let result =
      ffi::PyObject_GetBuffer(item.as_ptr(), &raw mut view, ffi::PyBUF_SIMPLE);
    if result == 0 {
      ffi::PyBuffer_Release(&raw mut view);
      return true;
    }
    ffi::PyErr_Clear();
    false
  }
}

pub fn hash_buffer_like(item: &Bound<'_, PyAny>) -> PyResult<u64> {
  let buffer = PyBuffer::<u8>::get(item)
    .map_err(|_| PyTypeError::new_err(crate::py_input::BUFFER_TYPE_ERROR))?;

  if !buffer.is_c_contiguous() {
    return Err(PyTypeError::new_err(crate::py_input::BUFFER_TYPE_ERROR));
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
