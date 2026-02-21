use pyo3::buffer::PyBuffer;
use pyo3::ffi;
use pyo3::prelude::*;

pub fn extract_prehashed_u64_ptr(
  object_ptr: *mut ffi::PyObject,
) -> PyResult<u64> {
  // SAFETY: object_ptr is a borrowed Python object pointer under the GIL.
  let value = unsafe { ffi::PyLong_AsUnsignedLongLong(object_ptr) };
  // SAFETY: checking the C-API error indicator must happen under the GIL.
  if value == u64::MAX && unsafe { !ffi::PyErr_Occurred().is_null() } {
    // SAFETY: clearing the conversion error keeps the Python exception surface stable.
    unsafe { ffi::PyErr_Clear() };
    return Err(crate::py_input::convert::py_err_to_type_error(
      crate::py_input::PREHASHED_TOKEN_TYPE_ERROR,
    ));
  }
  Ok(value)
}

pub fn try_extend_prehashed_u64_buffer(
  document: &Bound<'_, PyAny>,
  output: &mut Vec<u64>,
) -> PyResult<bool> {
  let Ok(buffer) = PyBuffer::<u64>::get(document) else {
    return Ok(false);
  };
  if !buffer.is_c_contiguous() {
    return Err(crate::py_input::convert::py_err_to_type_error(
      crate::py_input::PREHASHED_TOKEN_TYPE_ERROR,
    ));
  }
  let item_count = buffer.item_count();
  if item_count == 0 {
    return Ok(true);
  }
  let data_ptr = buffer.buf_ptr().cast::<u64>();
  if data_ptr.is_null() {
    return Err(crate::py_input::convert::py_err_to_type_error(
      crate::py_input::PREHASHED_TOKEN_TYPE_ERROR,
    ));
  }
  let values = unsafe { std::slice::from_raw_parts(data_ptr, item_count) };
  output.extend_from_slice(values);
  Ok(true)
}
