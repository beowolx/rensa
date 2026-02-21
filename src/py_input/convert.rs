use pyo3::exceptions::PyTypeError;
use pyo3::ffi;
use pyo3::prelude::*;

#[cold]
#[inline(never)]
pub fn py_err_to_type_error(message: &'static str) -> PyErr {
  PyTypeError::new_err(message)
}

#[inline]
pub fn py_ssize_to_usize(value: ffi::Py_ssize_t) -> PyResult<usize> {
  usize::try_from(value)
    .map_err(|_| py_err_to_type_error(crate::py_input::SEQUENCE_SIZE_ERROR))
}
