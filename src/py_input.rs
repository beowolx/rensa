use crate::utils::calculate_hash_fast;
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyTypeError;
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyIterator, PyList, PyMemoryView, PyTuple};
use std::os::raw::c_char;

const TOKEN_TYPE_ERROR: &str =
  "each item must be str, bytes, bytearray, or a C-contiguous u8 buffer";
const BYTE_TOKEN_TYPE_ERROR: &str =
  "each item must be bytes, bytearray, or a C-contiguous u8 buffer";
const PREHASHED_TOKEN_TYPE_ERROR: &str =
  "each item must be an unsigned 64-bit integer";
const BUFFER_TYPE_ERROR: &str =
  "buffer inputs must be C-contiguous and byte-sized (u8)";
const SEQUENCE_SIZE_ERROR: &str = "sequence size does not fit in usize";

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

#[inline]
fn py_err_to_type_error(message: &'static str) -> PyErr {
  PyTypeError::new_err(message)
}

#[inline]
fn py_ssize_to_usize(value: ffi::Py_ssize_t) -> PyResult<usize> {
  usize::try_from(value).map_err(|_| py_err_to_type_error(SEQUENCE_SIZE_ERROR))
}

fn extract_prehashed_u64_ptr(object_ptr: *mut ffi::PyObject) -> PyResult<u64> {
  // SAFETY: object_ptr is a borrowed Python object pointer under the GIL.
  let value = unsafe { ffi::PyLong_AsUnsignedLongLong(object_ptr) };
  // SAFETY: checking the C-API error indicator must happen under the GIL.
  if value == u64::MAX && unsafe { !ffi::PyErr_Occurred().is_null() } {
    // SAFETY: clearing the conversion error keeps the Python exception surface stable.
    unsafe { ffi::PyErr_Clear() };
    return Err(py_err_to_type_error(PREHASHED_TOKEN_TYPE_ERROR));
  }
  Ok(value)
}

fn try_extend_prehashed_u64_buffer(
  document: &Bound<'_, PyAny>,
  output: &mut Vec<u64>,
) -> PyResult<bool> {
  let Ok(buffer) = PyBuffer::<u64>::get(document) else {
    return Ok(false);
  };
  if !buffer.is_c_contiguous() {
    return Err(py_err_to_type_error(PREHASHED_TOKEN_TYPE_ERROR));
  }
  let values = unsafe {
    std::slice::from_raw_parts(
      buffer.buf_ptr().cast::<u64>(),
      buffer.item_count(),
    )
  };
  output.extend_from_slice(values);
  Ok(true)
}

unsafe fn hash_unicode_ptr(
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
    let length =
      py_ssize_to_usize(unsafe { ffi::PyUnicode_GET_LENGTH(object_ptr) })?;
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
  let utf8_len = py_ssize_to_usize(length)?;
  // SAFETY: pointer returned by CPython stays valid while object is alive.
  let bytes =
    unsafe { std::slice::from_raw_parts(utf8_ptr.cast::<u8>(), utf8_len) };
  Ok(calculate_hash_fast(bytes))
}

unsafe fn hash_bytes_ptr(
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
  let byte_len = py_ssize_to_usize(length)?;
  // SAFETY: pointer/length are valid for bytes object.
  let bytes =
    unsafe { std::slice::from_raw_parts(data_ptr.cast::<u8>(), byte_len) };
  Ok(calculate_hash_fast(bytes))
}

unsafe fn hash_bytearray_ptr(
  py: Python<'_>,
  object_ptr: *mut ffi::PyObject,
) -> PyResult<u64> {
  // SAFETY: caller ensures object_ptr is bytearray.
  let data_ptr = unsafe { ffi::PyByteArray_AsString(object_ptr) };
  if data_ptr.is_null() {
    return Err(PyErr::fetch(py));
  }
  // SAFETY: caller ensures object_ptr is bytearray.
  let length = py_ssize_to_usize(unsafe { ffi::PyByteArray_Size(object_ptr) })?;
  // SAFETY: pointer/length are valid while object is alive.
  let bytes =
    unsafe { std::slice::from_raw_parts(data_ptr.cast::<u8>(), length) };
  Ok(calculate_hash_fast(bytes))
}

fn hash_token_ptr(
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
    let item = Bound::from_borrowed_ptr(py, object_ptr);
    if has_buffer_protocol(&item) {
      return hash_buffer_like(&item);
    }
  }
  Err(py_err_to_type_error(TOKEN_TYPE_ERROR))
}

fn hash_byte_token_ptr(
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
    let item = Bound::from_borrowed_ptr(py, object_ptr);
    if has_buffer_protocol(&item) {
      return hash_buffer_like(&item);
    }
  }
  Err(py_err_to_type_error(BYTE_TOKEN_TYPE_ERROR))
}

#[allow(clippy::too_many_lines)]
fn try_extend_tokens_from_fast_sequence(
  document: &Bound<'_, PyAny>,
  output: &mut Vec<u64>,
) -> PyResult<bool> {
  let py = document.py();
  // SAFETY: checking runtime type and using borrowed sequence APIs under GIL.
  unsafe {
    let object_ptr = document.as_ptr();
    if ffi::PyList_Check(object_ptr) != 0 {
      let length = ffi::PyList_GET_SIZE(object_ptr);
      let length_usize = py_ssize_to_usize(length)?;
      output.reserve(length_usize);
      if length == 0 {
        return Ok(true);
      }
      let first_item = ffi::PyList_GET_ITEM(object_ptr, 0);
      if ffi::PyUnicode_Check(first_item) != 0 {
        let mut index: ffi::Py_ssize_t = 0;
        while index < length {
          let item_ptr = ffi::PyList_GET_ITEM(object_ptr, index);
          if ffi::PyUnicode_Check(item_ptr) != 0 {
            output.push(hash_unicode_ptr(py, item_ptr)?);
          } else {
            output.push(hash_token_ptr(py, item_ptr)?);
          }
          index += 1;
        }
        return Ok(true);
      }
      if ffi::PyBytes_Check(first_item) != 0 {
        let mut index: ffi::Py_ssize_t = 0;
        while index < length {
          let item_ptr = ffi::PyList_GET_ITEM(object_ptr, index);
          if ffi::PyBytes_Check(item_ptr) != 0 {
            output.push(hash_bytes_ptr(py, item_ptr)?);
          } else {
            output.push(hash_token_ptr(py, item_ptr)?);
          }
          index += 1;
        }
        return Ok(true);
      }
      if ffi::PyByteArray_Check(first_item) != 0 {
        let mut index: ffi::Py_ssize_t = 0;
        while index < length {
          let item_ptr = ffi::PyList_GET_ITEM(object_ptr, index);
          if ffi::PyByteArray_Check(item_ptr) != 0 {
            output.push(hash_bytearray_ptr(py, item_ptr)?);
          } else {
            output.push(hash_token_ptr(py, item_ptr)?);
          }
          index += 1;
        }
        return Ok(true);
      }
      let mut index: ffi::Py_ssize_t = 0;
      while index < length {
        let item_ptr = ffi::PyList_GET_ITEM(object_ptr, index);
        output.push(hash_token_ptr(py, item_ptr)?);
        index += 1;
      }
      return Ok(true);
    }

    if ffi::PyTuple_Check(object_ptr) != 0 {
      let length = ffi::PyTuple_GET_SIZE(object_ptr);
      let length_usize = py_ssize_to_usize(length)?;
      output.reserve(length_usize);
      if length == 0 {
        return Ok(true);
      }
      let first_item = ffi::PyTuple_GET_ITEM(object_ptr, 0);
      if ffi::PyUnicode_Check(first_item) != 0 {
        let mut index: ffi::Py_ssize_t = 0;
        while index < length {
          let item_ptr = ffi::PyTuple_GET_ITEM(object_ptr, index);
          if ffi::PyUnicode_Check(item_ptr) != 0 {
            output.push(hash_unicode_ptr(py, item_ptr)?);
          } else {
            output.push(hash_token_ptr(py, item_ptr)?);
          }
          index += 1;
        }
        return Ok(true);
      }
      if ffi::PyBytes_Check(first_item) != 0 {
        let mut index: ffi::Py_ssize_t = 0;
        while index < length {
          let item_ptr = ffi::PyTuple_GET_ITEM(object_ptr, index);
          if ffi::PyBytes_Check(item_ptr) != 0 {
            output.push(hash_bytes_ptr(py, item_ptr)?);
          } else {
            output.push(hash_token_ptr(py, item_ptr)?);
          }
          index += 1;
        }
        return Ok(true);
      }
      if ffi::PyByteArray_Check(first_item) != 0 {
        let mut index: ffi::Py_ssize_t = 0;
        while index < length {
          let item_ptr = ffi::PyTuple_GET_ITEM(object_ptr, index);
          if ffi::PyByteArray_Check(item_ptr) != 0 {
            output.push(hash_bytearray_ptr(py, item_ptr)?);
          } else {
            output.push(hash_token_ptr(py, item_ptr)?);
          }
          index += 1;
        }
        return Ok(true);
      }
      let mut index: ffi::Py_ssize_t = 0;
      while index < length {
        let item_ptr = ffi::PyTuple_GET_ITEM(object_ptr, index);
        output.push(hash_token_ptr(py, item_ptr)?);
        index += 1;
      }
      return Ok(true);
    }
  }
  Ok(false)
}

#[allow(clippy::cast_possible_truncation)]
#[inline]
const fn sampled_index(sample_idx: usize, total: usize, limit: usize) -> usize {
  (((sample_idx as u128 * 2 + 1) * total as u128) / (limit as u128 * 2))
    as usize
}

#[allow(clippy::too_many_lines)]
#[allow(clippy::cast_possible_wrap)]
fn try_extend_tokens_from_fast_sequence_sampled(
  document: &Bound<'_, PyAny>,
  output: &mut Vec<u64>,
  max_tokens: usize,
) -> PyResult<bool> {
  if max_tokens == 0 {
    return Ok(false);
  }
  let py = document.py();
  // SAFETY: checking runtime type and using borrowed sequence APIs under GIL.
  unsafe {
    let object_ptr = document.as_ptr();
    if ffi::PyList_Check(object_ptr) != 0 {
      let length = ffi::PyList_GET_SIZE(object_ptr);
      let length_usize = py_ssize_to_usize(length)?;
      let take_count = length_usize.min(max_tokens);
      output.reserve(take_count);
      if length_usize == 0 {
        return Ok(true);
      }
      if length_usize <= max_tokens {
        return try_extend_tokens_from_fast_sequence(document, output);
      }
      let first_item = ffi::PyList_GET_ITEM(object_ptr, 0);
      if ffi::PyUnicode_Check(first_item) != 0 {
        for sample_idx in 0..max_tokens {
          let index = sampled_index(sample_idx, length_usize, max_tokens);
          let item_ptr =
            ffi::PyList_GET_ITEM(object_ptr, index as ffi::Py_ssize_t);
          if ffi::PyUnicode_Check(item_ptr) != 0 {
            output.push(hash_unicode_ptr(py, item_ptr)?);
          } else {
            output.push(hash_token_ptr(py, item_ptr)?);
          }
        }
      } else if ffi::PyBytes_Check(first_item) != 0 {
        for sample_idx in 0..max_tokens {
          let index = sampled_index(sample_idx, length_usize, max_tokens);
          let item_ptr =
            ffi::PyList_GET_ITEM(object_ptr, index as ffi::Py_ssize_t);
          if ffi::PyBytes_Check(item_ptr) != 0 {
            output.push(hash_bytes_ptr(py, item_ptr)?);
          } else {
            output.push(hash_token_ptr(py, item_ptr)?);
          }
        }
      } else if ffi::PyByteArray_Check(first_item) != 0 {
        for sample_idx in 0..max_tokens {
          let index = sampled_index(sample_idx, length_usize, max_tokens);
          let item_ptr =
            ffi::PyList_GET_ITEM(object_ptr, index as ffi::Py_ssize_t);
          if ffi::PyByteArray_Check(item_ptr) != 0 {
            output.push(hash_bytearray_ptr(py, item_ptr)?);
          } else {
            output.push(hash_token_ptr(py, item_ptr)?);
          }
        }
      } else {
        for sample_idx in 0..max_tokens {
          let index = sampled_index(sample_idx, length_usize, max_tokens);
          let item_ptr =
            ffi::PyList_GET_ITEM(object_ptr, index as ffi::Py_ssize_t);
          output.push(hash_token_ptr(py, item_ptr)?);
        }
      }
      return Ok(true);
    }

    if ffi::PyTuple_Check(object_ptr) != 0 {
      let length = ffi::PyTuple_GET_SIZE(object_ptr);
      let length_usize = py_ssize_to_usize(length)?;
      let take_count = length_usize.min(max_tokens);
      output.reserve(take_count);
      if length_usize == 0 {
        return Ok(true);
      }
      if length_usize <= max_tokens {
        return try_extend_tokens_from_fast_sequence(document, output);
      }
      let first_item = ffi::PyTuple_GET_ITEM(object_ptr, 0);
      if ffi::PyUnicode_Check(first_item) != 0 {
        for sample_idx in 0..max_tokens {
          let index = sampled_index(sample_idx, length_usize, max_tokens);
          let item_ptr =
            ffi::PyTuple_GET_ITEM(object_ptr, index as ffi::Py_ssize_t);
          if ffi::PyUnicode_Check(item_ptr) != 0 {
            output.push(hash_unicode_ptr(py, item_ptr)?);
          } else {
            output.push(hash_token_ptr(py, item_ptr)?);
          }
        }
      } else if ffi::PyBytes_Check(first_item) != 0 {
        for sample_idx in 0..max_tokens {
          let index = sampled_index(sample_idx, length_usize, max_tokens);
          let item_ptr =
            ffi::PyTuple_GET_ITEM(object_ptr, index as ffi::Py_ssize_t);
          if ffi::PyBytes_Check(item_ptr) != 0 {
            output.push(hash_bytes_ptr(py, item_ptr)?);
          } else {
            output.push(hash_token_ptr(py, item_ptr)?);
          }
        }
      } else if ffi::PyByteArray_Check(first_item) != 0 {
        for sample_idx in 0..max_tokens {
          let index = sampled_index(sample_idx, length_usize, max_tokens);
          let item_ptr =
            ffi::PyTuple_GET_ITEM(object_ptr, index as ffi::Py_ssize_t);
          if ffi::PyByteArray_Check(item_ptr) != 0 {
            output.push(hash_bytearray_ptr(py, item_ptr)?);
          } else {
            output.push(hash_token_ptr(py, item_ptr)?);
          }
        }
      } else {
        for sample_idx in 0..max_tokens {
          let index = sampled_index(sample_idx, length_usize, max_tokens);
          let item_ptr =
            ffi::PyTuple_GET_ITEM(object_ptr, index as ffi::Py_ssize_t);
          output.push(hash_token_ptr(py, item_ptr)?);
        }
      }
      return Ok(true);
    }
  }
  Ok(false)
}

fn try_extend_byte_tokens_from_fast_sequence(
  document: &Bound<'_, PyAny>,
  output: &mut Vec<u64>,
) -> PyResult<bool> {
  let py = document.py();
  // SAFETY: checking runtime type and using borrowed sequence APIs under GIL.
  unsafe {
    let object_ptr = document.as_ptr();
    if ffi::PyList_Check(object_ptr) != 0 {
      let length = ffi::PyList_GET_SIZE(object_ptr);
      output.reserve(py_ssize_to_usize(length)?);
      if length == 0 {
        return Ok(true);
      }
      let first_item = ffi::PyList_GET_ITEM(object_ptr, 0);
      if ffi::PyBytes_Check(first_item) != 0 {
        let mut index: ffi::Py_ssize_t = 0;
        while index < length {
          let item_ptr = ffi::PyList_GET_ITEM(object_ptr, index);
          if ffi::PyBytes_Check(item_ptr) != 0 {
            output.push(hash_bytes_ptr(py, item_ptr)?);
          } else {
            output.push(hash_byte_token_ptr(py, item_ptr)?);
          }
          index += 1;
        }
        return Ok(true);
      }
      if ffi::PyByteArray_Check(first_item) != 0 {
        let mut index: ffi::Py_ssize_t = 0;
        while index < length {
          let item_ptr = ffi::PyList_GET_ITEM(object_ptr, index);
          if ffi::PyByteArray_Check(item_ptr) != 0 {
            output.push(hash_bytearray_ptr(py, item_ptr)?);
          } else {
            output.push(hash_byte_token_ptr(py, item_ptr)?);
          }
          index += 1;
        }
        return Ok(true);
      }
      let mut index: ffi::Py_ssize_t = 0;
      while index < length {
        let item_ptr = ffi::PyList_GET_ITEM(object_ptr, index);
        output.push(hash_byte_token_ptr(py, item_ptr)?);
        index += 1;
      }
      return Ok(true);
    }
    if ffi::PyTuple_Check(object_ptr) != 0 {
      let length = ffi::PyTuple_GET_SIZE(object_ptr);
      output.reserve(py_ssize_to_usize(length)?);
      if length == 0 {
        return Ok(true);
      }
      let first_item = ffi::PyTuple_GET_ITEM(object_ptr, 0);
      if ffi::PyBytes_Check(first_item) != 0 {
        let mut index: ffi::Py_ssize_t = 0;
        while index < length {
          let item_ptr = ffi::PyTuple_GET_ITEM(object_ptr, index);
          if ffi::PyBytes_Check(item_ptr) != 0 {
            output.push(hash_bytes_ptr(py, item_ptr)?);
          } else {
            output.push(hash_byte_token_ptr(py, item_ptr)?);
          }
          index += 1;
        }
        return Ok(true);
      }
      if ffi::PyByteArray_Check(first_item) != 0 {
        let mut index: ffi::Py_ssize_t = 0;
        while index < length {
          let item_ptr = ffi::PyTuple_GET_ITEM(object_ptr, index);
          if ffi::PyByteArray_Check(item_ptr) != 0 {
            output.push(hash_bytearray_ptr(py, item_ptr)?);
          } else {
            output.push(hash_byte_token_ptr(py, item_ptr)?);
          }
          index += 1;
        }
        return Ok(true);
      }
      let mut index: ffi::Py_ssize_t = 0;
      while index < length {
        let item_ptr = ffi::PyTuple_GET_ITEM(object_ptr, index);
        output.push(hash_byte_token_ptr(py, item_ptr)?);
        index += 1;
      }
      return Ok(true);
    }
  }
  Ok(false)
}

pub fn hash_token(item: &Bound<'_, PyAny>) -> PyResult<u64> {
  hash_token_ptr(item.py(), item.as_ptr())
}

pub fn hash_single_bufferlike(
  items: &Bound<'_, PyAny>,
) -> PyResult<Option<u64>> {
  // SAFETY: runtime type checks with borrowed pointer under GIL.
  unsafe {
    let object_ptr = items.as_ptr();
    if ffi::PyUnicode_Check(object_ptr) != 0 {
      return Ok(None);
    }
    if ffi::PyBytes_Check(object_ptr) != 0 {
      return hash_bytes_ptr(items.py(), object_ptr).map(Some);
    }
    if ffi::PyByteArray_Check(object_ptr) != 0 {
      return hash_bytearray_ptr(items.py(), object_ptr).map(Some);
    }
  }

  if has_buffer_protocol(items) {
    return hash_buffer_like(items).map(Some);
  }

  Ok(None)
}

/// Visits token hashes from one Python document input.
///
/// Supports:
/// - single bytes-like buffers as a single token
/// - iterables of `str`/bytes-like tokens
pub fn for_each_token_hash<F>(
  document: &Bound<'_, PyAny>,
  mut visitor: F,
) -> PyResult<()>
where
  F: FnMut(u64) -> PyResult<()>,
{
  let mut hashed = Vec::new();
  if try_extend_tokens_from_fast_sequence(document, &mut hashed)? {
    for token_hash in hashed {
      visitor(token_hash)?;
    }
    return Ok(());
  }

  if let Some(single_hash) = hash_single_bufferlike(document)? {
    visitor(single_hash)?;
    return Ok(());
  }

  let iterator = PyIterator::from_object(document)?;
  for item in iterator {
    visitor(hash_token(&item?)?)?;
  }
  Ok(())
}

/// Extends `output` with token hashes from one Python document input.
pub fn extend_token_hashes_from_document(
  document: &Bound<'_, PyAny>,
  output: &mut Vec<u64>,
) -> PyResult<()> {
  if try_extend_tokens_from_fast_sequence(document, output)? {
    return Ok(());
  }

  if let Some(single_hash) = hash_single_bufferlike(document)? {
    output.push(single_hash);
    return Ok(());
  }

  let iterator = PyIterator::from_object(document)?;
  for item in iterator {
    output.push(hash_token(&item?)?);
  }
  Ok(())
}

/// Extends `output` with token hashes from one Python document input with an
/// optional maximum sampled token budget for list/tuple documents.
pub fn extend_token_hashes_from_document_with_limit(
  document: &Bound<'_, PyAny>,
  output: &mut Vec<u64>,
  max_tokens: Option<usize>,
) -> PyResult<()> {
  if let Some(limit) = max_tokens {
    if try_extend_tokens_from_fast_sequence_sampled(document, output, limit)? {
      return Ok(());
    }
  }
  extend_token_hashes_from_document(document, output)
}

/// Returns list/tuple length when the document is a fast sequence.
pub fn fast_sequence_length(
  document: &Bound<'_, PyAny>,
) -> PyResult<Option<usize>> {
  // SAFETY: runtime type checks with borrowed pointer under GIL.
  unsafe {
    let object_ptr = document.as_ptr();
    if ffi::PyList_Check(object_ptr) != 0 {
      return py_ssize_to_usize(ffi::PyList_GET_SIZE(object_ptr)).map(Some);
    }
    if ffi::PyTuple_Check(object_ptr) != 0 {
      return py_ssize_to_usize(ffi::PyTuple_GET_SIZE(object_ptr)).map(Some);
    }
  }
  Ok(None)
}

/// Extends `output` with pre-hashed `u64` token values from one document input.
pub fn extend_prehashed_token_values_from_document(
  document: &Bound<'_, PyAny>,
  output: &mut Vec<u64>,
) -> PyResult<()> {
  if try_extend_prehashed_u64_buffer(document, output)? {
    return Ok(());
  }

  if let Ok(py_list) = document.cast::<PyList>() {
    output.reserve(py_list.len());
    // SAFETY: list access is guarded by CPython list checks and GIL.
    unsafe {
      let object_ptr = document.as_ptr();
      let length = ffi::PyList_GET_SIZE(object_ptr);
      let mut index: ffi::Py_ssize_t = 0;
      while index < length {
        let item_ptr = ffi::PyList_GET_ITEM(object_ptr, index);
        output.push(extract_prehashed_u64_ptr(item_ptr)?);
        index += 1;
      }
    }
    return Ok(());
  }

  if let Ok(py_tuple) = document.cast::<PyTuple>() {
    output.reserve(py_tuple.len());
    // SAFETY: tuple access is guarded by CPython tuple checks and GIL.
    unsafe {
      let object_ptr = document.as_ptr();
      let length = ffi::PyTuple_GET_SIZE(object_ptr);
      let mut index: ffi::Py_ssize_t = 0;
      while index < length {
        let item_ptr = ffi::PyTuple_GET_ITEM(object_ptr, index);
        output.push(extract_prehashed_u64_ptr(item_ptr)?);
        index += 1;
      }
    }
    return Ok(());
  }

  let iterator = PyIterator::from_object(document)?;
  for item in iterator {
    let item = item?;
    output.push(extract_prehashed_u64_ptr(item.as_ptr())?);
  }
  Ok(())
}

/// Extends `output` with token hashes from byte-only tokens.
///
/// Supports single bytes-like buffers and iterables of bytes-like buffers.
pub fn extend_byte_token_hashes_from_document(
  document: &Bound<'_, PyAny>,
  output: &mut Vec<u64>,
) -> PyResult<()> {
  if try_extend_byte_tokens_from_fast_sequence(document, output)? {
    return Ok(());
  }

  // SAFETY: runtime type checks with borrowed pointer under GIL.
  unsafe {
    let object_ptr = document.as_ptr();
    if ffi::PyBytes_Check(object_ptr) != 0 {
      output.push(hash_bytes_ptr(document.py(), object_ptr)?);
      return Ok(());
    }
    if ffi::PyByteArray_Check(object_ptr) != 0 {
      output.push(hash_bytearray_ptr(document.py(), object_ptr)?);
      return Ok(());
    }
  }
  if has_buffer_protocol(document) {
    output.push(hash_buffer_like(document)?);
    return Ok(());
  }

  let iterator = PyIterator::from_object(document)?;
  for item in iterator {
    let item = item?;
    output.push(hash_byte_token_ptr(item.py(), item.as_ptr())?);
  }
  Ok(())
}
