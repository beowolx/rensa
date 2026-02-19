use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyIterator, PyList, PyTuple};

mod buffer;
mod convert;
mod fast_sequence;
mod prehashed;
mod ptr_hash;

const TOKEN_TYPE_ERROR: &str =
  "each item must be str, bytes, bytearray, or a C-contiguous u8 buffer";
const BYTE_TOKEN_TYPE_ERROR: &str =
  "each item must be bytes, bytearray, or a C-contiguous u8 buffer";
const PREHASHED_TOKEN_TYPE_ERROR: &str =
  "each item must be an unsigned 64-bit integer";
const BUFFER_TYPE_ERROR: &str =
  "buffer inputs must be C-contiguous and byte-sized (u8)";
const SEQUENCE_SIZE_ERROR: &str = "sequence size does not fit in usize";

pub fn hash_token(item: &Bound<'_, PyAny>) -> PyResult<u64> {
  ptr_hash::hash_token_ptr(item.py(), item.as_ptr())
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
      return ptr_hash::hash_bytes_ptr(items.py(), object_ptr).map(Some);
    }
    if ffi::PyByteArray_Check(object_ptr) != 0 {
      return ptr_hash::hash_bytearray_ptr(items.py(), object_ptr).map(Some);
    }
  }

  if buffer::has_buffer_protocol(items) {
    return buffer::hash_buffer_like(items).map(Some);
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
  if fast_sequence::try_for_each_token_hash_from_fast_sequence(
    document,
    &mut visitor,
  )? {
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
  if fast_sequence::try_extend_tokens_from_fast_sequence(document, output)? {
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
    if fast_sequence::try_extend_tokens_from_fast_sequence_sampled(
      document, output, limit,
    )? {
      return Ok(());
    }
  }
  extend_token_hashes_from_document(document, output)
}

/// Returns list/tuple length when the document is a fast sequence.
pub fn fast_sequence_length(
  document: &Bound<'_, PyAny>,
) -> PyResult<Option<usize>> {
  fast_sequence::fast_sequence_length(document)
}

/// Extends `output` with pre-hashed `u64` token values from one document input.
pub fn extend_prehashed_token_values_from_document(
  document: &Bound<'_, PyAny>,
  output: &mut Vec<u64>,
) -> PyResult<()> {
  if prehashed::try_extend_prehashed_u64_buffer(document, output)? {
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
        output.push(prehashed::extract_prehashed_u64_ptr(item_ptr)?);
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
        output.push(prehashed::extract_prehashed_u64_ptr(item_ptr)?);
        index += 1;
      }
    }
    return Ok(());
  }

  let iterator = PyIterator::from_object(document)?;
  for item in iterator {
    let item = item?;
    output.push(prehashed::extract_prehashed_u64_ptr(item.as_ptr())?);
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
  if fast_sequence::try_extend_byte_tokens_from_fast_sequence(document, output)?
  {
    return Ok(());
  }

  if let Some(single_hash) = hash_single_bufferlike(document)? {
    output.push(single_hash);
    return Ok(());
  }

  let iterator = PyIterator::from_object(document)?;
  for item in iterator {
    let item = item?;
    output.push(ptr_hash::hash_byte_token_ptr(item.py(), item.as_ptr())?);
  }
  Ok(())
}
