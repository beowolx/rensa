use crate::py_input::convert::py_ssize_to_usize;
use crate::py_input::ptr_hash::{
  hash_byte_token_ptr, hash_bytearray_ptr, hash_bytes_ptr, hash_token_ptr,
  hash_unicode_ptr,
};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::PyAny;

#[derive(Clone, Copy)]
enum FastSequenceKind {
  List,
  Tuple,
}

impl FastSequenceKind {
  #[inline]
  unsafe fn from_object_ptr(object_ptr: *mut ffi::PyObject) -> Option<Self> {
    if ffi::PyList_Check(object_ptr) != 0 {
      return Some(Self::List);
    }
    if ffi::PyTuple_Check(object_ptr) != 0 {
      return Some(Self::Tuple);
    }
    None
  }
}

#[derive(Clone, Copy)]
enum TokenHashMode {
  Unicode,
  Bytes,
  ByteArray,
  Generic,
}

impl TokenHashMode {
  #[inline]
  unsafe fn from_first_item(first_item: *mut ffi::PyObject) -> Self {
    if ffi::PyUnicode_Check(first_item) != 0 {
      return Self::Unicode;
    }
    if ffi::PyBytes_Check(first_item) != 0 {
      return Self::Bytes;
    }
    if ffi::PyByteArray_Check(first_item) != 0 {
      return Self::ByteArray;
    }
    Self::Generic
  }
}

#[derive(Clone, Copy)]
enum ByteTokenHashMode {
  Bytes,
  ByteArray,
  Generic,
}

impl ByteTokenHashMode {
  #[inline]
  unsafe fn from_first_item(first_item: *mut ffi::PyObject) -> Self {
    if ffi::PyBytes_Check(first_item) != 0 {
      return Self::Bytes;
    }
    if ffi::PyByteArray_Check(first_item) != 0 {
      return Self::ByteArray;
    }
    Self::Generic
  }
}

pub fn try_extend_tokens_from_fast_sequence(
  document: &Bound<'_, PyAny>,
  output: &mut Vec<u64>,
) -> PyResult<bool> {
  let py = document.py();
  // SAFETY: runtime type checks and borrowed sequence access happen under GIL.
  unsafe {
    let object_ptr = document.as_ptr();
    let Some(kind) = FastSequenceKind::from_object_ptr(object_ptr) else {
      return Ok(false);
    };
    extend_tokens_from_sequence(py, kind, object_ptr, output)?;
  }
  Ok(true)
}

pub fn try_for_each_token_hash_from_fast_sequence<F>(
  document: &Bound<'_, PyAny>,
  visitor: &mut F,
) -> PyResult<bool>
where
  F: FnMut(u64) -> PyResult<()>,
{
  let py = document.py();
  // SAFETY: runtime type checks and borrowed sequence access happen under GIL.
  unsafe {
    let object_ptr = document.as_ptr();
    let Some(kind) = FastSequenceKind::from_object_ptr(object_ptr) else {
      return Ok(false);
    };
    for_each_token_hash_in_sequence(py, kind, object_ptr, visitor)?;
  }
  Ok(true)
}

unsafe fn for_each_token_hash_in_sequence<F>(
  py: Python<'_>,
  kind: FastSequenceKind,
  object_ptr: *mut ffi::PyObject,
  visitor: &mut F,
) -> PyResult<()>
where
  F: FnMut(u64) -> PyResult<()>,
{
  match kind {
    FastSequenceKind::List => {
      for_each_token_hash_in_list(py, object_ptr, visitor)
    }
    FastSequenceKind::Tuple => {
      for_each_token_hash_in_tuple(py, object_ptr, visitor)
    }
  }
}

unsafe fn for_each_token_hash_in_list<F>(
  py: Python<'_>,
  object_ptr: *mut ffi::PyObject,
  visitor: &mut F,
) -> PyResult<()>
where
  F: FnMut(u64) -> PyResult<()>,
{
  let length = ffi::PyList_GET_SIZE(object_ptr);
  if length == 0 {
    return Ok(());
  }

  let first_item = ffi::PyList_GET_ITEM(object_ptr, 0);
  let mode = TokenHashMode::from_first_item(first_item);
  let first_type_ptr = ffi::Py_TYPE(first_item);

  let mut index: ffi::Py_ssize_t = 0;
  match mode {
    TokenHashMode::Unicode => {
      while index < length {
        let item_ptr = ffi::PyList_GET_ITEM(object_ptr, index);
        let item_type_ptr = ffi::Py_TYPE(item_ptr);
        visitor(if item_type_ptr == first_type_ptr {
          hash_unicode_ptr(py, item_ptr)?
        } else {
          hash_token_ptr(py, item_ptr)?
        })?;
        index += 1;
      }
    }
    TokenHashMode::Bytes => {
      while index < length {
        let item_ptr = ffi::PyList_GET_ITEM(object_ptr, index);
        let item_type_ptr = ffi::Py_TYPE(item_ptr);
        visitor(if item_type_ptr == first_type_ptr {
          hash_bytes_ptr(py, item_ptr)?
        } else {
          hash_token_ptr(py, item_ptr)?
        })?;
        index += 1;
      }
    }
    TokenHashMode::ByteArray => {
      while index < length {
        let item_ptr = ffi::PyList_GET_ITEM(object_ptr, index);
        let item_type_ptr = ffi::Py_TYPE(item_ptr);
        visitor(if item_type_ptr == first_type_ptr {
          hash_bytearray_ptr(py, item_ptr)?
        } else {
          hash_token_ptr(py, item_ptr)?
        })?;
        index += 1;
      }
    }
    TokenHashMode::Generic => {
      while index < length {
        let item_ptr = ffi::PyList_GET_ITEM(object_ptr, index);
        visitor(hash_token_ptr(py, item_ptr)?)?;
        index += 1;
      }
    }
  }

  Ok(())
}

unsafe fn for_each_token_hash_in_tuple<F>(
  py: Python<'_>,
  object_ptr: *mut ffi::PyObject,
  visitor: &mut F,
) -> PyResult<()>
where
  F: FnMut(u64) -> PyResult<()>,
{
  let length = ffi::PyTuple_GET_SIZE(object_ptr);
  if length == 0 {
    return Ok(());
  }

  let first_item = ffi::PyTuple_GET_ITEM(object_ptr, 0);
  let mode = TokenHashMode::from_first_item(first_item);
  let first_type_ptr = ffi::Py_TYPE(first_item);

  let mut index: ffi::Py_ssize_t = 0;
  match mode {
    TokenHashMode::Unicode => {
      while index < length {
        let item_ptr = ffi::PyTuple_GET_ITEM(object_ptr, index);
        let item_type_ptr = ffi::Py_TYPE(item_ptr);
        visitor(if item_type_ptr == first_type_ptr {
          hash_unicode_ptr(py, item_ptr)?
        } else {
          hash_token_ptr(py, item_ptr)?
        })?;
        index += 1;
      }
    }
    TokenHashMode::Bytes => {
      while index < length {
        let item_ptr = ffi::PyTuple_GET_ITEM(object_ptr, index);
        let item_type_ptr = ffi::Py_TYPE(item_ptr);
        visitor(if item_type_ptr == first_type_ptr {
          hash_bytes_ptr(py, item_ptr)?
        } else {
          hash_token_ptr(py, item_ptr)?
        })?;
        index += 1;
      }
    }
    TokenHashMode::ByteArray => {
      while index < length {
        let item_ptr = ffi::PyTuple_GET_ITEM(object_ptr, index);
        let item_type_ptr = ffi::Py_TYPE(item_ptr);
        visitor(if item_type_ptr == first_type_ptr {
          hash_bytearray_ptr(py, item_ptr)?
        } else {
          hash_token_ptr(py, item_ptr)?
        })?;
        index += 1;
      }
    }
    TokenHashMode::Generic => {
      while index < length {
        let item_ptr = ffi::PyTuple_GET_ITEM(object_ptr, index);
        visitor(hash_token_ptr(py, item_ptr)?)?;
        index += 1;
      }
    }
  }

  Ok(())
}

unsafe fn extend_tokens_from_sequence(
  py: Python<'_>,
  kind: FastSequenceKind,
  object_ptr: *mut ffi::PyObject,
  output: &mut Vec<u64>,
) -> PyResult<()> {
  match kind {
    FastSequenceKind::List => extend_tokens_from_list(py, object_ptr, output),
    FastSequenceKind::Tuple => extend_tokens_from_tuple(py, object_ptr, output),
  }
}

unsafe fn extend_tokens_from_list(
  py: Python<'_>,
  object_ptr: *mut ffi::PyObject,
  output: &mut Vec<u64>,
) -> PyResult<()> {
  let length = ffi::PyList_GET_SIZE(object_ptr);
  output.reserve(py_ssize_to_usize(length)?);
  if length == 0 {
    return Ok(());
  }

  let first_item = ffi::PyList_GET_ITEM(object_ptr, 0);
  let mode = TokenHashMode::from_first_item(first_item);
  let first_type_ptr = ffi::Py_TYPE(first_item);

  let mut index: ffi::Py_ssize_t = 0;
  match mode {
    TokenHashMode::Unicode => {
      while index < length {
        let item_ptr = ffi::PyList_GET_ITEM(object_ptr, index);
        let item_type_ptr = ffi::Py_TYPE(item_ptr);
        output.push(if item_type_ptr == first_type_ptr {
          hash_unicode_ptr(py, item_ptr)?
        } else {
          hash_token_ptr(py, item_ptr)?
        });
        index += 1;
      }
    }
    TokenHashMode::Bytes => {
      while index < length {
        let item_ptr = ffi::PyList_GET_ITEM(object_ptr, index);
        let item_type_ptr = ffi::Py_TYPE(item_ptr);
        output.push(if item_type_ptr == first_type_ptr {
          hash_bytes_ptr(py, item_ptr)?
        } else {
          hash_token_ptr(py, item_ptr)?
        });
        index += 1;
      }
    }
    TokenHashMode::ByteArray => {
      while index < length {
        let item_ptr = ffi::PyList_GET_ITEM(object_ptr, index);
        let item_type_ptr = ffi::Py_TYPE(item_ptr);
        output.push(if item_type_ptr == first_type_ptr {
          hash_bytearray_ptr(py, item_ptr)?
        } else {
          hash_token_ptr(py, item_ptr)?
        });
        index += 1;
      }
    }
    TokenHashMode::Generic => {
      while index < length {
        let item_ptr = ffi::PyList_GET_ITEM(object_ptr, index);
        output.push(hash_token_ptr(py, item_ptr)?);
        index += 1;
      }
    }
  }

  Ok(())
}

unsafe fn extend_tokens_from_tuple(
  py: Python<'_>,
  object_ptr: *mut ffi::PyObject,
  output: &mut Vec<u64>,
) -> PyResult<()> {
  let length = ffi::PyTuple_GET_SIZE(object_ptr);
  output.reserve(py_ssize_to_usize(length)?);
  if length == 0 {
    return Ok(());
  }

  let first_item = ffi::PyTuple_GET_ITEM(object_ptr, 0);
  let mode = TokenHashMode::from_first_item(first_item);
  let first_type_ptr = ffi::Py_TYPE(first_item);

  let mut index: ffi::Py_ssize_t = 0;
  match mode {
    TokenHashMode::Unicode => {
      while index < length {
        let item_ptr = ffi::PyTuple_GET_ITEM(object_ptr, index);
        let item_type_ptr = ffi::Py_TYPE(item_ptr);
        output.push(if item_type_ptr == first_type_ptr {
          hash_unicode_ptr(py, item_ptr)?
        } else {
          hash_token_ptr(py, item_ptr)?
        });
        index += 1;
      }
    }
    TokenHashMode::Bytes => {
      while index < length {
        let item_ptr = ffi::PyTuple_GET_ITEM(object_ptr, index);
        let item_type_ptr = ffi::Py_TYPE(item_ptr);
        output.push(if item_type_ptr == first_type_ptr {
          hash_bytes_ptr(py, item_ptr)?
        } else {
          hash_token_ptr(py, item_ptr)?
        });
        index += 1;
      }
    }
    TokenHashMode::ByteArray => {
      while index < length {
        let item_ptr = ffi::PyTuple_GET_ITEM(object_ptr, index);
        let item_type_ptr = ffi::Py_TYPE(item_ptr);
        output.push(if item_type_ptr == first_type_ptr {
          hash_bytearray_ptr(py, item_ptr)?
        } else {
          hash_token_ptr(py, item_ptr)?
        });
        index += 1;
      }
    }
    TokenHashMode::Generic => {
      while index < length {
        let item_ptr = ffi::PyTuple_GET_ITEM(object_ptr, index);
        output.push(hash_token_ptr(py, item_ptr)?);
        index += 1;
      }
    }
  }

  Ok(())
}

#[derive(Clone, Copy)]
struct MidpointSampler {
  q: usize,
  r: usize,
  step_div: usize,
  step_mod: usize,
  denom: usize,
}

impl MidpointSampler {
  #[inline]
  fn new(total: usize, limit: usize) -> Self {
    debug_assert!(limit > 0);
    debug_assert!(total >= limit);

    let denom = limit * 2;
    let total_div = total / limit;
    let total_rem = total - total_div * limit;
    let q = total_div / 2;
    let r = if (total_div & 1) == 0 {
      total_rem
    } else {
      limit + total_rem
    };
    let step_div = total_div;
    let step_mod = total_rem * 2;

    Self {
      q,
      r,
      step_div,
      step_mod,
      denom,
    }
  }

  #[inline]
  const fn next(&mut self) -> usize {
    let index = self.q;
    self.r += self.step_mod;
    self.q += self.step_div;
    if self.r >= self.denom {
      self.r -= self.denom;
      self.q += 1;
    }
    index
  }
}

pub fn try_extend_tokens_from_fast_sequence_sampled(
  document: &Bound<'_, PyAny>,
  output: &mut Vec<u64>,
  max_tokens: usize,
) -> PyResult<bool> {
  let py = document.py();
  // SAFETY: runtime type checks and borrowed sequence access happen under GIL.
  unsafe {
    let object_ptr = document.as_ptr();
    let Some(kind) = FastSequenceKind::from_object_ptr(object_ptr) else {
      return Ok(false);
    };
    extend_tokens_from_sequence_sampled(
      py, kind, object_ptr, output, max_tokens,
    )?;
  }
  Ok(true)
}

unsafe fn extend_tokens_from_sequence_sampled(
  py: Python<'_>,
  kind: FastSequenceKind,
  object_ptr: *mut ffi::PyObject,
  output: &mut Vec<u64>,
  max_tokens: usize,
) -> PyResult<()> {
  if max_tokens == 0 {
    return Ok(());
  }

  match kind {
    FastSequenceKind::List => {
      extend_tokens_from_list_sampled(py, object_ptr, output, max_tokens)
    }
    FastSequenceKind::Tuple => {
      extend_tokens_from_tuple_sampled(py, object_ptr, output, max_tokens)
    }
  }
}

unsafe fn extend_tokens_from_list_sampled(
  py: Python<'_>,
  object_ptr: *mut ffi::PyObject,
  output: &mut Vec<u64>,
  max_tokens: usize,
) -> PyResult<()> {
  let length = ffi::PyList_GET_SIZE(object_ptr);
  let length_usize = py_ssize_to_usize(length)?;
  output.reserve(length_usize.min(max_tokens));
  if length_usize == 0 {
    return Ok(());
  }
  if length_usize <= max_tokens {
    return extend_tokens_from_list(py, object_ptr, output);
  }

  let first_item = ffi::PyList_GET_ITEM(object_ptr, 0);
  let mode = TokenHashMode::from_first_item(first_item);
  let first_type_ptr = ffi::Py_TYPE(first_item);
  let mut sampler = MidpointSampler::new(length_usize, max_tokens);
  match mode {
    TokenHashMode::Unicode => {
      for _ in 0..max_tokens {
        let index = sampler.next();
        debug_assert!(index <= ffi::Py_ssize_t::MAX as usize);
        #[allow(clippy::cast_possible_wrap)]
        let index_ssize = index as ffi::Py_ssize_t;
        let item_ptr = ffi::PyList_GET_ITEM(object_ptr, index_ssize);
        let item_type_ptr = ffi::Py_TYPE(item_ptr);
        output.push(if item_type_ptr == first_type_ptr {
          hash_unicode_ptr(py, item_ptr)?
        } else {
          hash_token_ptr(py, item_ptr)?
        });
      }
    }
    TokenHashMode::Bytes => {
      for _ in 0..max_tokens {
        let index = sampler.next();
        debug_assert!(index <= ffi::Py_ssize_t::MAX as usize);
        #[allow(clippy::cast_possible_wrap)]
        let index_ssize = index as ffi::Py_ssize_t;
        let item_ptr = ffi::PyList_GET_ITEM(object_ptr, index_ssize);
        let item_type_ptr = ffi::Py_TYPE(item_ptr);
        output.push(if item_type_ptr == first_type_ptr {
          hash_bytes_ptr(py, item_ptr)?
        } else {
          hash_token_ptr(py, item_ptr)?
        });
      }
    }
    TokenHashMode::ByteArray => {
      for _ in 0..max_tokens {
        let index = sampler.next();
        debug_assert!(index <= ffi::Py_ssize_t::MAX as usize);
        #[allow(clippy::cast_possible_wrap)]
        let index_ssize = index as ffi::Py_ssize_t;
        let item_ptr = ffi::PyList_GET_ITEM(object_ptr, index_ssize);
        let item_type_ptr = ffi::Py_TYPE(item_ptr);
        output.push(if item_type_ptr == first_type_ptr {
          hash_bytearray_ptr(py, item_ptr)?
        } else {
          hash_token_ptr(py, item_ptr)?
        });
      }
    }
    TokenHashMode::Generic => {
      for _ in 0..max_tokens {
        let index = sampler.next();
        debug_assert!(index <= ffi::Py_ssize_t::MAX as usize);
        #[allow(clippy::cast_possible_wrap)]
        let index_ssize = index as ffi::Py_ssize_t;
        let item_ptr = ffi::PyList_GET_ITEM(object_ptr, index_ssize);
        output.push(hash_token_ptr(py, item_ptr)?);
      }
    }
  }

  Ok(())
}

unsafe fn extend_tokens_from_tuple_sampled(
  py: Python<'_>,
  object_ptr: *mut ffi::PyObject,
  output: &mut Vec<u64>,
  max_tokens: usize,
) -> PyResult<()> {
  let length = ffi::PyTuple_GET_SIZE(object_ptr);
  let length_usize = py_ssize_to_usize(length)?;
  output.reserve(length_usize.min(max_tokens));
  if length_usize == 0 {
    return Ok(());
  }
  if length_usize <= max_tokens {
    return extend_tokens_from_tuple(py, object_ptr, output);
  }

  let first_item = ffi::PyTuple_GET_ITEM(object_ptr, 0);
  let mode = TokenHashMode::from_first_item(first_item);
  let first_type_ptr = ffi::Py_TYPE(first_item);
  let mut sampler = MidpointSampler::new(length_usize, max_tokens);
  match mode {
    TokenHashMode::Unicode => {
      for _ in 0..max_tokens {
        let index = sampler.next();
        debug_assert!(index <= ffi::Py_ssize_t::MAX as usize);
        #[allow(clippy::cast_possible_wrap)]
        let index_ssize = index as ffi::Py_ssize_t;
        let item_ptr = ffi::PyTuple_GET_ITEM(object_ptr, index_ssize);
        let item_type_ptr = ffi::Py_TYPE(item_ptr);
        output.push(if item_type_ptr == first_type_ptr {
          hash_unicode_ptr(py, item_ptr)?
        } else {
          hash_token_ptr(py, item_ptr)?
        });
      }
    }
    TokenHashMode::Bytes => {
      for _ in 0..max_tokens {
        let index = sampler.next();
        debug_assert!(index <= ffi::Py_ssize_t::MAX as usize);
        #[allow(clippy::cast_possible_wrap)]
        let index_ssize = index as ffi::Py_ssize_t;
        let item_ptr = ffi::PyTuple_GET_ITEM(object_ptr, index_ssize);
        let item_type_ptr = ffi::Py_TYPE(item_ptr);
        output.push(if item_type_ptr == first_type_ptr {
          hash_bytes_ptr(py, item_ptr)?
        } else {
          hash_token_ptr(py, item_ptr)?
        });
      }
    }
    TokenHashMode::ByteArray => {
      for _ in 0..max_tokens {
        let index = sampler.next();
        debug_assert!(index <= ffi::Py_ssize_t::MAX as usize);
        #[allow(clippy::cast_possible_wrap)]
        let index_ssize = index as ffi::Py_ssize_t;
        let item_ptr = ffi::PyTuple_GET_ITEM(object_ptr, index_ssize);
        let item_type_ptr = ffi::Py_TYPE(item_ptr);
        output.push(if item_type_ptr == first_type_ptr {
          hash_bytearray_ptr(py, item_ptr)?
        } else {
          hash_token_ptr(py, item_ptr)?
        });
      }
    }
    TokenHashMode::Generic => {
      for _ in 0..max_tokens {
        let index = sampler.next();
        debug_assert!(index <= ffi::Py_ssize_t::MAX as usize);
        #[allow(clippy::cast_possible_wrap)]
        let index_ssize = index as ffi::Py_ssize_t;
        let item_ptr = ffi::PyTuple_GET_ITEM(object_ptr, index_ssize);
        output.push(hash_token_ptr(py, item_ptr)?);
      }
    }
  }

  Ok(())
}

pub fn try_extend_byte_tokens_from_fast_sequence(
  document: &Bound<'_, PyAny>,
  output: &mut Vec<u64>,
) -> PyResult<bool> {
  let py = document.py();
  // SAFETY: runtime type checks and borrowed sequence access happen under GIL.
  unsafe {
    let object_ptr = document.as_ptr();
    let Some(kind) = FastSequenceKind::from_object_ptr(object_ptr) else {
      return Ok(false);
    };
    extend_byte_tokens_from_sequence(py, kind, object_ptr, output)?;
  }
  Ok(true)
}

unsafe fn extend_byte_tokens_from_sequence(
  py: Python<'_>,
  kind: FastSequenceKind,
  object_ptr: *mut ffi::PyObject,
  output: &mut Vec<u64>,
) -> PyResult<()> {
  match kind {
    FastSequenceKind::List => {
      extend_byte_tokens_from_list(py, object_ptr, output)
    }
    FastSequenceKind::Tuple => {
      extend_byte_tokens_from_tuple(py, object_ptr, output)
    }
  }
}

unsafe fn extend_byte_tokens_from_list(
  py: Python<'_>,
  object_ptr: *mut ffi::PyObject,
  output: &mut Vec<u64>,
) -> PyResult<()> {
  let length = ffi::PyList_GET_SIZE(object_ptr);
  output.reserve(py_ssize_to_usize(length)?);
  if length == 0 {
    return Ok(());
  }

  let first_item = ffi::PyList_GET_ITEM(object_ptr, 0);
  let mode = ByteTokenHashMode::from_first_item(first_item);
  let first_type_ptr = ffi::Py_TYPE(first_item);

  let mut index: ffi::Py_ssize_t = 0;
  match mode {
    ByteTokenHashMode::Bytes => {
      while index < length {
        let item_ptr = ffi::PyList_GET_ITEM(object_ptr, index);
        let item_type_ptr = ffi::Py_TYPE(item_ptr);
        output.push(if item_type_ptr == first_type_ptr {
          hash_bytes_ptr(py, item_ptr)?
        } else {
          hash_byte_token_ptr(py, item_ptr)?
        });
        index += 1;
      }
    }
    ByteTokenHashMode::ByteArray => {
      while index < length {
        let item_ptr = ffi::PyList_GET_ITEM(object_ptr, index);
        let item_type_ptr = ffi::Py_TYPE(item_ptr);
        output.push(if item_type_ptr == first_type_ptr {
          hash_bytearray_ptr(py, item_ptr)?
        } else {
          hash_byte_token_ptr(py, item_ptr)?
        });
        index += 1;
      }
    }
    ByteTokenHashMode::Generic => {
      while index < length {
        let item_ptr = ffi::PyList_GET_ITEM(object_ptr, index);
        output.push(hash_byte_token_ptr(py, item_ptr)?);
        index += 1;
      }
    }
  }

  Ok(())
}

unsafe fn extend_byte_tokens_from_tuple(
  py: Python<'_>,
  object_ptr: *mut ffi::PyObject,
  output: &mut Vec<u64>,
) -> PyResult<()> {
  let length = ffi::PyTuple_GET_SIZE(object_ptr);
  output.reserve(py_ssize_to_usize(length)?);
  if length == 0 {
    return Ok(());
  }

  let first_item = ffi::PyTuple_GET_ITEM(object_ptr, 0);
  let mode = ByteTokenHashMode::from_first_item(first_item);
  let first_type_ptr = ffi::Py_TYPE(first_item);

  let mut index: ffi::Py_ssize_t = 0;
  match mode {
    ByteTokenHashMode::Bytes => {
      while index < length {
        let item_ptr = ffi::PyTuple_GET_ITEM(object_ptr, index);
        let item_type_ptr = ffi::Py_TYPE(item_ptr);
        output.push(if item_type_ptr == first_type_ptr {
          hash_bytes_ptr(py, item_ptr)?
        } else {
          hash_byte_token_ptr(py, item_ptr)?
        });
        index += 1;
      }
    }
    ByteTokenHashMode::ByteArray => {
      while index < length {
        let item_ptr = ffi::PyTuple_GET_ITEM(object_ptr, index);
        let item_type_ptr = ffi::Py_TYPE(item_ptr);
        output.push(if item_type_ptr == first_type_ptr {
          hash_bytearray_ptr(py, item_ptr)?
        } else {
          hash_byte_token_ptr(py, item_ptr)?
        });
        index += 1;
      }
    }
    ByteTokenHashMode::Generic => {
      while index < length {
        let item_ptr = ffi::PyTuple_GET_ITEM(object_ptr, index);
        output.push(hash_byte_token_ptr(py, item_ptr)?);
        index += 1;
      }
    }
  }

  Ok(())
}

pub fn fast_sequence_length(
  document: &Bound<'_, PyAny>,
) -> PyResult<Option<usize>> {
  // SAFETY: runtime type checks with borrowed pointer under GIL.
  unsafe {
    let object_ptr = document.as_ptr();
    let Some(kind) = FastSequenceKind::from_object_ptr(object_ptr) else {
      return Ok(None);
    };
    let length = match kind {
      FastSequenceKind::List => ffi::PyList_GET_SIZE(object_ptr),
      FastSequenceKind::Tuple => ffi::PyTuple_GET_SIZE(object_ptr),
    };
    py_ssize_to_usize(length).map(Some)
  }
}
