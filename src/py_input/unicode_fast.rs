use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::PyString;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::OnceLock;

const MAX_PLAUSIBLE_DATA_OFFSET: usize = 512;

#[derive(Clone, Copy)]
pub struct UnicodeFastPath {
  state_mask: u32,
  state_value: u32,
  data_offset: usize,
}

static FAST_PATH: OnceLock<Option<UnicodeFastPath>> = OnceLock::new();

pub fn init(py: Python<'_>) {
  let _ = FAST_PATH.get_or_init(|| calibrate(py));
}

#[inline]
fn fast_path() -> Option<&'static UnicodeFastPath> {
  FAST_PATH.get().and_then(Option::as_ref)
}

struct ProbeFacts {
  state: u32,
  utf8_offset: Option<usize>,
  length_matches_bytes: bool,
}

fn probe_string(string: &Bound<'_, PyString>) -> PyResult<ProbeFacts> {
  let object_ptr = string.as_ptr();
  let mut utf8_len: ffi::Py_ssize_t = 0;
  let utf8_ptr =
    unsafe { ffi::PyUnicode_AsUTF8AndSize(object_ptr, &raw mut utf8_len) };
  if utf8_ptr.is_null() {
    return Err(PyErr::fetch(string.py()));
  }

  let (state, length) = unsafe {
    let ascii_object = object_ptr.cast::<ffi::PyASCIIObject>();
    (
      std::ptr::read_volatile(&raw const (*ascii_object).state),
      std::ptr::read_volatile(&raw const (*ascii_object).length),
    )
  };

  let base_address = object_ptr as usize;
  let utf8_address = utf8_ptr as usize;
  let utf8_offset = utf8_address
    .checked_sub(base_address)
    .filter(|&offset| offset > 0 && offset <= MAX_PLAUSIBLE_DATA_OFFSET);

  Ok(ProbeFacts {
    state,
    utf8_offset,
    length_matches_bytes: length == utf8_len,
  })
}

fn calibrate(py: Python<'_>) -> Option<UnicodeFastPath> {
  if std::env::var("RENSA_ASCII_FAST_PATH").is_ok_and(|value| value == "0") {
    return None;
  }
  let implementation_is_cpython = py
    .import("sys")
    .and_then(|sys| sys.getattr("implementation"))
    .and_then(|implementation| implementation.getattr("name"))
    .and_then(|name| name.extract::<String>())
    .is_ok_and(|name| name == "cpython");
  if !implementation_is_cpython {
    return None;
  }

  let ascii_probes = [
    PyString::new(py, "rensa ascii fast path calibration probe"),
    PyString::intern(py, "rensa_ascii_probe_interned"),
    PyString::new(py, "z"),
    PyString::new(py, ""),
  ];
  let non_ascii_probes = [
    PyString::new(py, "caf\u{e9}"),
    PyString::new(py, "\u{3c0}\u{3c1}"),
    PyString::new(py, "\u{1f600}"),
  ];

  let mut ascii_facts = Vec::with_capacity(ascii_probes.len());
  for probe in &ascii_probes {
    ascii_facts.push(probe_string(probe).ok()?);
  }

  let data_offset = ascii_facts[0].utf8_offset?;
  for facts in &ascii_facts {
    if facts.utf8_offset != Some(data_offset) || !facts.length_matches_bytes {
      return None;
    }
  }

  let base_state = ascii_facts[0].state;
  let mut varying_bits = 0u32;
  for facts in &ascii_facts {
    varying_bits |= facts.state ^ base_state;
  }
  let state_mask = !varying_bits;
  let state_value = base_state & state_mask;
  if state_mask == 0 {
    return None;
  }

  for probe in &non_ascii_probes {
    let facts = probe_string(probe).ok()?;
    if facts.state & state_mask == state_value {
      return None;
    }
  }

  Some(UnicodeFastPath {
    state_mask,
    state_value,
    data_offset,
  })
}

#[allow(clippy::cast_sign_loss)]
#[allow(clippy::inline_always)]
#[inline(always)]
pub unsafe fn compact_ascii_bytes(
  object_ptr: *mut ffi::PyObject,
) -> Option<(*const u8, usize)> {
  let fast_path = fast_path()?;
  unsafe {
    let ascii_object = object_ptr.cast::<ffi::PyASCIIObject>();
    let state_ptr = (&raw const (*ascii_object).state).cast::<AtomicU32>();
    let state = (*state_ptr).load(Ordering::Relaxed);
    if state & fast_path.state_mask != fast_path.state_value {
      return None;
    }
    let length = std::ptr::read(&raw const (*ascii_object).length);
    debug_assert!(length >= 0);
    Some((
      object_ptr.cast::<u8>().add(fast_path.data_offset),
      length as usize,
    ))
  }
}
