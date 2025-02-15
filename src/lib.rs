#![deny(clippy::all)]
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]
#![allow(clippy::unsafe_derive_deserialize)]
#![allow(clippy::cast_precision_loss)]

mod cminhash;
mod lsh;
mod rminhash;
mod utils;

use pyo3::prelude::*;

/// Python module for `MinHash` and LSH implementations
///
/// # Errors
/// Returns an error if the module initialization fails or classes cannot be added
#[pymodule]
pub fn rensa(m: &Bound<'_, PyModule>) -> PyResult<()> {
  m.add_class::<rminhash::RMinHash>()?;
  m.add_class::<cminhash::CMinHash>()?;
  m.add_class::<lsh::RMinHashLSH>()?;
  Ok(())
}
