#![deny(clippy::all)]
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]
#![allow(clippy::unsafe_derive_deserialize)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::needless_pass_by_value)]

mod cminhash;
mod lsh;
mod opt_dens_minhash;
mod rminhash;
mod utils;
mod dedup;

pub use cminhash::CMinHash;
pub use dedup::{CMinHashDeduper, OptDensDeduper, RMinHashDeduper};
pub use lsh::RMinHashLSH;
pub use opt_dens_minhash::OptDensMinHash;
pub use rminhash::RMinHash;

use pyo3::prelude::*;

/// Python module for MinHash and LSH implementations
///
/// # Errors
/// Returns an error if the module initialization fails or classes cannot be added
#[pymodule]
pub fn rensa(m: &Bound<'_, PyModule>) -> PyResult<()> {
  m.add_class::<RMinHash>()?;
  m.add_class::<OptDensMinHash>()?;
  m.add_class::<CMinHash>()?;
  m.add_class::<RMinHashLSH>()?;
  m.add_class::<RMinHashDeduper>()?;
  m.add_class::<CMinHashDeduper>()?;
  m.add_class::<OptDensDeduper>()?;
  Ok(())
}
