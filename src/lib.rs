#![deny(clippy::all)]
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]

#[cfg(feature = "mimalloc")]
use mimalloc::MiMalloc;

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

mod cminhash;
mod env;
mod inline_dedup;
mod lsh;
mod py_input;
mod rminhash;
mod simd;
mod utils;

pub use cminhash::CMinHash;
pub use inline_dedup::CMinHashDeduplicator;
pub use inline_dedup::RMinHashDeduplicator;
pub use lsh::RMinHashLSH;
pub use rminhash::RMinHash;
pub use rminhash::RMinHashDigestMatrix;

use pyo3::prelude::*;

/// Python module for `MinHash` and LSH implementations
///
/// # Errors
/// Returns an error if the module initialization fails or classes cannot be added
#[pymodule(gil_used = true)]
pub fn rensa(m: &Bound<'_, PyModule>) -> PyResult<()> {
  m.add_class::<RMinHash>()?;
  m.add_class::<RMinHashDigestMatrix>()?;
  m.add_class::<CMinHash>()?;
  m.add_class::<RMinHashLSH>()?;
  m.add_class::<RMinHashDeduplicator>()?;
  m.add_class::<CMinHashDeduplicator>()?;
  Ok(())
}
