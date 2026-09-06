//! Practical C-MinHash using two seeded 64-bit permutations.
//!
//! Xiaoyun Li and Ping Li, *C-MinHash: Improving Minwise Hashing with
//! Circulant Permutation*, ICML 2022:
//! <https://proceedings.mlr.press/v162/li22m.html>.
//!
//! For each token hash `x`, the signature takes the minimum of
//! `pi(sigma(x) - (k + 1))`, with subtraction modulo 2^64. The subtraction
//! rotates the input coordinates of `pi`, as in the paper's right circulant
//! shifts. Independently seeded `SplitMix64` finalizers approximate the paper's
//! uniformly random permutations; its exact variance theorem does not apply
//! to this restricted pseudorandom family.

use pyo3::prelude::*;
use rand_core::{RngCore, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use serde::{Deserialize, Deserializer, Serialize};

mod batch;
mod core;
mod py;

const HASH_BATCH_SIZE: usize = 32;
const STATE_VERSION: u32 = 0x434d_4802;
const PY_STATE_PREFIX: &[u8] = b"Rensa:CMinHash:2\0";
type ReduceResult = (Py<PyAny>, (usize, u64), Py<PyAny>);

/// A C-MinHash sketch with seeded pseudorandom circulant permutations.
#[derive(Serialize, Clone)]
#[pyclass(module = "rensa", skip_from_py_object)]
pub struct CMinHash {
  version: u32,
  num_perm: usize,
  seed: u64,
  hash_values: Vec<u64>,
  #[serde(skip)]
  sigma_key: u64,
  #[serde(skip)]
  pi_key: u64,
}

#[derive(Deserialize)]
struct CMinHashState {
  #[serde(deserialize_with = "deserialize_version")]
  version: u32,
  num_perm: usize,
  seed: u64,
  hash_values: Vec<u64>,
}

fn deserialize_version<'de, D>(deserializer: D) -> Result<u32, D::Error>
where
  D: Deserializer<'de>,
{
  let version = u32::deserialize(deserializer)?;
  if version != STATE_VERSION {
    return Err(serde::de::Error::custom(
      "unsupported CMinHash state version; rebuild the sketch from its tokens",
    ));
  }
  Ok(version)
}

pub(in crate::cminhash) struct CMinHashParams {
  pub(in crate::cminhash) sigma_key: u64,
  pub(in crate::cminhash) pi_key: u64,
}

impl CMinHashParams {
  pub(in crate::cminhash) fn new(seed: u64) -> Self {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    Self {
      sigma_key: rng.next_u64(),
      pi_key: rng.next_u64(),
    }
  }
}

impl<'de> Deserialize<'de> for CMinHash {
  fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
  where
    D: Deserializer<'de>,
  {
    let state = CMinHashState::deserialize(deserializer)?;
    if state.num_perm == 0 || state.hash_values.len() != state.num_perm {
      return Err(serde::de::Error::custom(
        "invalid CMinHash state: num_perm must equal the nonempty hash_values length",
      ));
    }
    let params = CMinHashParams::new(state.seed);
    Ok(Self {
      version: state.version,
      num_perm: state.num_perm,
      seed: state.seed,
      hash_values: state.hash_values,
      sigma_key: params.sigma_key,
      pi_key: params.pi_key,
    })
  }
}

#[cfg(test)]
mod tests {
  use crate::cminhash::{CMinHash, STATE_VERSION};
  use serde::Serialize;

  #[test]
  fn serialized_sketch_restores_permutations_for_further_updates(
  ) -> pyo3::PyResult<()> {
    let mut original = CMinHash::new(129, 42)?;
    original.update_iter(["first", "second"]);
    let bytes = postcard::to_allocvec(&original).unwrap();
    let mut restored: CMinHash = postcard::from_bytes(&bytes).unwrap();
    assert_eq!(restored.hash_values, original.hash_values);
    original.update_iter(["third", "fourth"]);
    restored.update_iter(["third", "fourth"]);
    assert_eq!(restored.hash_values, original.hash_values);
    Ok(())
  }

  #[test]
  fn serialization_rejects_legacy_or_invalid_states() {
    #[derive(Serialize)]
    struct LegacyState {
      num_perm: usize,
      seed: u64,
      hash_values: Vec<u64>,
      sigma_a: u64,
      sigma_b: u64,
      pi_c: u64,
      pi_d: u64,
    }
    let legacy = LegacyState {
      num_perm: 128,
      seed: 42,
      hash_values: vec![u64::MAX; 128],
      sigma_a: 1,
      sigma_b: 2,
      pi_c: 3,
      pi_d: 4,
    };
    let bytes = postcard::to_allocvec(&legacy).unwrap();
    assert!(postcard::from_bytes::<CMinHash>(&bytes).is_err());

    for (version, num_perm, values) in [
      (STATE_VERSION + 1, 1_usize, vec![0_u64]),
      (STATE_VERSION, 0, vec![]),
      (STATE_VERSION, 2, vec![0]),
    ] {
      let bytes =
        postcard::to_allocvec(&(version, num_perm, 42_u64, values)).unwrap();
      assert!(postcard::from_bytes::<CMinHash>(&bytes).is_err());
    }
  }
}
