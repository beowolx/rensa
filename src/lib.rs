use murmurhash3::murmurhash3_x86_32;
use pyo3::prelude::*;
use rand::prelude::*;
use std::collections::HashMap;

#[pyclass]
struct CMinHash {
    num_perm: usize,
    seed: u32,
    hash_values: Vec<u32>,
    a: Vec<u32>,
    b: Vec<u32>,
}

#[pymethods]
impl CMinHash {
    #[new]
    fn new(num_perm: usize, seed: u32) -> Self {
        let mut rng = StdRng::seed_from_u64(seed as u64);
        let a: Vec<u32> = (0..num_perm).map(|_| rng.gen()).collect();
        let b: Vec<u32> = (0..num_perm).map(|_| rng.gen()).collect();

        CMinHash {
            num_perm,
            seed,
            hash_values: vec![u32::MAX; num_perm],
            a,
            b,
        }
    }

    fn update(&mut self, items: Vec<String>) {
        for item in items {
            let item_hash = murmurhash3_x86_32(item.as_bytes(), self.seed);
            for i in 0..self.num_perm {
                let hash = self.a[i].wrapping_mul(item_hash).wrapping_add(self.b[i]);
                self.hash_values[i] = self.hash_values[i].min(hash);
            }
        }
    }

    fn digest(&self) -> Vec<u32> {
        self.hash_values.clone()
    }

    fn jaccard(&self, other: &CMinHash) -> f64 {
        let equal_count = self
            .hash_values
            .iter()
            .zip(&other.hash_values)
            .filter(|&(a, b)| a == b)
            .count();
        equal_count as f64 / self.num_perm as f64
    }
}

#[pyclass]
struct CMinHashLSH {
    threshold: f64,
    num_perm: usize,
    num_bands: usize,
    band_size: usize,
    hash_tables: Vec<HashMap<u64, Vec<usize>>>,
}

#[pymethods]
impl CMinHashLSH {
    #[new]
    fn new(threshold: f64, num_perm: usize, num_bands: usize) -> Self {
        CMinHashLSH {
            threshold,
            num_perm,
            num_bands,
            band_size: num_perm / num_bands,
            hash_tables: vec![HashMap::new(); num_bands],
        }
    }

    fn insert(&mut self, key: usize, minhash: &CMinHash) {
        for (i, table) in self.hash_tables.iter_mut().enumerate() {
            let start = i * self.band_size;
            let end = start + self.band_size;
            let band: Vec<u8> = minhash.digest()[start..end]
                .iter()
                .flat_map(|&x| x.to_le_bytes().to_vec())
                .collect();
            let band_hash = murmurhash3_x86_32(&band, 0) as u64;
            table.entry(band_hash).or_insert_with(Vec::new).push(key);
        }
    }

    fn query(&self, minhash: &CMinHash) -> Vec<usize> {
        let mut candidates = Vec::new();
        for (i, table) in self.hash_tables.iter().enumerate() {
            let start = i * self.band_size;
            let end = start + self.band_size;
            let band: Vec<u8> = minhash.digest()[start..end]
                .iter()
                .flat_map(|&x| x.to_le_bytes().to_vec())
                .collect();
            let band_hash = murmurhash3_x86_32(&band, 0) as u64;
            if let Some(keys) = table.get(&band_hash) {
                candidates.extend(keys);
            }
        }
        candidates.sort();
        candidates.dedup();
        candidates
    }

    fn is_similar(&self, minhash1: &CMinHash, minhash2: &CMinHash) -> bool {
        let jaccard = minhash1.jaccard(minhash2);
        jaccard >= self.threshold
    }

    fn get_num_perm(&self) -> usize {
        self.num_perm
    }

    fn get_num_bands(&self) -> usize {
        self.num_bands
    }
}

#[pymodule]
fn rensa(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CMinHash>()?;
    m.add_class::<CMinHashLSH>()?;
    Ok(())
}
