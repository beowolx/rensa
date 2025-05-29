## Table of Contents

- [Rensa: A novel high-performance MinHash Implementation in Rust](#rensa-a-novel-high-performance-minhash-implementation-in-rust)
  - [Introduction](#introduction)
  - [Technical Implementation](#technical-implementation)
    - [R-MinHash (Original Rensa Variant)](#r-minhash-original-rensa-variant)
    - [C-MinHash (Based on the C-MinHash Paper)](#c-minhash-based-on-the-c-minhash-paper)
  - [Installation](#installation)
  - [Usage Example](#usage-example)
    - [Using C-MinHash for Similarity](#using-c-minhash-for-similarity)
  - [Algorithm Comparison: R-MinHash vs. C-MinHash vs. Datasketch](#algorithm-comparison-r-minhash-vs-c-minhash-vs-datasketch)
  - [Benchmark Results](#benchmark-results)
    - [Speed](#speed)
    - [Accuracy (Deduplication Results)](#accuracy-deduplication-results)
  - [Running the Benchmarks](#running-the-benchmarks)
  - [Limitations and Future Work](#limitations-and-future-work)
  - [Contributing](#contributing)
  - [License](#license)

# Rensa: A novel high-performance MinHash Implementation in Rust

## Introduction

Rensa (Swedish for "clean") is a high-performance MinHash implementation written in Rust with Python bindings. It's designed for efficient similarity estimation and deduplication of large datasets.

Rensa implements a variant of the MinHash algorithm that combines ideas from traditional MinHash and the C-MinHash algorithm proposed in the paper [C-MinHash: Rigorously Reducing K Permutations to Two](https://arxiv.org/abs/2109.03337) to create a novel MinHash implementation that I call `R-MinHash`.

Rensa is particularly useful in scenarios where you need to:

- Quickly estimate the similarity between large sets of data
- Deduplicate large datasets
- Perform locality-sensitive hashing (LSH) for approximate nearest neighbor search

Use cases include:
- Content deduplication in large document collections
- Identifying similar items in recommendation systems
- Clustering of high-dimensional data
- Near-duplicate detection in web crawling

## Technical Implementation

Rensa offers two high-performance MinHash variants in Rust: `R-MinHash` (its original novel approach) and `C-MinHash` (an implementation more closely following the C-MinHash paper). Both are designed for efficient similarity estimation and leverage common strategies for speed and memory efficiency:
- **Fast Hash Functions**: Rensa employs fast, non-cryptographic hash functions (based on FxHash) for processing input items.
- **Memory-Efficient Data Structures**: Implementations use compact data structures to minimize memory usage while maintaining fast access times.
- **Optimized Routines**: Core operations are optimized using techniques like batch processing and vectorized operations where appropriate.

### R-MinHash (Original Rensa Variant)

This variant was Rensa's initial novel approach. Key aspects of Rensa's `RMinHash` implementation include:

1.  **Efficient Permutation Generation**: Instead of storing full permutations or using k independent hash functions, Rensa's `RMinHash` uses a unique pair of random numbers (a, b) for each of the `num_perm` permutations. These are used to generate hash values on-the-fly for each item.

2.  **Simplified Approach**: While inspired by ideas related to C-MinHash, `RMinHash` is a distinct, simpler approach.
    - It does not apply an initial global permutation (σ) to the input data's hash in the same way as described in the C-MinHash paper for its primary permutation step.
    - It uses `num_perm` distinct pairs of random numbers (a, b) to simulate `num_perm` independent hash functions, rather than deriving them from a smaller set of parameters in a circulant manner.

3.  **Trade-off**: `RMinHash`'s approach trades some of the potential variance reduction benefits of more complex MinHash schemes (like full C-MinHash) for simplicity and good performance. It still offers better performance than traditional MinHash in many scenarios.

### C-MinHash (Based on the C-MinHash Paper)

Rensa also includes `CMinHash`, an implementation more directly aligned with the principles of the C-MinHash algorithm from the paper "[C-MinHash: Rigorously Reducing K Permutations to Two](https://arxiv.org/abs/2109.03337)". Key aspects of this implementation are:

1.  **Two-Stage Hashing**: It utilizes two sets of universal hash function parameters for its permutation scheme:
    - An initial hash transformation (σ) is applied to the hash of each input item using parameters `sigma_a` and `sigma_b`.
    - A second pair of parameters, `pi_c` and `pi_d`, are used in combination with the σ-transformed item hash to generate the `num_perm` values in the MinHash signature. Specifically, for the `k`-th hash slot (where `k` is from 0 to `num_perm-1`), the value is derived from `(pi_c * sigma_transformed_hash + (pi_c * k + pi_d))`. The `(pi_c * k + pi_d)` terms are precomputed for each `k` to enhance efficiency.
2.  **Highly Optimized Routines**: The `update` and `jaccard` methods in `CMinHash` are heavily optimized. This includes batch processing of input items, structuring calculations to improve cache utilization, and using vectorized operations (e.g., processing data in fixed-size chunks like blocks of 16 or 8) for faster computations.
3.  **Performance Focus**: This implementation is specifically engineered for maximum single-threaded performance through these aggressive optimizations and careful memory access patterns.

Rensa's Locality-Sensitive Hashing (LSH) implementation, `RMinHashLSH`, currently utilizes the `RMinHash` variant for its index.

These design choices result in a MinHash implementation that is fast, memory-efficient, and suitable for large-scale similarity estimation and deduplication tasks. While Rensa may not provide the same theoretical guarantees as full C-MinHash, our benchmarks show that it offers significant performance improvements over traditional MinHash implementations like `datasketch`.

## Installation

You can install Rensa using `pip`. It's available in all platforms:

```bash
pip install rensa
```

## Usage Example

Here's an example of how to use Rensa to deduplicate a dataset:

```python
from datasets import load_dataset
from rensa import RMinHash, CMinHash
from tqdm import tqdm

def rensa_r_minhash(text, num_perm=128):
    m = RMinHash(num_perm=num_perm, seed=42)
    m.update(text.split())
    return m

def rensa_c_minhash(text, num_perm=128):
    m = CMinHash(num_perm=num_perm, seed=42)
    m.update(text.split())
    return m

def deduplicate_dataset(dataset, minhash_func, num_perm=128, desc="Deduplicating"):
    unique_hashes = set()
    deduplicated_indices = []
    
    for idx, example in tqdm(enumerate(dataset), total=len(dataset), desc=desc):
        minhash = minhash_func(example["sql"], num_perm)
        hash_tuple = tuple(minhash.digest())
        
        if hash_tuple not in unique_hashes:
            unique_hashes.add(hash_tuple)
            deduplicated_indices.append(idx)
    
    return deduplicated_indices

def main():
    print("Loading dataset...")
    sql_dataset = load_dataset("gretelai/synthetic_text_to_sql", split="train")
    
    print("Deduplicating dataset with R-MinHash...")
    deduplicated_indices_r = deduplicate_dataset(sql_dataset, rensa_r_minhash, desc="R-MinHash Deduplication")
    deduplicated_dataset_r = sql_dataset.select(deduplicated_indices_r)
    
    print("Deduplicating dataset with C-MinHash...")
    deduplicated_indices_c = deduplicate_dataset(sql_dataset, rensa_c_minhash, desc="C-MinHash Deduplication")
    deduplicated_dataset_c = sql_dataset.select(deduplicated_indices_c)

    print("--- R-MinHash Deduplication Results ---")
    print(f"Original dataset size: {len(sql_dataset)}")
    print(f"Deduplicated dataset size (R-MinHash): {len(deduplicated_dataset_r)}")
    print(f"Rows removed (R-MinHash): {len(sql_dataset) - len(deduplicated_dataset_r)}")

    print("--- C-MinHash Deduplication Results ---")
    # Note: C-MinHash might yield different deduplication counts due to its hashing nature
    print(f"Original dataset size: {len(sql_dataset)}")
    print(f"Deduplicated dataset size (C-MinHash): {len(deduplicated_dataset_c)}")
    print(f"Rows removed (C-MinHash): {len(sql_dataset) - len(deduplicated_dataset_c)}")

if __name__ == "__main__":
    main()
```

### Using C-MinHash for Similarity

Here's a more direct example of using `CMinHash` for calculating Jaccard similarity:

```python
from rensa import CMinHash

# Example texts
text1 = "This is an example sentence for CMinHash."
text2 = "This is another example sentence, slightly different from the first."

# Initialize CMinHash objects
num_permutations = 128
seed = 42
c_minhash1 = CMinHash(num_perm=num_permutations, seed=seed)
c_minhash2 = CMinHash(num_perm=num_permutations, seed=seed)

# Update with words from each text
c_minhash1.update(text1.split())
c_minhash2.update(text2.split())

# Calculate Jaccard similarity
similarity = c_minhash1.jaccard(c_minhash2)
print(f"Jaccard similarity between the two texts using C-MinHash: {similarity:.4f}")

# Get signatures
signature1 = c_minhash1.digest()
# print(f"C-MinHash signature 1: {signature1}")
```

## Algorithm Comparison: R-MinHash vs. C-MinHash vs. Datasketch

Rensa offers two MinHash implementations, `RMinHash` and `CMinHash`, each with different trade-offs compared to each other and the popular `datasketch` library.

Based on the latest `advanced_benchmark.py` results (averaged over 5 runs on the `gretelai/synthetic_text_to_sql` dataset, 100,000 rows):

*   **Speed**:
    *   **`CMinHash`** is consistently the fastest. At 256 permutations, it achieves an average execution time of **6.95 seconds**.
    *   **`RMinHash`** is also very fast. At 256 permutations, it averages **7.58 seconds**.
    *   **`datasketch`** is considerably slower. At 256 permutations, it averages **96.87 seconds**.
    This makes `CMinHash` up to approximately **13.94x faster** than `datasketch`, and `RMinHash` up to approximately **12.78x faster** than `datasketch` (both at 256 permutations).

*   **Accuracy (Jaccard Similarity of Deduplicated Sets, 128 permutations)**:
    *   **`RMinHash`** produces deduplication results identical to `datasketch` (Jaccard similarity of **1.0000** between their output sets of unique items, with 99262 common items).
    *   **`CMinHash`** yields slightly different deduplication results. The Jaccard similarity between its output set and those from `datasketch` or `RMinHash` is **0.9896** (with 98231 common items). This indicates that while highly effective for similarity estimation, its resulting signatures might differ slightly from traditional MinHash under certain conditions.

*   **Recommendation**:
    *   For most use cases, **`RMinHash`** provides an excellent balance of high speed (up to ~12.8x faster than `datasketch`) and accuracy (matching `datasketch`'s deduplication results).
    *   If absolute maximum throughput is the primary concern, **`CMinHash`** offers the best performance (up to ~13.9x faster than `datasketch`), with a minor trade-off in exact deduplication results compared to `datasketch`/`RMinHash`.
    *   If you require features beyond core MinHash generation or need to integrate with an existing `datasketch` ecosystem, `datasketch` remains a comprehensive option, albeit slower for MinHash operations.

## Benchmark Results

The results below are from the `advanced_benchmark.py` script, averaged over 5 runs on the `gretelai/synthetic_text_to_sql` dataset (100,000 rows).

![Graph with benchmark results that demonstrate that Rensa is 12x faster](https://github.com/beowolx/rensa/assets/61982523/c793ad0d-0cfd-4ec5-8d4b-4e1b02feda5a)

### Speed

Average execution time in seconds for deduplicating the dataset.

| Permutations | Datasketch Time (s) | Rensa R-MinHash Time (s) | Rensa C-MinHash Time (s) | R-MinHash Speedup (vs DS) | C-MinHash Speedup (vs DS) |
|--------------|---------------------|--------------------------|--------------------------|---------------------------|---------------------------|
| 64           | 39.65               | 6.20                     | 6.07                     | 6.40x                     | 6.53x                     |
| 128          | 57.74               | 6.39                     | 6.15                     | 9.04x                     | 9.39x                     |
| 256          | 96.87               | 7.58                     | 6.95                     | 12.78x                    | 13.94x                    |


### Accuracy (Deduplication Results)

Jaccard similarity of the deduplicated document indices produced by each method (using 128 permutations), compared against `datasketch` as the baseline or against each other.

-   **Datasketch vs Rensa (RMinHash)**: Jaccard Similarity **1.0000** (Intersection: 99262 identical items)
-   **Datasketch vs Rensa (CMinHash)**: Jaccard Similarity **0.9896** (Intersection: 98231 items)
-   **Rensa (RMinHash) vs Rensa (CMinHash)**: Jaccard Similarity **0.9896** (Intersection: 98231 items)

This confirms that `RMinHash` produces identical deduplication results to `datasketch`, while `CMinHash` is highly similar but not identical.

## Running the Benchmarks

To run the benchmarks yourself, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/beowolx/rensa.git
   cd rensa
   ```

2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the simple benchmark:
   ```bash
   python benchmarks/simple_benchmark.py
   ```

5. Run the advanced benchmark:
   ```bash
   python benchmarks/advanced_benchmark.py
   ```

The `simple_benchmark.py` script provides a basic comparison of deduplication performance between Rensa and `datasketch`. The `advanced_benchmark.py` script offers a more comprehensive analysis, including multiple runs with different numbers of permutations, memory usage tracking, and detailed profiling information.

## Limitations and Future Work

While Rensa offers significant performance improvements, it has some limitations compared to `datasketch`:

1. **Feature set**: Rensa currently implements only the core MinHash (`RMinHash`, `CMinHash`) and LSH (for `RMinHash`) functionality. It doesn't include some of the advanced features found in `datasketch` like HyperLogLog, etc.

2. **Customization**: `datasketch` offers more options for customizing the hash functions and other parameters. Rensa's implementations are more fixed for performance but offer `seed` and `num_perm` customization.

3. **Theoretical guarantees**: 
    - `RMinHash`, due to its simplified permutation generation, may not provide the same level of variance reduction as theoretically optimal MinHash or the full C-MinHash algorithm in all scenarios.
    - `CMinHash` is designed to be a more faithful implementation of the C-MinHash paper's principles, aiming for stronger theoretical backing regarding its reduction of k permutations to two.

Future work on Rensa may include:

- Adding more advanced features and customization options
- Further optimizing performance for specific use cases and data types

Despite these limitations, Rensa's performance benefits make it an excellent choice for applications where speed and efficiency are critical, especially when working with large datasets.

## Contributing

Contributions to Rensa are welcome! Please feel free to submit pull requests, report bugs, or suggest features through the GitHub issue tracker.

## License

Rensa is released under the MIT License. See the LICENSE file for details.
