## Table of Contents

- [Rensa: A novel high-performance MinHash Implementation in Rust](#rensa-a-novel-high-performance-minhash-implementation-in-rust)
  - [Introduction](#introduction)
  - [Technical Implementation](#technical-implementation)
    - [R-MinHash (Original Rensa Variant)](#r-minhash-original-rensa-variant)
    - [C-MinHash (Based on the C-MinHash Paper)](#c-minhash-based-on-the-c-minhash-paper)
    - [OptDensMinHash (Based on Optimal Densification)](#optdensminhash-based-on-optimal-densification)
  - [Installation](#installation)
  - [Core Concepts: Direct MinHash vs. LSH-based Deduplication](#core-concepts-direct-minhash-vs-lsh-based-deduplication)
    - [Direct MinHash Deduplication](#direct-minhash-deduplication)
    - [LSH-based Deduplication](#lsh-based-deduplication)
    - [When to Use Which](#when-to-use-which)
  - [Usage Example](#usage-example)
    - [Deduplicating with Direct MinHash](#deduplicating-with-direct-minhash)
    - [Using C-MinHash for Similarity](#using-c-minhash-for-similarity)
    - [Deduplicating with RMinHashLSH](#deduplicating-with-rminhashlsh)
  - [Algorithm Comparison: R-MinHash vs. C-MinHash vs. OptDensMinHash vs. Datasketch](#algorithm-comparison-r-minhash-vs-c-minhash-vs-optdensminhash-vs-datasketch)
  - [Benchmark Results](#benchmark-results)
    - [MinHash Implementations Speed](#minhash-implementations-speed)
    - [MinHash Implementations Accuracy (Deduplication Results)](#minhash-implementations-accuracy-deduplication-results)
    - [LSH Performance (RMinHashLSH vs. Datasketch MinHashLSH)](#lsh-performance-rminhashlsh-vs-datasketch-minhashlsh)
  - [Running the Benchmarks](#running-the-benchmarks)
  - [Limitations and Future Work](#limitations-and-future-work)
  - [Contributing](#contributing)
  - [License](#license)

# Rensa: A novel high-performance MinHash Implementation in Rust

## Introduction

Rensa (Swedish for "clean") is a high-performance MinHash suite written in Rust with Python bindings. It's designed for efficient similarity estimation and deduplication of large datasets.

Rensa initially implemented a variant of the MinHash algorithm (`R-MinHash`) that combined ideas from traditional MinHash and the C-MinHash algorithm. It now also offers a more direct `C-MinHash` implementation and `OptDensMinHash` which uses optimal densification.

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

Rensa offers three high-performance MinHash variants in Rust: `R-MinHash` (its original novel approach), `C-MinHash` (an implementation closely following the C-MinHash paper), and `OptDensMinHash` (based on optimal densification techniques). All are designed for efficient similarity estimation and leverage common strategies for speed and memory efficiency:
- **Fast Hash Functions**: Rensa employs fast, non-cryptographic hash functions (based on FxHash or Murmur3) for processing input items.
- **Memory-Efficient Data Structures**: Implementations use compact data structures to minimize memory usage while maintaining fast access times.
- **Optimized Routines**: Core operations are optimized using techniques like batch processing and vectorized operations where appropriate.

### R-MinHash (Original Rensa Variant)

This variant was Rensa's initial novel approach. Key aspects of Rensa's `RMinHash` implementation include:

1.  **Efficient Permutation Generation**: Instead of storing full permutations or using k independent hash functions, Rensa's `RMinHash` uses a unique pair of random numbers (a, b) for each of the `num_perm` permutations. These are used to generate hash values on-the-fly for each item.

2.  **Simplified Approach**: While inspired by ideas related to C-MinHash, `RMinHash` is a distinct, simpler approach.
    - It does not apply an initial global permutation (σ) to the input data's hash in the same way as described in the C-MinHash paper for its primary permutation step.
    - It uses `num_perm` distinct pairs of random numbers (a, b) to simulate `num_perm` independent hash functions, rather than deriving them from a smaller set of parameters in a circulant manner.

3.  **Trade-off**: `RMinHash`'s approach trades some of the potential variance reduction benefits of more complex MinHash schemes (like full C-MinHash) for simplicity and good performance. It still offers better performance than traditional MinHash in many scenarios.

Rensa's Locality-Sensitive Hashing (LSH) implementation, `RMinHashLSH`, currently utilizes the `RMinHash` variant for its index.

### C-MinHash (Based on the C-MinHash Paper)

Rensa also includes `CMinHash`, an implementation more directly aligned with the principles of the C-MinHash algorithm from the paper "[C-MinHash: Rigorously Reducing K Permutations to Two](https://arxiv.org/abs/2109.03337)". Key aspects of this implementation are:

1.  **Two-Stage Hashing**: It utilizes two sets of universal hash function parameters for its permutation scheme:
    - An initial hash transformation (σ) is applied to the hash of each input item using parameters `sigma_a` and `sigma_b`.
    - A second pair of parameters, `pi_c` and `pi_d`, are used in combination with the σ-transformed item hash to generate the `num_perm` values in the MinHash signature. Specifically, for the `k`-th hash slot (where `k` is from 0 to `num_perm-1`), the value is derived from `(pi_c * sigma_transformed_hash + (pi_c * k + pi_d))`. The `(pi_c * k + pi_d)` terms are precomputed for each `k` to enhance efficiency.
2.  **Highly Optimized Routines**: The `update` and `jaccard` methods in `CMinHash` are heavily optimized. This includes batch processing of input items, structuring calculations to improve cache utilization, and using vectorized operations (e.g., processing data in fixed-size chunks like blocks of 16 or 8) for faster computations.
3.  **Performance Focus**: This implementation is specifically engineered for maximum single-threaded performance through these aggressive optimizations and careful memory access patterns.

### OptDensMinHash (Based on Optimal Densification)

Rensa also provides `OptDensMinHash`, which implements MinHash enhanced by an optimal densification strategy. This approach aims to improve accuracy, especially for sparse datasets or smaller numbers of permutations, by ensuring that MinHash signatures are always fully populated.

1.  **Densification**: If, after processing all input items, some slots in the MinHash signature remain empty (i.e., no item hashed to them as the minimum), this algorithm fills these empty slots using values from other, non-empty slots in a principled manner. This "densification" ensures a complete signature.
2.  **Theoretical Basis**: The core ideas are drawn from research on densified MinHash algorithms, such as:
    - Shrivastava, A. (2017). Optimal Densification for Fast and Accurate Minwise Hashing. *PMLR*.
    - Mai, T., et al. (2020). On densification for MinWise Hashing. *PMLR*.
3.  **Usage**: `OptDensMinHash` is designed for unweighted data. The densification process is automatically triggered internally when the signature is requested (e.g., via `digest()` or `jaccard()`).

These design choices result in a suite of MinHash implementations that are fast, memory-efficient, and suitable for large-scale similarity estimation and deduplication tasks. Benchmarks show that Rensa's implementations offer significant performance improvements over traditional MinHash libraries like `datasketch`.

## Installation

You can install Rensa using `pip`. It's available in all platforms:

```bash
pip install rensa
```

## Core Concepts: Direct MinHash vs. LSH-based Deduplication

Rensa provides tools for both direct MinHash signature comparison and Locality Sensitive Hashing (LSH) for more scalable deduplication. Understanding the difference helps in choosing the right approach.

### Direct MinHash Deduplication

  * **How it works**:
    1.  Generate a MinHash signature (e.g., using `RMinHash`, `CMinHash`, or `OptDensMinHash`) for every item in your dataset.
    2.  To find exact duplicates (based on the MinHash signature), you can store the signatures (e.g., as tuples) in a set or dictionary and identify items that produce identical signatures.
    3.  To find similar items, you would compute the Jaccard similarity between the MinHash signatures of pairs of items.
  * **Pros**:
      * Conceptually simple for finding items with identical MinHash signatures.
      * Gives precise Jaccard similarity estimates between any two chosen MinHash signatures.
  * **Cons**:
      * Finding all similar pairs by computing Jaccard similarity between all MinHash signatures can be computationally expensive ($O(N^2)$ comparisons for $N$ items), making it unsuitable for very large datasets if broad similarity search is needed.
  * **Example**: See the "Deduplicating with Direct MinHash" example below.

### LSH-based Deduplication

  * **How it works**:
    1.  Generate MinHash signatures for all items (Rensa's `RMinHashLSH` uses `RMinHash`).
    2.  Index these MinHash signatures into an LSH data structure (e.g., `RMinHashLSH`). LSH groups similar signatures into common "buckets" based on bands of their hash values.
    3.  For each item, query the LSH index. The LSH index returns a small set of *candidate* items that are likely to be similar to the query item.
    4.  Compute the true Jaccard similarity (using their MinHash signatures) only between the query item and its candidates. This significantly reduces the number of pairwise comparisons.
  * **Pros**:
      * Much faster for finding similar items in large datasets because it avoids most pairwise comparisons.
      * Scales well for approximate nearest neighbor searches.
  * **Cons**:
      * Probabilistic: It might miss some similar pairs (false negatives) or identify some non-similar items as candidates (false positives for candidacy, which are then typically filtered by a final Jaccard check).
      * Requires tuning parameters like the LSH similarity threshold, the number of bands, and the final Jaccard similarity threshold for verification.
  * **Example**: See the "Deduplicating with RMinHashLSH" example below.

### When to Use Which

  * **Direct MinHash**:
      * Smaller datasets where $O(N^2)$ comparisons (or a smarter selection of pairs) are feasible.
      * When you need to find exact matches based on MinHash signatures.
      * When you need to compute Jaccard similarity for specific, pre-selected pairs.
  * **LSH-based Deduplication**:
      * Large datasets where comparing all pairs is too slow.
      * When you need to find approximately similar items efficiently (approximate nearest neighbor search).
      * When performance and scalability for finding potential duplicates are critical.

## Usage Example

### Deduplicating with Direct MinHash

Here's an example of how to use Rensa's MinHash implementations (e.g., `RMinHash`, `CMinHash`) for direct deduplication:

```python
from datasets import load_dataset
from rensa import RMinHash, CMinHash # Or OptDensMinHash
from tqdm import tqdm

# Define a function to generate MinHash (works for RMinHash, CMinHash)
def generate_minhash_signature(text, minhash_class, num_perm=128, seed=42):
    m = minhash_class(num_perm=num_perm, seed=seed)
    m.update(text.split())
    return m

def deduplicate_dataset_direct(dataset, text_column="sql", minhash_class=RMinHash, num_perm=128, desc="Deduplicating"):
    unique_hashes = set()
    deduplicated_indices = []
   
    for idx, example in tqdm(enumerate(dataset), total=len(dataset), desc=desc):
        minhash_obj = generate_minhash_signature(example[text_column], minhash_class, num_perm)
        hash_tuple = tuple(minhash_obj.digest())
       
        if hash_tuple not in unique_hashes:
            unique_hashes.add(hash_tuple)
            deduplicated_indices.append(idx)
           
    return deduplicated_indices

def main_direct_deduplication():
    print("Loading dataset...")
    sql_dataset_dict = load_dataset("gretelai/synthetic_text_to_sql")
    sql_dataset = sql_dataset_dict["train"]
   
    print("Deduplicating dataset with R-MinHash...")
    deduplicated_indices_r = deduplicate_dataset_direct(
        sql_dataset,
        text_column="sql",
        minhash_class=RMinHash,
        desc="R-MinHash Deduplication"
    )
    deduplicated_dataset_r = sql_dataset.select(deduplicated_indices_r)
   
    print(f"Original dataset size: {len(sql_dataset)}")
    print(f"Deduplicated dataset size (R-MinHash): {len(deduplicated_dataset_r)}")
    print(f"Rows removed (R-MinHash): {len(sql_dataset) - len(deduplicated_dataset_r)}")

    # Example with C-MinHash
    # print("Deduplicating dataset with C-MinHash...")
    # deduplicated_indices_c = deduplicate_dataset_direct(
    #     sql_dataset,
    #     text_column="sql",
    #     minhash_class=CMinHash,
    #     desc="C-MinHash Deduplication"
    # )
    # deduplicated_dataset_c = sql_dataset.select(deduplicated_indices_c)
    # print(f"Deduplicated dataset size (C-MinHash): {len(deduplicated_dataset_c)}")

if __name__ == "__main__":
    main_direct_deduplication()
```

### Using C-MinHash for Similarity

Here's a more direct example of using `CMinHash` for calculating Jaccard similarity:

```python
from rensa import CMinHash

# Example texts
text1 = "This is an example sentence for CMinHash."
text2 = "This is another example sentence, slightly different from the first."

# Initialize CMinHash objects
num_permutations = 256
seed = 12345
c_minhash1 = CMinHash(num_perm=num_permutations, seed=seed)
c_minhash2 = CMinHash(num_perm=num_permutations, seed=seed)

# Update with words from each text
c_minhash1.update(text1.split())
c_minhash2.update(text2.split())

# Calculate Jaccard similarity
similarity = c_minhash1.jaccard(c_minhash2)
print(f"Estimated Jaccard similarity (CMinHash, {num_permutations} perm): {similarity:.4f}")

# Get signatures
signature1 = c_minhash1.digest()
# print(f"C-MinHash signature 1: {signature1}")
```

### Deduplicating with RMinHashLSH
Here's an example of how to use `RMinHashLSH` for deduplicating a dataset. This approach is more efficient for larger datasets. Key LSH parameters are set to example values within the function.

```python
from datasets import load_dataset
from rensa import RMinHash, RMinHashLSH
from tqdm import tqdm

def deduplicate_dataset_with_lsh_simple(dataset, text_column="sql"):
    num_perm = 128
    seed = 42
    lsh_threshold = 0.8
    num_bands = 16 
    final_jaccard_threshold = 0.85

    if num_perm % num_bands != 0:
        raise ValueError(f"num_bands ({num_bands}) must divide num_perm ({num_perm}).")

    minhashes = {} 
   
    for idx, example in tqdm(enumerate(dataset), total=len(dataset), desc="1. Generating RMinHashes"):
        text_content = str(example[text_column])
        tokens = text_content.split()
        m = RMinHash(num_perm=num_perm, seed=seed)
        m.update(tokens)
        minhashes[idx] = m

    lsh_index = RMinHashLSH(threshold=lsh_threshold, num_perm=num_perm, num_bands=num_bands)
    for doc_id, rminhash_obj in tqdm(minhashes.items(), desc="2. Indexing into LSH"):
        lsh_index.insert(doc_id, rminhash_obj)

    to_remove = set()
    sorted_doc_ids = sorted(minhashes.keys())

    for doc_id in tqdm(sorted_doc_ids, desc="3. Querying LSH & Deduplicating"):
        if doc_id in to_remove:
            continue

        query_minhash = minhashes[doc_id]
        candidate_ids = lsh_index.query(query_minhash)

        for candidate_id in candidate_ids:
            if candidate_id == doc_id or candidate_id in to_remove:
                continue
            
            candidate_minhash = minhashes[candidate_id]
            actual_jaccard = query_minhash.jaccard(candidate_minhash)

            if actual_jaccard >= final_jaccard_threshold:
                # Keep the item with the smaller original index
                if doc_id < candidate_id:
                    to_remove.add(candidate_id)
                else:
                    to_remove.add(doc_id)
                    break 
   
    deduplicated_indices = [idx for idx in sorted_doc_ids if idx not in to_remove]
    return deduplicated_indices

def main_lsh_deduplication_simple():
    print("Loading dataset...")
    try:
        sql_dataset_dict = load_dataset("gretelai/synthetic_text_to_sql")
        sql_dataset = sql_dataset_dict["train"]
    except Exception as e:
        print(f"Failed to load dataset: {e}. Ensure 'datasets' is installed or use a local dataset.")
        return

    print("Deduplicating dataset with RMinHashLSH...")
   
    deduplicated_indices_lsh = deduplicate_dataset_with_lsh_simple(
        sql_dataset,
        text_column="sql"
    )
    deduplicated_dataset_lsh = sql_dataset.select(deduplicated_indices_lsh)

    print(f"Original dataset size (train split): {len(sql_dataset)}")
    print(f"Deduplicated dataset size (RMinHashLSH): {len(deduplicated_dataset_lsh)}")
    print(f"Rows removed (RMinHashLSH): {len(sql_dataset) - len(deduplicated_dataset_lsh)}")

if __name__ == "__main__":
    main_lsh_deduplication_simple()
```

## Algorithm Comparison: R-MinHash vs. C-MinHash vs. OptDensMinHash vs. Datasketch

Rensa offers three MinHash implementations (`RMinHash`, `CMinHash`, `OptDensMinHash`), each with different trade-offs compared to each other and the popular `datasketch` library.

Based on the latest `advanced_benchmark.py` results (averaged over 5 runs on the `gretelai/synthetic_text_to_sql` dataset, 100,000 rows, in a Macbook Pro M2 32GB):

  * **Speed (at 256 permutations)**:

      * **`CMinHash`** is consistently the fastest. Average execution time: **5.47 seconds**.
      * **`RMinHash`** is also very fast. Average execution time: **5.58 seconds**.
      * **`OptDensMinHash`** is fast. Average execution time: **12.36 seconds**.
      * **`datasketch`** is considerably slower. Average execution time: **92.45 seconds**.
        This makes `CMinHash` up to approximately **16.90x faster** than `datasketch`, `RMinHash` up to approximately **16.57x faster**, and `OptDensMinHash` up to approximately **7.48x faster** than `datasketch` (all at 256 permutations).

  * **Accuracy (Jaccard Similarity of Deduplicated Sets vs. Datasketch, 128 permutations)**:

      * **`RMinHash`** produces deduplication results identical to `datasketch` (Jaccard similarity of **1.0000** between their output sets of unique items, with 99262 common items).
      * **`OptDensMinHash`** yields results very close to `datasketch`. The Jaccard similarity is **0.9997** (with 99233 common items with Datasketch).
      * **`CMinHash`** also yields results very close to `datasketch`. The Jaccard similarity is **0.9996** (with 99223 common items with Datasketch).
        This indicates that while all Rensa variants are highly effective for similarity estimation, `RMinHash` perfectly matches `datasketch`'s deduplication output in this benchmark, while `CMinHash` and `OptDensMinHash` produce extremely similar results.

  * **Recommendation**:

      * For most use cases, **`RMinHash`** provides an excellent balance of high speed (up to ~16.6x faster than `datasketch`) and accuracy (matching `datasketch`'s deduplication results). **It remains the generally recommended algorithm.**
      * If absolute maximum throughput is the primary concern, **`CMinHash`** offers the best performance (up to ~16.9x faster than `datasketch`), with a negligible difference in exact deduplication results compared to `datasketch`/`RMinHash`.
      * **`OptDensMinHash`** offers a good balance of speed and high accuracy, and might be particularly beneficial for datasets with high sparsity or when using fewer permutations, due to its densification strategy.
      * If you require features beyond core MinHash generation or need to integrate with an existing `datasketch` ecosystem, `datasketch` remains a comprehensive option, albeit slower for MinHash operations.

## Benchmark Results

The results below are from the `advanced_benchmark.py` script, averaged over 5 runs on the `gretelai/synthetic_text_to_sql` dataset (100,000 rows).

![Graph with benchmark results that demonstrate that Rensa is 12x faster](./assets/final_benchmark_comparison.png)


### MinHash Implementations Speed

Average execution time in seconds for deduplicating the dataset.

| Permutations | Datasketch Time (s) | Rensa R-MinHash Time (s) | Rensa C-MinHash Time (s) | Rensa OptDensMinHash Time (s) | R-MinHash Speedup (vs DS) | C-MinHash Speedup (vs DS) | OptDensMinHash Speedup (vs DS) |
| ------------ | ------------------- | ------------------------ | ------------------------ | ----------------------------- | ------------------------- | ------------------------- | ------------------------------ |
| 64           | 37.89               | 4.80                     | 4.76                     | 6.28                          | 7.90x                     | 7.96x                     | 6.03x                          |
| 128          | 56.59               | 5.15                     | 5.04                     | 8.37                          | 11.00x                    | 11.23x                    | 6.76x                          |
| 256          | 92.45               | 5.58                     | 5.47                     | 12.36                         | 16.57x                    | 16.90x                    | 7.48x                          |

### MinHash Implementations Accuracy (Deduplication Results)

Jaccard similarity of the deduplicated document indices produced by each method (using 128 permutations), compared against `datasketch` as the baseline or against each other.

  - **Datasketch vs Rensa (RMinHash)**: Jaccard Similarity **1.0000** (Intersection: 99262 identical items)
  - **Datasketch vs Rensa (CMinHash)**: Jaccard Similarity **0.9996** (Intersection: 99223 items)
  - **Datasketch vs Rensa (OptDensMinHash)**: Jaccard Similarity **0.9997** (Intersection: 99233 items)
  - **Rensa (RMinHash) vs Rensa (CMinHash)**: Jaccard Similarity **0.9996** (Intersection: 99223 items)
  - **Rensa (RMinHash) vs Rensa (OptDensMinHash)**: Jaccard Similarity **0.9997** (Intersection: 99233 items)
  - **Rensa (CMinHash) vs Rensa (OptDensMinHash)**: Jaccard Similarity **0.9993** (Intersection: 99194 items)

This confirms that `RMinHash` produces identical deduplication results to `datasketch` in this benchmark, while `CMinHash` and `OptDensMinHash` are highly similar.

### LSH Performance (RMinHashLSH vs. Datasketch MinHashLSH)

The following results are from `benchmarks/lsh_benchmark.py` using the `