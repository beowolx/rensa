# Rensa: High-Performance MinHash Implementation in Rust

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

Key aspects of Rensa's implementation include:

1. **Efficient permutation generation**: Instead of storing full permutations or using k independent hash functions, Rensa uses a pair of random numbers (a, b) to generate permutations on-the-fly. This approach significantly reduces memory usage while maintaining the algorithm's effectiveness.

2. **Simplified C-MinHash**: While inspired by C-MinHash, Rensa's implementation differs in a few key ways:
   - It does not apply an initial independent permutation (σ) to the input data.
   - Instead of using circulant permutations (π_k) for each hash value, Rensa uses the same pair of random numbers (a, b) for all permutations.

3. **Trade-off between memory and variance reduction**: Rensa's approach trades some of the variance reduction benefits of full C-MinHash for improved memory efficiency and simplicity. While it may not achieve the same level of variance reduction as C-MinHash, it still offers better performance than traditional MinHash in many scenarios.

4. **Fast hash function**: Rensa uses the [fxhash](https://crates.io/crates/fxhash) crate which implements the FxHash algorithm, a fast, non-cryptographic hash function, to further optimize performance.

5. **Vectorized operations**: The R-MinHash computation is optimized using vector operations, allowing for efficient parallel processing of multiple hash values.

6. **Memory-efficient data structures**: The implementation uses compact data structures to minimize memory usage while maintaining fast access times.

7. **Efficient LSH implementation**: The LSH index uses a band-based approach with optimized data structures for fast insertion and query operations.

These design choices result in a MinHash implementation that is fast, memory-efficient, and suitable for large-scale similarity estimation and deduplication tasks. While Rensa may not provide the same theoretical guarantees as full C-MinHash, our benchmarks show that it offers significant performance improvements over traditional MinHash implementations like `datasketch`.

## Installation

You can install Rensa using pip:

```bash
pip install rensa
```

## Usage Example

Here's an example of how to use Rensa to deduplicate a dataset:

```python
from datasets import load_dataset
from rensa import RMinHash, RMinHashLSH

# Load the dataset
dataset = load_dataset("gretelai/synthetic_text_to_sql", split="train")

# Initialize MinHash and LSH
num_perm = 128
threshold = 0.5
minhash_lsh = RMinHashLSH(threshold=threshold, num_perm=num_perm, num_bands=16)

# Deduplicate the dataset
unique_indices = set()
for idx, example in enumerate(dataset):
    minhash = RMinHash(num_perm=num_perm, seed=42)
    minhash.update(example["sql"].split())
    
    if not minhash_lsh.query(minhash):
        minhash_lsh.insert(idx, minhash)
        unique_indices.add(idx)

# Create a new dataset with only unique items
deduplicated_dataset = dataset.select(list(unique_indices))

print(f"Original dataset size: {len(dataset)}")
print(f"Deduplicated dataset size: {len(deduplicated_dataset)}")
```

## Benchmark Results

I've conducted extensive benchmarks comparing Rensa to the popular `datasketch` library. Here are the key findings:

1. **Speed**: Rensa consistently outperforms `datasketch` in terms of speed, with performance improvements of 2.5-3 times faster across different numbers of permutations.

2. **Memory Usage**: Memory usage is comparable between Rensa and `datasketch`, with Rensa using slightly less memory for smaller numbers of permutations.

3. **Scalability**: Both implementations show linear growth in time and memory usage as the number of permutations increases, but Rensa maintains its performance advantage across the scale.

4. **Accuracy**: Despite the simplified implementation, Rensa achieves the same deduplication results to `datasketch`, with a high Jaccard similarity between the deduplicated sets produced by both libraries.

[INSERT GRAPH HERE]

These results demonstrate that Rensa offers significant performance benefits while maintaining accuracy, making it an excellent choice for large-scale similarity estimation and deduplication tasks.

## Running the Benchmarks

To run the benchmarks yourself, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/beowolx/rensa.git
   cd rensa
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the simple benchmark:
   ```bash
   python benchmarks/simple_benchmark.py
   ```

4. Run the advanced benchmark:
   ```bash
   python benchmarks/advanced_benchmark.py
   ```

The `simple_benchmark.py` script provides a basic comparison of deduplication performance between Rensa and `datasketch`. The `advanced_benchmark.py` script offers a more comprehensive analysis, including multiple runs with different numbers of permutations, memory usage tracking, and detailed profiling information.

## Limitations and Future Work

While Rensa offers significant performance improvements, it has some limitations compared to `datasketch`:

1. **Feature set**: Rensa currently implements only the core MinHash and LSH functionality. It doesn't include some of the advanced features found in `datasketch`.

2. **Customization**: `datasketch` offers more options for customizing the hash functions and other parameters, while Rensa currently has a more fixed implementation.

3. **Theoretical guarantees**: Due to the simplified C-MinHash implementation, Rensa may not provide the same level of variance reduction as the full C-MinHash algorithm in all scenarios.

Future work on Rensa may include:

- Adding more advanced features and customization options
- Further optimizing performance for specific use cases and data types

Despite these limitations, Rensa's performance benefits make it an excellent choice for applications where speed and efficiency are critical, especially when working with large datasets.

## Contributing

Contributions to Rensa are welcome! Please feel free to submit pull requests, report bugs, or suggest features through the GitHub issue tracker.

## License

Rensa is released under the MIT License. See the LICENSE file for details.