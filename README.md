# Rensa

[![CI](https://github.com/beowolx/rensa/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/beowolx/rensa/actions/workflows/CI.yml)
[![PyPI](https://img.shields.io/pypi/v/rensa.svg)](https://pypi.org/project/rensa/)
[![Python](https://img.shields.io/badge/python-%3E%3D3.8-blue.svg)](https://pypi.org/project/rensa/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Rensa is a MinHash library written in Rust with Python bindings. It estimates
Jaccard similarity from compact signatures and finds near-duplicate documents
using locality-sensitive hashing (LSH). Rensa supports batch and streaming
deduplication on Linux, macOS and Windows.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1o1nzwXWAa8kdkEJljbJFW1VuI-3VZLUn?usp=sharing)
Try it in [Maxime Labonne's AutoDedup notebook](#playground).

## Contents

- [Quick comparison with other tools](#quick-comparison-with-other-tools)
- [Why should I use Rensa?](#why-should-i-use-rensa)
- [Why shouldn't I use Rensa?](#why-shouldnt-i-use-rensa)
- [Is it really faster than everything else?](#is-it-really-faster-than-everything-else)
- [Playground](#playground)
- [Installation](#installation)
- [Usage](#usage)

## Quick comparison with other tools

Rensa's rho batch pipeline averaged **600× faster than Datasketch** and
**12× faster than FastSketch**.

![Deduplication time for Rensa compared with Datasketch and FastSketch across seven datasets](assets/bench_time_full_suite.png)

| Comparison | Average speedup |
| --- | ---: |
| Rensa vs. [Datasketch](https://ekzhu.com/datasketch/) | **608.52× faster** |
| Rensa vs. [FastSketch](https://github.com/pzcddm/FastSketchLSH) | **11.92× faster** |

The [benchmark](benchmarks/full_benchmark.py) measures sketch construction and
duplicate detection together across seven datasets, using 128 signature slots,
eight LSH bands, a 0.8 threshold, and both one-thread and eight-thread runs. It
uses Rensa's rho pipeline, which samples token positions to increase throughput.
The [performance section](#is-it-really-faster-than-everything-else) explains how
that differs from full-set MinHash.

## Why should I use Rensa?

- You want to find near-duplicates in a dataset without comparing every pair of
  documents. With LSH enabled, `RMinHashDeduplicator` finds candidate matches and
  checks their estimated similarity against your threshold.
- You want to process a batch or keep filtering documents as they arrive. The
  batch APIs build sketches from token lists in one call; streaming
  deduplicators retain their indexes between calls.
- You need compact signatures. R-MinHash uses 32 bits per slot, so a 128-slot
  signature occupies **512 bytes**, excluding object overhead. The signature
  stays the same size as the document grows.
- You need to update a sketch incrementally. Full-set R-MinHash gives the same
  signature when tokens are reordered, repeated or split across updates.
- You already tokenize your data in Python. Rensa accepts token iterables,
  including strings and bytes, and provides prehashed input APIs when you want
  to reuse token hashes.

Rensa provides full-set **R-MinHash** and **C-MinHash** for similarity estimation,
along with a separate **rho pipeline** for faster bulk filtering. Choose full-set
R-MinHash when token order should have no effect on the result. Rho samples token
positions, so document length and token order affect which duplicates it finds.

## Why shouldn't I use Rensa?

- You need exact similarity or guaranteed duplicate detection. MinHash estimates
  similarity, and LSH can miss matching pairs. Use exact set comparisons if
  those errors are unacceptable.
- You need semantic similarity. Rensa compares token overlap. Two paraphrases
  can mean the same thing and share very few tokens.
- You need other sketches, such as Weighted MinHash or HyperLogLog.
  [Datasketch](https://ekzhu.com/datasketch/) supports both.
- You have a workload where another implementation is faster. Document size,
  signature size and how much work you send through each Python call all matter.
  If Rensa is unexpectedly slow, please [open an issue](https://github.com/beowolx/rensa/issues)
  with a small reproducer.

## Is it really faster than everything else?

Yes, by a wide margin in the batch benchmarks: about **600× faster than
Datasketch** and **12× faster than FastSketch** on average. These results measure
the complete rho batch pipeline, from sketch construction to duplicate detection.

Rensa is fast because:

- **Fewer hash evaluations.** Classical MinHash evaluates a hash function for
  every signature slot for each token. R-MinHash adapts
  [Fast Similarity Sketching](https://arxiv.org/abs/1704.04370): each token chooses
  a slot and rank in up to three seeded bucket rounds. Only slots still empty
  after those rounds need their own fallback hash over all tokens. Dense inputs
  can fill every slot during the bucket rounds and skip fallback entirely.
- **Skipping updates that cannot change the result.** For sufficiently long,
  dense inputs with power-of-two signature sizes, small signature values
  establish an upper bound. R-MinHash can then discard updates that cannot
  improve any slot and skip later rounds.
  It produces the same signature as processing the full token set.
- **SIMD where it pays off.** Runtime dispatch selects NEON, AVX2 or AVX-512
  kernels when available. A few empty slots are handled directly; larger
  fallback sets use SIMD batches. Some conditional bucket loops remain scalar
  because scattering vector results would cost more than it saves.
- **Less work at the Python boundary.** A CPython fast path reads compact ASCII
  strings directly. Specialized list and tuple paths avoid temporary Rust
  strings and a Python call for every token. Token hashing matches
  `rustc_hash::FxHasher` without trait dispatch in the hot path, and prehashed
  input skips token hashing altogether.
- **Less repeated setup.** Sketches with the same `(num_perm, seed)` share
  fallback parameters and their SIMD layouts. Short-input paths keep scratch
  space on the stack, and standard builds use mimalloc for heap allocation.
- **Batch processing in Rust.** Batch APIs write contiguous signature matrices
  and can distribute large batches across Rayon workers. Native LSH processes a
  whole matrix in one call, avoiding a Python sketch object and query call for
  each document.

The rho pipeline samples token positions for bulk filtering. Full-set R-MinHash
uses every token and gives the same signature when tokens are reordered or
repeated. The [accuracy benchmark](benchmarks/accuracy_benchmark.py) checks
estimation error and retrieval against exact Jaccard, including pairs whose
tokens have been reordered.

Rensa also implements a practical variant of
[C-MinHash](https://proceedings.mlr.press/v162/li22m.html), using two seeded
nonlinear permutations and 64-bit signature slots. Its streaming deduplicator
indexes candidate signatures and stops comparing a pair as soon as it cannot
meet the threshold.

## Playground

If you'd like to try Rensa before installing it,
[Maxime Labonne](https://github.com/mlabonne) created an
[AutoDedup notebook](https://colab.research.google.com/drive/1o1nzwXWAa8kdkEJljbJFW1VuI-3VZLUn?usp=sharing)
that runs in Google Colab.

Choose a Hugging Face dataset, the column to deduplicate, a split and the MinHash
implementation, then run the notebook. It reports how many rows remain and shows
removed samples alongside the records they matched, so you can inspect what the
deduplicator considered a duplicate.

## Installation

```bash
uv add rensa
```

## Usage

### Estimate similarity

```python
from rensa import RMinHash

first, second = RMinHash.from_token_sets(
    ["the quick brown fox".split(), "the quick brown dog".split()],
    num_perm=128,
    seed=42,
)
print(first.jaccard(second))
```

Use `update(tokens)` to extend a sketch incrementally. `CMinHash` supports the
same similarity interface.

### Keep unique documents

```python
from rensa import RMinHashDeduplicator

documents = [
    "select id from users",
    "select name from orders",
    "select id from users",
]
dedup = RMinHashDeduplicator(threshold=0.8, num_perm=128, use_lsh=True)
keep = dedup.add_pairs(
    (str(i), text.split()) for i, text in enumerate(documents)
)

print(keep)  # [True, True, False]
```

The deduplicator keeps its index across calls, so the same API works for batches
and streams. LSH candidate retrieval is approximate; the deduplicator checks
candidates against the sketch similarity threshold.
