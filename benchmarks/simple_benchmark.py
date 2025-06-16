import time

from datasets import load_dataset
from datasketch import MinHash
from rensa import CMinHash, RMinHash
from tqdm import tqdm


def datasketch_minhash(text, num_perm=128):
    m = MinHash(num_perm=num_perm)
    for word in text.split():
        m.update(word.encode("utf-8"))
    return m


def rensa_minhash(text, num_perm=128):
    m = RMinHash(num_perm=num_perm, seed=42)
    m.update(text.split())
    return m


def cminimash(text, num_perm=128):
    m = CMinHash(num_perm=num_perm, seed=42)
    m.update(text.split())
    return m




def benchmark_deduplication(dataset, minhash_func, num_perm=128, desc="Processing"):
    start_time = time.time()

    unique_hashes = set()
    deduplicated_indices = []

    for idx, example in tqdm(enumerate(dataset), total=len(dataset), desc=desc):
        minhash = minhash_func(example["sql"], num_perm)

        if isinstance(minhash, MinHash):
            hash_tuple = tuple(minhash.digest())
        else:
            hash_tuple = tuple(minhash.digest())

        if hash_tuple not in unique_hashes:
            unique_hashes.add(hash_tuple)
            deduplicated_indices.append(idx)

    end_time = time.time()

    return {
        "time": end_time - start_time,
        "deduplicated_count": len(deduplicated_indices),
        "deduplicated_indices": set(deduplicated_indices),
    }


def run_benchmark():
    print("Loading dataset...")
    sql_dataset = load_dataset("gretelai/synthetic_text_to_sql", split="train")

    print("\nRunning Datasketch benchmark...")
    datasketch_results = benchmark_deduplication(
        sql_dataset, datasketch_minhash, desc="Datasketch"
    )

    print("\nRunning Rensa (R-MinHash) benchmark...")
    rensa_results = benchmark_deduplication(
        sql_dataset, rensa_minhash, desc="Rensa R-MinHash"
    )

    print("\nRunning Rensa (C-MinHash) benchmark...")
    cminimash_results = benchmark_deduplication(
        sql_dataset, cminimash, desc="Rensa C-MinHash"
    )


    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Total rows in dataset: {len(sql_dataset)}")

    results = [
        ("Datasketch", datasketch_results),
        ("R-MinHash", rensa_results),
        ("C-MinHash", cminimash_results),
    ]

    for name, result in results:
        print(f"\n{name}:")
        print(f"  Time: {result['time']:.2f} seconds")
        print(
            f"  Rows removed: {len(sql_dataset) - result['deduplicated_count']}")
        print(f"  Rows remaining: {result['deduplicated_count']}")

    print("\n" + "=" * 60)
    print("ACCURACY COMPARISON")
    print("=" * 60)

    # Compare results between methods
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            name1, result1 = results[i]
            name2, result2 = results[j]
            common = result1["deduplicated_indices"].intersection(
                result2["deduplicated_indices"]
            )
            jaccard = len(common) / len(
                result1["deduplicated_indices"].union(
                    result2["deduplicated_indices"])
            )
            print(
                f"Jaccard similarity between {name1} and {name2}: {jaccard:.4f}")

    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)

    # Sort by time
    sorted_results = sorted(
        [(name, r["time"]) for name, r in results], key=lambda x: x[1]
    )

    fastest_name, fastest_time = sorted_results[0]
    print(f"\nFastest method: {fastest_name} ({fastest_time:.2f}s)")

    for i in range(1, len(sorted_results)):
        name, time = sorted_results[i]
        speedup = time / fastest_time
        print(f"{fastest_name} is {speedup:.2f}x faster than {name}")

    # Additional statistics
    print("\n" + "=" * 60)
    print("DETAILED PERFORMANCE TABLE")
    print("=" * 60)
    print(f"{'Method':<20} {'Time (s)':<12} {'Speedup vs Datasketch':<25}")
    print("-" * 57)

    datasketch_time = datasketch_results["time"]
    for name, result in results:
        speedup = datasketch_time / result["time"]
        print(f"{name:<20} {result['time']:<12.2f} {speedup:<25.2f}x")


if __name__ == "__main__":
    run_benchmark()
