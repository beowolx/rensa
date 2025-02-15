import time
from datasets import load_dataset
from datasketch import MinHash
from rensa import CMinHash
from tqdm import tqdm


def datasketch_minhash(text, num_perm=128):
    m = MinHash(num_perm=num_perm)
    for word in text.split():
        m.update(word.encode('utf-8'))
    return m


def cminhash_minhash(text, num_perm=128):
    m = CMinHash(num_perm=num_perm, seed=42)
    m.update(text.split())
    return m


def benchmark_deduplication(dataset, minhash_func, num_perm=128, desc="Processing"):
    start_time = time.time()

    unique_hashes = set()
    deduplicated_indices = []

    for idx, example in tqdm(enumerate(dataset), total=len(dataset), desc=desc):
        minhash = minhash_func(example["sql"], num_perm)
        # Both Datasketch and CMinHash return a list of integers as their digest.
        hash_tuple = tuple(minhash.digest())

        if hash_tuple not in unique_hashes:
            unique_hashes.add(hash_tuple)
            deduplicated_indices.append(idx)

    end_time = time.time()

    return {
        "time": end_time - start_time,
        "deduplicated_count": len(deduplicated_indices),
        "deduplicated_indices": set(deduplicated_indices)
    }


def run_benchmark():
    print("Loading dataset...")
    sql_dataset = load_dataset("gretelai/synthetic_text_to_sql", split="train")

    print("Running Datasketch benchmark...")
    datasketch_results = benchmark_deduplication(
        sql_dataset, datasketch_minhash, desc="Datasketch")

    print("Running C-MinHash (Rensa) benchmark...")
    cminhash_results = benchmark_deduplication(
        sql_dataset, cminhash_minhash, desc="C-MinHash")

    print("\nBenchmark Results:")
    print(f"Total rows: {len(sql_dataset)}")

    print("\nDatasketch:")
    print(f"Time: {datasketch_results['time']:.2f} seconds")
    print(
        f"Rows removed: {len(sql_dataset) - datasketch_results['deduplicated_count']}")
    print(f"Rows remaining: {datasketch_results['deduplicated_count']}")

    print("\nC-MinHash:")
    print(f"Time: {cminhash_results['time']:.2f} seconds")
    print(
        f"Rows removed: {len(sql_dataset) - cminhash_results['deduplicated_count']}")
    print(f"Rows remaining: {cminhash_results['deduplicated_count']}")

    print("\nComparison:")
    common_indices = datasketch_results['deduplicated_indices'].intersection(
        cminhash_results['deduplicated_indices'])
    print(f"Rows remaining in both: {len(common_indices)}")
    print(
        f"Rows removed only by Datasketch: {len(datasketch_results['deduplicated_indices'] - cminhash_results['deduplicated_indices'])}")
    print(
        f"Rows removed only by C-MinHash: {len(cminhash_results['deduplicated_indices'] - datasketch_results['deduplicated_indices'])}")

    jaccard_similarity = len(common_indices) / len(
        datasketch_results['deduplicated_indices'].union(cminhash_results['deduplicated_indices']))
    print(f"Jaccard similarity of deduplicated sets: {jaccard_similarity:.4f}")

    # Performance comparison
    print("\nPerformance Comparison:")
    if datasketch_results['time'] < cminhash_results['time']:
        faster = "Datasketch"
        slower = "C-MinHash"
        time_diff = cminhash_results['time'] - datasketch_results['time']
        speedup = cminhash_results['time'] / datasketch_results['time']
    else:
        faster = "C-MinHash"
        slower = "Datasketch"
        time_diff = datasketch_results['time'] - cminhash_results['time']
        speedup = datasketch_results['time'] / cminhash_results['time']

    print(f"{faster} ran faster than {slower} by {time_diff:.2f} seconds.")
    print(f"{faster} is {speedup:.2f}x faster than {slower}.")


if __name__ == "__main__":
    run_benchmark()
