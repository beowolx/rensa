import time
import statistics
from datasets import load_dataset
from rensa import RMinHash, CMinHash
import memory_profiler
import cProfile
import pstats
import io


def rminhash_minhash(text, num_perm=128):
    m = RMinHash(num_perm=num_perm, seed=42)
    m.update(text.split())
    return m


def cminhash_minhash(text, num_perm=128):
    m = CMinHash(num_perm=num_perm, seed=42)
    m.update(text.split())
    return m


def benchmark_deduplication(dataset, minhash_func, num_perm=128):
    unique_hashes = set()
    deduplicated_indices = []

    for idx, example in enumerate(dataset):
        minhash = minhash_func(example["sql"], num_perm)
        hash_tuple = tuple(minhash.digest())

        if hash_tuple not in unique_hashes:
            unique_hashes.add(hash_tuple)
            deduplicated_indices.append(idx)

    return set(deduplicated_indices)


def measure_time_and_memory(func, *args, **kwargs):
    start_time = time.time()
    memory_usage = memory_profiler.memory_usage(
        (func, args, kwargs), interval=0.1, timeout=None)
    end_time = time.time()

    return end_time - start_time, max(memory_usage) - min(memory_usage)


def profile_func(func, *args, **kwargs):
    pr = cProfile.Profile()
    pr.enable()
    result = func(*args, **kwargs)
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(10)
    return result, s.getvalue()


def run_benchmark(num_runs=5, perm_values=[64, 128, 256]):
    print("Loading dataset...")
    sql_dataset = load_dataset("gretelai/synthetic_text_to_sql", split="train")

    results = {
        "rminhash": {perm: {"time": [], "memory": []} for perm in perm_values},
        "cminhash": {perm: {"time": [], "memory": []} for perm in perm_values}
    }

    for num_perm in perm_values:
        print(f"\nBenchmarking with {num_perm} permutations:")
        for run in range(num_runs):
            print(f"  Run {run + 1}/{num_runs}")

            for lib, minhash_func in [("rminhash", rminhash_minhash), ("cminhash", cminhash_minhash)]:
                time_taken, memory_used = measure_time_and_memory(
                    benchmark_deduplication, sql_dataset, minhash_func, num_perm)
                results[lib][num_perm]["time"].append(time_taken)
                results[lib][num_perm]["memory"].append(memory_used)

    print("\nBenchmark Results:")
    for num_perm in perm_values:
        print(f"\nResults for {num_perm} permutations:")
        for lib in ["rminhash", "cminhash"]:
            times = results[lib][num_perm]["time"]
            memories = results[lib][num_perm]["memory"]
            print(f"  {lib.upper()}:")
            print(
                f"    Time (seconds): mean={statistics.mean(times):.2f}, std={statistics.stdev(times):.2f}")
            print(
                f"    Memory (MB): mean={statistics.mean(memories):.2f}, std={statistics.stdev(memories):.2f}")

    # Correctness check
    print("\nChecking correctness...")
    rminhash_result = benchmark_deduplication(sql_dataset, rminhash_minhash)
    cminhash_result = benchmark_deduplication(sql_dataset, cminhash_minhash)
    jaccard_similarity = len(rminhash_result.intersection(
        cminhash_result)) / len(rminhash_result.union(cminhash_result))
    print(f"Jaccard similarity of deduplicated sets: {jaccard_similarity:.4f}")

    # Profiling
    print("\nProfiling R-MinHash:")
    _, rminhash_profile = profile_func(
        benchmark_deduplication, sql_dataset, rminhash_minhash)
    print(rminhash_profile)

    print("\nProfiling C-MinHash:")
    _, cminhash_profile = profile_func(
        benchmark_deduplication, sql_dataset, cminhash_minhash)
    print(cminhash_profile)


if __name__ == "__main__":
    run_benchmark()
