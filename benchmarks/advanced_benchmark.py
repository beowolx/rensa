import cProfile
import io
import os
import pstats
import statistics
import time

import matplotlib.pyplot as plt
from datasets import load_dataset
from datasketch import MinHash
from rensa import CMinHash, OptDensMinHash, RMinHash  # type: ignore


def datasketch_minhash(text, num_perm=128):
    m = MinHash(num_perm=num_perm)
    for word in text.split():
        m.update(word.encode("utf-8"))
    return m


def rensa_minhash(text, num_perm=128):
    m = RMinHash(num_perm=num_perm, seed=42)
    m.update(text.split())
    return m


def cminhash_minhash(text, num_perm=128):
    m = CMinHash(num_perm=num_perm, seed=42)
    m.update(text.split())
    return m


def optdens_minhash(text, num_perm=128):
    m = OptDensMinHash(num_perm=num_perm, seed=42)
    m.update(text.split())
    return m


def benchmark_deduplication(dataset, minhash_func, num_perm=128):
    unique_hashes = set()
    deduplicated_indices = []

    for idx, example in enumerate(dataset):
        minhash = minhash_func(example["sql"], num_perm)

        if isinstance(minhash, MinHash):
            hash_tuple = tuple(minhash.digest())
        else:  # RMinHash
            hash_tuple = tuple(minhash.digest())

        if hash_tuple not in unique_hashes:
            unique_hashes.add(hash_tuple)
            deduplicated_indices.append(idx)

    return set(deduplicated_indices)


def measure_time_and_memory(func, *args, **kwargs):
    start_time = time.time()
    func(*args, **kwargs)
    end_time = time.time()

    return end_time - start_time


def profile_func(func, *args, **kwargs):
    pr = cProfile.Profile()
    pr.enable()
    result = func(*args, **kwargs)
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(10)
    return result, s.getvalue()


def calculate_jaccard(set1, set2):
    intersection_len = len(set1.intersection(set2))
    union_len = len(set1.union(set2))
    return intersection_len / union_len if union_len > 0 else 0.0


def run_benchmark(num_runs=5, perm_values=[64, 128, 256]):
    print("Loading dataset...")
    sql_dataset = load_dataset("gretelai/synthetic_text_to_sql", split="train")

    results = {
        "datasketch": {perm: {"time": []} for perm in perm_values},
        "rensa": {perm: {"time": []} for perm in perm_values},
        "cminhash": {perm: {"time": []} for perm in perm_values},
        "optdens": {perm: {"time": []} for perm in perm_values},
    }

    minhash_methods = [
        ("datasketch", datasketch_minhash, "Datasketch"),
        ("rensa", rensa_minhash, "Rensa (RMinHash)"),
        ("cminhash", cminhash_minhash, "Rensa (CMinHash)"),
        ("optdens", optdens_minhash, "Rensa (OptDensMinHash)"),
    ]

    for num_perm in perm_values:
        print(f"\nBenchmarking with {num_perm} permutations:")
        for run in range(num_runs):
            print(f"  Run {run + 1}/{num_runs}")

            for lib_name, minhash_func, _ in minhash_methods:
                print(f"    Benchmarking {lib_name}...")
                time_taken = measure_time_and_memory(
                    benchmark_deduplication, sql_dataset, minhash_func, num_perm
                )
                results[lib_name][num_perm]["time"].append(time_taken)

    print("\nBenchmark Results (Averages over runs):")
    for num_perm in perm_values:
        print(f"\nResults for {num_perm} permutations:")
        for lib_name, _, display_name in minhash_methods:
            times = results[lib_name][num_perm]["time"]
            print(f"  {display_name}:")
            if times:
                print(
                    f"    Time (seconds): mean={statistics.mean(times):.2f}, std={statistics.stdev(times) if len(times) > 1 else 0.0:.2f}"
                )
            else:
                print("    Time (seconds): No data")

    default_perm_for_correctness = 128
    print(
        f"\nChecking correctness (using {default_perm_for_correctness} permutations for all):"
    )

    print("  Running deduplication for correctness checks...")
    ds_deduped_indices = benchmark_deduplication(
        sql_dataset, datasketch_minhash, default_perm_for_correctness
    )
    rensa_deduped_indices = benchmark_deduplication(
        sql_dataset, rensa_minhash, default_perm_for_correctness
    )
    cmin_deduped_indices = benchmark_deduplication(
        sql_dataset, cminhash_minhash, default_perm_for_correctness
    )
    optdens_deduped_indices = benchmark_deduplication(
        sql_dataset, optdens_minhash, default_perm_for_correctness
    )

    print("\n  Jaccard Similarities of Deduplicated Sets:")

    all_deduped_sets = [
        ("Datasketch", ds_deduped_indices),
        ("Rensa (RMinHash)", rensa_deduped_indices),
        ("Rensa (CMinHash)", cmin_deduped_indices),
        ("Rensa (OptDensMinHash)", optdens_deduped_indices),
    ]

    for i in range(len(all_deduped_sets)):
        for j in range(i + 1, len(all_deduped_sets)):
            name1, set1 = all_deduped_sets[i]
            name2, set2 = all_deduped_sets[j]
            jaccard = calculate_jaccard(set1, set2)
            intersection_count = len(set1.intersection(set2))
            print(
                f"    {name1} vs {name2}: {jaccard:.4f} (Intersection: {intersection_count})"
            )

    print(f"\nProfiling (using {default_perm_for_correctness} permutations):")

    for lib_name, minhash_func, display_name in minhash_methods:
        print(f"\nProfiling {display_name}:")
        _, profile_output = profile_func(
            benchmark_deduplication,
            sql_dataset,
            minhash_func,
            default_perm_for_correctness,
        )
        print(profile_output)

    plot_and_save_results(results, perm_values, minhash_methods)


def plot_and_save_results(results, perm_values, minhash_methods, output_dir="assets"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig, ax_time = plt.subplots(1, 1, figsize=(12, 8))
    plt.style.use("seaborn-v0_8-whitegrid")

    colors = ["blue", "red", "green", "purple", "orange"]
    markers = ["o", "s", "D", "^", "v"]

    datasketch_times = [
        statistics.mean(results["datasketch"][p]["time"])
        for p in perm_values
        if results["datasketch"][p]["time"]
    ]

    for i, (lib_name, _, display_name) in enumerate(minhash_methods):
        mean_times = [
            statistics.mean(results[lib_name][p]["time"])
            for p in perm_values
            if results[lib_name][p]["time"]
        ]
        if not mean_times:
            continue

        ax_time.plot(
            perm_values,
            mean_times,
            marker=markers[i % len(markers)],
            linestyle="-",
            color=colors[i % len(colors)],
            label=display_name,
        )

        if lib_name != "datasketch" and datasketch_times:
            for j, p_val in enumerate(perm_values):
                if (
                    j < len(mean_times)
                    and j < len(datasketch_times)
                    and mean_times[j] > 0
                ):
                    speedup = datasketch_times[j] / mean_times[j]
                    ax_time.text(
                        p_val,
                        mean_times[j],
                        f"{speedup:.2f}x faster",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        color=colors[i % len(colors)],
                    )

    ax_time.set_xlabel("Number of Permutations")
    ax_time.set_ylabel("Execution Time (seconds)")
    ax_time.set_title("Execution Time Comparison")
    ax_time.legend()
    ax_time.grid(True, which="both", ls="--", c="0.7")

    fig.suptitle("Rensa vs Datasketch Performance Comparison",
                 fontsize=16, y=0.95)
    plt.tight_layout(rect=(0, 0, 1, 0.93))

    plot_path = os.path.join(output_dir, "benchmark_comparison.png")
    plt.savefig(plot_path)
    print(f"\nPlot saved to {plot_path}")


if __name__ == "__main__":
    run_benchmark()
