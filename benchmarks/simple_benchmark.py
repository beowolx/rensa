import time

from datasets import load_dataset
from datasketch import MinHash
from rensa import CMinHash, RMinHash  # type: ignore
from tqdm import tqdm


def datasketch_minhash(text, num_perm=128):
    m = MinHash(num_perm=num_perm)
    for word in text.split():
        m.update(word.encode('utf-8'))
    return m


def rensa_minhash(text, num_perm=128):
    m = RMinHash(num_perm=num_perm, seed=42)
    m.update(text.split())
    return m


def cminhash_double_hash_minhash(text, num_perm=128):
    m = CMinHash(num_perm=num_perm, seed=42, mode="double_hash")
    m.update(text.split())
    return m


def cminhash_circulant_minhash(text, num_perm=128):
    m = CMinHash(num_perm=num_perm, seed=42, mode="circulant")
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
        else:  # RMinHash
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


def calculate_jaccard(set1_indices, set2_indices):
    intersection_len = len(set1_indices.intersection(set2_indices))
    union_len = len(set1_indices.union(set2_indices))
    return intersection_len / union_len if union_len > 0 else 0.0


def print_performance_comparison_pair(name1, time1, name2, time2):
    if time1 == time2:
        print(f"{name1} and {name2} had the same performance: {time1:.2f}s.")
        return

    if time1 < time2:
        faster_name, slower_name = name1, name2
        faster_time, slower_time = time1, time2
    else:
        faster_name, slower_name = name2, name1
        faster_time, slower_time = time2, time1

    time_diff = slower_time - faster_time
    if faster_time > 0:
        speedup = slower_time / faster_time
        print(
            f"{faster_name} is {speedup:.2f}x faster than {slower_name} (by {time_diff:.2f}s).")
    else:
        print(
            f"{faster_name} finished almost instantly, {slower_name} took {time_diff:.2f}s.")


def run_benchmark():
    print("Loading dataset...")
    sql_dataset = load_dataset("gretelai/synthetic_text_to_sql", split="train")
    dataset_len = len(sql_dataset)  # type: ignore

    print("Running Datasketch benchmark...")
    datasketch_results = benchmark_deduplication(
        sql_dataset, datasketch_minhash, desc="Datasketch")

    print("Running Rensa benchmark...")
    rensa_results = benchmark_deduplication(
        sql_dataset, rensa_minhash, desc="Rensa")

    print("Running CMinHash (Double Hash) benchmark...")
    cminhash_double_hash_results = benchmark_deduplication(
        sql_dataset, cminhash_double_hash_minhash, desc="CMinHash (DH)")

    print("Running CMinHash (Circulant) benchmark...")
    cminhash_circulant_results = benchmark_deduplication(
        sql_dataset, cminhash_circulant_minhash, desc="CMinHash (Circ)")

    print("\nBenchmark Results:")
    print(f"Total rows: {dataset_len}")

    print("\nDatasketch:")
    print(f"Time: {datasketch_results['time']:.2f} seconds")
    print(
        f"Rows removed: {dataset_len - datasketch_results['deduplicated_count']}")
    print(f"Rows remaining: {datasketch_results['deduplicated_count']}")

    print("\nRensa:")
    print(f"Time: {rensa_results['time']:.2f} seconds")
    print(f"Rows removed: {dataset_len - rensa_results['deduplicated_count']}")
    print(f"Rows remaining: {rensa_results['deduplicated_count']}")

    print("\nCMinHash (Double Hash):")
    print(f"Time: {cminhash_double_hash_results['time']:.2f} seconds")
    print(
        f"Rows removed: {dataset_len - cminhash_double_hash_results['deduplicated_count']}")
    print(
        f"Rows remaining: {cminhash_double_hash_results['deduplicated_count']}")

    print("\nCMinHash (Circulant):")
    print(f"Time: {cminhash_circulant_results['time']:.2f} seconds")
    print(
        f"Rows removed: {dataset_len - cminhash_circulant_results['deduplicated_count']}")
    print(
        f"Rows remaining: {cminhash_circulant_results['deduplicated_count']}")

    print("\nComparison Metrics:")

    ds_indices = datasketch_results['deduplicated_indices']
    rensa_indices = rensa_results['deduplicated_indices']
    cmin_dh_indices = cminhash_double_hash_results['deduplicated_indices']
    cmin_circ_indices = cminhash_circulant_results['deduplicated_indices']

    print("\n  Jaccard Similarities (of deduplicated sets):")

    def print_jaccard(name1, set1, name2, set2):
        jaccard = calculate_jaccard(set1, set2)
        intersection_count = len(set1.intersection(set2))
        print(
            f"    {name1} vs {name2}: {jaccard:.4f} (Intersection: {intersection_count})")

    print_jaccard("Datasketch", ds_indices, "Rensa", rensa_indices)
    print_jaccard("Datasketch", ds_indices, "CMinHash (DH)", cmin_dh_indices)
    print_jaccard("Datasketch", ds_indices,
                  "CMinHash (Circ)", cmin_circ_indices)
    print_jaccard("Rensa", rensa_indices, "CMinHash (DH)", cmin_dh_indices)
    print_jaccard("Rensa", rensa_indices, "CMinHash (Circ)", cmin_circ_indices)
    print_jaccard("CMinHash (DH)", cmin_dh_indices,
                  "CMinHash (Circ)", cmin_circ_indices)

    print("\n  Differences in Deduplicated Sets (Unique Rows Kept):")

    def print_set_differences(name1, set1, name2, set2):
        diff1_vs_2 = len(set1 - set2)
        diff2_vs_1 = len(set2 - set1)
        # "Rows removed by X compared to Y" means "Rows kept by Y but not X"
        print(f"    Rows kept by {name1} but not {name2}: {diff1_vs_2}")
        print(f"    Rows kept by {name2} but not {name1}: {diff2_vs_1}")

    print("    Datasketch vs Rensa:")
    print_set_differences("Datasketch", ds_indices, "Rensa", rensa_indices)
    print("    Datasketch vs CMinHash (DH):")
    print_set_differences("Datasketch", ds_indices,
                          "CMinHash (DH)", cmin_dh_indices)
    print("    Datasketch vs CMinHash (Circulant):")
    print_set_differences("Datasketch", ds_indices,
                          "CMinHash (Circ)", cmin_circ_indices)
    print("    Rensa vs CMinHash (DH):")
    print_set_differences("Rensa", rensa_indices,
                          "CMinHash (DH)", cmin_dh_indices)
    print("    Rensa vs CMinHash (Circulant):")
    print_set_differences("Rensa", rensa_indices,
                          "CMinHash (Circ)", cmin_circ_indices)

    print("\nPerformance Comparison:")
    all_results_timed = {
        "Datasketch": datasketch_results['time'],
        "Rensa": rensa_results['time'],
        "CMinHash (DH)": cminhash_double_hash_results['time'],
        "CMinHash (Circulant)": cminhash_circulant_results['time'],
    }

    # Specific comparisons requested
    print_performance_comparison_pair("Datasketch", all_results_timed["Datasketch"],
                                      "CMinHash (DH)", all_results_timed["CMinHash (DH)"])
    print_performance_comparison_pair("Datasketch", all_results_timed["Datasketch"],
                                      "CMinHash (Circulant)", all_results_timed["CMinHash (Circulant)"])
    print_performance_comparison_pair("Rensa", all_results_timed["Rensa"],
                                      "CMinHash (DH)", all_results_timed["CMinHash (DH)"])
    print_performance_comparison_pair("Rensa", all_results_timed["Rensa"],
                                      "CMinHash (Circulant)", all_results_timed["CMinHash (Circulant)"])

    # Additional useful comparisons
    print_performance_comparison_pair("Datasketch", all_results_timed["Datasketch"],
                                      "Rensa", all_results_timed["Rensa"])
    print_performance_comparison_pair("CMinHash (DH)", all_results_timed["CMinHash (DH)"],
                                      "CMinHash (Circulant)", all_results_timed["CMinHash (Circulant)"])

    print("\nSorted Performance (Fastest to Slowest):")
    sorted_timings = sorted(all_results_timed.items(),
                            key=lambda item: item[1])
    for name, timing in sorted_timings:
        print(f"  {name}: {timing:.2f} seconds")


if __name__ == "__main__":
    run_benchmark()
