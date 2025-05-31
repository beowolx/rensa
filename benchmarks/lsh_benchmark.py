import argparse
import time

from datasets import load_dataset
from datasketch import MinHash, MinHashLSH
from rensa import RMinHash, RMinHashLSH
from tqdm import tqdm


def create_rensa_minhash(text, num_perm, seed):
    m = RMinHash(num_perm=num_perm, seed=seed)
    m.update(text.split())
    return m


def create_datasketch_minhash(text, num_perm):
    m = MinHash(num_perm=num_perm)
    for word in text.split():
        m.update(word.encode("utf-8"))
    return m


# Calculate optimal number of bands (similar to datasketch's approach)
def calculate_optimal_num_bands(threshold, num_perm):
    """Calculate the optimal number of bands to achieve the desired threshold."""
    # This approximates datasketch's internal calculation
    # For a threshold t, we want to find b (bands) and r (rows per band) such that:
    # - b * r = num_perm
    # - A pair with Jaccard similarity s has probability 1-(1-s^r)^b of being a candidate
    # - We want this probability to be high when s >= threshold

    best_num_bands = 1
    best_error = float("inf")

    for b in range(1, num_perm + 1):
        if num_perm % b != 0:
            continue
        r = num_perm // b
        prob_at_threshold = 1 - (1 - threshold**r) ** b
        error = abs(prob_at_threshold - 0.5)
        if error < best_error:
            best_error = error
            best_num_bands = b

    return best_num_bands


def deduplicate_with_rensa_lsh(
    dataset,
    num_perm,
    seed,
    lsh_threshold,
    num_bands,
    final_jaccard_threshold,
    limit=None,
):
    print(
        f"\nRensa LSH Deduplication (num_perm={num_perm}, lsh_threshold={lsh_threshold}, "
        f"num_bands={num_bands}, rows_per_band={num_perm // num_bands}, final_jaccard_threshold={final_jaccard_threshold})"
    )
    start_time = time.time()

    if limit:
        dataset = dataset.select(range(limit))
        print(f"Processing a limited dataset of {limit} rows.")

    # Phase 1: Generate MinHashes
    print("Phase 1: Generating Rensa MinHashes...")
    phase1_start = time.time()
    minhashes = {}
    for idx, example in tqdm(
        enumerate(dataset), total=len(dataset), desc="Rensa MinHashing"
    ):
        minhashes[idx] = create_rensa_minhash(example["sql"], num_perm, seed)
    phase1_time = time.time() - phase1_start

    # Phase 2: Build LSH Index
    print("Phase 2: Building Rensa LSH index...")
    phase2_start = time.time()
    lsh_index = RMinHashLSH(
        threshold=lsh_threshold, num_perm=num_perm, num_bands=num_bands
    )
    for doc_id, rminhash_obj in tqdm(
        minhashes.items(), desc="Inserting into Rensa LSH"
    ):
        lsh_index.insert(doc_id, rminhash_obj)
    phase2_time = time.time() - phase2_start

    # Phase 3: Query and Deduplicate
    print("Phase 3: Querying Rensa LSH and deduplicating...")
    phase3_start = time.time()
    to_remove = set()
    sorted_doc_ids = sorted(minhashes.keys())
    total_candidates_checked = 0

    for doc_id in tqdm(sorted_doc_ids, desc="Rensa LSH Querying"):
        if doc_id in to_remove:
            continue

        query_minhash = minhashes[doc_id]
        candidate_ids = lsh_index.query(query_minhash)
        total_candidates_checked += len(candidate_ids)

        for candidate_id in candidate_ids:
            if candidate_id == doc_id or candidate_id in to_remove:
                continue

            if candidate_id not in minhashes:
                continue

            candidate_minhash = minhashes[candidate_id]
            actual_jaccard = query_minhash.jaccard(candidate_minhash)

            if actual_jaccard >= final_jaccard_threshold:
                if doc_id < candidate_id:
                    to_remove.add(candidate_id)
                else:
                    to_remove.add(doc_id)
                    break

    phase3_time = time.time() - phase3_start
    kept_indices = set(sorted_doc_ids) - to_remove
    total_time = time.time() - start_time

    return {
        "total_time": total_time,
        "phase1_time": phase1_time,
        "phase2_time": phase2_time,
        "phase3_time": phase3_time,
        "kept_indices": kept_indices,
        "removed_count": len(to_remove),
        "kept_count": len(kept_indices),
        "total_candidates": total_candidates_checked,
        "avg_candidates_per_query": total_candidates_checked / len(sorted_doc_ids)
        if sorted_doc_ids
        else 0,
    }


def deduplicate_with_datasketch_lsh(
    dataset, num_perm, lsh_threshold, final_jaccard_threshold, limit=None
):
    print(
        f"\nDatasketch LSH Deduplication (num_perm={num_perm}, lsh_threshold={lsh_threshold}, "
        f"final_jaccard_threshold={final_jaccard_threshold})"
    )
    # Note: datasketch automatically calculates num_bands internally
    start_time = time.time()

    if limit:
        dataset = dataset.select(range(limit))
        print(f"Processing a limited dataset of {limit} rows.")

    # Phase 1: Generate MinHashes
    print("Phase 1: Generating Datasketch MinHashes...")
    phase1_start = time.time()
    minhashes = {}
    for idx, example in tqdm(
        enumerate(dataset), total=len(dataset), desc="Datasketch MinHashing"
    ):
        minhashes[idx] = create_datasketch_minhash(example["sql"], num_perm)
    phase1_time = time.time() - phase1_start

    # Phase 2: Build LSH Index
    print("Phase 2: Building Datasketch LSH index...")
    phase2_start = time.time()
    lsh = MinHashLSH(threshold=lsh_threshold, num_perm=num_perm)

    for doc_id, minhash_obj in tqdm(
        minhashes.items(), desc="Inserting into Datasketch LSH"
    ):
        lsh.insert(str(doc_id), minhash_obj)
    phase2_time = time.time() - phase2_start

    # Phase 3: Query and Deduplicate
    print("Phase 3: Querying Datasketch LSH and deduplicating...")
    phase3_start = time.time()
    to_remove = set()
    sorted_doc_ids = sorted(minhashes.keys())
    total_candidates_checked = 0

    for doc_id in tqdm(sorted_doc_ids, desc="Datasketch LSH Querying"):
        if doc_id in to_remove:
            continue

        query_minhash = minhashes[doc_id]
        candidate_keys = lsh.query(query_minhash)
        total_candidates_checked += len(candidate_keys)

        for candidate_key in candidate_keys:
            candidate_id = int(candidate_key)
            if candidate_id == doc_id or candidate_id in to_remove:
                continue

            if candidate_id not in minhashes:
                continue

            candidate_minhash = minhashes[candidate_id]
            actual_jaccard = query_minhash.jaccard(candidate_minhash)

            if actual_jaccard >= final_jaccard_threshold:
                if doc_id < candidate_id:
                    to_remove.add(candidate_id)
                else:
                    to_remove.add(doc_id)
                    break

    phase3_time = time.time() - phase3_start
    kept_indices = set(sorted_doc_ids) - to_remove
    total_time = time.time() - start_time

    return {
        "total_time": total_time,
        "phase1_time": phase1_time,
        "phase2_time": phase2_time,
        "phase3_time": phase3_time,
        "kept_indices": kept_indices,
        "removed_count": len(to_remove),
        "kept_count": len(kept_indices),
        "total_candidates": total_candidates_checked,
        "avg_candidates_per_query": total_candidates_checked / len(sorted_doc_ids)
        if sorted_doc_ids
        else 0,
    }


def run_lsh_benchmark(args):
    print("Loading dataset...")
    sql_dataset_full = load_dataset("gretelai/synthetic_text_to_sql", split="train")
    dataset_limit = (
        args.limit if args.limit and args.limit < len(sql_dataset_full) else None
    )

    print(
        f"Using {dataset_limit if dataset_limit else len(sql_dataset_full)} rows for the benchmark."
    )

    # Parameters
    NUM_PERM = args.num_perm
    SEED = 42
    LSH_THRESHOLD = args.lsh_threshold
    FINAL_JACCARD_THRESHOLD = args.final_jaccard_threshold

    # Calculate optimal number of bands for fair comparison
    if args.num_bands:
        NUM_BANDS_RENSA = args.num_bands
    else:
        NUM_BANDS_RENSA = calculate_optimal_num_bands(LSH_THRESHOLD, NUM_PERM)
        print(
            f"\nCalculated optimal num_bands for threshold {LSH_THRESHOLD}: {NUM_BANDS_RENSA}"
        )

    # Run benchmarks
    rensa_lsh_results = deduplicate_with_rensa_lsh(
        sql_dataset_full,
        NUM_PERM,
        SEED,
        LSH_THRESHOLD,
        NUM_BANDS_RENSA,
        FINAL_JACCARD_THRESHOLD,
        limit=dataset_limit,
    )

    datasketch_lsh_results = deduplicate_with_datasketch_lsh(
        sql_dataset_full,
        NUM_PERM,
        LSH_THRESHOLD,
        FINAL_JACCARD_THRESHOLD,
        limit=dataset_limit,
    )

    # Print results
    print("\n" + "=" * 60)
    print("LSH BENCHMARK RESULTS")
    print("=" * 60)
    original_size = dataset_limit if dataset_limit else len(sql_dataset_full)
    print(f"Original dataset size: {original_size}")

    print("\nRensa RMinHashLSH:")
    print(f"  Total Time: {rensa_lsh_results['total_time']:.2f} seconds")
    print(f"    - MinHash generation: {rensa_lsh_results['phase1_time']:.2f}s")
    print(f"    - LSH index building: {rensa_lsh_results['phase2_time']:.2f}s")
    print(f"    - Query & deduplication: {rensa_lsh_results['phase3_time']:.2f}s")
    print(f"  Rows kept: {rensa_lsh_results['kept_count']}")
    print(f"  Rows removed: {rensa_lsh_results['removed_count']}")
    print(
        f"  Avg candidates per query: {rensa_lsh_results['avg_candidates_per_query']:.2f}"
    )

    print("\nDatasketch MinHashLSH:")
    print(f"  Total Time: {datasketch_lsh_results['total_time']:.2f} seconds")
    print(f"    - MinHash generation: {datasketch_lsh_results['phase1_time']:.2f}s")
    print(f"    - LSH index building: {datasketch_lsh_results['phase2_time']:.2f}s")
    print(f"    - Query & deduplication: {datasketch_lsh_results['phase3_time']:.2f}s")
    print(f"  Rows kept: {datasketch_lsh_results['kept_count']}")
    print(f"  Rows removed: {datasketch_lsh_results['removed_count']}")
    print(
        f"  Avg candidates per query: {datasketch_lsh_results['avg_candidates_per_query']:.2f}"
    )

    # Accuracy comparison
    intersection_kept = len(
        rensa_lsh_results["kept_indices"].intersection(
            datasketch_lsh_results["kept_indices"]
        )
    )
    union_kept = len(
        rensa_lsh_results["kept_indices"].union(datasketch_lsh_results["kept_indices"])
    )
    jaccard_kept_sets = intersection_kept / union_kept if union_kept > 0 else 0.0

    print("\n" + "=" * 60)
    print("ACCURACY COMPARISON (Jaccard of Kept Sets)")
    print("=" * 60)
    print(
        f"Jaccard similarity between Rensa and Datasketch kept sets: {jaccard_kept_sets:.4f}"
    )
    print(f"  Intersection size: {intersection_kept}")
    print(f"  Union size: {union_kept}")
    print(
        f"  Rensa kept: {rensa_lsh_results['kept_count']}, "
        f"Datasketch kept: {datasketch_lsh_results['kept_count']}"
    )

    # Check if results are identical
    if jaccard_kept_sets >= 0.99:
        print("\n✓ Both algorithms produced NEARLY IDENTICAL deduplication results!")
    else:
        print("\n✗ Algorithms produced DIFFERENT deduplication results.")
        # Show some differences
        rensa_only = (
            rensa_lsh_results["kept_indices"] - datasketch_lsh_results["kept_indices"]
        )
        datasketch_only = (
            datasketch_lsh_results["kept_indices"] - rensa_lsh_results["kept_indices"]
        )
        print(f"  Documents kept only by Rensa: {len(rensa_only)}")
        print(f"  Documents kept only by Datasketch: {len(datasketch_only)}")

    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)

    # Overall speedup
    if datasketch_lsh_results["total_time"] > 0:
        overall_speedup = (
            datasketch_lsh_results["total_time"] / rensa_lsh_results["total_time"]
        )
        if overall_speedup > 1:
            print(
                f"Rensa LSH was {overall_speedup:.2f}x faster overall than Datasketch LSH."
            )
        else:
            print(
                f"Datasketch LSH was {1 / overall_speedup:.2f}x faster overall than Rensa LSH."
            )

    # Phase-by-phase comparison
    print("\nPhase-by-phase speedup (Datasketch time / Rensa time):")
    phases = ["phase1_time", "phase2_time", "phase3_time"]
    phase_names = ["MinHash generation", "LSH index building", "Query & deduplication"]

    for phase, name in zip(phases, phase_names):
        if rensa_lsh_results[phase] > 0:
            speedup = datasketch_lsh_results[phase] / rensa_lsh_results[phase]
            print(f"  {name}: {speedup:.2f}x")

    # Efficiency comparison
    print("\nEfficiency metrics:")
    print(
        f"  Rensa average candidates per query: {rensa_lsh_results['avg_candidates_per_query']:.2f}"
    )
    print(
        f"  Datasketch average candidates per query: {datasketch_lsh_results['avg_candidates_per_query']:.2f}"
    )

    if rensa_lsh_results["avg_candidates_per_query"] > 0:
        candidate_ratio = (
            datasketch_lsh_results["avg_candidates_per_query"]
            / rensa_lsh_results["avg_candidates_per_query"]
        )
        print(f"  Candidate generation ratio: {candidate_ratio:.2f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark LSH deduplication with Rensa and Datasketch."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of dataset rows to process for faster testing.",
    )
    parser.add_argument(
        "--num_perm", type=int, default=128, help="Number of permutations for MinHash."
    )
    parser.add_argument(
        "--lsh_threshold",
        type=float,
        default=0.8,
        help="LSH threshold parameter for candidate generation.",
    )
    parser.add_argument(
        "--num_bands",
        type=int,
        default=None,
        help="Number of bands for Rensa LSH (default: calculated optimally based on threshold).",
    )
    parser.add_argument(
        "--final_jaccard_threshold",
        type=float,
        default=0.85,
        help="Final Jaccard similarity threshold for deduplication.",
    )

    cli_args = parser.parse_args()
    run_lsh_benchmark(cli_args)
