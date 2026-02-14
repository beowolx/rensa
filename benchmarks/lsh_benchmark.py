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


def calculate_optimal_num_bands(threshold, num_perm):
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


def deduplicate_with_lsh(
    dataset,
    *,
    label,
    minhash_factory,
    lsh_factory,
    insert_fn,
    query_fn,
    final_jaccard_threshold,
    limit=None,
):
    start_time = time.time()

    if limit:
        dataset = dataset.select(range(limit))
        print(f"Processing a limited dataset of {limit} rows.")

    print(f"Phase 1: Generating {label} MinHashes...")
    phase1_start = time.time()
    minhashes = {}
    for idx, example in tqdm(
        enumerate(dataset), total=len(dataset), desc=f"{label} MinHashing"
    ):
        minhashes[idx] = minhash_factory(example["sql"])
    phase1_time = time.time() - phase1_start

    print(f"Phase 2: Building {label} LSH index...")
    phase2_start = time.time()
    lsh_index = lsh_factory()
    for doc_id, minhash_obj in tqdm(
        minhashes.items(), desc=f"Inserting into {label} LSH"
    ):
        insert_fn(lsh_index, doc_id, minhash_obj)
    phase2_time = time.time() - phase2_start

    print(f"Phase 3: Querying {label} LSH and deduplicating...")
    phase3_start = time.time()
    to_remove = set()
    sorted_doc_ids = sorted(minhashes.keys())
    total_candidates_checked = 0

    for doc_id in tqdm(sorted_doc_ids, desc=f"{label} LSH Querying"):
        if doc_id in to_remove:
            continue

        query_minhash = minhashes[doc_id]
        candidate_ids = query_fn(lsh_index, query_minhash)
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


def run_lsh_benchmark(args):
    print("Loading dataset...")
    sql_dataset_full = load_dataset("gretelai/synthetic_text_to_sql", split="train")
    dataset_limit = (
        args.limit if args.limit and args.limit < len(sql_dataset_full) else None
    )

    print(
        f"Using {dataset_limit if dataset_limit else len(sql_dataset_full)} rows for the benchmark."
    )

    NUM_PERM = args.num_perm
    LSH_THRESHOLD = args.lsh_threshold
    FINAL_JACCARD_THRESHOLD = args.final_jaccard_threshold

    if args.num_bands:
        NUM_BANDS_RENSA = args.num_bands
    else:
        NUM_BANDS_RENSA = calculate_optimal_num_bands(LSH_THRESHOLD, NUM_PERM)
        print(
            f"\nCalculated optimal num_bands for threshold {LSH_THRESHOLD}: {NUM_BANDS_RENSA}"
        )

    print(
        f"\nRensa LSH Deduplication (num_perm={NUM_PERM}, lsh_threshold={LSH_THRESHOLD}, "
        f"num_bands={NUM_BANDS_RENSA}, rows_per_band={NUM_PERM // NUM_BANDS_RENSA}, "
        f"final_jaccard_threshold={FINAL_JACCARD_THRESHOLD})"
    )
    rensa_lsh_results = deduplicate_with_lsh(
        sql_dataset_full,
        label="Rensa",
        minhash_factory=lambda text: create_rensa_minhash(text, NUM_PERM, 42),
        lsh_factory=lambda: RMinHashLSH(
            threshold=LSH_THRESHOLD, num_perm=NUM_PERM, num_bands=NUM_BANDS_RENSA
        ),
        insert_fn=lambda lsh, doc_id, mh: lsh.insert(doc_id, mh),
        query_fn=lambda lsh, mh: lsh.query(mh),
        final_jaccard_threshold=FINAL_JACCARD_THRESHOLD,
        limit=dataset_limit,
    )

    print(
        f"\nDatasketch LSH Deduplication (num_perm={NUM_PERM}, lsh_threshold={LSH_THRESHOLD}, "
        f"final_jaccard_threshold={FINAL_JACCARD_THRESHOLD})"
    )
    datasketch_lsh_results = deduplicate_with_lsh(
        sql_dataset_full,
        label="Datasketch",
        minhash_factory=lambda text: create_datasketch_minhash(text, NUM_PERM),
        lsh_factory=lambda: MinHashLSH(threshold=LSH_THRESHOLD, num_perm=NUM_PERM),
        insert_fn=lambda lsh, doc_id, mh: lsh.insert(str(doc_id), mh),
        query_fn=lambda lsh, mh: [int(k) for k in lsh.query(mh)],
        final_jaccard_threshold=FINAL_JACCARD_THRESHOLD,
        limit=dataset_limit,
    )

    print("\n" + "=" * 60)
    print("LSH BENCHMARK RESULTS")
    print("=" * 60)
    original_size = dataset_limit if dataset_limit else len(sql_dataset_full)
    print(f"Original dataset size: {original_size}")

    for label, results in [
        ("Rensa RMinHashLSH", rensa_lsh_results),
        ("Datasketch MinHashLSH", datasketch_lsh_results),
    ]:
        print(f"\n{label}:")
        print(f"  Total Time: {results['total_time']:.2f} seconds")
        print(f"    - MinHash generation: {results['phase1_time']:.2f}s")
        print(f"    - LSH index building: {results['phase2_time']:.2f}s")
        print(f"    - Query & deduplication: {results['phase3_time']:.2f}s")
        print(f"  Rows kept: {results['kept_count']}")
        print(f"  Rows removed: {results['removed_count']}")
        print(
            f"  Avg candidates per query: {results['avg_candidates_per_query']:.2f}"
        )

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

    if jaccard_kept_sets >= 0.99:
        print("\nBoth algorithms produced nearly identical deduplication results.")
    else:
        print("\nAlgorithms produced different deduplication results.")
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

    print("\nPhase-by-phase speedup (Datasketch time / Rensa time):")
    phases = [
        ("phase1_time", "MinHash generation"),
        ("phase2_time", "LSH index building"),
        ("phase3_time", "Query & deduplication"),
    ]

    for phase_key, phase_name in phases:
        if rensa_lsh_results[phase_key] > 0:
            speedup = datasketch_lsh_results[phase_key] / rensa_lsh_results[phase_key]
            print(f"  {phase_name}: {speedup:.2f}x")

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
