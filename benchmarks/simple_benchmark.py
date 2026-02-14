import argparse
import json
import os
import platform
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Callable

from datasets import load_dataset
from datasketch import MinHash
from rensa import CMinHash, RMinHash
from tqdm import tqdm

DEFAULT_DATASET = "gretelai/synthetic_text_to_sql"
DEFAULT_SPLIT = "train"
DEFAULT_REVISION = "740ab236e64503fba51be1101df7a1be83bf455d"
DEFAULT_TEXT_COLUMN = "sql"
DEFAULT_NUM_PERM = 128
DEFAULT_LSH_THRESHOLD = 0.97
DEFAULT_FINAL_JACCARD_THRESHOLD = 0.95
DEFAULT_WARMUP_RUNS = 0
DEFAULT_MEASURED_RUNS = 1


MinHashFactory = Callable[[str, int], object]


def datasketch_minhash(text: str, num_perm: int = DEFAULT_NUM_PERM) -> MinHash:
    minhash = MinHash(num_perm=num_perm)
    minhash.update_batch(token.encode("utf-8") for token in text.split())
    return minhash


def rensa_minhash(text: str, num_perm: int = DEFAULT_NUM_PERM) -> RMinHash:
    minhash = RMinHash(num_perm=num_perm, seed=42)
    minhash.update(text.split())
    return minhash


def cminhash_minhash(text: str, num_perm: int = DEFAULT_NUM_PERM) -> CMinHash:
    minhash = CMinHash(num_perm=num_perm, seed=42)
    minhash.update(text.split())
    return minhash


def digest_tuple(minhash: object) -> tuple[int, ...]:
    if hasattr(minhash, "digest_u64"):
        return tuple(minhash.digest_u64())
    return tuple(minhash.digest())


def calculate_optimal_num_bands(threshold: float, num_perm: int) -> int:
    best_num_bands = 1
    best_error = float("inf")

    for num_bands in range(1, num_perm + 1):
        if num_perm % num_bands != 0:
            continue

        rows_per_band = num_perm // num_bands
        candidate_probability = 1 - (1 - threshold**rows_per_band) ** num_bands
        error = abs(candidate_probability - 0.5)
        if error < best_error:
            best_error = error
            best_num_bands = num_bands

    return best_num_bands


def benchmark_deduplication(
    texts: list[str],
    minhash_factory: MinHashFactory,
    *,
    num_perm: int,
    num_bands: int,
    final_jaccard_threshold: float,
    desc: str,
    disable_progress: bool,
) -> dict[str, object]:
    start_time = perf_counter()

    rows_per_band = num_perm // num_bands
    band_tables: list[dict[tuple[int, ...], list[int]]] = [
        defaultdict(list) for _ in range(num_bands)
    ]
    kept_minhashes: dict[int, object] = {}
    kept_indices: set[int] = set()

    total_candidates = 0

    iterator = tqdm(
        enumerate(texts),
        total=len(texts),
        desc=desc,
        disable=disable_progress,
    )

    for idx, text in iterator:
        minhash = minhash_factory(text, num_perm)
        signature = digest_tuple(minhash)

        if len(signature) != num_perm:
            raise ValueError(
                f"MinHash digest length mismatch: expected {num_perm}, got {len(signature)}"
            )

        candidate_ids: set[int] = set()
        for band_index in range(num_bands):
            start = band_index * rows_per_band
            stop = start + rows_per_band
            band_key = signature[start:stop]
            candidate_ids.update(band_tables[band_index].get(band_key, []))

        total_candidates += len(candidate_ids)

        duplicate_found = False
        for candidate_id in candidate_ids:
            candidate_minhash = kept_minhashes[candidate_id]
            if minhash.jaccard(candidate_minhash) >= final_jaccard_threshold:
                duplicate_found = True
                break

        if duplicate_found:
            continue

        kept_indices.add(idx)
        kept_minhashes[idx] = minhash

        for band_index in range(num_bands):
            start = band_index * rows_per_band
            stop = start + rows_per_band
            band_key = signature[start:stop]
            band_tables[band_index][band_key].append(idx)

    elapsed = perf_counter() - start_time
    return {
        "time": elapsed,
        "deduplicated_count": len(kept_indices),
        "deduplicated_indices": kept_indices,
        "total_candidates": total_candidates,
        "avg_candidates_per_row": total_candidates / len(texts) if texts else 0.0,
    }


def materialize_texts(dataset: object, text_column: str) -> list[str]:
    return [str(row[text_column]) for row in dataset]


def calculate_jaccard(set_a: set[int], set_b: set[int]) -> float:
    union_size = len(set_a.union(set_b))
    if union_size == 0:
        return 0.0
    return len(set_a.intersection(set_b)) / union_size


def rotate_methods(
    methods: list[tuple[str, str, MinHashFactory]],
    rotation: int,
) -> list[tuple[str, str, MinHashFactory]]:
    offset = rotation % len(methods)
    return methods[offset:] + methods[:offset]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Datasketch vs R-MinHash vs C-MinHash using "
            "threshold-based LSH-style deduplication."
        ),
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--text-column", default=DEFAULT_TEXT_COLUMN)
    parser.add_argument("--num-perm", type=int, default=DEFAULT_NUM_PERM)
    parser.add_argument("--lsh-threshold", type=float, default=DEFAULT_LSH_THRESHOLD)
    parser.add_argument(
        "--num-bands",
        type=int,
        help="Optional fixed number of LSH bands. If omitted, it is calculated from --lsh-threshold.",
    )
    parser.add_argument(
        "--final-jaccard-threshold",
        type=float,
        default=DEFAULT_FINAL_JACCARD_THRESHOLD,
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        help="Optional cap for dataset rows used in benchmark runs.",
    )
    parser.add_argument("--warmup-runs", type=int, default=DEFAULT_WARMUP_RUNS)
    parser.add_argument("--measured-runs", type=int, default=DEFAULT_MEASURED_RUNS)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--disable-progress",
        action="store_true",
        default=bool(os.environ.get("CI")),
        help="Disable tqdm progress bars (default true when CI env var is present).",
    )
    return parser.parse_args()


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    print("Loading dataset...")
    dataset = load_dataset(
        args.dataset,
        split=args.split,
        revision=args.revision,
    )
    if args.max_rows is not None and args.max_rows < len(dataset):
        dataset = dataset.select(range(args.max_rows))
        print(f"Using dataset subset with {len(dataset)} rows.")
    texts = materialize_texts(dataset, args.text_column)
    print(f"Materialized {len(texts)} rows from column '{args.text_column}'.")

    if args.num_bands is None:
        num_bands = calculate_optimal_num_bands(args.lsh_threshold, args.num_perm)
        print(
            f"Using calculated num_bands={num_bands} "
            f"for lsh_threshold={args.lsh_threshold:.3f}."
        )
    else:
        num_bands = args.num_bands
        print(f"Using fixed num_bands={num_bands}.")

    methods: list[tuple[str, str, MinHashFactory]] = [
        ("datasketch", "Datasketch", datasketch_minhash),
        ("r_minhash", "R-MinHash", rensa_minhash),
        ("c_minhash", "C-MinHash", cminhash_minhash),
    ]

    for warmup_index in range(args.warmup_runs):
        print(f"\nWarmup run {warmup_index + 1}/{args.warmup_runs}...")
        ordered_methods = rotate_methods(methods, warmup_index)
        for _, display_name, factory in ordered_methods:
            benchmark_deduplication(
                texts,
                factory,
                num_perm=args.num_perm,
                num_bands=num_bands,
                final_jaccard_threshold=args.final_jaccard_threshold,
                desc=f"Warmup {display_name}",
                disable_progress=args.disable_progress,
            )

    measured_runs: list[dict[str, object]] = []
    last_run_sets: dict[str, set[int]] = {}

    for run_index in range(args.measured_runs):
        print(f"\nMeasured run {run_index + 1}/{args.measured_runs}...")
        ordered_methods = rotate_methods(methods, run_index)

        run_results: dict[str, dict[str, object]] = {}
        for key, display_name, factory in ordered_methods:
            print(f"Running {display_name} benchmark...")
            result = benchmark_deduplication(
                texts,
                factory,
                num_perm=args.num_perm,
                num_bands=num_bands,
                final_jaccard_threshold=args.final_jaccard_threshold,
                desc=f"Run {run_index + 1} {display_name}",
                disable_progress=args.disable_progress,
            )
            run_results[key] = result

        measured_runs.append(
            {
                "times_seconds": {
                    "datasketch": run_results["datasketch"]["time"],
                    "r_minhash": run_results["r_minhash"]["time"],
                    "c_minhash": run_results["c_minhash"]["time"],
                },
                "rows_remaining": {
                    "datasketch": run_results["datasketch"]["deduplicated_count"],
                    "r_minhash": run_results["r_minhash"]["deduplicated_count"],
                    "c_minhash": run_results["c_minhash"]["deduplicated_count"],
                },
                "avg_candidates_per_row": {
                    "datasketch": run_results["datasketch"]["avg_candidates_per_row"],
                    "r_minhash": run_results["r_minhash"]["avg_candidates_per_row"],
                    "c_minhash": run_results["c_minhash"]["avg_candidates_per_row"],
                },
                "total_candidates": {
                    "datasketch": run_results["datasketch"]["total_candidates"],
                    "r_minhash": run_results["r_minhash"]["total_candidates"],
                    "c_minhash": run_results["c_minhash"]["total_candidates"],
                },
            }
        )

        last_run_sets = {
            "datasketch": run_results["datasketch"]["deduplicated_indices"],
            "r_minhash": run_results["r_minhash"]["deduplicated_indices"],
            "c_minhash": run_results["c_minhash"]["deduplicated_indices"],
        }

    datasketch_times = [run["times_seconds"]["datasketch"] for run in measured_runs]
    rminhash_times = [run["times_seconds"]["r_minhash"] for run in measured_runs]
    cminhash_times = [run["times_seconds"]["c_minhash"] for run in measured_runs]

    datasketch_median = statistics.median(datasketch_times)
    rminhash_median = statistics.median(rminhash_times)
    cminhash_median = statistics.median(cminhash_times)

    speedup_r = datasketch_median / rminhash_median
    speedup_c = datasketch_median / cminhash_median

    ds_set = last_run_sets["datasketch"]
    r_set = last_run_sets["r_minhash"]
    c_set = last_run_sets["c_minhash"]

    ds_vs_r_jaccard = calculate_jaccard(ds_set, r_set)
    ds_vs_c_jaccard = calculate_jaccard(ds_set, c_set)
    r_vs_c_jaccard = calculate_jaccard(r_set, c_set)

    ds_vs_r_symdiff = len(ds_set.symmetric_difference(r_set))
    ds_vs_c_symdiff = len(ds_set.symmetric_difference(c_set))
    r_vs_c_symdiff = len(r_set.symmetric_difference(c_set))

    total_rows = len(texts)

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Total rows in dataset: {total_rows}")

    final_rows_remaining = measured_runs[-1]["rows_remaining"]
    final_times = measured_runs[-1]["times_seconds"]
    final_avg_candidates = measured_runs[-1]["avg_candidates_per_row"]

    print("\nDatasketch:")
    print(f"  Time: {final_times['datasketch']:.2f} seconds")
    print(f"  Rows removed: {total_rows - final_rows_remaining['datasketch']}")
    print(f"  Rows remaining: {final_rows_remaining['datasketch']}")
    print(f"  Avg candidates per row: {final_avg_candidates['datasketch']:.2f}")

    print("\nR-MinHash:")
    print(f"  Time: {final_times['r_minhash']:.2f} seconds")
    print(f"  Rows removed: {total_rows - final_rows_remaining['r_minhash']}")
    print(f"  Rows remaining: {final_rows_remaining['r_minhash']}")
    print(f"  Avg candidates per row: {final_avg_candidates['r_minhash']:.2f}")

    print("\nC-MinHash:")
    print(f"  Time: {final_times['c_minhash']:.2f} seconds")
    print(f"  Rows removed: {total_rows - final_rows_remaining['c_minhash']}")
    print(f"  Rows remaining: {final_rows_remaining['c_minhash']}")
    print(f"  Avg candidates per row: {final_avg_candidates['c_minhash']:.2f}")

    print("\n" + "=" * 60)
    print("ACCURACY COMPARISON")
    print("=" * 60)
    print(f"Jaccard similarity between Datasketch and R-MinHash: {ds_vs_r_jaccard:.4f}")
    print(f"Jaccard similarity between Datasketch and C-MinHash: {ds_vs_c_jaccard:.4f}")
    print(f"Jaccard similarity between R-MinHash and C-MinHash: {r_vs_c_jaccard:.4f}")

    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"{'Method':<20} {'Median Time (s)':<18} {'Speedup vs Datasketch':<25}")
    print("-" * 67)
    print(f"{'Datasketch':<20} {datasketch_median:<18.2f} {1.0:<25.2f}x")
    print(f"{'R-MinHash':<20} {rminhash_median:<18.2f} {speedup_r:<25.2f}x")
    print(f"{'C-MinHash':<20} {cminhash_median:<18.2f} {speedup_c:<25.2f}x")

    payload: dict[str, object] = {
        "metadata": {
            "dataset": args.dataset,
            "split": args.split,
            "revision": args.revision,
            "text_column": args.text_column,
            "num_perm": args.num_perm,
            "lsh_threshold": args.lsh_threshold,
            "num_bands": num_bands,
            "rows_per_band": args.num_perm // num_bands,
            "final_jaccard_threshold": args.final_jaccard_threshold,
            "max_rows_requested": args.max_rows,
            "rows_used": len(texts),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        "runs": measured_runs,
        "summary": {
            "median_times_seconds": {
                "datasketch": datasketch_median,
                "r_minhash": rminhash_median,
                "c_minhash": cminhash_median,
            },
            "speedup_vs_datasketch": {
                "r_minhash": speedup_r,
                "c_minhash": speedup_c,
            },
        },
        "accuracy": {
            "jaccard": {
                "datasketch_vs_r_minhash": ds_vs_r_jaccard,
                "datasketch_vs_c_minhash": ds_vs_c_jaccard,
                "r_minhash_vs_c_minhash": r_vs_c_jaccard,
            },
            "set_equality": {
                "datasketch_equals_r_minhash": ds_set == r_set,
            },
            "set_diff_counts": {
                "ds_vs_r_symdiff_count": ds_vs_r_symdiff,
                "ds_vs_c_symdiff_count": ds_vs_c_symdiff,
                "r_vs_c_symdiff_count": r_vs_c_symdiff,
            },
            "rows_removed": {
                "datasketch": total_rows - len(ds_set),
                "r_minhash": total_rows - len(r_set),
                "c_minhash": total_rows - len(c_set),
            },
        },
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"\nWrote benchmark JSON to {args.output_json}")

    return payload


def main() -> None:
    args = parse_args()

    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs must be >= 0")
    if args.measured_runs <= 0:
        raise ValueError("--measured-runs must be > 0")
    if args.num_perm <= 0:
        raise ValueError("--num-perm must be > 0")
    if args.max_rows is not None and args.max_rows <= 0:
        raise ValueError("--max-rows must be > 0 when provided")
    if not 0.0 <= args.lsh_threshold <= 1.0:
        raise ValueError("--lsh-threshold must be in [0.0, 1.0]")
    if not 0.0 <= args.final_jaccard_threshold <= 1.0:
        raise ValueError("--final-jaccard-threshold must be in [0.0, 1.0]")

    if args.num_bands is not None:
        if args.num_bands <= 0:
            raise ValueError("--num-bands must be > 0 when provided")
        if args.num_bands > args.num_perm:
            raise ValueError("--num-bands must be <= --num-perm")
        if args.num_perm % args.num_bands != 0:
            raise ValueError("--num-bands must divide --num-perm")

    run_benchmark(args)


if __name__ == "__main__":
    main()
