import argparse
import json
import os
import platform
import statistics
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
DEFAULT_NUM_PERM = 128
DEFAULT_WARMUP_RUNS = 0
DEFAULT_MEASURED_RUNS = 1


MinHashFactory = Callable[[str, int], object]


def datasketch_minhash(text: str, num_perm: int = DEFAULT_NUM_PERM) -> MinHash:
    minhash = MinHash(num_perm=num_perm)
    for word in text.split():
        minhash.update(word.encode("utf-8"))
    return minhash


def rensa_minhash(text: str, num_perm: int = DEFAULT_NUM_PERM) -> RMinHash:
    minhash = RMinHash(num_perm=num_perm, seed=42)
    minhash.update(text.split())
    return minhash


def cminhash_minhash(text: str, num_perm: int = DEFAULT_NUM_PERM) -> CMinHash:
    minhash = CMinHash(num_perm=num_perm, seed=42)
    minhash.update(text.split())
    return minhash


def cminimash(text: str, num_perm: int = DEFAULT_NUM_PERM) -> CMinHash:
    # Backward-compatible alias used by older benchmark scripts/imports.
    return cminhash_minhash(text, num_perm)


def benchmark_deduplication(
    dataset: object,
    minhash_factory: MinHashFactory,
    *,
    num_perm: int,
    desc: str,
    disable_progress: bool,
) -> dict[str, object]:
    start_time = perf_counter()

    unique_hashes: set[tuple[int, ...]] = set()
    deduplicated_indices: set[int] = set()

    iterator = tqdm(
        enumerate(dataset),
        total=len(dataset),
        desc=desc,
        disable=disable_progress,
    )

    for idx, example in iterator:
        minhash = minhash_factory(example["sql"], num_perm)
        digest = tuple(minhash.digest())

        if digest not in unique_hashes:
            unique_hashes.add(digest)
            deduplicated_indices.add(idx)

    elapsed = perf_counter() - start_time
    return {
        "time": elapsed,
        "deduplicated_count": len(deduplicated_indices),
        "deduplicated_indices": deduplicated_indices,
    }


def calculate_jaccard(set_a: set[int], set_b: set[int]) -> float:
    union_size = len(set_a.union(set_b))
    if union_size == 0:
        return 0.0
    return len(set_a.intersection(set_b)) / union_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Datasketch vs R-MinHash vs C-MinHash with optional JSON output.",
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--num-perm", type=int, default=DEFAULT_NUM_PERM)
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

    methods: list[tuple[str, str, MinHashFactory]] = [
        ("datasketch", "Datasketch", datasketch_minhash),
        ("r_minhash", "R-MinHash", rensa_minhash),
        ("c_minhash", "C-MinHash", cminhash_minhash),
    ]

    for warmup_index in range(args.warmup_runs):
        print(f"\nWarmup run {warmup_index + 1}/{args.warmup_runs}...")
        for _, display_name, factory in methods:
            benchmark_deduplication(
                dataset,
                factory,
                num_perm=args.num_perm,
                desc=f"Warmup {display_name}",
                disable_progress=args.disable_progress,
            )

    measured_runs: list[dict[str, object]] = []
    last_run_sets: dict[str, set[int]] = {}

    for run_index in range(args.measured_runs):
        print(f"\nMeasured run {run_index + 1}/{args.measured_runs}...")
        run_results: dict[str, dict[str, object]] = {}
        for key, display_name, factory in methods:
            print(f"Running {display_name} benchmark...")
            result = benchmark_deduplication(
                dataset,
                factory,
                num_perm=args.num_perm,
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

    total_rows = len(dataset)

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Total rows in dataset: {total_rows}")

    final_rows_remaining = measured_runs[-1]["rows_remaining"]
    final_times = measured_runs[-1]["times_seconds"]

    print("\nDatasketch:")
    print(f"  Time: {final_times['datasketch']:.2f} seconds")
    print(f"  Rows removed: {total_rows - final_rows_remaining['datasketch']}")
    print(f"  Rows remaining: {final_rows_remaining['datasketch']}")

    print("\nR-MinHash:")
    print(f"  Time: {final_times['r_minhash']:.2f} seconds")
    print(f"  Rows removed: {total_rows - final_rows_remaining['r_minhash']}")
    print(f"  Rows remaining: {final_rows_remaining['r_minhash']}")

    print("\nC-MinHash:")
    print(f"  Time: {final_times['c_minhash']:.2f} seconds")
    print(f"  Rows removed: {total_rows - final_rows_remaining['c_minhash']}")
    print(f"  Rows remaining: {final_rows_remaining['c_minhash']}")

    print("\n" + "=" * 60)
    print("ACCURACY COMPARISON")
    print("=" * 60)
    print(f"Jaccard similarity between Datasketch and R-MinHash: {ds_vs_r_jaccard:.4f}")
    print(f"Jaccard similarity between Datasketch and C-MinHash: {ds_vs_c_jaccard:.4f}")
    print(f"Jaccard similarity between R-MinHash and C-MinHash: {r_vs_c_jaccard:.4f}")

    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)
    print(f"Median Datasketch time: {datasketch_median:.2f}s")
    print(f"Median R-MinHash time: {rminhash_median:.2f}s")
    print(f"Median C-MinHash time: {cminhash_median:.2f}s")
    print(f"R-MinHash speedup vs Datasketch (median): {speedup_r:.2f}x")
    print(f"C-MinHash speedup vs Datasketch (median): {speedup_c:.2f}x")

    print("\n" + "=" * 60)
    print("DETAILED PERFORMANCE TABLE")
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
            "num_perm": args.num_perm,
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

    run_benchmark(args)


if __name__ == "__main__":
    main()
