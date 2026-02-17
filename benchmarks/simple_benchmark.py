from __future__ import annotations

import argparse
import json
import os
import pickle
import platform
import statistics
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from full_benchmark import (
    DATASET_PRESETS,
    DEFAULT_CACHE_DIR,
    DEFAULT_NGRAM_SIZE,
    DEFAULT_NUM_BANDS,
    DEFAULT_NUM_PERM,
    DEFAULT_SEED,
    DEFAULT_THRESHOLD,
    THREAD_ENV_VARS,
    jaccard_similarity,
    load_or_prepare_token_cache,
    mismatch_stats,
    resolve_max_rows,
    run_datasketch,
    run_fastsketch,
    run_rensa,
)

ENGINE_KEYS = ("datasketch", "fastsketch", "rensa_r", "rensa_c")
DEFAULT_DATASET_KEY = "ag_news"
DEFAULT_MAX_ROWS = 20_000
DEFAULT_WARMUP_RUNS = 0
DEFAULT_MEASURED_RUNS = 1
DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parents[1] / ".bench" / "profiling" / "simple_benchmark.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simple single-thread benchmark comparing Datasketch, FastSketch, "
            "Rensa R-MinHash, and Rensa C-MinHash on one dataset preset."
        )
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET_KEY,
        choices=sorted(DATASET_PRESETS.keys()),
        help="Dataset preset key.",
    )
    parser.add_argument("--num-perm", type=int, default=DEFAULT_NUM_PERM)
    parser.add_argument("--num-bands", type=int, default=DEFAULT_NUM_BANDS)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=DEFAULT_MAX_ROWS,
        help=(
            "Optional global cap applied as an upper bound to this dataset preset default. "
            "Set 0 to disable the global cap."
        ),
    )
    parser.add_argument("--warmup-runs", type=int, default=DEFAULT_WARMUP_RUNS)
    parser.add_argument("--measured-runs", type=int, default=DEFAULT_MEASURED_RUNS)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--ngram-size", type=int, default=DEFAULT_NGRAM_SIZE)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def pin_single_thread_env() -> dict[str, str]:
    pinned: dict[str, str] = {}
    for key in THREAD_ENV_VARS:
        os.environ[key] = "1"
        pinned[key] = os.environ[key]
    return pinned


def run_rensa_c(
    token_sets: list[list[str]],
    num_perm: int,
    threshold: float,
    seed: int,
) -> tuple[dict[str, Any], list[bool]]:
    from rensa import CMinHashDeduplicator  # type: ignore

    entries = [(str(index), tokens) for index, tokens in enumerate(token_sets)]

    start = perf_counter()
    deduper = CMinHashDeduplicator(threshold=threshold, num_perm=num_perm, seed=seed)
    inserted_flags = [bool(value) for value in deduper.add_pairs(entries)]
    elapsed = perf_counter() - start

    duplicate_flags = [not inserted for inserted in inserted_flags]
    rows_removed = sum(1 for is_duplicate in duplicate_flags if is_duplicate)
    metrics = {
        "sketch": elapsed,
        "build": 0.0,
        "query": 0.0,
        "total": elapsed,
        "rows_removed": rows_removed,
        "rows_remaining": len(duplicate_flags) - rows_removed,
        "total_candidates": None,
        "avg_candidates_per_row": None,
    }
    return metrics, duplicate_flags


def rotate_order(methods: list[str], offset: int) -> list[str]:
    position = offset % len(methods)
    return methods[position:] + methods[:position]


def run_engine(
    engine: str,
    token_sets: list[list[str]],
    num_perm: int,
    num_bands: int,
    threshold: float,
    seed: int,
) -> tuple[dict[str, Any], list[bool]]:
    if engine == "datasketch":
        return run_datasketch(
            token_sets=token_sets,
            num_perm=num_perm,
            num_bands=num_bands,
            threshold=threshold,
            seed=seed,
        )
    if engine == "fastsketch":
        return run_fastsketch(
            token_sets=token_sets,
            num_perm=num_perm,
            num_bands=num_bands,
            seed=seed,
            threads=1,
        )
    if engine == "rensa_r":
        return run_rensa(
            token_sets=token_sets,
            num_perm=num_perm,
            num_bands=num_bands,
            threshold=threshold,
            seed=seed,
        )
    if engine == "rensa_c":
        return run_rensa_c(
            token_sets=token_sets,
            num_perm=num_perm,
            threshold=threshold,
            seed=seed,
        )
    raise ValueError(f"Unknown engine: {engine}")


def run_once(
    token_sets: list[list[str]],
    order: list[str],
    num_perm: int,
    num_bands: int,
    threshold: float,
    seed: int,
) -> dict[str, Any]:
    results: dict[str, dict[str, Any]] = {}
    flags_by_engine: dict[str, list[bool]] = {}

    for engine in order:
        metrics, flags = run_engine(
            engine=engine,
            token_sets=token_sets,
            num_perm=num_perm,
            num_bands=num_bands,
            threshold=threshold,
            seed=seed,
        )
        results[engine] = metrics
        flags_by_engine[engine] = flags

    kept_sets = {
        engine: {index for index, is_duplicate in enumerate(flags) if not is_duplicate}
        for engine, flags in flags_by_engine.items()
    }

    datasketch_flags = flags_by_engine["datasketch"]
    payload = {
        "rows": len(token_sets),
        "order": order,
        "engines": results,
        "accuracy": {
            "jaccard": {
                "datasketch_vs_fastsketch": jaccard_similarity(
                    kept_sets["datasketch"], kept_sets["fastsketch"]
                ),
                "datasketch_vs_rensa_r": jaccard_similarity(
                    kept_sets["datasketch"], kept_sets["rensa_r"]
                ),
                "datasketch_vs_rensa_c": jaccard_similarity(
                    kept_sets["datasketch"], kept_sets["rensa_c"]
                ),
                "rensa_r_vs_rensa_c": jaccard_similarity(
                    kept_sets["rensa_r"], kept_sets["rensa_c"]
                ),
                "rensa_r_vs_fastsketch": jaccard_similarity(
                    kept_sets["rensa_r"], kept_sets["fastsketch"]
                ),
                "rensa_c_vs_fastsketch": jaccard_similarity(
                    kept_sets["rensa_c"], kept_sets["fastsketch"]
                ),
            },
            "mismatch_vs_datasketch": {
                "fastsketch": mismatch_stats(datasketch_flags, flags_by_engine["fastsketch"]),
                "rensa_r": mismatch_stats(datasketch_flags, flags_by_engine["rensa_r"]),
                "rensa_c": mismatch_stats(datasketch_flags, flags_by_engine["rensa_c"]),
            },
        },
    }
    return payload


def median(values: list[float]) -> float:
    return statistics.median(values)


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        raise ValueError("No measured runs to summarize")

    engine_summary: dict[str, Any] = {}
    for engine in ENGINE_KEYS:
        totals = [run["engines"][engine]["total"] for run in runs]
        sketches = [run["engines"][engine]["sketch"] for run in runs]
        builds = [run["engines"][engine]["build"] for run in runs]
        queries = [run["engines"][engine]["query"] for run in runs]
        rows_removed = [run["engines"][engine]["rows_removed"] for run in runs]
        rows_remaining = [run["engines"][engine]["rows_remaining"] for run in runs]

        engine_summary[engine] = {
            "median_total": median(totals),
            "median_sketch": median(sketches),
            "median_build": median(builds),
            "median_query": median(queries),
            "median_rows_removed": int(round(median(rows_removed))),
            "median_rows_remaining": int(round(median(rows_remaining))),
        }

    datasketch_total = engine_summary["datasketch"]["median_total"]
    speedup_vs_datasketch = {}
    for engine in ENGINE_KEYS:
        engine_total = engine_summary[engine]["median_total"]
        speedup_vs_datasketch[engine] = datasketch_total / engine_total if engine_total else None

    jaccard_keys = (
        "datasketch_vs_fastsketch",
        "datasketch_vs_rensa_r",
        "datasketch_vs_rensa_c",
        "rensa_r_vs_rensa_c",
        "rensa_r_vs_fastsketch",
        "rensa_c_vs_fastsketch",
    )
    mismatch_keys = ("fastsketch", "rensa_r", "rensa_c")

    return {
        "engine_medians": engine_summary,
        "speedup_vs_datasketch": speedup_vs_datasketch,
        "accuracy": {
            "median_jaccard": {
                key: median([run["accuracy"]["jaccard"][key] for run in runs]) for key in jaccard_keys
            },
            "mismatch_vs_datasketch": {
                engine: {
                    "median_count": int(
                        round(
                            median(
                                [
                                    run["accuracy"]["mismatch_vs_datasketch"][engine]["count"]
                                    for run in runs
                                ]
                            )
                        )
                    ),
                    "median_rate": median(
                        [run["accuracy"]["mismatch_vs_datasketch"][engine]["rate"] for run in runs]
                    ),
                    "median_false_positive": int(
                        round(
                            median(
                                [
                                    run["accuracy"]["mismatch_vs_datasketch"][engine]["false_positive"]
                                    for run in runs
                                ]
                            )
                        )
                    ),
                    "median_false_negative": int(
                        round(
                            median(
                                [
                                    run["accuracy"]["mismatch_vs_datasketch"][engine]["false_negative"]
                                    for run in runs
                                ]
                            )
                        )
                    ),
                }
                for engine in mismatch_keys
            },
        },
    }


def print_summary(summary: dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print("Simple Benchmark (single-thread)")
    print("=" * 80)
    print(
        f"{'Engine':<14} {'Median Total(s)':<16} {'Sketch':<10} {'Build':<10} {'Query':<10} {'Speedup vs DS':<14}"
    )
    print("-" * 80)

    for engine in ENGINE_KEYS:
        engine_stats = summary["engine_medians"][engine]
        speedup = summary["speedup_vs_datasketch"][engine]
        speedup_label = f"{speedup:.2f}x" if speedup is not None else "n/a"
        print(
            f"{engine:<14} "
            f"{engine_stats['median_total']:<16.4f} "
            f"{engine_stats['median_sketch']:<10.4f} "
            f"{engine_stats['median_build']:<10.4f} "
            f"{engine_stats['median_query']:<10.4f} "
            f"{speedup_label:<14}"
        )

    print("\nAccuracy (median Jaccard of kept sets):")
    for key, value in summary["accuracy"]["median_jaccard"].items():
        print(f"  {key}: {value:.6f}")

    print("\nMismatch vs Datasketch duplicate flags (median):")
    for engine in ("fastsketch", "rensa_r", "rensa_c"):
        mismatch = summary["accuracy"]["mismatch_vs_datasketch"][engine]
        print(
            f"  {engine}: count={mismatch['median_count']} "
            f"rate={mismatch['median_rate']:.6f} "
            f"fp={mismatch['median_false_positive']} "
            f"fn={mismatch['median_false_negative']}"
        )


def main(args: argparse.Namespace) -> None:
    if args.num_perm <= 0:
        raise ValueError("--num-perm must be > 0")
    if args.num_bands <= 0:
        raise ValueError("--num-bands must be > 0")
    if args.num_bands > args.num_perm:
        raise ValueError("--num-bands must be <= --num-perm")
    if args.num_perm % args.num_bands != 0:
        raise ValueError("--num-bands must divide --num-perm")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be in [0, 1]")
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs must be >= 0")
    if args.measured_runs <= 0:
        raise ValueError("--measured-runs must be > 0")
    if args.ngram_size <= 0:
        raise ValueError("--ngram-size must be > 0")

    global_max_rows = args.max_rows
    if global_max_rows is not None:
        if global_max_rows < 0:
            raise ValueError("--max-rows must be >= 0")
        if global_max_rows == 0:
            global_max_rows = None

    spec = DATASET_PRESETS[args.dataset]
    effective_max_rows = resolve_max_rows(spec, global_max_rows=global_max_rows, dataset_max_rows={})

    token_cache, row_count, token_cache_sha256 = load_or_prepare_token_cache(
        cache_dir=args.cache_dir,
        spec=spec,
        max_rows=effective_max_rows,
        ngram_size=args.ngram_size,
    )
    with token_cache.open("rb") as handle:
        token_sets: list[list[str]] = pickle.load(handle)

    thread_env = pin_single_thread_env()

    print(
        f"Prepared dataset '{spec.key}' rows={row_count} "
        f"effective_max_rows={effective_max_rows if effective_max_rows is not None else 'all'}"
    )
    print(f"Token cache: {token_cache}")

    methods = list(ENGINE_KEYS)
    for warmup_index in range(args.warmup_runs):
        order = rotate_order(methods, warmup_index)
        _ = run_once(
            token_sets=token_sets,
            order=order,
            num_perm=args.num_perm,
            num_bands=args.num_bands,
            threshold=args.threshold,
            seed=args.seed,
        )

    measured_runs: list[dict[str, Any]] = []
    for run_index in range(args.measured_runs):
        order = rotate_order(methods, run_index)
        run_payload = run_once(
            token_sets=token_sets,
            order=order,
            num_perm=args.num_perm,
            num_bands=args.num_bands,
            threshold=args.threshold,
            seed=args.seed,
        )
        measured_runs.append(run_payload)

        print(
            f"run={run_index} order={order} "
            f"ds={run_payload['engines']['datasketch']['total']:.3f}s "
            f"fs={run_payload['engines']['fastsketch']['total']:.3f}s "
            f"rr={run_payload['engines']['rensa_r']['total']:.3f}s "
            f"rc={run_payload['engines']['rensa_c']['total']:.3f}s"
        )

    summary = summarize_runs(measured_runs)
    print_summary(summary)

    payload = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "config": {
            "dataset": spec.key,
            "dataset_spec": {
                "hf_dataset": spec.hf_dataset,
                "split": spec.split,
                "text_columns": list(spec.text_columns),
                "default_max_rows": spec.default_max_rows,
                "streaming": spec.streaming,
                "revision": spec.revision,
            },
            "rows": row_count,
            "effective_max_rows": effective_max_rows,
            "num_perm": args.num_perm,
            "num_bands": args.num_bands,
            "rows_per_band": args.num_perm // args.num_bands,
            "threshold": args.threshold,
            "seed": args.seed,
            "warmup_runs": args.warmup_runs,
            "measured_runs": args.measured_runs,
            "ngram_size": args.ngram_size,
            "thread_env": thread_env,
            "token_cache": str(token_cache),
            "token_cache_sha256": token_cache_sha256,
        },
        "runs": measured_runs,
        "summary": summary,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"\nWrote benchmark JSON to {args.output_json}")


if __name__ == "__main__":
    parsed_args = parse_args()
    main(parsed_args)
