from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
import random
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

from datasets import load_dataset
from datasketch import MinHash, MinHashLSH

THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

ENGINE_KEYS = ("datasketch", "fastsketch", "rensa")
FALLBACK_TEXT_COLUMNS = ("text", "content", "document", "body", "raw")

DEFAULT_DATASETS = (
    "finefineweb,codefeedback,ag_news,pinecone,shuyuej,books3,bookcorpusopen"
)
DEFAULT_THREADS = "1,8"
DEFAULT_NUM_PERM = 128
DEFAULT_NUM_BANDS = 8
DEFAULT_THRESHOLD = 0.8
DEFAULT_SEED = 12345
DEFAULT_WARMUP_RUNS = 0
DEFAULT_REPETITIONS = 1
DEFAULT_NGRAM_SIZE = 3

DEFAULT_CACHE_DIR = (
    Path(__file__).resolve().parents[1] / ".bench" / "local" / "cache" / "full_benchmark"
)
DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parents[1] / ".bench" / "profiling" / "full_benchmark.json"
)


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    hf_dataset: str
    split: str
    text_columns: tuple[str, ...]
    default_max_rows: int | None
    revision: str | None = None
    streaming: bool = False


DATASET_PRESETS: dict[str, DatasetSpec] = {
    "finefineweb": DatasetSpec(
        key="finefineweb",
        hf_dataset="m-a-p/FineFineWeb",
        split="train",
        text_columns=("text",),
        default_max_rows=200_000,
        streaming=True,
    ),
    "codefeedback": DatasetSpec(
        key="codefeedback",
        hf_dataset="m-a-p/CodeFeedback-Filtered-Instruction",
        split="train",
        text_columns=("query",),
        default_max_rows=156_526,
    ),
    "ag_news": DatasetSpec(
        key="ag_news",
        hf_dataset="fancyzhx/ag_news",
        split="train",
        text_columns=("text",),
        default_max_rows=120_000,
    ),
    "pinecone": DatasetSpec(
        key="pinecone",
        hf_dataset="pinecone/core-2020-05-10-deduplication",
        split="train",
        text_columns=("processed_title", "processed_abstract"),
        default_max_rows=100_000,
    ),
    "shuyuej": DatasetSpec(
        key="shuyuej",
        hf_dataset="shuyuej/pretraining-dataset",
        split="train",
        text_columns=("text",),
        default_max_rows=37_777,
        # JSON source can fail with non-streaming pyarrow parsing on some snapshots.
        streaming=True,
    ),
    "bookcorpusopen": DatasetSpec(
        key="bookcorpusopen",
        hf_dataset="lucadiliello/bookcorpusopen",
        split="train",
        text_columns=("text",),
        default_max_rows=1_000,
        streaming=True,
    ),
    "books3": DatasetSpec(
        key="books3",
        hf_dataset="P1ayer-1/books-3-textbooks",
        split="train",
        text_columns=("text",),
        default_max_rows=1_000,
        streaming=True,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fair full benchmark across Datasketch, FastSketch, and Rensa. "
            "Uses per-run process isolation (all engines in one subprocess), "
            "randomized engine order per repetition, "
            "and explicit single-thread plus multi-thread lanes."
        ),
    )
    parser.add_argument("--datasets", default=DEFAULT_DATASETS)
    parser.add_argument("--threads", default=DEFAULT_THREADS)
    parser.add_argument("--num-perm", type=int, default=DEFAULT_NUM_PERM)
    parser.add_argument("--num-bands", type=int, default=DEFAULT_NUM_BANDS)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--warmup-runs", type=int, default=DEFAULT_WARMUP_RUNS)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help=(
            "Optional global row cap applied as an upper bound to all dataset defaults. "
            "Set 0 to disable the global cap."
        ),
    )
    parser.add_argument(
        "--dataset-max-rows",
        type=str,
        default="",
        help=(
            "Per-dataset cap overrides using key=value pairs, separated by commas. "
            "Example: finefineweb=300000,books3=2000,bookcorpusopen=0 (0 means uncapped)."
        ),
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--ngram-size", type=int, default=DEFAULT_NGRAM_SIZE)

    parser.add_argument("--_run-once", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--token-cache", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--order", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--run-threads", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--expected-token-cache-sha", type=str, default="", help=argparse.SUPPRESS)

    return parser.parse_args()


def sanitized_fragment(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value)


def parse_dataset_keys(value: str) -> list[str]:
    keys = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = [item for item in keys if item not in DATASET_PRESETS]
    if unknown:
        valid = ", ".join(sorted(DATASET_PRESETS))
        raise ValueError(f"Unknown dataset preset(s): {unknown}. Valid presets: {valid}")
    return keys


def parse_threads(value: str) -> list[int]:
    threads = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not threads:
        raise ValueError("--threads must include at least one positive integer")
    for thread_count in threads:
        if thread_count <= 0:
            raise ValueError("--threads values must be positive integers")
    return threads


def parse_dataset_max_rows(value: str) -> dict[str, int | None]:
    overrides: dict[str, int | None] = {}
    if not value.strip():
        return overrides

    pairs = [item.strip() for item in value.split(",") if item.strip()]
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(
                "--dataset-max-rows must use key=value pairs separated by commas"
            )
        key_raw, rows_raw = pair.split("=", 1)
        key = key_raw.strip().lower()
        if key not in DATASET_PRESETS:
            valid = ", ".join(sorted(DATASET_PRESETS))
            raise ValueError(
                f"Unknown dataset in --dataset-max-rows: '{key}'. Valid presets: {valid}"
            )
        rows = int(rows_raw.strip())
        if rows < 0:
            raise ValueError(
                f"Row cap for dataset '{key}' must be >= 0, got {rows}"
            )
        overrides[key] = None if rows == 0 else rows

    return overrides


def resolve_max_rows(
    spec: DatasetSpec,
    global_max_rows: int | None,
    dataset_max_rows: Mapping[str, int | None],
) -> int | None:
    resolved = spec.default_max_rows
    if spec.key in dataset_max_rows:
        resolved = dataset_max_rows[spec.key]
    if global_max_rows is not None:
        resolved = global_max_rows if resolved is None else min(resolved, global_max_rows)
    return resolved


def dataset_cache_path(
    cache_dir: Path,
    spec: DatasetSpec,
    max_rows: int | None,
    ngram_size: int,
) -> Path:
    row_suffix = "all" if max_rows is None else str(max_rows)
    revision_suffix = "none" if spec.revision is None else sanitized_fragment(spec.revision)
    columns_fragment = "__".join(sanitized_fragment(column) for column in spec.text_columns)
    name_fragment = sanitized_fragment(spec.hf_dataset)
    filename = (
        f"{spec.key}__{name_fragment}__{spec.split}__{columns_fragment}"
        f"__ng{ngram_size}__stream{int(spec.streaming)}"
        f"__rev_{revision_suffix}__rows_{row_suffix}.pkl"
    )
    return cache_dir / filename


def cached_row_count_from_cache_path(cache_path: Path) -> int | None:
    suffix_marker = "__rows_"
    name = cache_path.name
    marker_index = name.rfind(suffix_marker)
    if marker_index == -1:
        return None
    suffix = name[marker_index + len(suffix_marker) :]
    if suffix.endswith(".pkl"):
        suffix = suffix[: -len(".pkl")]
    return int(suffix) if suffix.isdigit() else None


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def extract_text(row: Mapping[str, Any], text_columns: tuple[str, ...]) -> str:
    selected_parts: list[str] = []
    for column in text_columns:
        value = row.get(column)
        if isinstance(value, str):
            if value.strip():
                selected_parts.append(value)
        elif value is not None:
            selected_parts.append(str(value))

    if selected_parts:
        return " ".join(selected_parts)

    for column in FALLBACK_TEXT_COLUMNS:
        value = row.get(column)
        if isinstance(value, str) and value.strip():
            return value

    parts = [value for value in row.values() if isinstance(value, str) and value.strip()]
    if parts:
        return " \n ".join(parts)

    for value in row.values():
        if value is not None:
            return str(value)

    return ""


def tokenize_to_ngrams(text: str, ngram_size: int) -> list[str]:
    tokens = [token for token in text.lower().split() if token]
    if ngram_size <= 1 or len(tokens) < ngram_size:
        return tokens
    return [" ".join(tokens[index : index + ngram_size]) for index in range(len(tokens) - ngram_size + 1)]


def select_evenly_spaced_indices(total_rows: int, target_rows: int) -> list[int]:
    if target_rows >= total_rows:
        return list(range(total_rows))
    return [(index * total_rows) // target_rows for index in range(target_rows)]


def load_token_sets_from_hf(
    spec: DatasetSpec,
    max_rows: int | None,
    ngram_size: int,
) -> list[list[str]]:
    load_kwargs: dict[str, Any] = {"split": spec.split}
    if spec.revision:
        load_kwargs["revision"] = spec.revision

    token_sets: list[list[str]] = []
    if spec.streaming:
        load_kwargs["streaming"] = True
        dataset_stream = load_dataset(spec.hf_dataset, **load_kwargs)
        for index, row in enumerate(dataset_stream):
            if max_rows is not None and index >= max_rows:
                break
            text = extract_text(row, spec.text_columns)
            token_sets.append(tokenize_to_ngrams(text, ngram_size))
        return token_sets

    dataset = load_dataset(spec.hf_dataset, **load_kwargs)
    if max_rows is not None and max_rows < len(dataset):
        indices = select_evenly_spaced_indices(len(dataset), max_rows)
        dataset = dataset.select(indices)

    for row in dataset:
        text = extract_text(row, spec.text_columns)
        token_sets.append(tokenize_to_ngrams(text, ngram_size))
    return token_sets


def load_or_prepare_token_cache(
    cache_dir: Path,
    spec: DatasetSpec,
    max_rows: int | None,
    ngram_size: int,
) -> tuple[Path, int, str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = dataset_cache_path(cache_dir, spec, max_rows, ngram_size)

    if cache_path.exists():
        row_count = cached_row_count_from_cache_path(cache_path)
        if row_count is None:
            with cache_path.open("rb") as handle:
                token_sets = pickle.load(handle)
            row_count = len(token_sets)
        return cache_path, row_count, sha256_file(cache_path)

    token_sets = load_token_sets_from_hf(spec, max_rows, ngram_size)
    with cache_path.open("wb") as handle:
        pickle.dump(token_sets, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return cache_path, len(token_sets), sha256_file(cache_path)


def thread_env(threads: int) -> dict[str, str]:
    env = os.environ.copy()
    value = str(threads)
    for key in THREAD_ENV_VARS:
        env[key] = value
    return env


def resolve_package_version(
    package_name: str,
    module_name: str | None = None,
) -> str | None:
    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        if module_name is None:
            return None
    try:
        module = __import__(module_name if module_name is not None else package_name)
    except Exception:
        return None
    version = getattr(module, "__version__", None)
    return str(version) if version is not None else None


def current_rensa_env() -> dict[str, str]:
    return {key: value for key, value in sorted(os.environ.items()) if key.startswith("RENSA_")}


def run_datasketch(
    token_sets: list[list[str]],
    num_perm: int,
    num_bands: int,
    threshold: float,
    seed: int,
) -> tuple[dict[str, Any], list[bool]]:
    rows_per_band = num_perm // num_bands
    _ = threshold  # `params=(b, r)` controls banding for datasketch in this benchmark.

    # Datasketch's batch APIs reuse initialized state and avoid repeated setup.
    sketch_start = perf_counter()
    if hasattr(MinHash, "generator"):
        byte_sets = ([token.encode("utf-8") for token in tokens] for tokens in token_sets)
        minhashes: list[MinHash] = list(
            MinHash.generator(byte_sets, num_perm=num_perm, seed=seed)
        )
    else:
        minhashes = []
        for tokens in token_sets:
            minhash = MinHash(num_perm=num_perm, seed=seed)
            minhash.update_batch(token.encode("utf-8") for token in tokens)
            minhashes.append(minhash)
    sketch_elapsed = perf_counter() - sketch_start

    lsh = MinHashLSH(
        num_perm=num_perm,
        params=(num_bands, rows_per_band),
    )

    build_start = perf_counter()
    for index, minhash in enumerate(minhashes):
        lsh.insert(index, minhash, check_duplication=False)
    build_elapsed = perf_counter() - build_start

    query_start = perf_counter()
    duplicate_flags: list[bool] = []
    total_candidates = 0
    for index, minhash in enumerate(minhashes):
        candidates = lsh.query(minhash)
        total_candidates += len(candidates)
        duplicate_flags.append(any(candidate != index for candidate in candidates))
    query_elapsed = perf_counter() - query_start

    rows = len(token_sets)
    rows_removed = sum(1 for is_duplicate in duplicate_flags if is_duplicate)
    metrics = {
        "sketch": sketch_elapsed,
        "build": build_elapsed,
        "query": query_elapsed,
        "total": sketch_elapsed + build_elapsed + query_elapsed,
        "rows_removed": rows_removed,
        "rows_remaining": rows - rows_removed,
        "total_candidates": total_candidates,
        "avg_candidates_per_row": (total_candidates / rows) if rows else 0.0,
    }
    return metrics, duplicate_flags


def run_fastsketch(
    token_sets: list[list[str]],
    num_perm: int,
    num_bands: int,
    seed: int,
    threads: int,
) -> tuple[dict[str, Any], list[bool]]:
    from FastSketchLSH import FastSimilaritySketch, LSH  # type: ignore

    sketcher = FastSimilaritySketch(sketch_size=num_perm, seed=seed)

    sketch_start = perf_counter()
    sketches = sketcher.sketch_batch(token_sets, num_threads=threads)
    sketch_elapsed = perf_counter() - sketch_start

    lsh = LSH(num_perm=num_perm, num_bands=num_bands, num_threads=threads)

    build_start = perf_counter()
    lsh.build_from_batch(sketches)
    build_elapsed = perf_counter() - build_start

    query_start = perf_counter()
    flat, indptr = lsh.batch_query_csr(sketches)
    row_count = len(indptr) - 1
    duplicate_flags = [
        int(indptr[index + 1] - indptr[index]) > 1 for index in range(row_count)
    ]
    query_elapsed = perf_counter() - query_start

    rows_removed = sum(1 for is_duplicate in duplicate_flags if is_duplicate)
    metrics = {
        "sketch": sketch_elapsed,
        "build": build_elapsed,
        "query": query_elapsed,
        "total": sketch_elapsed + build_elapsed + query_elapsed,
        "rows_removed": rows_removed,
        "rows_remaining": row_count - rows_removed,
        "total_candidates": int(len(flat)),
        "avg_candidates_per_row": (len(flat) / row_count) if row_count else 0.0,
    }
    return metrics, duplicate_flags


def run_rensa(
    token_sets: list[list[str]],
    num_perm: int,
    num_bands: int,
    threshold: float,
    seed: int,
) -> tuple[dict[str, Any], list[bool]]:
    from rensa import RMinHash, RMinHashLSH  # type: ignore

    matrix: Any | None = None
    minhashes: list[Any] | None = None
    sketch_start = perf_counter()
    if hasattr(RMinHash, "digest_matrix_from_token_sets_rho"):
        probes = int(os.environ.get("RENSA_RHO_PROBES", "4"))
        matrix = RMinHash.digest_matrix_from_token_sets_rho(
            token_sets,
            num_perm=num_perm,
            seed=seed,
            probes=probes,
        )
    elif hasattr(RMinHash, "digest_matrix_from_token_sets"):
        matrix = RMinHash.digest_matrix_from_token_sets(
            token_sets,
            num_perm=num_perm,
            seed=seed,
        )
    else:
        # Compatibility fallback for older branch checkouts in CI perf comparisons.
        minhashes = []
        for tokens in token_sets:
            minhash = RMinHash(num_perm=num_perm, seed=seed)
            minhash.update(tokens)
            minhashes.append(minhash)
    sketch_elapsed = perf_counter() - sketch_start

    lsh = RMinHashLSH(threshold=threshold, num_perm=num_perm, num_bands=num_bands)

    query_elapsed = 0.0
    if matrix is not None and hasattr(lsh, "query_duplicate_flags_matrix_one_shot"):
        build_start = perf_counter()
        duplicate_flags = [
            bool(value) for value in lsh.query_duplicate_flags_matrix_one_shot(matrix)
        ]
        build_elapsed = perf_counter() - build_start
    elif matrix is not None and hasattr(lsh, "insert_matrix_and_query_duplicate_flags"):
        build_start = perf_counter()
        duplicate_flags = [
            bool(value)
            for value in lsh.insert_matrix_and_query_duplicate_flags(matrix, start_key=0)
        ]
        build_elapsed = perf_counter() - build_start
    elif matrix is not None and hasattr(lsh, "insert_matrix") and hasattr(lsh, "query_duplicate_flags_matrix"):
        build_start = perf_counter()
        lsh.insert_matrix(matrix, start_key=0)
        build_elapsed = perf_counter() - build_start

        query_start = perf_counter()
        duplicate_flags = [bool(value) for value in lsh.query_duplicate_flags_matrix(matrix)]
        query_elapsed = perf_counter() - query_start
    elif minhashes is not None:
        build_start = perf_counter()
        for index, minhash in enumerate(minhashes):
            lsh.insert(index, minhash)
        build_elapsed = perf_counter() - build_start

        query_start = perf_counter()
        duplicate_flags = []
        for minhash in minhashes:
            candidates = lsh.query(minhash)
            duplicate_flags.append(len(candidates) > 1)
        query_elapsed = perf_counter() - query_start
    else:
        raise RuntimeError(
            "Unsupported rensa API shape: no compatible matrix or per-item benchmark path"
        )

    rows = len(duplicate_flags)
    rows_removed = sum(1 for is_duplicate in duplicate_flags if is_duplicate)
    metrics = {
        "sketch": sketch_elapsed,
        "build": build_elapsed,
        "query": query_elapsed,
        "total": sketch_elapsed + build_elapsed + query_elapsed,
        "rows_removed": rows_removed,
        "rows_remaining": rows - rows_removed,
        "total_candidates": None,
        "avg_candidates_per_row": None,
    }
    return metrics, duplicate_flags


def run_engine(
    engine: str,
    token_sets: list[list[str]],
    num_perm: int,
    num_bands: int,
    threshold: float,
    seed: int,
    threads: int,
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
            threads=threads,
        )
    if engine == "rensa":
        return run_rensa(
            token_sets=token_sets,
            num_perm=num_perm,
            num_bands=num_bands,
            threshold=threshold,
            seed=seed,
        )
    raise ValueError(f"Unknown engine: {engine}")


def mismatch_stats(reference_flags: list[bool], candidate_flags: list[bool]) -> dict[str, Any]:
    if len(reference_flags) != len(candidate_flags):
        raise ValueError("Flag vector length mismatch")

    mismatches = 0
    false_positive = 0
    false_negative = 0
    for reference, candidate in zip(reference_flags, candidate_flags):
        if reference == candidate:
            continue
        mismatches += 1
        if candidate and not reference:
            false_positive += 1
        else:
            false_negative += 1

    total = len(reference_flags)
    return {
        "count": mismatches,
        "rate": (mismatches / total) if total else 0.0,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def jaccard_similarity(set_a: set[int], set_b: set[int]) -> float:
    union_size = len(set_a.union(set_b))
    if union_size == 0:
        return 1.0
    return len(set_a.intersection(set_b)) / union_size


def run_once(args: argparse.Namespace) -> None:
    if args.token_cache is None:
        raise ValueError("--token-cache is required in --_run-once mode")
    if not args.order:
        raise ValueError("--order is required in --_run-once mode")

    ordered_engines = [item.strip().lower() for item in args.order.split(",") if item.strip()]
    if set(ordered_engines) != set(ENGINE_KEYS):
        raise ValueError(
            f"--order must contain exactly these engines: {ENGINE_KEYS}, got {ordered_engines}"
        )

    expected_threads = str(args.run_threads)
    thread_assertions = {
        key: os.environ.get(key, "") == expected_threads for key in THREAD_ENV_VARS
    }
    if not all(thread_assertions.values()):
        raise RuntimeError(
            "Thread environment assertion failed in run-once lane: "
            f"expected all vars to be '{expected_threads}', got {thread_assertions}"
        )

    token_cache_sha = sha256_file(args.token_cache)
    if args.expected_token_cache_sha and token_cache_sha != args.expected_token_cache_sha:
        raise RuntimeError(
            "Token cache hash mismatch in run-once lane: "
            f"expected {args.expected_token_cache_sha}, got {token_cache_sha}"
        )

    with args.token_cache.open("rb") as handle:
        token_sets: list[list[str]] = pickle.load(handle)

    results: dict[str, dict[str, Any]] = {}
    flags_by_engine: dict[str, list[bool]] = {}
    for engine in ordered_engines:
        metrics, flags = run_engine(
            engine=engine,
            token_sets=token_sets,
            num_perm=args.num_perm,
            num_bands=args.num_bands,
            threshold=args.threshold,
            seed=args.seed,
            threads=args.run_threads,
        )
        results[engine] = metrics
        flags_by_engine[engine] = flags

    kept_sets = {
        engine: {index for index, is_duplicate in enumerate(flags) if not is_duplicate}
        for engine, flags in flags_by_engine.items()
    }

    datasketch_flags = flags_by_engine["datasketch"]
    fastsketch_flags = flags_by_engine["fastsketch"]
    payload = {
        "rows": len(token_sets),
        "threads": args.run_threads,
        "order": ordered_engines,
        "token_cache_sha256": token_cache_sha,
        "thread_env_assertions": thread_assertions,
        "engines": results,
        "accuracy": {
            "jaccard": {
                "datasketch_vs_rensa": jaccard_similarity(
                    kept_sets["datasketch"], kept_sets["rensa"]
                ),
                "datasketch_vs_fastsketch": jaccard_similarity(
                    kept_sets["datasketch"], kept_sets["fastsketch"]
                ),
                "rensa_vs_fastsketch": jaccard_similarity(
                    kept_sets["rensa"], kept_sets["fastsketch"]
                ),
            },
            "mismatch_vs_datasketch": {
                "rensa": mismatch_stats(datasketch_flags, flags_by_engine["rensa"]),
                "fastsketch": mismatch_stats(datasketch_flags, flags_by_engine["fastsketch"]),
            },
            "mismatch_vs_fastsketch": {
                "rensa": mismatch_stats(fastsketch_flags, flags_by_engine["rensa"]),
                "datasketch": mismatch_stats(fastsketch_flags, flags_by_engine["datasketch"]),
            },
        },
    }
    print(json.dumps(payload))


def run_once_subprocess(
    script_path: Path,
    token_cache: Path,
    token_cache_sha256: str,
    num_perm: int,
    num_bands: int,
    threshold: float,
    seed: int,
    threads: int,
    order: list[str],
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(script_path),
        "--_run-once",
        "--token-cache",
        str(token_cache),
        "--expected-token-cache-sha",
        token_cache_sha256,
        "--num-perm",
        str(num_perm),
        "--num-bands",
        str(num_bands),
        "--threshold",
        str(threshold),
        "--seed",
        str(seed),
        "--run-threads",
        str(threads),
        "--order",
        ",".join(order),
    ]
    completed = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        env=thread_env(threads),
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Run failed (threads={threads}, order={order}):\n"
            f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
        )
    return json.loads(completed.stdout)


def median(values: list[float]) -> float:
    return statistics.median(values)


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        raise ValueError("No measured runs to summarize")

    cache_hashes = {run["token_cache_sha256"] for run in runs}
    if len(cache_hashes) != 1:
        raise ValueError(f"Token cache hash mismatch across runs: {sorted(cache_hashes)}")

    thread_env_ok = all(
        all(bool(value) for value in run["thread_env_assertions"].values()) for run in runs
    )
    if not thread_env_ok:
        raise ValueError("Thread environment assertions failed in one or more runs")

    engine_summary: dict[str, Any] = {}
    for engine in ENGINE_KEYS:
        totals = [run["engines"][engine]["total"] for run in runs]
        sketches = [run["engines"][engine]["sketch"] for run in runs]
        builds = [run["engines"][engine]["build"] for run in runs]
        queries = [run["engines"][engine]["query"] for run in runs]
        rows_removed = [run["engines"][engine]["rows_removed"] for run in runs]
        rows_remaining = [run["engines"][engine]["rows_remaining"] for run in runs]
        avg_candidates = [
            run["engines"][engine]["avg_candidates_per_row"]
            for run in runs
            if run["engines"][engine]["avg_candidates_per_row"] is not None
        ]

        engine_summary[engine] = {
            "median_total": median(totals),
            "median_sketch": median(sketches),
            "median_build": median(builds),
            "median_query": median(queries),
            "median_rows_removed": int(round(median(rows_removed))),
            "median_rows_remaining": int(round(median(rows_remaining))),
            "median_avg_candidates_per_row": median(avg_candidates) if avg_candidates else None,
        }

    datasketch_total = engine_summary["datasketch"]["median_total"]
    speedup_vs_datasketch = {}
    for engine in ENGINE_KEYS:
        engine_total = engine_summary[engine]["median_total"]
        speedup_vs_datasketch[engine] = datasketch_total / engine_total if engine_total else None

    accuracy_jaccard = {
        key: median([run["accuracy"]["jaccard"][key] for run in runs])
        for key in (
            "datasketch_vs_rensa",
            "datasketch_vs_fastsketch",
            "rensa_vs_fastsketch",
        )
    }

    mismatch_summary: dict[str, Any] = {}
    for engine in ("rensa", "fastsketch"):
        mismatch_summary[engine] = {
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

    mismatch_vs_fastsketch_summary: dict[str, Any] = {}
    for engine in ("rensa", "datasketch"):
        mismatch_vs_fastsketch_summary[engine] = {
            "median_count": int(
                round(
                    median(
                        [
                            run["accuracy"]["mismatch_vs_fastsketch"][engine]["count"]
                            for run in runs
                        ]
                    )
                )
            ),
            "median_rate": median(
                [run["accuracy"]["mismatch_vs_fastsketch"][engine]["rate"] for run in runs]
            ),
            "median_false_positive": int(
                round(
                    median(
                        [
                            run["accuracy"]["mismatch_vs_fastsketch"][engine]["false_positive"]
                            for run in runs
                        ]
                    )
                )
            ),
            "median_false_negative": int(
                round(
                    median(
                        [
                            run["accuracy"]["mismatch_vs_fastsketch"][engine]["false_negative"]
                            for run in runs
                        ]
                    )
                )
            ),
        }

    return {
        "engine_medians": engine_summary,
        "speedup_vs_datasketch": speedup_vs_datasketch,
        "accuracy": {
            "median_jaccard": accuracy_jaccard,
            "mismatch_vs_datasketch": mismatch_summary,
            "mismatch_vs_fastsketch": mismatch_vs_fastsketch_summary,
        },
        "fairness": {
            "token_cache_hash_consistent": True,
            "token_cache_sha256": next(iter(cache_hashes)),
            "thread_env_assertions_all_true": True,
        },
    }


def print_section_summary(dataset_key: str, threads: int, summary: dict[str, Any]) -> None:
    medians = summary["engine_medians"]
    speedups = summary["speedup_vs_datasketch"]
    accuracy = summary["accuracy"]

    print("\n" + "=" * 80)
    print(f"{dataset_key.upper()} | threads={threads}")
    print("=" * 80)
    print(
        f"{'Engine':<14} {'Median Total(s)':<16} {'Sketch':<10} {'Build':<10} {'Query':<10} {'Speedup vs DS':<14}"
    )
    print("-" * 80)
    for engine in ENGINE_KEYS:
        engine_stats = medians[engine]
        speedup = speedups[engine]
        speedup_label = f"{speedup:.2f}x" if speedup is not None else "n/a"
        print(
            f"{engine:<14} "
            f"{engine_stats['median_total']:<16.4f} "
            f"{engine_stats['median_sketch']:<10.4f} "
            f"{engine_stats['median_build']:<10.4f} "
            f"{engine_stats['median_query']:<10.4f} "
            f"{speedup_label:<14}"
        )

    print("\nRows removed (median):")
    for engine in ENGINE_KEYS:
        print(
            f"  {engine}: {medians[engine]['median_rows_removed']} "
            f"(remaining {medians[engine]['median_rows_remaining']})"
        )

    print("\nAccuracy (median Jaccard of kept sets):")
    print(
        f"  Datasketch vs Rensa: {accuracy['median_jaccard']['datasketch_vs_rensa']:.6f}"
    )
    print(
        f"  Datasketch vs FastSketch: {accuracy['median_jaccard']['datasketch_vs_fastsketch']:.6f}"
    )
    print(
        f"  Rensa vs FastSketch: {accuracy['median_jaccard']['rensa_vs_fastsketch']:.6f}"
    )

    print("\nMismatch vs Datasketch duplicate flags (median):")
    for engine in ("rensa", "fastsketch"):
        mismatch = accuracy["mismatch_vs_datasketch"][engine]
        print(
            f"  {engine}: count={mismatch['median_count']} "
            f"rate={mismatch['median_rate']:.6f} "
            f"fp={mismatch['median_false_positive']} "
            f"fn={mismatch['median_false_negative']}"
        )

    print("\nMismatch vs FastSketch duplicate flags (median):")
    for engine in ("rensa", "datasketch"):
        mismatch = accuracy["mismatch_vs_fastsketch"][engine]
        print(
            f"  {engine}: count={mismatch['median_count']} "
            f"rate={mismatch['median_rate']:.6f} "
            f"fp={mismatch['median_false_positive']} "
            f"fn={mismatch['median_false_negative']}"
        )


def main(args: argparse.Namespace) -> None:
    if args._run_once:
        run_once(args)
        return

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
    if args.repetitions <= 0:
        raise ValueError("--repetitions must be > 0")
    if args.ngram_size <= 0:
        raise ValueError("--ngram-size must be > 0")

    global_max_rows = args.max_rows
    if global_max_rows is not None:
        if global_max_rows < 0:
            raise ValueError("--max-rows must be >= 0")
        if global_max_rows == 0:
            global_max_rows = None

    dataset_keys = parse_dataset_keys(args.datasets)
    thread_modes = parse_threads(args.threads)
    dataset_max_rows_overrides = parse_dataset_max_rows(args.dataset_max_rows)

    print(
        f"Config: num_perm={args.num_perm} threshold={args.threshold} "
        f"num_bands={args.num_bands} rows_per_band={args.num_perm // args.num_bands} "
        f"ngram_size={args.ngram_size} warmup_runs={args.warmup_runs} repetitions={args.repetitions}"
    )
    print(f"Datasets: {dataset_keys}")
    print(f"Thread modes: {thread_modes}")
    if args.warmup_runs == 0 and args.repetitions == 1:
        print(
            "Stability note: warmup_runs=0 and repetitions=1 is the fastest config "
            "but can be noisy. For publication-grade comparisons, prefer warmup_runs>=1 "
            "and repetitions>=3."
        )
    print(
        "Datasketch lane note: Datasketch does not expose an explicit thread count API; "
        "lane thread env vars are still pinned for symmetry."
    )

    script_path = Path(__file__).resolve()
    full_results: list[dict[str, Any]] = []

    for dataset_key in dataset_keys:
        spec = DATASET_PRESETS[dataset_key]
        effective_max_rows = resolve_max_rows(
            spec,
            global_max_rows=global_max_rows,
            dataset_max_rows=dataset_max_rows_overrides,
        )
        token_cache, row_count, token_cache_sha256 = load_or_prepare_token_cache(
            cache_dir=args.cache_dir,
            spec=spec,
            max_rows=effective_max_rows,
            ngram_size=args.ngram_size,
        )

        print(
            f"\nPrepared dataset '{dataset_key}' -> rows={row_count}, "
            f"effective_max_rows={effective_max_rows if effective_max_rows is not None else 'all'}, "
            f"cache={token_cache}"
        )

        for threads in thread_modes:
            runs: list[dict[str, Any]] = []
            rng = random.Random(args.seed + sum(ord(ch) for ch in dataset_key) + threads)
            total_iterations = args.warmup_runs + args.repetitions

            for iteration in range(total_iterations):
                order = list(ENGINE_KEYS)
                rng.shuffle(order)
                run_payload = run_once_subprocess(
                    script_path=script_path,
                    token_cache=token_cache,
                    token_cache_sha256=token_cache_sha256,
                    num_perm=args.num_perm,
                    num_bands=args.num_bands,
                    threshold=args.threshold,
                    seed=args.seed,
                    threads=threads,
                    order=order,
                )
                if iteration >= args.warmup_runs:
                    runs.append(run_payload)

                print(
                    f"{dataset_key.upper()} t={threads} iter={iteration} order={order} "
                    f"ds={run_payload['engines']['datasketch']['total']:.3f}s "
                    f"fs={run_payload['engines']['fastsketch']['total']:.3f}s "
                    f"rs={run_payload['engines']['rensa']['total']:.3f}s"
                )

            summary = summarize_runs(runs)
            print_section_summary(dataset_key, threads, summary)

            full_results.append(
                {
                    "dataset": asdict(spec),
                    "threads": threads,
                    "rows": row_count,
                    "effective_max_rows": effective_max_rows,
                    "token_cache": str(token_cache),
                    "token_cache_sha256": token_cache_sha256,
                    "runs": runs,
                    "summary": summary,
                }
            )

    payload = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "library_versions": {
                "datasketch": resolve_package_version("datasketch"),
                "datasets": resolve_package_version("datasets"),
                "rensa": resolve_package_version("rensa", module_name="rensa"),
                "fastsketch": resolve_package_version(
                    "FastSketchLSH", module_name="FastSketchLSH"
                ),
            },
        },
        "config": {
            "datasets": dataset_keys,
            "threads": thread_modes,
            "num_perm": args.num_perm,
            "num_bands": args.num_bands,
            "rows_per_band": args.num_perm // args.num_bands,
            "threshold": args.threshold,
            "seed": args.seed,
            "warmup_runs": args.warmup_runs,
            "repetitions": args.repetitions,
            "ngram_size": args.ngram_size,
            "max_rows": global_max_rows,
            "dataset_max_rows": dataset_max_rows_overrides,
            "cache_dir": str(args.cache_dir),
            "rensa_env": current_rensa_env(),
            "datasketch_threading_note": (
                "Datasketch does not expose explicit thread count controls; "
                "lane env vars are pinned for process symmetry."
            ),
        },
        "results": full_results,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"\nWrote benchmark JSON to {args.output_json}")


if __name__ == "__main__":
    parsed_args = parse_args()
    main(parsed_args)
