from __future__ import annotations

import argparse
import hashlib
import inspect
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
DEFAULT_RENSA_SKETCH_MODE = "rho_matrix"
DEFAULT_RENSA_QUERY_MODE = "matrix_one_shot"
DEFAULT_RENSA_RHO_PROBES = 4
TOKEN_CACHE_SCHEMA_VERSION = 2
TOKENIZER_FINGERPRINT_FRAGMENT_LEN = 12

DEFAULT_CACHE_DIR = (
    Path(__file__).resolve().parents[1] / ".bench" / "local" / "cache" / "full_benchmark"
)
DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parents[1] / ".bench" / "profiling" / "full_benchmark.json"
)
DEFAULT_BASELINE_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "baselines"
    / "full_benchmark_run.json"
)

RENSA_SKETCH_MODES = ("compat", "rho_matrix", "matrix", "per_item")
RENSA_QUERY_MODES = (
    "compat",
    "matrix_one_shot",
    "matrix_insert_query",
    "matrix_build_query",
    "per_item",
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
        revision="7fd92dc825a75cbff271a5a52eea0eda91a2c112",
        streaming=True,
    ),
    "codefeedback": DatasetSpec(
        key="codefeedback",
        hf_dataset="m-a-p/CodeFeedback-Filtered-Instruction",
        split="train",
        text_columns=("query",),
        default_max_rows=156_526,
        revision="a08c213a9748c66c15d0225814be80a2e77adf4a",
    ),
    "ag_news": DatasetSpec(
        key="ag_news",
        hf_dataset="fancyzhx/ag_news",
        split="train",
        text_columns=("text",),
        default_max_rows=120_000,
        revision="eb185aade064a813bc0b7f42de02595523103ca4",
    ),
    "pinecone": DatasetSpec(
        key="pinecone",
        hf_dataset="pinecone/core-2020-05-10-deduplication",
        split="train",
        text_columns=("processed_title", "processed_abstract"),
        default_max_rows=100_000,
        revision="2170a0eb8d75233132608787a4519f25bcc47ad7",
    ),
    "shuyuej": DatasetSpec(
        key="shuyuej",
        hf_dataset="shuyuej/pretraining-dataset",
        split="train",
        text_columns=("text",),
        default_max_rows=37_777,
        revision="c0d79e736bafdd0d7ff9a0b9b08d5c58a1254109",
        # JSON source can fail with non-streaming pyarrow parsing on some snapshots.
        streaming=True,
    ),
    "bookcorpusopen": DatasetSpec(
        key="bookcorpusopen",
        hf_dataset="lucadiliello/bookcorpusopen",
        split="train",
        text_columns=("text",),
        default_max_rows=1_000,
        revision="edb74e6c88abb38f0a0fc993a7068ab00a32db45",
        streaming=True,
    ),
    "books3": DatasetSpec(
        key="books3",
        hf_dataset="P1ayer-1/books-3-textbooks",
        split="train",
        text_columns=("text",),
        default_max_rows=1_000,
        revision="c3711e9461ad3de0a2e95855ad34a1ff97e0b799",
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
    parser.add_argument(
        "--rensa-sketch-mode",
        choices=RENSA_SKETCH_MODES,
        default=DEFAULT_RENSA_SKETCH_MODE,
    )
    parser.add_argument(
        "--rensa-query-mode",
        choices=RENSA_QUERY_MODES,
        default=DEFAULT_RENSA_QUERY_MODE,
    )
    parser.add_argument(
        "--rensa-rho-probes",
        type=int,
        default=DEFAULT_RENSA_RHO_PROBES,
    )
    parser.add_argument("--baseline-json", type=Path, default=DEFAULT_BASELINE_PATH)

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
    tokenizer_fragment = tokenizer_fingerprint()[:TOKENIZER_FINGERPRINT_FRAGMENT_LEN]
    filename = (
        f"{spec.key}__{name_fragment}__{spec.split}__{columns_fragment}"
        f"__ng{ngram_size}__stream{int(spec.streaming)}"
        f"__rev_{revision_suffix}"
        f"__cache_v{TOKEN_CACHE_SCHEMA_VERSION}"
        f"__tok_{tokenizer_fragment}__rows_{row_suffix}.pkl"
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


def sha256_token_sets(token_sets: list[list[str]]) -> str:
    hasher = hashlib.sha256()
    for token_set in token_sets:
        hasher.update(len(token_set).to_bytes(8, "little"))
        for token in token_set:
            encoded = token.encode("utf-8")
            hasher.update(len(encoded).to_bytes(8, "little"))
            hasher.update(encoded)
        hasher.update(b"\xff")
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


_TOKENIZER_FINGERPRINT: str | None = None


def tokenizer_fingerprint() -> str:
    global _TOKENIZER_FINGERPRINT
    if _TOKENIZER_FINGERPRINT is not None:
        return _TOKENIZER_FINGERPRINT

    hasher = hashlib.sha256()
    components = (
        str(TOKEN_CACHE_SCHEMA_VERSION),
        inspect.getsource(extract_text),
        inspect.getsource(tokenize_to_ngrams),
        repr(FALLBACK_TEXT_COLUMNS),
    )
    for component in components:
        hasher.update(component.encode("utf-8"))
        hasher.update(b"\0")
    _TOKENIZER_FINGERPRINT = hasher.hexdigest()
    return _TOKENIZER_FINGERPRINT


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


def build_token_cache_payload(
    spec: DatasetSpec,
    max_rows: int | None,
    ngram_size: int,
    token_sets: list[list[str]],
) -> dict[str, Any]:
    return {
        "schema_version": TOKEN_CACHE_SCHEMA_VERSION,
        "tokenizer_fingerprint": tokenizer_fingerprint(),
        "dataset_key": spec.key,
        "hf_dataset": spec.hf_dataset,
        "split": spec.split,
        "text_columns": list(spec.text_columns),
        "revision": spec.revision,
        "streaming": spec.streaming,
        "max_rows": max_rows,
        "ngram_size": ngram_size,
        "token_sets_sha256": sha256_token_sets(token_sets),
        "token_sets": token_sets,
    }


def load_token_cache_payload(
    cache_path: Path,
    spec: DatasetSpec | None = None,
    max_rows: int | None = None,
    ngram_size: int | None = None,
) -> dict[str, Any] | None:
    with cache_path.open("rb") as handle:
        payload = pickle.load(handle)

    if isinstance(payload, list):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("schema_version") != TOKEN_CACHE_SCHEMA_VERSION:
        return None
    if payload.get("tokenizer_fingerprint") != tokenizer_fingerprint():
        return None

    token_sets = payload.get("token_sets")
    if not isinstance(token_sets, list):
        return None
    if not isinstance(payload.get("token_sets_sha256"), str):
        payload["token_sets_sha256"] = sha256_token_sets(token_sets)

    if spec is None:
        return payload

    expected = {
        "dataset_key": spec.key,
        "hf_dataset": spec.hf_dataset,
        "split": spec.split,
        "text_columns": list(spec.text_columns),
        "revision": spec.revision,
        "streaming": spec.streaming,
        "max_rows": max_rows,
        "ngram_size": ngram_size,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            return None
    return payload


def load_or_prepare_token_cache(
    cache_dir: Path,
    spec: DatasetSpec,
    max_rows: int | None,
    ngram_size: int,
) -> tuple[Path, int, str, str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = dataset_cache_path(cache_dir, spec, max_rows, ngram_size)

    if cache_path.exists():
        payload = load_token_cache_payload(
            cache_path,
            spec=spec,
            max_rows=max_rows,
            ngram_size=ngram_size,
        )
        if payload is not None:
            row_count = len(payload["token_sets"])
            return (
                cache_path,
                row_count,
                sha256_file(cache_path),
                payload["token_sets_sha256"],
            )

    token_sets = load_token_sets_from_hf(spec, max_rows, ngram_size)
    payload = build_token_cache_payload(spec, max_rows, ngram_size, token_sets)
    with cache_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return (
        cache_path,
        len(token_sets),
        sha256_file(cache_path),
        payload["token_sets_sha256"],
    )


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


_RENSA_BUILD_CHECKED = False


def assert_rensa_release_build() -> None:
    global _RENSA_BUILD_CHECKED
    if _RENSA_BUILD_CHECKED:
        return
    _RENSA_BUILD_CHECKED = True

    try:
        import rensa  # type: ignore
    except Exception:
        # Import failures will surface later with a clearer stack trace.
        return

    get_build_info = getattr(rensa, "get_build_info", None)
    if get_build_info is None:
        # Older revisions don't expose build info, so we can't validate profile.
        return

    info = get_build_info()
    profile = info.get("profile")
    debug_assertions = bool(info.get("debug_assertions"))
    if debug_assertions or profile != "release":
        raise RuntimeError(
            "Benchmarks must run against a release build of rensa. "
            f"Detected profile={profile!r}, debug_assertions={debug_assertions}. "
            "Rebuild with: maturin develop --release --uv --manifest-path crates/rensa-py/Cargo.toml"
        )


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
        "phase_model": "split",
        "sketch": sketch_elapsed,
        "build": build_elapsed,
        "query": query_elapsed,
        "fused_build_query": None,
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
        "phase_model": "split",
        "sketch": sketch_elapsed,
        "build": build_elapsed,
        "query": query_elapsed,
        "fused_build_query": None,
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
    sketch_mode: str = "compat",
    query_mode: str = "compat",
    rho_probes: int = DEFAULT_RENSA_RHO_PROBES,
) -> tuple[dict[str, Any], list[bool]]:
    assert_rensa_release_build()
    from rensa import RMinHash, RMinHashLSH  # type: ignore

    matrix: Any | None = None
    minhashes: list[Any] | None = None
    sketch_start = perf_counter()
    if sketch_mode == "compat":
        if hasattr(RMinHash, "digest_matrix_from_token_sets_rho"):
            matrix = RMinHash.digest_matrix_from_token_sets_rho(
                token_sets,
                num_perm=num_perm,
                seed=seed,
                probes=rho_probes,
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
    elif sketch_mode == "rho_matrix":
        digest_matrix_from_token_sets_rho = getattr(
            RMinHash, "digest_matrix_from_token_sets_rho", None
        )
        if digest_matrix_from_token_sets_rho is None:
            raise RuntimeError("Requested rensa sketch mode 'rho_matrix' is unavailable")
        matrix = RMinHash.digest_matrix_from_token_sets_rho(
            token_sets,
            num_perm=num_perm,
            seed=seed,
            probes=rho_probes,
        )
    elif sketch_mode == "matrix":
        digest_matrix_from_token_sets = getattr(
            RMinHash, "digest_matrix_from_token_sets", None
        )
        if digest_matrix_from_token_sets is None:
            raise RuntimeError("Requested rensa sketch mode 'matrix' is unavailable")
        matrix = RMinHash.digest_matrix_from_token_sets(
            token_sets,
            num_perm=num_perm,
            seed=seed,
        )
    elif sketch_mode == "per_item":
        minhashes = []
        for tokens in token_sets:
            minhash = RMinHash(num_perm=num_perm, seed=seed)
            minhash.update(tokens)
            minhashes.append(minhash)
    else:
        raise ValueError(f"Unknown rensa sketch mode: {sketch_mode}")
    sketch_elapsed = perf_counter() - sketch_start

    lsh = RMinHashLSH(
        threshold=threshold,
        num_perm=num_perm,
        num_bands=num_bands,
        seed=seed,
    )

    build_elapsed = 0.0
    query_elapsed = 0.0
    fused_build_query_elapsed: float | None = None
    phase_model = "split"
    if query_mode == "compat":
        if matrix is not None and hasattr(lsh, "query_duplicate_flags_matrix_one_shot"):
            fused_start = perf_counter()
            duplicate_flags = [
                bool(value) for value in lsh.query_duplicate_flags_matrix_one_shot(matrix)
            ]
            fused_build_query_elapsed = perf_counter() - fused_start
            phase_model = "fused_build_query"
        elif matrix is not None and hasattr(lsh, "insert_matrix_and_query_duplicate_flags"):
            fused_start = perf_counter()
            duplicate_flags = [
                bool(value)
                for value in lsh.insert_matrix_and_query_duplicate_flags(matrix, start_key=0)
            ]
            fused_build_query_elapsed = perf_counter() - fused_start
            phase_model = "fused_build_query"
        elif matrix is not None and hasattr(lsh, "insert_matrix") and hasattr(
            lsh, "query_duplicate_flags_matrix"
        ):
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
    elif query_mode == "matrix_one_shot":
        if matrix is None:
            raise RuntimeError("Rensa query mode 'matrix_one_shot' requires matrix sketch mode")
        query_duplicate_flags_matrix_one_shot = getattr(
            lsh, "query_duplicate_flags_matrix_one_shot", None
        )
        if query_duplicate_flags_matrix_one_shot is None:
            raise RuntimeError("Requested rensa query mode 'matrix_one_shot' is unavailable")
        fused_start = perf_counter()
        duplicate_flags = [bool(value) for value in query_duplicate_flags_matrix_one_shot(matrix)]
        fused_build_query_elapsed = perf_counter() - fused_start
        phase_model = "fused_build_query"
    elif query_mode == "matrix_insert_query":
        if matrix is None:
            raise RuntimeError(
                "Rensa query mode 'matrix_insert_query' requires matrix sketch mode"
            )
        insert_matrix_and_query_duplicate_flags = getattr(
            lsh, "insert_matrix_and_query_duplicate_flags", None
        )
        if insert_matrix_and_query_duplicate_flags is None:
            raise RuntimeError(
                "Requested rensa query mode 'matrix_insert_query' is unavailable"
            )
        fused_start = perf_counter()
        duplicate_flags = [
            bool(value) for value in insert_matrix_and_query_duplicate_flags(matrix, start_key=0)
        ]
        fused_build_query_elapsed = perf_counter() - fused_start
        phase_model = "fused_build_query"
    elif query_mode == "matrix_build_query":
        if matrix is None:
            raise RuntimeError("Rensa query mode 'matrix_build_query' requires matrix sketch mode")
        insert_matrix = getattr(lsh, "insert_matrix", None)
        query_duplicate_flags_matrix = getattr(lsh, "query_duplicate_flags_matrix", None)
        if insert_matrix is None or query_duplicate_flags_matrix is None:
            raise RuntimeError("Requested rensa query mode 'matrix_build_query' is unavailable")
        build_start = perf_counter()
        insert_matrix(matrix, start_key=0)
        build_elapsed = perf_counter() - build_start

        query_start = perf_counter()
        duplicate_flags = [bool(value) for value in query_duplicate_flags_matrix(matrix)]
        query_elapsed = perf_counter() - query_start
    elif query_mode == "per_item":
        if minhashes is None:
            raise RuntimeError("Rensa query mode 'per_item' requires per-item sketch mode")
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
        raise ValueError(f"Unknown rensa query mode: {query_mode}")

    rows = len(duplicate_flags)
    rows_removed = sum(1 for is_duplicate in duplicate_flags if is_duplicate)
    metrics = {
        "phase_model": phase_model,
        "sketch": sketch_elapsed,
        "build": build_elapsed,
        "query": query_elapsed,
        "fused_build_query": fused_build_query_elapsed,
        "total": sketch_elapsed + build_elapsed + query_elapsed + (fused_build_query_elapsed or 0.0),
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
    rensa_sketch_mode: str,
    rensa_query_mode: str,
    rensa_rho_probes: int,
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
            sketch_mode=rensa_sketch_mode,
            query_mode=rensa_query_mode,
            rho_probes=rensa_rho_probes,
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

    payload = load_token_cache_payload(args.token_cache)
    if payload is None:
        raise RuntimeError(
            "Token cache is stale or uses an unsupported schema. Rebuild it with the current "
            "benchmarks/full_benchmark.py cache loader."
        )
    token_sets: list[list[str]] = payload["token_sets"]
    token_sets_sha = payload["token_sets_sha256"]

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
            rensa_sketch_mode=args.rensa_sketch_mode,
            rensa_query_mode=args.rensa_query_mode,
            rensa_rho_probes=args.rensa_rho_probes,
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
        "token_sets_sha256": token_sets_sha,
        "token_cache_schema_version": TOKEN_CACHE_SCHEMA_VERSION,
        "tokenizer_fingerprint": tokenizer_fingerprint(),
        "rensa_modes": {
            "sketch_mode": args.rensa_sketch_mode,
            "query_mode": args.rensa_query_mode,
            "rho_probes": args.rensa_rho_probes,
        },
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
    rensa_sketch_mode: str,
    rensa_query_mode: str,
    rensa_rho_probes: int,
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
        "--rensa-sketch-mode",
        rensa_sketch_mode,
        "--rensa-query-mode",
        rensa_query_mode,
        "--rensa-rho-probes",
        str(rensa_rho_probes),
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
    token_set_hashes = {run["token_sets_sha256"] for run in runs}
    if len(token_set_hashes) != 1:
        raise ValueError(
            f"Token set hash mismatch across runs: {sorted(token_set_hashes)}"
        )

    thread_env_ok = all(
        all(bool(value) for value in run["thread_env_assertions"].values()) for run in runs
    )
    if not thread_env_ok:
        raise ValueError("Thread environment assertions failed in one or more runs")

    engine_summary: dict[str, Any] = {}
    for engine in ENGINE_KEYS:
        phase_models = {run["engines"][engine]["phase_model"] for run in runs}
        if len(phase_models) != 1:
            raise ValueError(
                f"Phase model mismatch across runs for {engine}: {sorted(phase_models)}"
            )
        totals = [run["engines"][engine]["total"] for run in runs]
        sketches = [run["engines"][engine]["sketch"] for run in runs]
        builds = [run["engines"][engine]["build"] for run in runs]
        queries = [run["engines"][engine]["query"] for run in runs]
        fused_build_queries = [
            run["engines"][engine]["fused_build_query"]
            for run in runs
            if run["engines"][engine]["fused_build_query"] is not None
        ]
        rows_removed = [run["engines"][engine]["rows_removed"] for run in runs]
        rows_remaining = [run["engines"][engine]["rows_remaining"] for run in runs]
        avg_candidates = [
            run["engines"][engine]["avg_candidates_per_row"]
            for run in runs
            if run["engines"][engine]["avg_candidates_per_row"] is not None
        ]

        engine_summary[engine] = {
            "phase_model": next(iter(phase_models)),
            "median_total": median(totals),
            "median_sketch": median(sketches),
            "median_build": median(builds),
            "median_query": median(queries),
            "median_fused_build_query": (
                median(fused_build_queries) if fused_build_queries else None
            ),
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
            "token_sets_sha256": next(iter(token_set_hashes)),
            "thread_env_assertions_all_true": True,
        },
    }


def canonicalize_dataset_fingerprint(
    fingerprint: dict[str, Any],
) -> dict[str, Any]:
    canonical = dict(fingerprint)
    token_sets_sha = canonical.get("token_sets_sha256")
    if not isinstance(token_sets_sha, str) or not token_sets_sha:
        raise ValueError(
            "Baseline dataset fingerprint is missing token_sets_sha256. "
            "Refresh the locked baseline artifact with the current benchmark script."
        )
    canonical.pop("token_cache_sha256", None)
    return canonical


def build_dataset_fingerprints(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fingerprints: dict[str, dict[str, Any]] = {}
    for result in results:
        dataset = result["dataset"]
        key = dataset["key"]
        fingerprint = {
            "key": key,
            "hf_dataset": dataset["hf_dataset"],
            "split": dataset["split"],
            "text_columns": list(dataset["text_columns"]),
            "revision": dataset["revision"],
            "streaming": bool(dataset["streaming"]),
            "effective_max_rows": result["effective_max_rows"],
            "token_cache_sha256": result["token_cache_sha256"],
            "token_sets_sha256": result["token_sets_sha256"],
            "token_cache_schema_version": TOKEN_CACHE_SCHEMA_VERSION,
            "tokenizer_fingerprint": tokenizer_fingerprint(),
        }
        existing = fingerprints.get(key)
        if existing is None:
            fingerprints[key] = fingerprint
            continue
        if existing != fingerprint:
            raise ValueError(
                f"Dataset fingerprint mismatch across thread lanes for '{key}': "
                f"{existing} != {fingerprint}"
            )
    return [fingerprints[key] for key in sorted(fingerprints)]


def aggregate_rensa_totals(results: list[dict[str, Any]]) -> dict[str, float]:
    totals_by_thread: dict[str, float] = {}
    for result in results:
        thread_key = str(result["threads"])
        totals_by_thread.setdefault(thread_key, 0.0)
        totals_by_thread[thread_key] += result["summary"]["engine_medians"]["rensa"]["median_total"]
    return totals_by_thread


def mean_accuracy_metric(
    results: list[dict[str, Any]],
    *,
    jaccard: bool,
) -> float:
    if not results:
        return 0.0
    if jaccard:
        values = [
            result["summary"]["accuracy"]["median_jaccard"]["datasketch_vs_rensa"]
            for result in results
        ]
    else:
        values = [
            result["summary"]["accuracy"]["mismatch_vs_datasketch"]["rensa"]["median_rate"]
            for result in results
        ]
    return statistics.fmean(values)


def validate_baseline_fingerprint(
    current_payload: dict[str, Any],
    baseline_payload: dict[str, Any],
) -> None:
    current_fingerprint = current_payload["input_fingerprint"]
    baseline_fingerprint = baseline_payload["input_fingerprint"]

    if current_fingerprint["ngram_size"] != baseline_fingerprint["ngram_size"]:
        raise ValueError(
            "Baseline ngram_size mismatch: "
            f"{current_fingerprint['ngram_size']} != {baseline_fingerprint['ngram_size']}"
        )
    if current_fingerprint["rensa"] != baseline_fingerprint["rensa"]:
        raise ValueError(
            "Baseline Rensa mode mismatch: "
            f"{current_fingerprint['rensa']} != {baseline_fingerprint['rensa']}"
        )

    current_datasets = current_fingerprint["datasets"]
    baseline_datasets = baseline_fingerprint["datasets"]
    canonical_current = [
        canonicalize_dataset_fingerprint(fingerprint)
        for fingerprint in current_datasets
    ]
    canonical_baseline = [
        canonicalize_dataset_fingerprint(fingerprint)
        for fingerprint in baseline_datasets
    ]
    if canonical_current != canonical_baseline:
        raise ValueError(
            "Baseline dataset fingerprint mismatch: "
            f"{canonical_current} != {canonical_baseline}"
        )


def compare_to_locked_baseline(
    payload: dict[str, Any],
    baseline_path: Path,
) -> dict[str, Any] | None:
    if not baseline_path.exists():
        return None

    baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    validate_baseline_fingerprint(payload, baseline_payload)

    baseline_config = baseline_payload["config"]
    current_config = payload["config"]
    comparable_fields = (
        "datasets",
        "threads",
        "num_perm",
        "num_bands",
        "threshold",
        "seed",
        "ngram_size",
    )
    for field in comparable_fields:
        if current_config[field] != baseline_config[field]:
            raise ValueError(
                f"Baseline config mismatch for {field}: "
                f"{current_config[field]!r} != {baseline_config[field]!r}"
            )

    current_totals = aggregate_rensa_totals(payload["results"])
    baseline_totals = {
        str(key): value
        for key, value in baseline_payload["locked_baseline"][
            "aggregate_rensa_median_total"
        ].items()
    }
    current_jaccard = mean_accuracy_metric(payload["results"], jaccard=True)
    current_mismatch = mean_accuracy_metric(payload["results"], jaccard=False)
    baseline_jaccard = baseline_payload["locked_baseline"]["mean_jaccard"]
    baseline_mismatch = baseline_payload["locked_baseline"]["mean_mismatch"]

    thread_checks = {
        thread: current_totals[thread] <= baseline_totals[thread]
        for thread in baseline_totals
    }
    fairness = {
        "thread_env_assertions_all_true": all(
            result["summary"]["fairness"]["thread_env_assertions_all_true"]
            for result in payload["results"]
        ),
        "token_cache_hash_consistent": all(
            result["summary"]["fairness"]["token_cache_hash_consistent"]
            for result in payload["results"]
        ),
    }
    accuracy = {
        "jaccard_non_regression": current_jaccard >= baseline_jaccard,
        "mismatch_non_regression": current_mismatch <= baseline_mismatch,
    }

    passed = all(thread_checks.values()) and all(fairness.values()) and all(accuracy.values())
    return {
        "baseline_path": str(baseline_path),
        "validated": True,
        "passed": passed,
        "aggregate_rensa_median_total": {
            "current": current_totals,
            "baseline": baseline_totals,
            "non_regression": thread_checks,
        },
        "accuracy": {
            "current_mean_jaccard": current_jaccard,
            "baseline_mean_jaccard": baseline_jaccard,
            "current_mean_mismatch": current_mismatch,
            "baseline_mean_mismatch": baseline_mismatch,
            **accuracy,
        },
        "fairness": fairness,
    }


def print_section_summary(dataset_key: str, threads: int, summary: dict[str, Any]) -> None:
    medians = summary["engine_medians"]
    speedups = summary["speedup_vs_datasketch"]
    accuracy = summary["accuracy"]

    print("\n" + "=" * 80)
    print(f"{dataset_key.upper()} | threads={threads}")
    print("=" * 80)
    print(f"{'Engine':<14} {'Median Total(s)':<16} {'Speedup vs DS':<14}")
    print("-" * 80)
    for engine in ENGINE_KEYS:
        engine_stats = medians[engine]
        speedup = speedups[engine]
        speedup_label = f"{speedup:.2f}x" if speedup is not None else "n/a"
        print(
            f"{engine:<14} "
            f"{engine_stats['median_total']:<16.4f} "
            f"{speedup_label:<14}"
        )

    print("\nEngine-specific phase medians (s):")
    print(
        f"{'Engine':<14} {'Model':<18} {'Sketch':<10} {'Build':<10} {'Query':<10} {'Fused':<10}"
    )
    print("-" * 80)
    for engine in ENGINE_KEYS:
        engine_stats = medians[engine]
        fused_value = engine_stats["median_fused_build_query"]
        fused_label = f"{fused_value:.4f}" if fused_value is not None else "n/a"
        print(
            f"{engine:<14} "
            f"{engine_stats['phase_model']:<18} "
            f"{engine_stats['median_sketch']:<10.4f} "
            f"{engine_stats['median_build']:<10.4f} "
            f"{engine_stats['median_query']:<10.4f} "
            f"{fused_label:<10}"
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
    if args.rensa_rho_probes <= 0:
        raise ValueError("--rensa-rho-probes must be > 0")
    if args.rensa_sketch_mode == "per_item" and args.rensa_query_mode != "per_item":
        raise ValueError(
            "--rensa-sketch-mode=per_item requires --rensa-query-mode=per_item"
        )
    if args.rensa_sketch_mode != "per_item" and args.rensa_query_mode == "per_item":
        raise ValueError(
            "--rensa-query-mode=per_item requires --rensa-sketch-mode=per_item"
        )
    if args.rensa_sketch_mode == "compat" and args.rensa_query_mode != "compat":
        raise ValueError(
            "--rensa-sketch-mode=compat requires --rensa-query-mode=compat"
        )
    if args.rensa_sketch_mode != "compat" and args.rensa_query_mode == "compat":
        raise ValueError(
            "--rensa-query-mode=compat requires --rensa-sketch-mode=compat"
        )

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
        f"ngram_size={args.ngram_size} warmup_runs={args.warmup_runs} repetitions={args.repetitions} "
        f"rensa_sketch_mode={args.rensa_sketch_mode} "
        f"rensa_query_mode={args.rensa_query_mode} "
        f"rensa_rho_probes={args.rensa_rho_probes}"
    )
    print(f"Datasets: {dataset_keys}")
    print(f"Thread modes: {thread_modes}")
    print("Dataset revisions are pinned for reproducibility.")
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
        (
            token_cache,
            row_count,
            token_cache_sha256,
            token_sets_sha256,
        ) = load_or_prepare_token_cache(
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
                    rensa_sketch_mode=args.rensa_sketch_mode,
                    rensa_query_mode=args.rensa_query_mode,
                    rensa_rho_probes=args.rensa_rho_probes,
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
                    "token_sets_sha256": token_sets_sha256,
                    "runs": runs,
                    "summary": summary,
                }
            )

    input_fingerprint = {
        "ngram_size": args.ngram_size,
        "rensa": {
            "sketch_mode": args.rensa_sketch_mode,
            "query_mode": args.rensa_query_mode,
            "rho_probes": args.rensa_rho_probes,
        },
        "datasets": build_dataset_fingerprints(full_results),
    }

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
            "rensa_sketch_mode": args.rensa_sketch_mode,
            "rensa_query_mode": args.rensa_query_mode,
            "rensa_rho_probes": args.rensa_rho_probes,
            "max_rows": global_max_rows,
            "dataset_max_rows": dataset_max_rows_overrides,
            "cache_dir": str(args.cache_dir),
            "baseline_json": str(args.baseline_json),
            "token_cache_schema_version": TOKEN_CACHE_SCHEMA_VERSION,
            "tokenizer_fingerprint": tokenizer_fingerprint(),
            "rensa_env": current_rensa_env(),
            "datasketch_threading_note": (
                "Datasketch does not expose explicit thread count controls; "
                "lane env vars are pinned for process symmetry."
            ),
        },
        "input_fingerprint": input_fingerprint,
        "results": full_results,
    }

    payload["baseline_comparison"] = compare_to_locked_baseline(
        payload,
        args.baseline_json,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"\nWrote benchmark JSON to {args.output_json}")

    comparison = payload["baseline_comparison"]
    if comparison is None:
        print(f"No baseline artifact found at {args.baseline_json}, skipped baseline comparison.")
        return

    print(
        "Baseline comparison: "
        f"threads={comparison['aggregate_rensa_median_total']['non_regression']} "
        f"accuracy={{'jaccard': {comparison['accuracy']['jaccard_non_regression']}, "
        f"'mismatch': {comparison['accuracy']['mismatch_non_regression']}}} "
        f"fairness={comparison['fairness']}"
    )
    if not comparison["passed"]:
        raise RuntimeError(
            "Locked baseline regression detected. "
            f"See {args.output_json} and {args.baseline_json} for details."
        )


if __name__ == "__main__":
    parsed_args = parse_args()
    main(parsed_args)
