#!/usr/bin/env python3
"""Compare native batch sketching on identical bytes or prehashed CSR inputs.

Each engine/size/signature-length case runs in a fresh process. Timing includes
parameter construction and native returned outputs, but excludes input encoding,
digest normalization, checksums, and returned-object destruction. Rho is a
sampled retrieval representation, so its speed needs separate accuracy results.
Prehashed mode reuses FastSketch's constructor outside timing and caps input size.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import pickle
import platform
import random
import subprocess
import sys
import tempfile
from pathlib import Path

# Set these before importing libraries that initialize their thread pools.
THREAD_ENV = (
    "RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)
for variable in THREAD_ENV:
    os.environ[variable] = "1"
os.environ["RENSA_PIPELINE_QUEUE_CAP"] = "0"

from datasketch import MinHash
from FastSketchLSH import FastSimilaritySketch
import numpy as np
from full_benchmark import sha256_file
from kernel_benchmark import cpu_name, documents, measure
from rensa import CMinHash, RMinHash

ENGINES = ("rensa_classic", "rensa_rho", "rensa_c", "datasketch", "fastsketch")


def mixed_sequence(count):
    # SplitMix's bijective finalizer produces unique, scattered uint64 values.
    values = np.arange(count, dtype=np.uint64)
    values += np.uint64(12345 + 0x9E3779B97F4A7C15)
    values ^= values >> np.uint64(30)
    values *= np.uint64(0xBF58476D1CE4E5B9)
    values ^= values >> np.uint64(27)
    values *= np.uint64(0x94D049BB133111EB)
    values ^= values >> np.uint64(31)
    return values


def prehashed_inputs(args, size):
    rows = min(args.rows, args.max_input_tokens // max(size, 1))
    cardinality = min(size, args.repeat_cardinality or size)
    if cardinality == size:
        flat = mixed_sequence(rows * size)
    else:
        vocabulary = mixed_sequence(rows * cardinality).reshape(rows, cardinality)
        flat = np.take(vocabulary, np.arange(size) % cardinality, axis=1).reshape(-1)
    offsets = np.arange(rows + 1, dtype=np.uint64) * np.uint64(size)
    flat.flags.writeable = offsets.flags.writeable = False
    digest = hashlib.sha256(b"rensa-prehashed-csr-v1\0")
    digest.update(rows.to_bytes(8, "little"))
    digest.update(size.to_bytes(8, "little"))
    for values in (flat, offsets):
        digest.update(memoryview(values.astype("<u8", copy=False)).cast("B"))
    return (flat, offsets), {
        "rows": rows, "tokens": rows * size, "tokens_per_row": size,
        "distinct_tokens_per_row": cardinality,
        "input_sha256": digest.hexdigest(), "input_bytes": flat.nbytes + offsets.nbytes,
        "token_cache": None, "token_cache_sha256": None,
    }


def inputs(args, size):
    if args.prehashed:
        return prehashed_inputs(args, size)
    cache_sha = None
    if args.token_cache:
        cache_sha = sha256_file(args.token_cache)
        with args.token_cache.open("rb") as handle:
            rows = pickle.load(handle)
    else:
        rows = documents(args.rows, size)
    rows = [
        [token.encode("utf-8") if isinstance(token, str) else token for token in row]
        for row in rows
    ]
    digest = hashlib.sha256()
    digest.update(len(rows).to_bytes(8, "little"))
    for row in rows:
        digest.update(len(row).to_bytes(8, "little"))
        for token in row:
            if not isinstance(token, bytes):
                raise TypeError("Token caches must contain rows of str or bytes tokens")
            digest.update(len(token).to_bytes(8, "little"))
            digest.update(token)
    return rows, {
        "rows": len(rows), "tokens": sum(map(len, rows)),
        "tokens_per_row": size, "input_sha256": digest.hexdigest(),
        "token_cache": str(args.token_cache) if args.token_cache else None,
        "token_cache_sha256": cache_sha,
    }


def operation(engine, rows, num_perm, seed, prehashed=False):
    if prehashed:
        flat, offsets = rows
        if engine == "rensa_classic":
            return (
                lambda: RMinHash.digest_matrix_from_flat_token_hashes(flat, offsets, num_perm, seed),
                lambda result: result.to_rows(),
            )
        sketcher = FastSimilaritySketch(num_perm, seed=seed)
        return (
            lambda: sketcher.batch_csr(flat, offsets, prehashed=True, num_threads=1),
            lambda result: result.tolist(),
        )
    if engine in ("rensa_classic", "rensa_rho"):
        method = (
            RMinHash.digest_matrix_from_token_sets if engine == "rensa_classic"
            else RMinHash.digest_matrix_from_token_sets_rho
        )
        return lambda: method(rows, num_perm, seed), lambda result: result.to_rows()
    if engine == "rensa_c":
        # Keep signatures native; digests64 would box every slot as a Python int.
        return (
            lambda: CMinHash.from_token_sets(rows, num_perm, seed),
            lambda result: [sketch.digest_u64() for sketch in result],
        )
    if engine == "datasketch":
        return (
            lambda: list(MinHash.generator(rows, num_perm=num_perm, seed=seed)),
            lambda result: [sketch.hashvalues.tolist() for sketch in result],
        )

    def run_fastsketch():
        sketcher = FastSimilaritySketch(num_perm, seed=seed)
        result = sketcher.batch(rows, num_threads=1)
        return sketcher, result  # Destroy both after the timed interval.

    return run_fastsketch, lambda result: result[1].tolist()


def isolated_case(args, engine, size, num_perm):
    with tempfile.TemporaryDirectory(prefix="rensa-sketch-") as directory:
        output = Path(directory) / "result.json"
        command = [
            sys.executable, str(Path(__file__).resolve()), "--in-process",
            "--output-json", str(output), "--engines", engine,
            "--num-perm", str(num_perm),
        ]
        if args.token_cache:
            command.extend(["--token-cache", str(args.token_cache)])
        else:
            command.extend(["--sizes", str(size)])
        if args.prehashed:
            command.append("--prehashed")
        if args.repeat_cardinality is not None:
            command.extend(["--repeat-cardinality", str(args.repeat_cardinality)])
        for name in ("rows", "max_input_tokens", "repetitions", "min_sample_seconds", "warmup_seconds", "seed"):
            command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
        completed = subprocess.run(command, text=True, capture_output=True)
        if completed.returncode:
            raise RuntimeError(
                f"Sketch subprocess failed ({engine}, tokens={size}, perm={num_perm}):\n"
                f"{completed.stderr}\n{completed.stdout}"
            )
        return json.loads(output.read_text())["results"][0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--token-cache", type=Path,
                        help="Existing local token pickle; uses all cached rows, ignoring sizes/rows.")
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--prehashed", action="store_true",
                        help="Compare Rensa/FastSketch CSR APIs using identical readonly uint64 buffers.")
    parser.add_argument("--max-input-tokens", type=int, default=8_388_608,
                        help="Cap prehashed tokens per case by reducing rows for long inputs.")
    parser.add_argument("--repeat-cardinality", type=int,
                        help="Repeat this many distinct values per prehashed row; default is all unique.")
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 128, 1024])
    parser.add_argument("--num-perm", type=int, nargs="+", default=[128])
    parser.add_argument("--engines", choices=ENGINES, nargs="+")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--min-sample-seconds", "--min-sample", type=float, default=0.1)
    parser.add_argument("--warmup-seconds", type=float, default=0.2)
    parser.add_argument("--in-process", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.engines is None:
        args.engines = ["rensa_classic", "fastsketch"] if args.prehashed else list(ENGINES)
    if min(args.rows, args.repetitions, args.max_input_tokens, *args.num_perm) <= 0 or min(args.sizes) < 0:
        parser.error("rows, repetitions, token budget and num-perm must be positive; sizes must be nonnegative")
    if args.prehashed:
        if args.token_cache or set(args.engines) - {"rensa_classic", "fastsketch"}:
            parser.error("prehashed mode supports only Rensa classic and FastSketch, without a token cache")
        if max(args.sizes) > args.max_input_tokens:
            parser.error("a prehashed row cannot exceed max-input-tokens")
    elif args.repeat_cardinality is not None:
        parser.error("repeat-cardinality requires prehashed mode")
    if args.repeat_cardinality is not None and args.repeat_cardinality <= 0:
        parser.error("repeat-cardinality must be positive")
    if "fastsketch" in args.engines and any(k & (k - 1) or k > 4096 for k in args.num_perm):
        parser.error("FastSketch requires power-of-two num-perm no greater than 4096")
    if not 0 <= args.seed <= 2**32 - 1:
        parser.error("seed must fit the shared unsigned 32-bit seed range")
    if not all(0 <= value < float("inf") for value in (args.min_sample_seconds, args.warmup_seconds)):
        parser.error("sample and warmup durations must be finite and nonnegative")
    report = {
        "schema_version": 1,
        "environment": {
            "platform": platform.platform(), "machine": platform.machine(),
            "cpu": cpu_name(), "python": platform.python_version(),
            "versions": {name: importlib.metadata.version(name)
                         for name in ("rensa", "datasketch", "FastSketchLSH", "numpy")},
            "cminhash_algorithm_version": getattr(CMinHash, "ALGORITHM_VERSION", None),
            "rminhash_algorithm_version": getattr(RMinHash, "ALGORITHM_VERSION", 1),
            "flags": {key: value for key, value in sorted(os.environ.items())
                      if key in THREAD_ENV or key.startswith("RENSA_")},
        },
        "methodology": {
            "input": "Identical UTF-8 bytes, same row/token order, no deduplication or sorting",
            "timed": "Parameter construction, batch sketching, native output allocation",
            "excluded": "Input preparation, output normalization/checksums, returned-object destruction",
            "isolation": "fresh process per engine/input/k" if not args.in_process else "single process",
            "engine_order": "deterministic shuffle per size/k" if args.prehashed else "configured order",
            "accuracy": "Reported separately; rho uses position sampling and is not a classic MinHash estimator",
            "validation": "Normalized SHA256 checked after warmup and every measured repetition",
            "apis": {
                "rensa_classic": "RMinHash.digest_matrix_from_token_sets",
                "rensa_rho": "RMinHash.digest_matrix_from_token_sets_rho",
                "rensa_c": "CMinHash.from_token_sets",
                "datasketch": "MinHash.generator (fully consumed)",
                "fastsketch": "FastSimilaritySketch(...).batch(..., num_threads=1)",
            },
        },
        "config": {key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items() if key != "output_json"},
        "results": [],
    }
    if args.prehashed:
        report["methodology"].update(
            input="Identical readonly contiguous uint64 values and CSR offsets; SplitMix64 sequence seeded with 12345",
            timed="Native batch call, internal per-call setup/copies, native output allocation",
            excluded="Input preparation, reusable FastSketch constructor, output normalization/checksums, returned-object destruction",
            apis={"rensa_classic": "RMinHash.digest_matrix_from_flat_token_hashes",
                  "fastsketch": "FastSimilaritySketch(...).batch_csr(..., prehashed=True, num_threads=1)"},
        )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    for size in ([None] if args.token_cache else args.sizes):
        if args.in_process:
            rows, input_metadata = inputs(args, size)
        for num_perm in args.num_perm:
            checksums = set()
            engines = list(args.engines)
            if args.prehashed:
                random.Random(f"{args.seed}:{size}:{num_perm}").shuffle(engines)
            for engine in engines:
                if args.in_process:
                    run, extract = operation(engine, rows, num_perm, args.seed, args.prehashed)
                    result = measure(run, extract, args.repetitions, args.min_sample_seconds, args.warmup_seconds)
                    result.update(input_metadata)
                else:
                    result = isolated_case(args, engine, size, num_perm)
                checksums.add(result["input_sha256"])
                result.update(engine=engine, num_perm=num_perm)
                report["results"].append(result)
                print(f"{engine:14s} rows={result['rows']:5d} tokens={size} perm={num_perm}: "
                      f"{result['median_seconds']:.6f}s", flush=True)
            if len(checksums) != 1:
                raise RuntimeError("Engines did not receive identical inputs")
            args.output_json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
