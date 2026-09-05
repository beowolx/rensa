#!/usr/bin/env python3
"""Compare native batch sketching on identical UTF-8 bytes, using one thread.

Each engine/size/signature-length case runs in a fresh process. Timing includes
parameter construction and native returned outputs, but excludes input encoding,
digest normalization, checksums, and returned-object destruction. Rho is a
sampled retrieval representation, so its speed needs separate accuracy results.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import pickle
import platform
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

from datasketch import MinHash
from FastSketchLSH import FastSimilaritySketch
from full_benchmark import sha256_file
from kernel_benchmark import cpu_name, documents, measure
from rensa import CMinHash, RMinHash

ENGINES = ("rensa_classic", "rensa_rho", "rensa_c", "datasketch", "fastsketch")


def inputs(args, size):
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


def operation(engine, rows, num_perm, seed):
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
        for name in ("rows", "repetitions", "min_sample_seconds", "warmup_seconds", "seed"):
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
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 128, 1024])
    parser.add_argument("--num-perm", type=int, nargs="+", default=[128])
    parser.add_argument("--engines", choices=ENGINES, nargs="+", default=list(ENGINES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--min-sample-seconds", "--min-sample", type=float, default=0.1)
    parser.add_argument("--warmup-seconds", type=float, default=0.2)
    parser.add_argument("--in-process", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.rows, args.repetitions, *args.sizes, *args.num_perm) <= 0:
        parser.error("rows, repetitions, sizes, and num-perm must be positive")
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
            "flags": {key: value for key, value in sorted(os.environ.items())
                      if key in THREAD_ENV or key.startswith("RENSA_")},
        },
        "methodology": {
            "input": "Identical UTF-8 bytes, same row/token order, no deduplication or sorting",
            "timed": "Parameter construction, batch sketching, native output allocation",
            "excluded": "Input preparation, output normalization/checksums, returned-object destruction",
            "isolation": "fresh process per engine/input/k" if not args.in_process else "single process",
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
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    for size in ([None] if args.token_cache else args.sizes):
        if args.in_process:
            rows, input_metadata = inputs(args, size)
        for num_perm in args.num_perm:
            checksums = set()
            for engine in args.engines:
                if args.in_process:
                    run, extract = operation(engine, rows, num_perm, args.seed)
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
                raise RuntimeError("Engines did not receive identical input bytes")
            args.output_json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
