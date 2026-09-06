#!/usr/bin/env python3
"""Deterministic, stdlib-only Rensa kernels; run the same file against both wheels.

Times include returned-object allocation, but exclude input generation, digest
extraction, correctness hashing, and result destruction. Update cases include
construction of fresh sketches. Query cases exclude index construction.
"""

import argparse
import hashlib
import importlib.machinery
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from rensa import CMinHash, CMinHashDeduplicator, RMinHash, RMinHashLSH

CASES = (
    "hash", "r_update", "c_update", "r_batch", "r_prehashed", "c_prehashed",
    "rho", "rho_dedup", "c_dedup", "query",
)


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def cpu_name():
    if platform.system() == "Darwin":
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True, capture_output=True
        )
        if result.returncode == 0:
            return result.stdout.strip()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    return platform.processor() or platform.machine()


def cpu_flags():
    """Report OS-advertised CPU features, not the binary's selected kernel."""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            name, _, value = line.partition(":")
            if name.strip() in ("flags", "Features"):
                return sorted(value.split())
    return None


def native_binary_sha256(distribution_name):
    distribution = importlib.metadata.distribution(distribution_name)
    return {
        str(path): hashlib.sha256(distribution.locate_file(path).read_bytes()).hexdigest()
        for path in distribution.files or ()
        if str(path).endswith(tuple(importlib.machinery.EXTENSION_SUFFIXES))
    }


def documents(rows, size):
    # Each consecutive group of four contains one exact duplicate pair and
    # two disjoint documents. Strings, row order, and overlap are reproducible.
    return [
        [f"document-{row - (row % 4 == 1)}-token-{token}" for token in range(size)]
        for row in range(rows)
    ]


def update_all(cls, docs, num_perm, seed):
    sketches = []
    for doc in docs:
        sketch = cls(num_perm, seed)
        sketch.update(doc)
        sketches.append(sketch)
    return sketches


def operation(case, docs, hashes, num_perm, seed):
    if case == "hash":
        return lambda: RMinHash.hash_token_sets(docs), lambda result: result
    if case == "c_dedup":
        signatures = CMinHash.from_token_sets(docs, num_perm, seed)
        entries = [(str(index), value) for index, value in enumerate(signatures)]
        def insert():
            index = CMinHashDeduplicator(0.8, num_perm, seed)
            flags = index.add_pairs(entries)
            return index, flags
        return insert, lambda result: [result[0].len(), result[1]]
    if case in ("r_update", "c_update"):
        cls = RMinHash if case == "r_update" else CMinHash
        extract = (lambda result: [s.digest() for s in result]) if case == "r_update" else (
            lambda result: [s.digest_u64() for s in result]
        )
        return lambda: update_all(cls, docs, num_perm, seed), extract
    if case == "c_prehashed":
        return lambda: CMinHash.digests64_from_token_hash_sets(hashes, num_perm, seed), lambda x: x
    if case == "query":
        sketches = RMinHash.from_token_sets(docs, num_perm, seed)
        index = RMinHashLSH(0.5, num_perm, 16)
        index.insert_many(sketches)
        return lambda: index.query_duplicate_flags(sketches), lambda x: x
    if case == "rho_dedup":
        def run():
            matrix = RMinHash.digest_matrix_from_token_sets_rho(docs, num_perm, seed)
            index = RMinHashLSH(0.5, num_perm, 16)
            flags = index.query_duplicate_flags_matrix_one_shot(matrix)
            return matrix, flags
        return run, lambda result: [result[0].to_rows(), result[1]]
    method, values = {
        "r_batch": (RMinHash.digest_matrix_from_token_sets, docs),
        "r_prehashed": (RMinHash.digest_matrix_from_token_hash_sets, hashes),
        "rho": (RMinHash.digest_matrix_from_token_sets_rho, docs),
    }[case]
    return lambda: method(values, num_perm, seed), lambda result: result.to_rows()


def measure(run, extract, repetitions, min_sample_seconds, warmup_seconds):
    result = run()  # Warm caches, lazy SIMD dispatch, and Rayon initialization.
    expected = fingerprint(extract(result))
    warmup_deadline = time.perf_counter() + warmup_seconds
    while time.perf_counter() < warmup_deadline:
        del result
        result = run()
    if fingerprint(extract(result)) != expected:
        raise RuntimeError("Non-deterministic output during warmup")
    del result
    samples = []
    iterations = []
    for _ in range(repetitions):
        elapsed = 0.0
        count = 0
        while elapsed < min_sample_seconds or count == 0:
            result = None
            start = time.perf_counter_ns()
            result = run()
            elapsed += (time.perf_counter_ns() - start) / 1e9
            count += 1
        samples.append(elapsed / count)
        iterations.append(count)
        if fingerprint(extract(result)) != expected:
            raise RuntimeError("Non-deterministic output across repetitions")
        del result
    return {"seconds": samples, "iterations": iterations, "median_seconds": statistics.median(samples), "sha256": expected}


def isolated_case(args, case, size, num_perm):
    with tempfile.TemporaryDirectory(prefix="rensa-kernel-") as directory:
        output = Path(directory) / "result.json"
        command = [
            sys.executable, str(Path(__file__).resolve()), "--in-process",
            "--output-json", str(output), "--cases", case, "--sizes", str(size),
            "--num-perm", str(num_perm),
        ]
        for name in ("rows", "repetitions", "min_sample_seconds", "warmup_seconds", "seed"):
            command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
        completed = subprocess.run(command, text=True, capture_output=True)
        if completed.returncode:
            raise RuntimeError(
                f"Kernel subprocess failed ({case}, tokens={size}, perm={num_perm}):\n"
                f"{completed.stderr}\n{completed.stdout}"
            )
        return json.loads(output.read_text())["results"][0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--min-sample-seconds", type=float, default=0.1)
    parser.add_argument("--warmup-seconds", type=float, default=0.2)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 8, 32, 128, 1024, 4096])
    parser.add_argument("--num-perm", type=int, nargs="+", default=[128, 512])
    parser.add_argument("--cases", choices=CASES, nargs="+", default=list(CASES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--in-process", action="store_true",
                        help="Run cases together for internal execution or diagnosis.")
    args = parser.parse_args()
    if min(args.rows, args.repetitions, *args.sizes, *args.num_perm) <= 0:
        parser.error("rows, repetitions, sizes, and num-perm must be positive")
    if not 0 <= args.seed <= 2**64 - 1:
        parser.error("seed must fit the unsigned 64-bit seed range")
    if not 0 <= args.warmup_seconds < float("inf"):
        parser.error("warmup-seconds must be finite and nonnegative")
    if not 0 <= args.min_sample_seconds < float("inf"):
        parser.error("min-sample-seconds must be finite and nonnegative")
    if any(n % 16 for n in args.num_perm) and any(case in args.cases for case in ("query", "rho_dedup")):
        parser.error("num-perm must divide evenly into the fixed 16-band LSH")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    report = {
        "schema_version": 1,
        "environment": {
            "platform": platform.platform(), "machine": platform.machine(),
            "cpu": cpu_name(), "logical_cpus": os.cpu_count(),
            "cpu_flags": cpu_flags(),
            "native_binary_sha256": {"rensa": native_binary_sha256("rensa")},
            "python": platform.python_version(), "rensa": importlib.metadata.version("rensa"),
            "cminhash_algorithm_version": getattr(CMinHash, "ALGORITHM_VERSION", 1),
            "rminhash_algorithm_version": getattr(RMinHash, "ALGORITHM_VERSION", 1),
            "flags": {key: value for key, value in sorted(os.environ.items())
                      if key.startswith(("RENSA_", "RAYON_")) or key in ("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS")},
        },
        "config": {key: value for key, value in vars(args).items() if key != "output_json"},
        "results": results,
    }
    for size in args.sizes:
        if args.in_process:
            docs = documents(args.rows, size)
            hashes = RMinHash.hash_token_sets(docs)
        for num_perm in args.num_perm:
            checksums = {}
            for case in args.cases:
                if args.in_process:
                    run, extract = operation(case, docs, hashes, num_perm, args.seed)
                    result = measure(run, extract, args.repetitions, args.min_sample_seconds, args.warmup_seconds)
                else:
                    result = isolated_case(args, case, size, num_perm)
                checksums[case] = result["sha256"]
                result.update(case=case, tokens_per_row=size, num_perm=num_perm)
                results.append(result)
                print(f"{case:12s} tokens={size:4d} perm={num_perm:3d}: {result['median_seconds']:.6f}s", flush=True)
            for equivalent in (("r_update", "r_batch", "r_prehashed"), ("c_update", "c_prehashed")):
                if len({checksums[name] for name in equivalent if name in checksums}) > 1:
                    raise RuntimeError(f"Digest mismatch among equivalent APIs: {equivalent}")
        args.output_json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
