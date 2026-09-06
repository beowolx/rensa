#!/usr/bin/env python3
"""Compare frozen R-MinHash builds in alternating intervals on identical inputs.

This is an additional diagnostic, with no performance thresholds. Both native
extensions stay loaded in one process; FastSketch is an unchanged timing control.
"""

import argparse
import importlib.machinery
import importlib.metadata
import importlib.util
import json
import os
import platform
import statistics
from pathlib import Path

# This import sets thread limits before initializing NumPy or native libraries.
from sketch_benchmark import THREAD_ENV, inputs, operation
from full_benchmark import sha256_file
from kernel_benchmark import cpu_flags, cpu_name, measure, native_binary_sha256


def load_extensions(base_path, head_path):
    if base_path.samefile(head_path):
        raise ValueError("base and head must be separate native extension files")
    modules = {}
    metadata = {}
    for name, path in (("base", base_path), ("head", head_path)):
        spec = importlib.util.spec_from_file_location(f"{name}.rensa", path)
        if spec is None or not isinstance(spec.loader, importlib.machinery.ExtensionFileLoader):
            raise ValueError(f"not a native extension file: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[name] = module
        metadata[name] = {
            "path": str(path.resolve()), "sha256": sha256_file(path),
            "algorithm_version": getattr(module.RMinHash, "ALGORITHM_VERSION", 1),
            "version": getattr(module, "__version__", None),
        }
    if modules["base"].RMinHash is modules["head"].RMinHash:
        raise ValueError("base and head resolved to the same RMinHash class")
    if metadata["base"]["algorithm_version"] != metadata["head"]["algorithm_version"]:
        raise ValueError("paired comparison requires the same R-MinHash algorithm version")
    return modules, metadata


def native_operation(module, rows, num_perm, seed, prehashed):
    # Bind the frozen class directly; the shared helper uses the installed class.
    method = (module.RMinHash.digest_matrix_from_flat_token_hashes if prehashed
              else module.RMinHash.digest_matrix_from_token_sets)
    values = rows if prehashed else (rows,)
    return lambda: method(*values, num_perm, seed), lambda result: result.to_rows()


def paired_measure(operations, repetitions, min_sample_seconds, warmup_seconds):
    samples = {}
    for engine, (run, extract) in operations.items():
        warmed = measure(run, extract, 1, 0, warmup_seconds)
        samples[engine] = {"seconds": [], "iterations": [], "sha256": warmed["sha256"]}
    if samples["base"]["sha256"] != samples["head"]["sha256"]:
        raise RuntimeError("Base/head signatures differ")

    # AB/BA alternates every cycle; FastSketch rotates through all three slots.
    orders = (
        ("base", "head", "fastsketch"), ("head", "base", "fastsketch"),
        ("fastsketch", "base", "head"), ("fastsketch", "head", "base"),
        ("base", "fastsketch", "head"), ("head", "fastsketch", "base"),
    )
    engine_order = []
    for cycle in range(repetitions):
        order = orders[cycle % len(orders)]
        engine_order.append(order)
        for engine in order:
            run, extract = operations[engine]
            result = measure(run, extract, 1, min_sample_seconds, 0)
            if result["sha256"] != samples[engine]["sha256"]:
                raise RuntimeError(f"{engine} signatures changed in paired cycle {cycle}")
            samples[engine]["seconds"].extend(result["seconds"])
            samples[engine]["iterations"].extend(result["iterations"])
    for result in samples.values():
        result["median_seconds"] = statistics.median(result["seconds"])
    head_over_base = [h / b for b, h in zip(samples["base"]["seconds"], samples["head"]["seconds"])]
    fastsketch_over_head = [f / h for f, h in zip(samples["fastsketch"]["seconds"], samples["head"]["seconds"])]
    return {
        "engine_order": engine_order, "samples": samples,
        "paired_head_over_base": head_over_base,
        "paired_fastsketch_over_head": fastsketch_over_head,
        "median_paired_head_over_base": statistics.median(head_over_base),
        "median_paired_fastsketch_over_head": statistics.median(fastsketch_over_head),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-extension", type=Path, required=True)
    parser.add_argument("--head-extension", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--prehashed", action="store_true")
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 128, 4096])
    parser.add_argument("--num-perm", type=int, nargs="+", default=[128, 512])
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--max-input-tokens", type=int, default=8_388_608)
    parser.add_argument("--repeat-cardinality", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repetitions", type=int, default=6)
    parser.add_argument("--min-sample-seconds", type=float, default=0.1)
    parser.add_argument("--warmup-seconds", type=float, default=0.2)
    args = parser.parse_args()
    if min(args.rows, args.max_input_tokens, args.repetitions, *args.num_perm) <= 0 or min(args.sizes) < 0:
        parser.error("rows, token budget, repetitions and num-perm must be positive; sizes must be nonnegative")
    if max(args.sizes) > args.max_input_tokens:
        parser.error("a row cannot exceed max-input-tokens")
    if any(k & (k - 1) or k > 4096 for k in args.num_perm):
        parser.error("FastSketch requires power-of-two num-perm no greater than 4096")
    if not args.prehashed and 0 in args.sizes:
        parser.error("FastSketch's byte API rejects empty rows; use prehashed mode")
    if args.repeat_cardinality is not None and (not args.prehashed or args.repeat_cardinality <= 0):
        parser.error("repeat-cardinality requires prehashed mode and a positive value")
    if not 0 <= args.seed <= 2**32 - 1:
        parser.error("seed must fit the shared unsigned 32-bit seed range")
    if not all(0 <= value < float("inf") for value in (args.min_sample_seconds, args.warmup_seconds)):
        parser.error("sample and warmup durations must be finite and nonnegative")
    try:
        modules, extensions = load_extensions(args.base_extension, args.head_extension)
    except (OSError, ImportError, ValueError) as error:
        parser.error(str(error))

    report = {
        "schema_version": 1, "extensions": extensions,
        "environment": {
            "platform": platform.platform(), "python": platform.python_version(),
            "cpu": cpu_name(), "cpu_flags": cpu_flags(),
            "versions": {name: importlib.metadata.version(name) for name in ("FastSketchLSH", "numpy")},
            "fastsketch_binary_sha256": native_binary_sha256("FastSketchLSH"),
            "flags": {key: value for key, value in sorted(os.environ.items())
                      if key in THREAD_ENV or key.startswith("RENSA_")},
        },
        "methodology": {
            "timed": "Native batch call, internal per-call setup, native output allocation",
            "excluded": "Input preparation, normalization/checksums, returned-object destruction; reusable FastSketch constructor only in prehashed mode",
            "order": "Alternating base/head order, with FastSketch position balanced over six cycles",
            "validation": "Existing normalized JSON fingerprint; exact R signatures and per-engine invariance after warmup and every sample",
        },
        "config": {key: value for key, value in vars(args).items()
                   if key not in ("base_extension", "head_extension", "output_json")},
        "results": [],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    for size in args.sizes:
        input_args = argparse.Namespace(**vars(args))
        input_args.token_cache = None
        input_args.rows = min(args.rows, args.max_input_tokens // max(size, 1))
        rows, metadata = inputs(input_args, size)
        for num_perm in args.num_perm:
            operations = {name: native_operation(module, rows, num_perm, args.seed, args.prehashed)
                          for name, module in modules.items()}
            operations["fastsketch"] = operation("fastsketch", rows, num_perm, args.seed, args.prehashed)
            result = paired_measure(operations, args.repetitions, args.min_sample_seconds, args.warmup_seconds)
            result.update(metadata, num_perm=num_perm)
            report["results"].append(result)
            print(f"tokens={size} k={num_perm}: paired head/base={result['median_paired_head_over_base']:.4f}, "
                  f"FastSketch/head={result['median_paired_fastsketch_over_head']:.4f}", flush=True)
            args.output_json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
