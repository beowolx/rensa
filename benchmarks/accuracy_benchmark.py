"""Compare sketch error and retrieval against exact Jaccard, without downloads."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path

for name in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
from datasketch import MinHash, MinHashLSH
from FastSketchLSH import FastSimilaritySketch, LSH, estimate_jaccard
from rensa import CMinHash, RMinHash, RMinHashLSH


def pairs(sizes, targets, repeats, orders):
    rows, cases = [], []
    rng = random.Random(20260905)
    for size in sizes:
        seen_overlaps = set()
        for target in targets:
            overlap = round(2 * size * target / (1 + target))
            if overlap in seen_overlaps:
                continue
            seen_overlaps.add(overlap)
            for order in orders:
                for repeat in range(repeats):
                    prefix = f"pair-{len(cases)}-"
                    common = [f"{prefix}shared-{i}" for i in range(overlap)]
                    left = common + [f"{prefix}left-{i}" for i in range(size - overlap)]
                    right = common + [f"{prefix}right-{i}" for i in range(size - overlap)]
                    if order == "shuffled":
                        rng.shuffle(left)
                        rng.shuffle(right)
                    elif order == "reversed":
                        right.reverse()
                    left_set, right_set = set(left), set(right)
                    exact = len(left_set & right_set) / len(left_set | right_set)
                    cases.append({"size": size, "target": target, "order": order,
                                  "repeat": repeat, "exact": exact})
                    rows.extend([left, right])
    return rows, cases


def sketch(engine, rows, k, seed):
    if engine == "rensa_classic":
        matrix = RMinHash.digest_matrix_from_token_sets(rows, k, seed)
        return np.asarray(matrix.to_rows(), dtype=np.uint64), matrix
    if engine == "rensa_c":
        return np.asarray(CMinHash.digests64_from_token_sets(rows, k, seed), dtype=np.uint64), None
    if engine == "datasketch":
        hashes = list(MinHash.generator(([token.encode() for token in row] for row in rows),
                                      num_perm=k, seed=seed))
        return np.asarray([mh.hashvalues for mh in hashes]), hashes
    if engine == "fastsketch":
        matrix = FastSimilaritySketch(k, seed=seed).batch(rows, num_threads=1)
        assert estimate_jaccard(matrix[0], matrix[1]) == float(np.mean(matrix[0] == matrix[1]))
        return matrix, matrix
    if engine == "rensa_rho":
        matrix = RMinHash.digest_matrix_from_token_sets_rho(rows, k, seed)
        return None, matrix
    raise ValueError(engine)


def summarize(estimates, exacts, k, threshold):
    errors = [estimate - exact for estimate, exact in zip(estimates, exacts)]
    bias = statistics.mean(errors)
    mse = statistics.mean(value * value for value in errors)
    reference_mse = statistics.mean(j * (1 - j) / k for j in exacts)
    return {"count": len(errors), "bias": bias, "rmse": math.sqrt(mse),
            "mae": statistics.mean(abs(value) for value in errors),
            "error_variance": statistics.pvariance(errors),
            "bias_standard_error": statistics.pstdev(errors) / math.sqrt(len(errors)),
            "ideal_independent_minhash_rmse": math.sqrt(reference_mse),
            "mse_over_ideal_minhash": mse / reference_mse if reference_mse else None,
            "threshold_classification": confusion(
                [j >= threshold for j in exacts],
                [estimate >= threshold for estimate in estimates])}


def accuracy(args):
    rows, cases = pairs(args.sizes, [0, .25, .5, .75, .79, .81, .9, 1],
                       args.repeats, ["shuffled"])
    result = {}
    for k in args.perms:
        for engine in args.engines:
            all_estimates, exacts = [], []
            grouped = defaultdict(lambda: ([], []))
            for seed in range(args.seeds):
                matrix, _ = sketch(engine, rows, k, seed)
                estimates = np.mean(matrix[0::2] == matrix[1::2], axis=1)
                for case, estimated in zip(cases, estimates):
                    all_estimates.append(float(estimated))
                    exacts.append(case["exact"])
                    cell = grouped[f'n={case["size"]},j={case["exact"]:.6f}']
                    cell[0].append(float(estimated))
                    cell[1].append(case["exact"])
            result[f"{engine}/k={k}"] = {
                **summarize(all_estimates, exacts, k, args.threshold),
                "by_case": {key: summarize(*values, k, args.threshold)
                            for key, values in grouped.items()}}
            print(f"accuracy {engine} k={k}: RMSE={result[f'{engine}/k={k}']['rmse']:.6f}", flush=True)
    return result


def confusion(truth, predicted):
    tp = sum(a and b for a, b in zip(truth, predicted))
    tn = sum(not a and not b for a, b in zip(truth, predicted))
    fp = sum(not a and b for a, b in zip(truth, predicted))
    fn = sum(a and not b for a, b in zip(truth, predicted))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "false_positive_rate": fp / (fp + tn) if fp + tn else None}


def retrieval(args):
    rows, cases = pairs(args.sizes, [0, .5, .7, .79, .81, .9, 1], args.repeats,
                       ["aligned", "shuffled", "reversed"])
    row_cases = [case for case in cases for _ in range(2)]
    truth = [case["exact"] >= args.threshold for case in row_cases]
    result = {}
    k = 128
    for engine in ["rensa_classic", "rensa_rho", "datasketch", "fastsketch"]:
        actual, all_truth = [], []
        grouped = defaultdict(lambda: ([], []))
        for seed in range(args.retrieval_seeds):
            matrix, native = sketch(engine, rows, k, seed)
            if engine in ("rensa_classic", "rensa_rho"):
                index = RMinHashLSH(threshold=args.threshold, num_perm=k, num_bands=8)
                flags = list(index.query_duplicate_flags_matrix_one_shot(native))
            elif engine == "datasketch":
                index = MinHashLSH(num_perm=k, params=(8, 16))
                for i, signature in enumerate(native):
                    index.insert(i, signature, check_duplication=False)
                flags = [any(candidate != i for candidate in index.query(signature))
                         for i, signature in enumerate(native)]
            elif engine == "fastsketch":
                index = LSH(num_perm=k, num_bands=8, num_threads=1)
                flags = list(index.insert_and_query_duplicates(matrix))
            flags = [bool(flag) for flag in flags]
            actual.extend(flags)
            all_truth.extend(truth)
            for expected, flag, case in zip(truth, flags, row_cases):
                key = f'n={case["size"]},j={case["exact"]:.6f},order={case["order"]}'
                grouped[key][0].append(expected)
                grouped[key][1].append(flag)
        result[engine] = {**confusion(all_truth, actual),
                          "by_case": {key: confusion(*values) for key, values in grouped.items()}}
        print(f"retrieval {engine}: {confusion(all_truth, actual)}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[2, 16, 128, 1024])
    parser.add_argument("--perms", type=int, nargs="+", default=[128, 512])
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--retrieval-seeds", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=.8)
    parser.add_argument("--engines", nargs="+", choices=["rensa_classic", "rensa_c", "datasketch", "fastsketch"],
                        default=["rensa_classic", "rensa_c", "datasketch", "fastsketch"])
    parser.add_argument("--skip-retrieval", action="store_true")
    args = parser.parse_args()
    if min(*args.sizes, *args.perms, args.seeds, args.repeats, args.retrieval_seeds) < 1:
        parser.error("sizes, permutations, seeds and repeats must be positive")
    if not 0 <= args.threshold <= 1:
        parser.error("threshold must be between zero and one")
    payload = {"config": {key: str(value) if isinstance(value, Path) else value
                           for key, value in vars(args).items()},
               "versions": {name: version(name) for name in ("rensa", "datasketch", "FastSketchLSH")},
               "cminhash_algorithm_version": getattr(CMinHash, "ALGORITHM_VERSION", 1),
               "environment": {key: value for key, value in sorted(os.environ.items())
                               if key.startswith(("RENSA_", "RAYON_"))},
               "retrieval_config": {"num_perm": 128, "num_bands": 8},
               "notes": ["Ground truth is exact intersection/union on string sets.",
                         "Different pair namespaces make non-mate pairs exactly disjoint.",
                         "All sketches use identical token lists and the same number of signature slots.",
                         "C-MinHash/FastSketch slots are 64-bit; classic/Datasketch slots have 32-bit values.",
                         "Rho is only evaluated as a retrieval pipeline, not as a slot-equality Jaccard estimator.",
                         "Datasketch and FastSketch retrieval are unverified 8-band/16-row candidates.",
                         "Threshold classification compares all four full-signature engines using the same estimate >= threshold rule.",
                         "C streaming removal semantics differ from query-all retrieval and are excluded from the retrieval section.",
                         "Retrieval counts rows having another row above the exact threshold; both members of a matching pair are positives.",
                         "Standard errors are descriptive; seeded trials share input pairs and are not a rigorous confidence interval."]}
    payload["accuracy"] = accuracy(args)
    if not args.skip_retrieval:
        payload["retrieval"] = retrieval(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
