"""Time + peak-memory benchmark for the Rensa batch dedup pipeline.

Measures, per thread lane and per phase:
- wall time (median over N subprocess runs)
- peak RSS growth (VmHWM delta) and retained RSS growth (VmRSS delta)

Phases:
- sketch: RMinHash.digest_matrix_from_token_sets_rho
- dedup:  RMinHashLSH.query_duplicate_flags_matrix_one_shot
- update_path: classic per-document RMinHash.update loop (subset of rows)

The corpus is synthetic and fully deterministic (seeded), with injected exact
and near duplicates so duplicate-flag precision/recall can be tracked as an
accuracy guardrail across code changes.

Each (lane, run) executes in a fresh subprocess so peak-RSS attribution is
clean and allocator state does not leak between runs.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

DEFAULT_ROWS = 20_000
DEFAULT_NUM_PERM = 128
DEFAULT_NUM_BANDS = 8
DEFAULT_THRESHOLD = 0.8
DEFAULT_SEED = 42
DEFAULT_PROBES = 4
DEFAULT_RUNS = 5
DEFAULT_THREAD_LANES = "1,8"
DEFAULT_UPDATE_PATH_ROWS = 2_000
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / ".bench" / "profiling"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="baseline", help="Tag stored in the output JSON.")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--num-perm", type=int, default=DEFAULT_NUM_PERM)
    parser.add_argument("--num-bands", type=int, default=DEFAULT_NUM_BANDS)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--probes", type=int, default=DEFAULT_PROBES)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument(
        "--thread-lanes",
        default=DEFAULT_THREAD_LANES,
        help="Comma-separated thread counts, e.g. '1,8'.",
    )
    parser.add_argument("--update-path-rows", type=int, default=DEFAULT_UPDATE_PATH_ROWS)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def build_corpus(rows: int, seed: int) -> tuple[list[list[str]], list[bool]]:
    """Deterministic synthetic corpus with injected duplicates.

    Returns (token_sets, should_flag) where should_flag[i] is True when row i
    belongs to a duplicate group (either as base or as copy).
    """
    rng = random.Random(seed)
    vocab = [f"w{i}" for i in range(32_768)]

    token_sets: list[list[str]] = []
    should_flag = [False] * rows

    for index in range(rows):
        draw = rng.random()
        if index > 100 and draw < 0.05:
            # Exact duplicate of an earlier row.
            base = rng.randrange(index)
            token_sets.append(list(token_sets[base]))
            should_flag[index] = True
            should_flag[base] = True
            continue
        if index > 100 and draw < 0.20:
            # Near duplicate: perturb ~5% of tokens of an earlier row.
            base = rng.randrange(index)
            tokens = list(token_sets[base])
            n_swap = max(1, len(tokens) // 20)
            for _ in range(n_swap):
                tokens[rng.randrange(len(tokens))] = rng.choice(vocab)
            token_sets.append(tokens)
            should_flag[index] = True
            should_flag[base] = True
            continue

        length_draw = rng.random()
        if length_draw < 0.60:
            length = rng.randint(8, 32)
        elif length_draw < 0.90:
            length = rng.randint(33, 96)
        else:
            length = rng.randint(97, 400)
        token_sets.append([rng.choice(vocab) for _ in range(length)])

    return token_sets, should_flag


def read_vm_status() -> dict[str, int]:
    """Read VmRSS/VmHWM in KiB from /proc/self/status (Linux only)."""
    values: dict[str, int] = {}
    with open("/proc/self/status", "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(("VmRSS:", "VmHWM:")):
                key, raw = line.split(":", 1)
                values[key] = int(raw.strip().split()[0])
    return values


class PhaseRecorder:
    def __init__(self) -> None:
        self.phases: dict[str, dict[str, float]] = {}
        self._start_vm: dict[str, int] = {}
        self._start_time = 0.0
        self._name = ""

    def start(self, name: str) -> None:
        self._name = name
        self._start_vm = read_vm_status()
        self._start_time = perf_counter()

    def stop(self) -> None:
        elapsed = perf_counter() - self._start_time
        vm = read_vm_status()
        self.phases[self._name] = {
            "elapsed_s": elapsed,
            "vm_hwm_delta_kib": vm["VmHWM"] - self._start_vm["VmHWM"],
            "vm_rss_delta_kib": vm["VmRSS"] - self._start_vm["VmRSS"],
            "vm_hwm_kib": vm["VmHWM"],
        }


def flag_accuracy(flags: list[bool], should_flag: list[bool]) -> dict[str, Any]:
    true_positive = sum(1 for f, t in zip(flags, should_flag) if f and t)
    false_positive = sum(1 for f, t in zip(flags, should_flag) if f and not t)
    false_negative = sum(1 for f, t in zip(flags, should_flag) if not f and t)
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    return {
        "flagged_rows": sum(flags),
        "expected_flagged_rows": sum(should_flag),
        "precision": precision,
        "recall": recall,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def run_worker(args: argparse.Namespace) -> None:
    from rensa import RMinHash, RMinHashLSH

    token_sets, should_flag = build_corpus(args.rows, args.seed)
    recorder = PhaseRecorder()

    recorder.start("sketch")
    matrix = RMinHash.digest_matrix_from_token_sets_rho(
        token_sets, num_perm=args.num_perm, seed=args.seed, probes=args.probes
    )
    recorder.stop()

    lsh = RMinHashLSH(threshold=args.threshold, num_perm=args.num_perm, num_bands=args.num_bands)
    recorder.start("dedup")
    flags = [bool(value) for value in lsh.query_duplicate_flags_matrix_one_shot(matrix)]
    recorder.stop()

    update_rows = min(args.update_path_rows, len(token_sets))
    recorder.start("update_path")
    digest_tail = 0
    for tokens in token_sets[:update_rows]:
        minhash = RMinHash(num_perm=args.num_perm, seed=args.seed)
        minhash.update(tokens)
        digest_tail ^= minhash.digest()[-1]
    recorder.stop()

    payload = {
        "phases": recorder.phases,
        "accuracy": flag_accuracy(flags, should_flag),
        "update_path_rows": update_rows,
        "update_path_digest_tail": digest_tail,
        "pipeline_total_s": recorder.phases["sketch"]["elapsed_s"] + recorder.phases["dedup"]["elapsed_s"],
    }
    print(json.dumps(payload))


def spawn_worker(args: argparse.Namespace, threads: int) -> dict[str, Any]:
    env = dict(os.environ)
    for key in THREAD_ENV_VARS:
        env[key] = str(threads)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--rows", str(args.rows),
        "--num-perm", str(args.num_perm),
        "--num-bands", str(args.num_bands),
        "--threshold", str(args.threshold),
        "--seed", str(args.seed),
        "--probes", str(args.probes),
        "--update-path-rows", str(args.update_path_rows),
    ]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
    return json.loads(result.stdout.splitlines()[-1])


def summarize_lane(runs: list[dict[str, Any]]) -> dict[str, Any]:
    phase_names = runs[0]["phases"].keys()
    summary: dict[str, Any] = {"phases": {}}
    for name in phase_names:
        summary["phases"][name] = {
            "median_elapsed_s": statistics.median(run["phases"][name]["elapsed_s"] for run in runs),
            "min_elapsed_s": min(run["phases"][name]["elapsed_s"] for run in runs),
            "median_vm_hwm_delta_kib": statistics.median(
                run["phases"][name]["vm_hwm_delta_kib"] for run in runs
            ),
            "median_vm_rss_delta_kib": statistics.median(
                run["phases"][name]["vm_rss_delta_kib"] for run in runs
            ),
        }
    summary["median_pipeline_total_s"] = statistics.median(run["pipeline_total_s"] for run in runs)
    summary["accuracy"] = runs[0]["accuracy"]
    return summary


def main() -> None:
    args = parse_args()
    if args.worker:
        run_worker(args)
        return

    lanes = [int(part) for part in args.thread_lanes.split(",") if part.strip()]
    lane_results: dict[str, Any] = {}
    for threads in lanes:
        runs = []
        for run_index in range(args.runs):
            payload = spawn_worker(args, threads)
            runs.append(payload)
            print(
                f"threads={threads} run={run_index} "
                f"sketch={payload['phases']['sketch']['elapsed_s']:.4f}s "
                f"dedup={payload['phases']['dedup']['elapsed_s']:.4f}s "
                f"update={payload['phases']['update_path']['elapsed_s']:.4f}s "
                f"sketch_hwm=+{payload['phases']['sketch']['vm_hwm_delta_kib']}KiB "
                f"dedup_hwm=+{payload['phases']['dedup']['vm_hwm_delta_kib']}KiB"
            )
        lane_results[str(threads)] = {"runs": runs, "summary": summarize_lane(runs)}

    print("\n" + "=" * 88)
    print(f"Perf + memory benchmark [{args.label}] rows={args.rows} runs={args.runs}")
    print("=" * 88)
    header = (
        f"{'lane':<6} {'phase':<12} {'median(s)':<11} {'min(s)':<11} "
        f"{'peakRSS delta':<14} {'retained delta':<14}"
    )
    print(header)
    print("-" * 88)
    for threads in lanes:
        summary = lane_results[str(threads)]["summary"]
        for name, stats in summary["phases"].items():
            print(
                f"{threads:<6} {name:<12} {stats['median_elapsed_s']:<11.4f} "
                f"{stats['min_elapsed_s']:<11.4f} "
                f"{stats['median_vm_hwm_delta_kib'] / 1024:<14.2f} "
                f"{stats['median_vm_rss_delta_kib'] / 1024:<14.2f}"
            )
        accuracy = summary["accuracy"]
        print(
            f"{threads:<6} accuracy: flagged={accuracy['flagged_rows']} "
            f"expected={accuracy['expected_flagged_rows']} "
            f"precision={accuracy['precision']:.4f} recall={accuracy['recall']:.4f}"
        )
    print("(peak/retained RSS deltas in MiB)")

    output_json = args.output_json or (DEFAULT_OUTPUT_DIR / f"perf_memory_{args.label}.json")
    payload = {
        "metadata": {
            "label": args.label,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "rensa_env": {
                key: value for key, value in sorted(os.environ.items()) if key.startswith("RENSA_")
            },
        },
        "config": {
            "rows": args.rows,
            "num_perm": args.num_perm,
            "num_bands": args.num_bands,
            "threshold": args.threshold,
            "seed": args.seed,
            "probes": args.probes,
            "runs": args.runs,
            "thread_lanes": lanes,
            "update_path_rows": args.update_path_rows,
        },
        "lanes": lane_results,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"\nWrote benchmark JSON to {output_json}")


if __name__ == "__main__":
    main()
