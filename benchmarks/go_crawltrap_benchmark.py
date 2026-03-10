from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / ".bench"
    / "profiling"
    / "go_crawltrap_benchmark.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the fixed-corpus Go crawler-trap benchmark command and write "
            "its JSON report to disk."
        ),
    )
    parser.add_argument(
        "--go-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "examples" / "go-crawltrap",
        help="Path to the examples/go-crawltrap Go module.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of end-to-end corpus iterations to run.",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="Optional path to the corpus fixture, relative to --go-dir unless absolute.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Where to write the benchmark JSON payload.",
    )
    parser.add_argument(
        "--print-stdout",
        action="store_true",
        help="Echo raw stdout from the Go benchmark command.",
    )
    return parser.parse_args()


def benchmark_summary(payload: dict[str, Any]) -> str:
    metrics = payload["metrics"]
    average_ms = metrics["average_duration_ns"] / 1_000_000.0
    total_ms = metrics["total_duration_ns"] / 1_000_000.0
    return (
        f"{payload['benchmark_name']}: {payload['config']['iterations']} iterations, "
        f"avg {average_ms:.3f} ms/run, total {total_ms:.3f} ms"
    )


def main() -> int:
    args = parse_args()
    output_json = args.output_json.resolve()

    command = [
        "go",
        "run",
        "./cmd/corpusbench",
        "-iterations",
        str(args.iterations),
        "-output",
        str(output_json),
    ]
    if args.corpus is not None:
        command.extend(["-corpus", str(args.corpus)])

    completed = subprocess.run(
        command,
        cwd=args.go_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        sys.stderr.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        return completed.returncode

    if not output_json.exists():
        raise SystemExit(
            f"Go benchmark command succeeded but did not write {output_json}"
        )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    print(f"Wrote Go crawltrap benchmark summary to {output_json}")
    print(benchmark_summary(payload))

    if args.print_stdout and completed.stdout.strip():
        print(completed.stdout.rstrip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
