import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate benchmark JSON results for absolute and regression modes.",
    )
    parser.add_argument("--mode", choices=["absolute", "compare"], required=True)
    parser.add_argument("--head-json", type=Path, required=True)
    parser.add_argument("--base-json", type=Path)

    parser.add_argument("--min-speedup", type=float, default=40.0)
    parser.add_argument("--max-slowdown-fraction", type=float, default=0.10)
    parser.add_argument("--min-jaccard-ds-r", type=float, default=0.999)
    parser.add_argument("--min-jaccard-ds-c", type=float, default=0.998)
    parser.add_argument("--min-jaccard-r-c", type=float, default=0.998)
    parser.add_argument(
        "--require-ds-r-set-equality",
        action="store_true",
        help=(
            "Require exact set equality between Datasketch and R-MinHash. "
            "Disabled by default for threshold-based deduplication benchmarks."
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _extract_value(payload: dict[str, object], path: tuple[str, ...]) -> object:
    current: object = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            dotted = ".".join(path)
            raise KeyError(f"Missing key: {dotted}")
        current = current[key]
    return current


def extract_float(payload: dict[str, object], path: tuple[str, ...]) -> float:
    value = _extract_value(payload, path)
    if not isinstance(value, (int, float)):
        dotted = ".".join(path)
        raise TypeError(f"Expected number at {dotted}, found {type(value).__name__}")
    return float(value)


def extract_bool(payload: dict[str, object], path: tuple[str, ...]) -> bool:
    value = _extract_value(payload, path)
    if not isinstance(value, bool):
        dotted = ".".join(path)
        raise TypeError(f"Expected bool at {dotted}, found {type(value).__name__}")
    return value


def format_table_line(check: str, actual: str, expected: str, status: str) -> str:
    return f"{check:<52} {actual:<20} {expected:<20} {status}"


def validate_absolute(
    *,
    head: dict[str, object],
    min_speedup: float,
    min_jaccard_ds_r: float,
    min_jaccard_ds_c: float,
    min_jaccard_r_c: float,
    require_ds_r_set_equality: bool,
) -> list[tuple[str, float, float, bool, str]]:
    checks: list[tuple[str, float, float, bool, str]] = []

    r_speedup = extract_float(head, ("summary", "speedup_vs_datasketch", "r_minhash"))
    c_speedup = extract_float(head, ("summary", "speedup_vs_datasketch", "c_minhash"))

    ds_r_j = extract_float(head, ("accuracy", "jaccard", "datasketch_vs_r_minhash"))
    ds_c_j = extract_float(head, ("accuracy", "jaccard", "datasketch_vs_c_minhash"))
    r_c_j = extract_float(head, ("accuracy", "jaccard", "r_minhash_vs_c_minhash"))

    checks.append((
        "R-MinHash speedup vs Datasketch",
        r_speedup,
        min_speedup,
        r_speedup >= min_speedup,
        "ratio",
    ))
    checks.append((
        "C-MinHash speedup vs Datasketch",
        c_speedup,
        min_speedup,
        c_speedup >= min_speedup,
        "ratio",
    ))
    checks.append((
        "Jaccard Datasketch vs R-MinHash",
        ds_r_j,
        min_jaccard_ds_r,
        ds_r_j >= min_jaccard_ds_r,
        "score",
    ))
    checks.append((
        "Jaccard Datasketch vs C-MinHash",
        ds_c_j,
        min_jaccard_ds_c,
        ds_c_j >= min_jaccard_ds_c,
        "score",
    ))
    checks.append((
        "Jaccard R-MinHash vs C-MinHash",
        r_c_j,
        min_jaccard_r_c,
        r_c_j >= min_jaccard_r_c,
        "score",
    ))
    if require_ds_r_set_equality:
        ds_equals_r = extract_bool(
            head,
            ("accuracy", "set_equality", "datasketch_equals_r_minhash"),
        )
        checks.append((
            "Set equality Datasketch vs R-MinHash",
            1.0 if ds_equals_r else 0.0,
            1.0,
            ds_equals_r,
            "bool",
        ))

    return checks


def validate_compare(
    *,
    head: dict[str, object],
    base: dict[str, object],
    max_slowdown_fraction: float,
) -> list[tuple[str, float, float, bool, str]]:
    checks: list[tuple[str, float, float, bool, str]] = []

    head_r = extract_float(head, ("summary", "median_times_seconds", "r_minhash"))
    base_r = extract_float(base, ("summary", "median_times_seconds", "r_minhash"))
    head_c = extract_float(head, ("summary", "median_times_seconds", "c_minhash"))
    base_c = extract_float(base, ("summary", "median_times_seconds", "c_minhash"))

    if base_r <= 0 or base_c <= 0:
        raise ValueError("Base median times must be > 0 for regression checks")

    r_slowdown = (head_r - base_r) / base_r
    c_slowdown = (head_c - base_c) / base_c

    checks.append((
        "R-MinHash slowdown vs base",
        r_slowdown,
        max_slowdown_fraction,
        r_slowdown <= max_slowdown_fraction,
        "fraction",
    ))
    checks.append((
        "C-MinHash slowdown vs base",
        c_slowdown,
        max_slowdown_fraction,
        c_slowdown <= max_slowdown_fraction,
        "fraction",
    ))

    return checks


def print_results(checks: list[tuple[str, float, float, bool, str]]) -> bool:
    print("=" * 120)
    print(format_table_line("Check", "Actual", "Expected", "Status"))
    print("-" * 120)

    all_passed = True
    for check_name, actual, expected, passed, value_type in checks:
        if value_type in {"ratio", "score"}:
            actual_str = f"{actual:.6f}"
            expected_str = f">={expected:.6f}"
        elif value_type == "fraction":
            actual_str = f"{actual * 100:.2f}%"
            expected_str = f"<={expected * 100:.2f}%"
        else:
            actual_str = "true" if actual == 1.0 else "false"
            expected_str = "true"

        status = "PASS" if passed else "FAIL"
        all_passed = all_passed and passed
        print(format_table_line(check_name, actual_str, expected_str, status))

    print("=" * 120)
    if all_passed:
        print("All benchmark checks passed.")
    else:
        print("Benchmark checks failed.")

    return all_passed


def main() -> None:
    args = parse_args()

    if args.mode == "compare" and args.base_json is None:
        raise ValueError("--base-json is required when --mode compare")

    head_payload = load_json(args.head_json)

    checks = validate_absolute(
        head=head_payload,
        min_speedup=args.min_speedup,
        min_jaccard_ds_r=args.min_jaccard_ds_r,
        min_jaccard_ds_c=args.min_jaccard_ds_c,
        min_jaccard_r_c=args.min_jaccard_r_c,
        require_ds_r_set_equality=args.require_ds_r_set_equality,
    )

    if args.mode == "compare":
        base_payload = load_json(args.base_json)
        checks.extend(
            validate_compare(
                head=head_payload,
                base=base_payload,
                max_slowdown_fraction=args.max_slowdown_fraction,
            )
        )

    success = print_results(checks)
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
