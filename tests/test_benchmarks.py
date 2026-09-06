import argparse
import hashlib
import importlib
import json
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("max_rows", [None, 2, 20])
def test_token_cache_reports_actual_rows_on_fresh_and_cached_loads(
    monkeypatch, tmp_path, max_rows
):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    benchmark = importlib.import_module("full_benchmark")
    tokens = [["first"], ["second"]]
    monkeypatch.setattr(benchmark, "load_token_sets_from_hf", lambda *args: tokens)
    spec = benchmark.DATASET_PRESETS["ag_news"]
    prepared = benchmark.load_or_prepare_token_cache(tmp_path, spec, max_rows, 3)

    def unexpected_download(*args):
        pytest.fail("An existing token cache must be reused")

    monkeypatch.setattr(benchmark, "load_token_sets_from_hf", unexpected_download)
    reused = benchmark.load_or_prepare_token_cache(tmp_path, spec, max_rows, 3)
    assert prepared == reused
    assert reused[1] == len(tokens)
    assert reused[2] == hashlib.sha256(reused[0].read_bytes()).hexdigest()


@pytest.mark.parametrize("module_name", ["simple_benchmark", "full_benchmark"])
def test_benchmark_summary_preserves_mismatch_rounding(module_name, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    benchmark = importlib.import_module(module_name)
    engines = benchmark.ENGINE_KEYS
    rensa = "rensa" if module_name == "full_benchmark" else "rensa_r"
    comparisons = {"mismatch_vs_datasketch": (rensa, "fastsketch")}
    if module_name == "full_benchmark":
        comparisons["mismatch_vs_fastsketch"] = (rensa, "datasketch")
    runs = [
        {
            "token_cache_sha256": "same-cache", "thread_env_assertions": {"threads": True},
            "engines": {engine: {"total": 4.0, "sketch": 1.0, "build": 1.0, "query": 2.0,
                                "rows_removed": 3, "rows_remaining": 7,
                                "avg_candidates_per_row": None} for engine in engines},
            "accuracy": {
                "jaccard": {f"datasketch_vs_{rensa}": 0.5, "datasketch_vs_fastsketch": 0.5,
                            f"{rensa}_vs_fastsketch": 0.5},
                **{comparison: {engine: stats for engine in compared}
                   for comparison, compared in comparisons.items()},
            },
        }
        for stats in (
            {"count": 1, "rate": 0.1, "false_positive": 0, "false_negative": 1},
            {"count": 4, "rate": 0.4, "false_positive": 1, "false_negative": 3},
        )
    ]
    summary = benchmark.summarize_runs(runs)
    for comparison, compared in comparisons.items():
        assert summary["accuracy"][comparison] == {
            engine: {"median_count": 2, "median_rate": 0.25,
                     "median_false_positive": 0, "median_false_negative": 2}
            for engine in compared
        }


def test_full_benchmark_rejects_empty_datasets_and_repeated_engines(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    benchmark = importlib.import_module("full_benchmark")
    with pytest.raises(ValueError, match="at least one dataset"):
        benchmark.parse_dataset_keys(" , ")
    with pytest.raises(ValueError, match="exactly these engines"):
        benchmark.run_once(argparse.Namespace(
            token_cache=Path("unused.pkl"), order="datasketch,fastsketch,rensa,rensa",
        ))


@pytest.mark.parametrize("module_name", ["simple_benchmark", "full_benchmark"])
@pytest.mark.parametrize("arguments, message", [
    (["--num-perm", "24"], "power-of-two"),
    (["--num-perm", "8192"], "4096"),
    (["--num-bands", "1"], "at least two bands"),
    (["--seed", "-1"], "unsigned 32-bit"),
    (["--seed", "4294967296"], "unsigned 32-bit"),
])
def test_dataset_benchmarks_reject_unsupported_engine_parameters(
    module_name, arguments, message, monkeypatch
):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    benchmark = importlib.import_module(module_name)
    monkeypatch.setattr(sys, "argv", [module_name, *arguments])
    monkeypatch.setattr(benchmark, "load_or_prepare_token_cache", lambda **kwargs: pytest.fail(
        "Invalid engine parameters must be rejected before loading a dataset"
    ))
    with pytest.raises(ValueError, match=message):
        benchmark.main(benchmark.parse_args())


@pytest.mark.parametrize("module_name, arguments, message", [
    ("accuracy_benchmark", ["--perms", "3"], "power-of-two"),
    ("accuracy_benchmark", ["--perms", "8192"], "4096"),
    ("kernel_benchmark", ["--seed", "-1"], "unsigned 64-bit"),
    ("kernel_benchmark", ["--seed", "18446744073709551616"], "unsigned 64-bit"),
])
def test_synthetic_benchmarks_reject_unsupported_engine_parameters(
    module_name, arguments, message, monkeypatch, capsys
):
    if module_name == "accuracy_benchmark":
        pytest.importorskip("FastSketchLSH")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    benchmark = importlib.import_module(module_name)
    monkeypatch.setattr(sys, "argv", [module_name, "--output-json", "unused.json", *arguments])
    monkeypatch.setattr(
        benchmark, "accuracy" if module_name == "accuracy_benchmark" else "isolated_case",
        lambda *args: pytest.fail("Invalid engine parameters must be rejected before a run"),
    )
    with pytest.raises(SystemExit) as error:
        benchmark.main()
    assert error.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.skipif(
    sys.implementation.name != "cpython", reason="Requires CPython reference counting"
)
def test_kernel_measure_destroys_results_outside_timed_intervals(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    benchmark = importlib.import_module("kernel_benchmark")
    events = []
    ticks = iter(range(8))

    class Result:
        def __init__(self):
            events.append("create")

        def __del__(self):
            events.append("destroy")

    def clock():
        tick = next(ticks)
        events.append("start" if tick % 2 == 0 else "stop")
        return tick * 1_000_000_000

    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: 0)
    monkeypatch.setattr(benchmark.time, "perf_counter_ns", clock)
    result = benchmark.measure(Result, lambda result: [1], 2, 2, 0)

    assert result["iterations"] == [2, 2]
    assert result["seconds"] == [1.0, 1.0]
    assert events == ["create", "destroy"] + ["start", "create", "stop", "destroy"] * 4


@pytest.mark.parametrize("module_name", ["simple_benchmark", "full_benchmark"])
def test_benchmark_records_order_sensitive_duplicate_decisions(
    module_name, monkeypatch, tmp_path, capsys
):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    benchmark = importlib.import_module(module_name)
    flags = {
        engine: [index % 2 == 0, index % 2 != 0]
        for index, engine in enumerate(benchmark.ENGINE_KEYS)
    }
    monkeypatch.setattr(
        benchmark, "run_engine", lambda **kwargs: ({}, flags[kwargs["engine"]])
    )
    token_sets = [["first"], ["second"]]
    if module_name == "simple_benchmark":
        result = benchmark.run_once(
            token_sets, list(benchmark.ENGINE_KEYS), 128, 8, 0.8, 42
        )
    else:
        token_cache = tmp_path / "tokens.pkl"
        token_cache.write_bytes(pickle.dumps(token_sets))
        for key in benchmark.THREAD_ENV_VARS:
            monkeypatch.setenv(key, "1")
        benchmark.run_once(argparse.Namespace(
            token_cache=token_cache,
            order=",".join(benchmark.ENGINE_KEYS),
            run_threads=1,
            expected_token_cache_sha=benchmark.sha256_file(token_cache),
            num_perm=128,
            num_bands=8,
            threshold=0.8,
            seed=42,
        ))
        result = json.loads(capsys.readouterr().out)

    assert result["duplicate_flags_sha256"] == {
        engine: hashlib.sha256(bytes(decisions)).hexdigest()
        for engine, decisions in flags.items()
    }
    assert len(set(result["duplicate_flags_sha256"].values())) == 2


def test_installed_fastsketch_detects_exact_duplicates(monkeypatch):
    pytest.importorskip("FastSketchLSH")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    benchmark = importlib.import_module("full_benchmark")
    metrics, flags = benchmark.run_fastsketch(
        [["alpha", "beta"], ["gamma", "delta"], ["alpha", "beta"]],
        num_perm=128, num_bands=8, seed=42, threads=1,
    )
    assert flags == [True, False, True]
    assert metrics["rows_removed"] == 2
    assert metrics["rows_remaining"] == 1


@pytest.fixture
def paired_benchmark(monkeypatch):
    pytest.importorskip("FastSketchLSH")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    return importlib.import_module("paired_sketch_benchmark")


@pytest.mark.parametrize("mode", [[], ["--prehashed"], ["--prehashed", "--repeat-cardinality", "2"]])
def test_paired_native_cli_uses_frozen_extensions(paired_benchmark, tmp_path, mode):
    module_path = importlib.import_module("rensa.rensa").__file__
    assert module_path is not None
    installed = Path(module_path)
    paths = []
    for label in ("base", "head"):
        path = tmp_path / label / installed.name
        path.parent.mkdir()
        shutil.copy2(installed, path)
        paths.append(path)
    output = tmp_path / "paired.json"
    sizes = [0, 8] if "--prehashed" in mode else [1, 8]
    subprocess.run([
        sys.executable, paired_benchmark.__file__,
        "--base-extension", str(paths[0]), "--head-extension", str(paths[1]),
        "--output-json", str(output), "--rows", "8", "--max-input-tokens", "16",
        "--sizes", *map(str, sizes), "--num-perm", "128",
        "--min-sample-seconds", "0.001", "--warmup-seconds", "0", *mode,
    ], check=True, text=True, capture_output=True, timeout=30)
    report = json.loads(output.read_text())
    assert report["config"]["repetitions"] == 6
    digest = hashlib.sha256(installed.read_bytes()).hexdigest()
    assert {entry["sha256"] for entry in report["extensions"].values()} == {digest}
    assert report["extensions"]["base"]["path"] != report["extensions"]["head"]["path"]
    assert report["environment"]["flags"]["RENSA_PIPELINE_QUEUE_CAP"] == "0"
    assert all(report["environment"]["flags"][name] == "1" for name in paired_benchmark.THREAD_ENV)
    assert len(report["results"]) == 2
    for row, size in zip(report["results"], sizes):
        assert row["rows"] == min(8, 16 // max(size, 1))
        assert len(row["input_sha256"]) == 64
        assert len({tuple(order) for order in row["engine_order"]}) == 6
        for cycle, order in enumerate(row["engine_order"]):
            assert (order.index("base") < order.index("head")) == (cycle % 2 == 0)
        assert row["samples"]["base"]["sha256"] == row["samples"]["head"]["sha256"]
        for sample in row["samples"].values():
            assert len(sample["seconds"]) == len(sample["iterations"]) == 6
            assert all(value > 0 for value in sample["seconds"])
        assert len(row["paired_head_over_base"]) == len(row["paired_fastsketch_over_head"]) == 6
        if "--repeat-cardinality" in mode:
            assert row["distinct_tokens_per_row"] == min(size, 2)


def test_paired_measure_rejects_different_r_signatures(paired_benchmark):
    operations = {
        "base": (lambda: [1], lambda result: result),
        "head": (lambda: [2], lambda result: result),
        "fastsketch": (lambda: [3], lambda result: result),
    }
    with pytest.raises(RuntimeError, match="Base/head signatures differ"):
        paired_benchmark.paired_measure(operations, 1, 0, 0)


@pytest.mark.parametrize("engine", ["base", "head", "fastsketch"])
def test_paired_measure_rejects_changes_between_intervals(paired_benchmark, engine):
    operations = {name: (lambda: [0], lambda result: result)
                  for name in ("base", "head", "fastsketch")}
    # Each invocation of measure sees locally stable output; the paired loop
    # must also check invariance between its warmup and subsequent intervals.
    values = iter([0, 0, 1, 1])
    operations[engine] = (lambda: [next(values)], lambda result: result)
    with pytest.raises(RuntimeError, match=f"{engine} signatures changed"):
        paired_benchmark.paired_measure(operations, 1, 0, 0)


@pytest.mark.parametrize("arguments, message", [
    (["--rows", "0"], "must be positive"),
    (["--sizes", "-1"], "sizes must be nonnegative"),
    (["--num-perm", "3"], "power-of-two"),
    (["--prehashed", "--sizes", "9", "--max-input-tokens", "8"], "cannot exceed"),
    (["--sizes", "0"], "rejects empty rows"),
    (["--repeat-cardinality", "2"], "requires prehashed mode"),
    (["--min-sample-seconds", "nan"], "finite and nonnegative"),
    (["--warmup-seconds", "-1"], "finite and nonnegative"),
    (["--seed", "4294967296"], "unsigned 32-bit"),
])
def test_paired_cli_input_guards(paired_benchmark, monkeypatch, capsys, arguments, message):
    monkeypatch.setattr(sys, "argv", [
        "paired_sketch_benchmark", "--base-extension", "base.so",
        "--head-extension", "head.so", "--output-json", "unused.json", *arguments,
    ])
    with pytest.raises(SystemExit) as error:
        paired_benchmark.main()
    assert error.value.code == 2
    assert message in capsys.readouterr().err
