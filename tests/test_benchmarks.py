import argparse
import hashlib
import importlib
import json
import pickle
from pathlib import Path

import pytest


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
