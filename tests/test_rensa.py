from array import array
import json
import os
import random
import subprocess
import sys

import pytest
from rensa import CMinHash, RMinHash, RMinHashLSH


def test_rminhash_creation():
    m = RMinHash(num_perm=16, seed=42)
    assert len(m.digest()) == 16


def test_minhash_rejects_zero_num_perm():
    with pytest.raises(ValueError, match="num_perm"):
        RMinHash(num_perm=0, seed=42)
    with pytest.raises(ValueError, match="num_perm"):
        CMinHash(num_perm=0, seed=42)


def test_rminhash_update_digest():
    m = RMinHash(num_perm=4, seed=42)
    initial_digest = m.digest()
    assert all(
        x == 4294967295 for x in initial_digest), "Digest should start at u32::MAX"
    m.update(["hello", "world"])
    updated_digest = m.digest()
    assert updated_digest != initial_digest, "Digest should change after update"


def test_rminhash_jaccard():
    m1 = RMinHash(num_perm=8, seed=100)
    m2 = RMinHash(num_perm=8, seed=100)
    m1.update(["apple", "banana", "cherry"])
    m2.update(["apple", "banana", "cherry"])
    sim = m1.jaccard(m2)
    assert abs(sim - 1.0) < 1e-9, "Identical sets => jaccard ~ 1.0"


def test_rminhash_jaccard_different():
    m1 = RMinHash(num_perm=8, seed=999)
    m2 = RMinHash(num_perm=8, seed=999)
    m1.update(["foo"])
    m2.update(["bar"])
    sim = m1.jaccard(m2)
    assert sim < 0.5, "Different sets => jaccard should be relatively low"


def test_rminhash_jaccard_rejects_num_perm_mismatch():
    m1 = RMinHash(num_perm=8, seed=1)
    m2 = RMinHash(num_perm=16, seed=1)
    with pytest.raises(ValueError, match="num_perm mismatch"):
        m1.jaccard(m2)


def test_cminhash_jaccard_rejects_num_perm_mismatch():
    m1 = CMinHash(num_perm=8, seed=1)
    m2 = CMinHash(num_perm=16, seed=1)
    with pytest.raises(ValueError, match="num_perm mismatch"):
        m1.jaccard(m2)


def test_rminhash_serialization_roundtrip():
    m = RMinHash(num_perm=5, seed=2023)
    m.update(["serialize", "this"])
    digest_before = m.digest()

    import pickle
    data = pickle.dumps(m)
    m2 = pickle.loads(data)

    assert m2.digest() == digest_before, "Deserialized RMinHash should have the same digest"


def test_rminhash_update_accepts_iterable_bytes_like_tokens():
    m = RMinHash(num_perm=32, seed=77)
    initial_digest = m.digest()

    m.update([
        b"alpha",
        bytearray(b"beta"),
        memoryview(b"gamma"),
        array("B", [100, 101, 102]),
    ])

    assert m.digest() != initial_digest


def test_cminhash_update_accepts_iterable_bytes_like_tokens():
    m = CMinHash(num_perm=32, seed=77)
    initial_digest = m.digest()

    m.update([
        b"alpha",
        bytearray(b"beta"),
        memoryview(b"gamma"),
        array("B", [100, 101, 102]),
    ])

    assert m.digest() != initial_digest


def test_rminhash_top_level_bytes_and_memoryview_are_single_tokens():
    bytes_direct = RMinHash(num_perm=32, seed=42)
    bytes_list = RMinHash(num_perm=32, seed=42)
    bytes_direct.update(b"abc")
    bytes_list.update([b"abc"])
    assert bytes_direct.digest() == bytes_list.digest()

    mv_direct = RMinHash(num_perm=32, seed=42)
    mv_list = RMinHash(num_perm=32, seed=42)
    mv = memoryview(b"abc")
    mv_direct.update(mv)
    mv_list.update([mv])
    assert mv_direct.digest() == mv_list.digest()


def test_cminhash_top_level_bytes_and_memoryview_are_single_tokens():
    bytes_direct = CMinHash(num_perm=32, seed=42)
    bytes_list = CMinHash(num_perm=32, seed=42)
    bytes_direct.update(b"abc")
    bytes_list.update([b"abc"])
    assert bytes_direct.digest() == bytes_list.digest()

    mv_direct = CMinHash(num_perm=32, seed=42)
    mv_list = CMinHash(num_perm=32, seed=42)
    mv = memoryview(b"abc")
    mv_direct.update(mv)
    mv_list.update([mv])
    assert mv_direct.digest() == mv_list.digest()


def test_rminhash_rejects_non_contiguous_memoryview():
    m = RMinHash(num_perm=32, seed=42)
    non_contiguous = memoryview(bytearray(b"abcd"))[::2]
    with pytest.raises(TypeError, match="C-contiguous and byte-sized"):
        m.update(non_contiguous)


def test_cminhash_rejects_non_contiguous_memoryview():
    m = CMinHash(num_perm=32, seed=42)
    non_contiguous = memoryview(bytearray(b"abcd"))[::2]
    with pytest.raises(TypeError, match="C-contiguous and byte-sized"):
        m.update(non_contiguous)


def test_rminhash_rejects_invalid_iterable_item_type():
    m = RMinHash(num_perm=32, seed=42)
    with pytest.raises(TypeError, match="each item must be"):
        m.update([123])


def test_cminhash_rejects_invalid_iterable_item_type():
    m = CMinHash(num_perm=32, seed=42)
    with pytest.raises(TypeError, match="each item must be"):
        m.update([123])


def test_rminhash_top_level_str_behavior_is_unchanged():
    direct = RMinHash(num_perm=32, seed=11)
    tokenized = RMinHash(num_perm=32, seed=11)
    direct.update("abc")
    tokenized.update(["a", "b", "c"])
    assert direct.digest() == tokenized.digest()


def test_cminhash_top_level_str_behavior_is_unchanged():
    direct = CMinHash(num_perm=32, seed=11)
    tokenized = CMinHash(num_perm=32, seed=11)
    direct.update("abc")
    tokenized.update(["a", "b", "c"])
    assert direct.digest() == tokenized.digest()


def test_rminhash_batch_builders_match_single_document_path():
    token_sets = [
        ["alpha", "beta", "gamma"],
        [b"one", bytearray(b"two"), memoryview(b"three")],
        "abc",
    ]
    scalar_digests = []
    for tokens in token_sets:
        m = RMinHash(num_perm=32, seed=123)
        m.update(tokens)
        scalar_digests.append(m.digest())

    built = RMinHash.from_token_sets(token_sets, num_perm=32, seed=123)
    assert [m.digest() for m in built] == scalar_digests

    digest_rows = RMinHash.digests_from_token_sets(
        token_sets, num_perm=32, seed=123
    )
    assert digest_rows == scalar_digests

    digest_matrix = RMinHash.digest_matrix_from_token_sets(
        token_sets, num_perm=32, seed=123
    )
    assert digest_matrix.len() == len(token_sets)
    assert digest_matrix.get_num_perm() == 32
    assert digest_matrix.to_rows() == scalar_digests


def test_rminhash_expert_batch_apis_match_default_digest_path():
    token_sets = [
        ["alpha", "beta", "gamma"],
        ["delta", "epsilon"],
        [b"one", bytearray(b"two"), memoryview(b"three")],
    ]

    default_matrix = RMinHash.digest_matrix_from_token_sets(
        token_sets, num_perm=64, seed=123
    )
    hashed_sets = RMinHash.hash_token_sets(token_sets)

    prehashed_matrix = RMinHash.digest_matrix_from_token_hash_sets(
        hashed_sets, num_perm=64, seed=123
    )
    byte_only_matrix = RMinHash.digest_matrix_from_token_byte_sets(
        [[b"alpha", b"beta"], [bytearray(b"gamma"), memoryview(b"delta")]],
        num_perm=64,
        seed=123,
    )
    byte_default_matrix = RMinHash.digest_matrix_from_token_sets(
        [[b"alpha", b"beta"], [bytearray(b"gamma"), memoryview(b"delta")]],
        num_perm=64,
        seed=123,
    )

    assert prehashed_matrix.to_rows() == default_matrix.to_rows()
    assert byte_only_matrix.to_rows() == byte_default_matrix.to_rows()


def test_rminhash_flat_token_hash_matrix_matches_default_path():
    token_sets = [
        ["alpha", "beta", "gamma"],
        ["delta", "epsilon"],
        [b"one", bytearray(b"two"), memoryview(b"three")],
    ]
    token_hash_sets = RMinHash.hash_token_sets(token_sets)
    row_offsets = [0]
    flat = []
    for row in token_hash_sets:
        flat.extend(row)
        row_offsets.append(len(flat))

    baseline = RMinHash.digest_matrix_from_token_sets(
        token_sets, num_perm=64, seed=123
    ).to_rows()
    from_flat_lists = RMinHash.digest_matrix_from_flat_token_hashes(
        flat, row_offsets, num_perm=64, seed=123
    ).to_rows()
    from_flat_buffers = RMinHash.digest_matrix_from_flat_token_hashes(
        array("Q", flat), array("I", row_offsets), num_perm=64, seed=123
    ).to_rows()

    assert from_flat_lists == baseline
    assert from_flat_buffers == baseline


def test_rminhash_rho_matrix_apis_are_consistent():
    token_sets = [
        ["alpha", "beta", "gamma"],
        ["delta", "epsilon"],
        [b"one", bytearray(b"two"), memoryview(b"three")],
    ]
    token_hash_sets = RMinHash.hash_token_sets(token_sets)
    row_offsets = [0]
    flat = []
    for row in token_hash_sets:
        flat.extend(row)
        row_offsets.append(len(flat))

    from_tokens = RMinHash.digest_matrix_from_token_sets_rho(
        token_sets, num_perm=64, seed=123, probes=2
    ).to_rows()
    from_hash_sets = RMinHash.digest_matrix_from_token_hash_sets_rho(
        token_hash_sets, num_perm=64, seed=123, probes=2
    ).to_rows()
    from_flat = RMinHash.digest_matrix_from_flat_token_hashes_rho(
        array("Q", flat), array("I", row_offsets), num_perm=64, seed=123, probes=2
    ).to_rows()

    assert from_tokens == from_hash_sets
    assert from_tokens == from_flat


def test_rminhash_rho_is_deterministic():
    token_sets = [["alpha", "beta"], ["gamma", "delta"]]
    first = RMinHash.digest_matrix_from_token_sets_rho(
        token_sets, num_perm=64, seed=123, probes=2
    ).to_rows()
    second = RMinHash.digest_matrix_from_token_sets_rho(
        token_sets, num_perm=64, seed=123, probes=2
    ).to_rows()
    assert first == second


def test_rminhash_rho_source_token_counts_are_reported():
    token_sets = [["alpha"], ["beta", "gamma", "delta"], []]
    matrix = RMinHash.digest_matrix_from_token_sets_rho(
        token_sets, num_perm=64, seed=123, probes=2
    )
    assert matrix.get_rho_source_token_counts() == [1, 3, 0]


def test_rminhash_rho_adaptive_budget_is_deterministic_across_thread_counts():
    code = """
import json
from rensa import RMinHash
token_sets = [
    ["tok" + str(i % 5) for i in range(4 + (doc_idx % 40))]
    for doc_idx in range(96)
]
matrix = RMinHash.digest_matrix_from_token_sets_rho(
    token_sets, num_perm=64, seed=42, probes=4
)
print(json.dumps(matrix.to_rows()))
"""

    def run_with_threads(threads: int) -> list[list[int]]:
        env = os.environ.copy()
        env["RAYON_NUM_THREADS"] = str(threads)
        env["RENSA_RHO_MEDIUM_TOKEN_BUDGET"] = "20"
        env["RENSA_RHO_MEDIUM_TOKEN_THRESHOLD"] = "96"
        proc = subprocess.run(
            [sys.executable, "-c", code],
            text=True,
            capture_output=True,
            env=env,
            check=True,
        )
        return json.loads(proc.stdout)

    assert run_with_threads(1) == run_with_threads(8)


def test_rminhashlsh_sparse_required_band_matches_is_monotonic():
    code = """
import json
from rensa import RMinHash, RMinHashLSH
token_sets = [[f"tok{idx % 97}"] for idx in range(2048)]
matrix = RMinHash.digest_matrix_from_token_sets_rho(
    token_sets, num_perm=128, seed=123, probes=4
)
lsh = RMinHashLSH(threshold=0.95, num_perm=128, num_bands=4)
flags = lsh.query_duplicate_flags_matrix_one_shot(matrix)
print(json.dumps(sum(bool(x) for x in flags)))
"""

    def run_with_required(required_matches: int) -> int:
        env = os.environ.copy()
        env["RENSA_RHO_SPARSE_OCCUPANCY_THRESHOLD"] = "128"
        env["RENSA_RHO_SPARSE_REQUIRED_BAND_MATCHES"] = str(required_matches)
        env["RENSA_RHO_SPARSE_VERIFY_ENABLE"] = "0"
        proc = subprocess.run(
            [sys.executable, "-c", code],
            text=True,
            capture_output=True,
            env=env,
            check=True,
        )
        return int(json.loads(proc.stdout))

    relaxed = run_with_required(1)
    strict = run_with_required(4)
    assert strict <= relaxed


def test_rminhashlsh_sparse_verify_disable_matches_threshold_zero():
    code = """
import json
from rensa import RMinHash, RMinHashLSH
token_sets = [[f"tok{idx % 97}"] for idx in range(2048)]
matrix = RMinHash.digest_matrix_from_token_sets_rho(
    token_sets, num_perm=128, seed=123, probes=4
)
lsh = RMinHashLSH(threshold=0.95, num_perm=128, num_bands=4)
flags = [bool(value) for value in lsh.query_duplicate_flags_matrix_one_shot(matrix)]
print(json.dumps(flags))
"""

    def run_with_verifier(enabled: int, threshold: float) -> list[bool]:
        env = os.environ.copy()
        env["RENSA_RHO_SPARSE_OCCUPANCY_THRESHOLD"] = "128"
        env["RENSA_RHO_SPARSE_REQUIRED_BAND_MATCHES"] = "2"
        env["RENSA_RHO_SPARSE_VERIFY_ENABLE"] = str(enabled)
        env["RENSA_RHO_SPARSE_VERIFY_THRESHOLD"] = str(threshold)
        proc = subprocess.run(
            [sys.executable, "-c", code],
            text=True,
            capture_output=True,
            env=env,
            check=True,
        )
        return json.loads(proc.stdout)

    phase2_flags = run_with_verifier(enabled=0, threshold=0.75)
    verifier_noop_flags = run_with_verifier(enabled=1, threshold=0.0)
    assert phase2_flags == verifier_noop_flags


def test_rminhashlsh_recall_rescue_window_only_relaxes_when_enabled():
    code = """
import json
from rensa import RMinHash, RMinHashLSH
token_sets = [
    [f"tok{(doc_idx + offset) % 509}" for offset in range(48)]
    for doc_idx in range(2048)
]
matrix = RMinHash.digest_matrix_from_token_sets_rho(
    token_sets, num_perm=128, seed=123, probes=4
)
lsh = RMinHashLSH(threshold=0.95, num_perm=128, num_bands=4)
flags = lsh.query_duplicate_flags_matrix_one_shot(matrix)
print(json.dumps(sum(bool(value) for value in flags)))
"""

    def run_with_env(
        enabled: int, min_tokens: int, max_tokens: int, required_matches: int
    ) -> int:
        env = os.environ.copy()
        env["RENSA_RHO_BAND_FOLD"] = "2"
        env["RENSA_RHO_MEDIUM_TOKEN_BUDGET"] = "64"
        env["RENSA_RHO_MEDIUM_TOKEN_THRESHOLD"] = "96"
        env["RENSA_RHO_SPARSE_VERIFY_ENABLE"] = "0"
        env["RENSA_RHO_RECALL_RESCUE_ENABLE"] = str(enabled)
        env["RENSA_RHO_RECALL_RESCUE_MIN_TOKENS"] = str(min_tokens)
        env["RENSA_RHO_RECALL_RESCUE_MAX_TOKENS"] = str(max_tokens)
        env["RENSA_RHO_RECALL_RESCUE_REQUIRED_BAND_MATCHES"] = str(
            required_matches
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            text=True,
            capture_output=True,
            env=env,
            check=True,
        )
        return int(json.loads(proc.stdout))

    disabled = run_with_env(enabled=0, min_tokens=17, max_tokens=96, required_matches=2)
    enabled = run_with_env(enabled=1, min_tokens=17, max_tokens=96, required_matches=2)
    out_of_window = run_with_env(
        enabled=1, min_tokens=97, max_tokens=128, required_matches=2
    )

    assert enabled >= disabled
    assert out_of_window == disabled


def test_cminhash_batch_builders_match_single_document_path():
    token_sets = [
        ["alpha", "beta", "gamma"],
        [b"one", bytearray(b"two"), memoryview(b"three")],
        "abc",
    ]
    scalar_digests = []
    scalar_digests64 = []
    for tokens in token_sets:
        m = CMinHash(num_perm=32, seed=123)
        m.update(tokens)
        scalar_digests.append(m.digest())
        scalar_digests64.append(m.digest_u64())

    built = CMinHash.from_token_sets(token_sets, num_perm=32, seed=123)
    assert [m.digest() for m in built] == scalar_digests
    assert [m.digest_u64() for m in built] == scalar_digests64

    digest_rows = CMinHash.digests_from_token_sets(
        token_sets, num_perm=32, seed=123
    )
    digest_rows64 = CMinHash.digests64_from_token_sets(
        token_sets, num_perm=32, seed=123
    )
    assert digest_rows == scalar_digests
    assert digest_rows64 == scalar_digests64


def test_cminhash_digests64_from_token_hash_sets_matches_default_path():
    token_sets = [
        ["alpha", "beta", "gamma"],
        ["delta", "epsilon"],
        [b"one", bytearray(b"two"), memoryview(b"three")],
    ]
    hashed_sets = RMinHash.hash_token_sets(token_sets)

    from_tokens = CMinHash.digests64_from_token_sets(
        token_sets, num_perm=64, seed=123
    )
    from_hashes = CMinHash.digests64_from_token_hash_sets(
        hashed_sets, num_perm=64, seed=123
    )

    assert from_hashes == from_tokens


def test_rminhash_batch_builders_reject_invalid_nested_tokens():
    with pytest.raises(TypeError, match="each item must be"):
        RMinHash.from_token_sets([["ok"], [123]], num_perm=32, seed=42)


def test_cminhash_batch_builders_reject_invalid_nested_tokens():
    with pytest.raises(TypeError, match="each item must be"):
        CMinHash.from_token_sets([["ok"], [123]], num_perm=32, seed=42)


def test_rminhash_digest_matrix_from_token_hash_sets_rejects_invalid_tokens():
    with pytest.raises(TypeError, match="unsigned 64-bit integer"):
        RMinHash.digest_matrix_from_token_hash_sets(
            [["ok", 123]], num_perm=32, seed=42
        )


@pytest.mark.parametrize(
    "row_offsets",
    [
        [],
        [1, 2],
        [0, 3, 2],
        [0, 1],
    ],
)
def test_rminhash_flat_token_hash_matrix_rejects_invalid_row_offsets(row_offsets):
    with pytest.raises(ValueError, match="row_offsets must start at 0"):
        RMinHash.digest_matrix_from_flat_token_hashes(
            [1, 2], row_offsets, num_perm=32, seed=42
        )


def test_rminhash_flat_token_hash_matrix_rejects_invalid_row_offset_types():
    with pytest.raises(ValueError, match="row_offsets must be an unsigned integer"):
        RMinHash.digest_matrix_from_flat_token_hashes(
            [1, 2], [0.0, 2.0], num_perm=32, seed=42
        )


def test_rminhash_flat_token_hash_matrix_rejects_non_contiguous_row_offset_buffer():
    offsets = memoryview(array("Q", [0, 1, 2, 3]))[::2]
    with pytest.raises(ValueError, match="row_offsets must be an unsigned integer"):
        RMinHash.digest_matrix_from_flat_token_hashes(
            [1, 2], offsets, num_perm=32, seed=42
        )


def test_rminhash_digest_matrix_from_token_byte_sets_rejects_str_tokens():
    with pytest.raises(TypeError, match="bytes"):
        RMinHash.digest_matrix_from_token_byte_sets(
            [["abc"]], num_perm=32, seed=42
        )


def test_rminhash_forced_kernel_modes_match_digest_output():
    token_sets = [["alpha", "beta"], ["gamma", "delta"], ["bytes", "token"]]
    code = f"""
import json
from rensa import RMinHash
token_sets = {token_sets!r}
digests = RMinHash.digests_from_token_sets(token_sets, num_perm=64, seed=42)
print(json.dumps(digests))
"""

    def run_with_kernel(kernel) -> list[list[int]]:
        env = os.environ.copy()
        if kernel is None:
            env.pop("RENSA_FORCE_KERNEL", None)
        else:
            env["RENSA_FORCE_KERNEL"] = kernel
        proc = subprocess.run(
            [sys.executable, "-c", code],
            text=True,
            capture_output=True,
            env=env,
            check=True,
        )
        return json.loads(proc.stdout)

    baseline = run_with_kernel(None)
    scalar = run_with_kernel("scalar")
    assert baseline == scalar

    if sys.platform == "darwin" and "arm64" in os.uname().machine:
        neon = run_with_kernel("neon")
        assert baseline == neon


def test_rminhash_digest_is_deterministic_across_rayon_thread_counts():
    code = """
import json
from rensa import RMinHash
token_sets = [["alpha", "beta", "gamma"]] * 32
digests = RMinHash.digests_from_token_sets(token_sets, num_perm=64, seed=42)
print(json.dumps(digests))
"""

    def run_with_threads(threads: int) -> list[list[int]]:
        env = os.environ.copy()
        env["RAYON_NUM_THREADS"] = str(threads)
        proc = subprocess.run(
            [sys.executable, "-c", code],
            text=True,
            capture_output=True,
            env=env,
            check=True,
        )
        return json.loads(proc.stdout)

    single = run_with_threads(1)
    multi = run_with_threads(8)
    assert single == multi


def test_rminhash_hash_token_sets_randomized_mixed_tokens_match_default_path():
    rng = random.Random(123)
    docs = []
    for _ in range(32):
        doc = []
        for _ in range(rng.randint(1, 12)):
            token_type = rng.randint(0, 2)
            base = f"tok-{rng.randint(0, 9999)}"
            if token_type == 0:
                doc.append(base)
            elif token_type == 1:
                doc.append(base.encode("utf-8"))
            else:
                doc.append(bytearray(base.encode("utf-8")))
        docs.append(doc)

    baseline = RMinHash.digest_matrix_from_token_sets(
        docs, num_perm=64, seed=42
    ).to_rows()
    prehashed = RMinHash.digest_matrix_from_token_hash_sets(
        RMinHash.hash_token_sets(docs), num_perm=64, seed=42
    ).to_rows()
    assert baseline == prehashed


def test_cminhash_digests64_from_token_hash_sets_rejects_invalid_tokens():
    with pytest.raises(TypeError, match="unsigned 64-bit integer"):
        CMinHash.digests64_from_token_hash_sets(
            [["ok", 123]], num_perm=32, seed=42
        )

# --------------------- RMinHashLSH Tests ---------------------


def test_rminhashlsh_basics():
    lsh = RMinHashLSH(0.5, 16, 4)
    assert lsh.get_num_perm() == 16
    assert lsh.get_num_bands() == 4


@pytest.mark.parametrize(
    "threshold,num_perm,num_bands,pattern",
    [
        (-0.1, 16, 4, "threshold"),
        (1.1, 16, 4, "threshold"),
        (0.5, 0, 1, "num_perm"),
        (0.5, 16, 0, "num_bands"),
        (0.5, 16, 17, "num_bands"),
        (0.5, 10, 3, "divisible"),
    ],
)
def test_rminhashlsh_rejects_invalid_parameters(
    threshold, num_perm, num_bands, pattern
):
    with pytest.raises(ValueError, match=pattern):
        RMinHashLSH(threshold, num_perm, num_bands)


def test_rminhashlsh_insert_and_query():
    lsh = RMinHashLSH(threshold=0.5, num_perm=8, num_bands=2)
    m1 = RMinHash(num_perm=8, seed=10)
    m2 = RMinHash(num_perm=8, seed=10)

    m1.update(["lsh", "python"])
    m2.update(["lsh", "python", "extra"])

    lsh.insert(100, m1)
    lsh.insert(200, m2)

    result1 = lsh.query(m1)
    # We'll at least get [100], but depending on collisions we might also get 200
    assert 100 in result1, "LSH should retrieve the same key for an identical query"


def test_rminhashlsh_insert_pairs_matches_single_insert():
    minhashes = []
    for tokens in (["a", "b"], ["a", "b", "c"], ["x", "y"]):
        m = RMinHash(num_perm=16, seed=99)
        m.update(tokens)
        minhashes.append(m)

    lsh_single = RMinHashLSH(threshold=0.7, num_perm=16, num_bands=4)
    for idx, minhash in enumerate(minhashes):
        lsh_single.insert(idx, minhash)

    lsh_pairs = RMinHashLSH(threshold=0.7, num_perm=16, num_bands=4)
    lsh_pairs.insert_pairs([(idx, minhash) for idx, minhash in enumerate(minhashes)])

    for minhash in minhashes:
        assert lsh_pairs.query(minhash) == lsh_single.query(minhash)


def test_rminhashlsh_insert_many_matches_insert_pairs():
    minhashes = []
    for tokens in (["a", "b"], ["a", "b", "c"], ["x", "y"]):
        m = RMinHash(num_perm=16, seed=99)
        m.update(tokens)
        minhashes.append(m)

    lsh_many = RMinHashLSH(threshold=0.7, num_perm=16, num_bands=4)
    lsh_many.insert_many(minhashes, start_key=100)

    lsh_pairs = RMinHashLSH(threshold=0.7, num_perm=16, num_bands=4)
    lsh_pairs.insert_pairs(
        [(100 + idx, minhash) for idx, minhash in enumerate(minhashes)]
    )

    for minhash in minhashes:
        assert lsh_many.query(minhash) == lsh_pairs.query(minhash)


def test_rminhashlsh_query_all_matches_single_query():
    minhashes = []
    lsh = RMinHashLSH(threshold=0.7, num_perm=16, num_bands=4)
    for idx, tokens in enumerate((["a", "b"], ["a", "b", "c"], ["x", "y"])):
        m = RMinHash(num_perm=16, seed=99)
        m.update(tokens)
        minhashes.append(m)
        lsh.insert(idx, m)

    expected = [lsh.query(minhash) for minhash in minhashes]
    assert lsh.query_all(minhashes) == expected


def test_rminhashlsh_query_duplicate_flags_matches_query_all():
    minhashes = []
    lsh = RMinHashLSH(threshold=0.7, num_perm=16, num_bands=4)
    for idx, tokens in enumerate((["a", "b"], ["a", "b", "c"], ["x", "y"])):
        m = RMinHash(num_perm=16, seed=99)
        m.update(tokens)
        minhashes.append(m)
        lsh.insert(idx, m)

    expected = [len(candidates) > 1 for candidates in lsh.query_all(minhashes)]
    assert lsh.query_duplicate_flags(minhashes) == expected


def test_rminhashlsh_matrix_methods_match_object_methods():
    token_sets = [
        ["alpha", "beta"],
        ["alpha", "beta", "gamma"],
        ["x", "y"],
    ]
    matrix = RMinHash.digest_matrix_from_token_sets(
        token_sets, num_perm=16, seed=99
    )
    minhashes = RMinHash.from_token_sets(token_sets, num_perm=16, seed=99)

    lsh_objects = RMinHashLSH(threshold=0.7, num_perm=16, num_bands=4)
    lsh_objects.insert_many(minhashes, start_key=0)

    lsh_matrix = RMinHashLSH(threshold=0.7, num_perm=16, num_bands=4)
    lsh_matrix.insert_matrix(matrix, start_key=0)

    assert lsh_objects.query_duplicate_flags(minhashes) == (
        lsh_matrix.query_duplicate_flags_matrix(matrix)
    )
    for minhash in minhashes:
        assert lsh_objects.query(minhash) == lsh_matrix.query(minhash)


def test_rminhashlsh_insert_matrix_and_query_duplicate_flags_ignores_future_overlap_key():
    num_perm = 64
    seed = 42
    start_key = 10
    overlapping_key = start_key + 2
    shared_tokens = [f"shared_{idx}" for idx in range(32)]

    matrix = RMinHash.digest_matrix_from_token_sets(
        [
            shared_tokens,
            [f"row1_{idx}" for idx in range(32)],
            [f"row2_{idx}" for idx in range(32)],
        ],
        num_perm=num_perm,
        seed=seed,
    )

    existing = RMinHash(num_perm=num_perm, seed=seed)
    existing.update(shared_tokens)

    lsh = RMinHashLSH(threshold=0.8, num_perm=num_perm, num_bands=1)
    lsh.insert(overlapping_key, existing)

    assert lsh.insert_matrix_and_query_duplicate_flags(matrix, start_key=start_key) == [
        True,
        False,
        False,
    ]


def test_rminhashlsh_one_shot_exposes_sparse_verify_stats():
    token_sets = [[f"tok{idx % 97}"] for idx in range(128)]
    matrix = RMinHash.digest_matrix_from_token_sets_rho(
        token_sets, num_perm=128, seed=42, probes=4
    )
    lsh = RMinHashLSH(threshold=0.95, num_perm=128, num_bands=4)
    lsh.query_duplicate_flags_matrix_one_shot(matrix)
    checks = lsh.get_last_one_shot_sparse_verify_checks()
    passes = lsh.get_last_one_shot_sparse_verify_passes()
    assert checks >= 0
    assert passes >= 0
    assert passes <= checks


def test_rminhashlsh_insert_pairs_rejects_malformed_entries():
    lsh = RMinHashLSH(threshold=0.7, num_perm=16, num_bands=4)
    minhash = RMinHash(num_perm=16, seed=99)
    minhash.update(["a", "b"])

    with pytest.raises(TypeError):
        lsh.insert_pairs([("bad-key", minhash)])


def test_rminhashlsh_query_all_rejects_num_perm_mismatch():
    lsh = RMinHashLSH(threshold=0.7, num_perm=16, num_bands=4)
    m16 = RMinHash(num_perm=16, seed=99)
    m16.update(["a", "b"])
    m8 = RMinHash(num_perm=8, seed=99)
    m8.update(["a", "b"])
    lsh.insert(1, m16)

    with pytest.raises(ValueError, match="MinHash has 8 permutations"):
        lsh.query_all([m16, m8])


def test_rminhashlsh_query_duplicate_flags_rejects_num_perm_mismatch():
    lsh = RMinHashLSH(threshold=0.7, num_perm=16, num_bands=4)
    m16 = RMinHash(num_perm=16, seed=99)
    m16.update(["a", "b"])
    m8 = RMinHash(num_perm=8, seed=99)
    m8.update(["a", "b"])
    lsh.insert(1, m16)

    with pytest.raises(ValueError, match="MinHash has 8 permutations"):
        lsh.query_duplicate_flags([m16, m8])


def test_rminhashlsh_rejects_signature_mismatch_on_insert_and_query():
    lsh = RMinHashLSH(threshold=0.5, num_perm=8, num_bands=2)
    good = RMinHash(num_perm=8, seed=10)
    bad = RMinHash(num_perm=16, seed=10)
    good.update(["lsh", "python"])
    bad.update(["lsh", "python"])

    lsh.insert(1, good)

    with pytest.raises(ValueError, match="MinHash has 16 permutations"):
        lsh.insert(2, bad)

    with pytest.raises(ValueError, match="MinHash has 16 permutations"):
        lsh.query(bad)


def test_rminhashlsh_is_similar():
    lsh = RMinHashLSH(threshold=0.8, num_perm=16, num_bands=4)
    m1 = RMinHash(num_perm=16, seed=555)
    m2 = RMinHash(num_perm=16, seed=555)

    m1.update(["similar", "hash", "test"])
    m2.update(["similar", "hash", "test", "almost"])

    assert lsh.is_similar(m1, m1), "Any object should be similar to itself"
    # This might or might not be above 0.8. We can verify by comparing directly:
    jaccard_value = m1.jaccard(m2)
    assert lsh.is_similar(m1, m2) == (
        jaccard_value >= 0.8), "Check LSH threshold vs real jaccard"
