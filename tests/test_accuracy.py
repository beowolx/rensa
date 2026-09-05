"""Accuracy against exact sets and signature invariants across kernel tails."""

import math
import statistics

import pytest

from rensa import CMinHash, RMinHash


@pytest.mark.parametrize("minhash_type", [RMinHash, CMinHash])
@pytest.mark.parametrize("num_perm", [1, 7, 17, 63, 129])
def test_signature_is_invariant_to_order_duplicates_and_update_boundaries(
    minhash_type, num_perm
):
    tokens = [f"token-{index}" for index in range(97)]
    tokens += ["", "café", "飛行機", "a" * 129]
    expected = minhash_type(num_perm=num_perm, seed=42)
    expected.update(tokens)

    reordered = minhash_type(num_perm=num_perm, seed=42)
    reordered.update(list(reversed(tokens)) * 3)
    incremental = minhash_type(num_perm=num_perm, seed=42)
    # Cross both the 32-token batching boundary and SIMD remainder lanes.
    for start, end in [(0, 1), (1, 32), (32, 33), (33, 65), (65, len(tokens))]:
        incremental.update(tokens[start:end])
    incremental.update([])
    incremental.update(tokens[:7])

    digest = "digest_u64" if minhash_type is CMinHash else "digest"
    assert getattr(reordered, digest)() == getattr(expected, digest)()
    assert getattr(incremental, digest)() == getattr(expected, digest)()
    assert expected.jaccard(reordered) == 1.0


@pytest.mark.parametrize("minhash_type", [RMinHash, CMinHash])
@pytest.mark.parametrize("overlap", [0, 32, 64, 85, 96])
def test_mean_jaccard_tracks_exact_set_similarity(minhash_type, overlap):
    left_tokens = [f"token-{index}" for index in range(96)]
    right_tokens = left_tokens[:overlap] + [
        f"other-{index}" for index in range(96 - overlap)
    ]
    left_set, right_set = set(left_tokens), set(right_tokens)
    exact = len(left_set & right_set) / len(left_set | right_set)
    estimates = []
    # Fixed independent seeds make this a reproducible aggregate accuracy check,
    # not a requirement that a single probabilistic sketch equal the ground truth.
    for seed in range(32):
        left = minhash_type(num_perm=256, seed=seed)
        right = minhash_type(num_perm=256, seed=seed)
        left.update(left_tokens)
        right.update(right_tokens)
        estimates.append(left.jaccard(right))
    assert abs(sum(estimates) / len(estimates) - exact) < 0.06


@pytest.mark.parametrize("prehashed", [False, True])
def test_cminhash_jaccard_error_reduces_with_more_permutations(prehashed):
    # Two-element sets expose correlated permutation lanes that a mean-only
    # check on larger sets misses. Sequential hashes also exercise structured
    # input without relying on the string hash to supply the initial shuffle.
    rows = [[0, 1], [1, 2]] if prehashed else [["left", "shared"], ["shared", "right"]]
    digest = (
        CMinHash.digests64_from_token_hash_sets
        if prehashed
        else CMinHash.digests64_from_token_sets
    )
    exact = 1 / 3
    errors_by_width = {}
    for num_perm in (128, 512):
        errors = []
        for seed in range(96):
            left, right = digest(rows, num_perm=num_perm, seed=seed)
            estimate = sum(a == b for a, b in zip(left, right)) / num_perm
            errors.append(estimate - exact)
        assert abs(statistics.mean(errors)) < 0.02
        errors_by_width[num_perm] = math.sqrt(statistics.mean(error**2 for error in errors))
    assert errors_by_width[128] < 0.07
    assert errors_by_width[512] < 0.04
    assert errors_by_width[512] < 0.75 * errors_by_width[128]


@pytest.mark.parametrize("minhash_type", [RMinHash, CMinHash])
def test_empty_signature_similarity(minhash_type):
    empty = minhash_type(num_perm=129, seed=42)
    other_empty = minhash_type(num_perm=129, seed=42)
    populated = minhash_type(num_perm=129, seed=42)
    populated.update(["nonempty"])
    assert empty.jaccard(other_empty) == 1.0
    assert empty.jaccard(populated) == 0.0


def test_rho_empty_identical_and_disjoint_sets():
    # Short documents use every token, avoiding the intentional position-based
    # sampling of longer documents when testing exact endpoint behavior.
    left = [f"left-{index}" for index in range(24)]
    right = [f"right-{index}" for index in range(24)]
    rows = RMinHash.digest_matrix_from_token_sets_rho(
        [[], [], left, left, right], num_perm=129, seed=42
    ).to_rows()
    assert rows[0] == rows[1] == [2**32 - 1] * 129
    assert rows[2] == rows[3]
    assert any(value != 2**32 - 1 for value in rows[2])
    # Matching empty buckets are not evidence that the source sets overlap.
    assert all(a != b or a == 2**32 - 1 for a, b in zip(rows[2], rows[4]))


@pytest.mark.parametrize("queue_capacity", [0, 2])
@pytest.mark.parametrize("streaming", [False, True])
def test_batch_chunk_reuse_preserves_each_signature(monkeypatch, queue_capacity, streaming):
    monkeypatch.setenv("RENSA_DOC_CHUNK_SIZE", "256")
    monkeypatch.setenv("RENSA_PIPELINE_QUEUE_CAP", str(queue_capacity))
    documents = [[f"row-{row}", f"group-{row % 7}"] for row in range(600)]
    expected = []
    for tokens in documents:
        sketch = RMinHash(num_perm=17, seed=42)
        sketch.update(tokens)
        expected.append(sketch.digest())
    source = (tokens for tokens in documents) if streaming else documents
    assert RMinHash.digest_matrix_from_token_sets(source, 17, 42).to_rows() == expected
