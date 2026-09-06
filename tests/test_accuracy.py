"""Accuracy against exact sets and signature invariants across kernel tails."""

import math
import statistics

import pytest

from rensa import CMinHash, RMinHash, RMinHashLSH


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


@pytest.mark.parametrize(
    "prehashed,left_size,right_size,overlap",
    [
        (False, 2, 2, 1),
        (True, 2, 2, 1),
        (False, 16, 1024, 16),
        (True, 64, 1024, 64),
        (False, 128, 160, 128),
        (False, 1024, 1280, 1024),
        (False, 2048, 3072, 1536),
    ],
)
def test_rminhash_accuracy_across_cardinalities(
    prehashed, left_size, right_size, overlap
):
    left = list(range(left_size))
    right = left[left_size - overlap:] + list(
        range(left_size, left_size + right_size - overlap)
    )
    rows = [left, right]
    if not prehashed:
        rows = [[f"cardinality-token-{value}" for value in row] for row in rows]
    digest = (
        RMinHash.digest_matrix_from_token_hash_sets
        if prehashed
        else RMinHash.digest_matrix_from_token_sets
    )
    exact = overlap / (left_size + right_size - overlap)
    mse_by_width = {}
    seed_count = 96
    for num_perm in (128, 512):
        errors = []
        for seed in range(seed_count):
            first, second = digest(rows, num_perm=num_perm, seed=seed).to_rows()
            estimate = sum(a == b for a, b in zip(first, second)) / num_perm
            errors.append(estimate - exact)
        # Compare aggregate error with classical independent MinHash variance.
        # The allowance covers finite fixed-seed sampling, not a changed target.
        reference_variance = exact * (1 - exact) / num_perm
        assert abs(statistics.mean(errors)) < 4 * math.sqrt(reference_variance / seed_count)
        mse_by_width[num_perm] = statistics.mean(error**2 for error in errors)
        assert mse_by_width[num_perm] < 1.65 * reference_variance
    assert mse_by_width[512] < 0.6 * mse_by_width[128]


@pytest.mark.parametrize("num_perm", [128, 129])
def test_rminhash_long_sets_preserve_order_duplicates_and_incremental_updates(num_perm):
    tokens = [f"long-token-{index}" for index in range(8193)]
    rows = RMinHash.digest_matrix_from_token_sets(
        [tokens, list(reversed(tokens)), tokens * 2], num_perm=num_perm, seed=42
    ).to_rows()
    assert rows[0] == rows[1] == rows[2]
    for chunk_size in (1, 31, 257, len(tokens)):
        incremental = RMinHash(num_perm=num_perm, seed=42)
        for start in range(0, len(tokens), chunk_size):
            incremental.update(tokens[start:start + chunk_size])
        incremental.update([])
        incremental.update(tokens[::3])
        assert incremental.digest() == rows[0]


def test_rminhash_batch_and_object_queries_agree_across_document_sizes():
    short = [f"short-{index}" for index in range(128)]
    long = [f"long-{index}" for index in range(2048)]
    documents = [
        short,
        short + [f"short-extra-{index}" for index in range(32)],
        long,
        long + [f"long-extra-{index}" for index in range(512)],
        list(reversed(long)),
        ["unrelated"],
    ]
    signatures = RMinHash.from_token_sets(documents, num_perm=128, seed=42)
    matrix = RMinHash.digest_matrix_from_token_sets(documents, num_perm=128, seed=42)
    index = RMinHashLSH(threshold=0.8, num_perm=128, num_bands=8)
    for key, signature in enumerate(signatures):
        index.insert(key, signature)
    expected = [
        any(candidate != key for candidate in index.query(signature))
        for key, signature in enumerate(signatures)
    ]
    batch_index = RMinHashLSH(threshold=0.8, num_perm=128, num_bands=8)
    assert batch_index.query_duplicate_flags_matrix_one_shot(matrix) == expected
    assert expected[2] and expected[4]
    assert not expected[5]
    for left, right in [(signatures[0], signatures[1]), (signatures[2], signatures[3])]:
        assert index.is_similar(left, right) == (left.jaccard(right) >= 0.8)


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
