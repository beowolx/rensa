import struct

import pytest

from rensa import RMinHash, RMinHashLSH


@pytest.mark.parametrize(
    "method", ["insert_many", "insert_matrix", "insert_matrix_and_query_duplicate_flags"]
)
def test_sequential_keys_reject_overflow_without_replacing_zero(method):
    maximum_key = (1 << (8 * struct.calcsize("P"))) - 1
    minhash = RMinHash(num_perm=16, seed=42)
    minhash.update(["alpha"])
    lsh = RMinHashLSH(threshold=0.8, num_perm=16, num_bands=4)
    existing = RMinHash(num_perm=16, seed=42)
    existing.update(["unrelated"])
    lsh.insert(0, existing)

    def rows(count):
        if method == "insert_many":
            return iter([minhash] * count)
        return RMinHash.digest_matrix_from_token_sets(
            [["alpha"]] * count, num_perm=16, seed=42
        )

    insert = getattr(lsh, method)
    with pytest.raises(ValueError, match="maximum key"):
        insert(rows(2), start_key=maximum_key)
    assert lsh.query(existing) == [0]
    if method != "insert_many":
        assert lsh.query(minhash) == []
    insert(rows(1), start_key=maximum_key)
    assert lsh.query(minhash) == [maximum_key]
    assert lsh.query(existing) == [0]


def test_duplicate_flags_count_distinct_keys_across_bands():
    minhash = RMinHash(num_perm=16, seed=42)
    minhash.update(["alpha"])
    matrix = RMinHash.digest_matrix_from_token_sets(
        [["alpha"]], num_perm=16, seed=42
    )
    lsh = RMinHashLSH(threshold=0.8, num_perm=16, num_bands=4)
    for keys, expected in [((), False), ((7,), False), ((7, 9), True)]:
        for key in keys:
            lsh.insert(key, minhash)
        assert lsh.query_duplicate_flags([minhash]) == [expected]
        assert lsh.query_duplicate_flags_matrix(matrix) == [expected]
        assert (len(lsh.query(minhash)) > 1) is expected
