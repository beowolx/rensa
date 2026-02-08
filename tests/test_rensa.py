from array import array

import pytest
from rensa import CMinHash, RMinHash, RMinHashLSH


def test_rminhash_creation():
    m = RMinHash(num_perm=16, seed=42)
    assert len(m.digest()) == 16


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

# --------------------- RMinHashLSH Tests ---------------------


def test_rminhashlsh_basics():
    lsh = RMinHashLSH(0.5, 16, 4)
    assert lsh.get_num_perm() == 16
    assert lsh.get_num_bands() == 4


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
