from rensa import RMinHash, RMinHashLSH, OptDensMinHash


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


def test_optdensminhash_similarity():
    m1 = OptDensMinHash(num_perm=8, seed=1)
    m2 = OptDensMinHash(num_perm=8, seed=1)
    m1.update(["apple", "banana"])
    m2.update(["apple", "banana"])
    assert abs(m1.jaccard(m2) - 1.0) < 1e-9


def test_optdensminhash_difference():
    m1 = OptDensMinHash(num_perm=8, seed=1)
    m2 = OptDensMinHash(num_perm=8, seed=1)
    m1.update(["foo"])
    m2.update(["bar"])
    assert m1.jaccard(m2) < 0.5
