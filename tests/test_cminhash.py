import pickle
import statistics
from rensa import CMinHash, RMinHash

# --- Basic Functionality Tests ---


def test_cminhash_creation():
    m = CMinHash(num_perm=16, seed=42)
    digest = m.digest()
    # Check that we have the correct number of hashes and all are initialized to u32::MAX (4294967295)
    assert len(digest) == 16
    assert all(
        x == 4294967295 for x in digest), "Initial digest should be u32::MAX"


def test_cminhash_update_digest():
    m = CMinHash(num_perm=4, seed=42)
    initial_digest = m.digest()
    m.update(["hello", "world"])
    updated_digest = m.digest()
    # Digest should change after an update.
    assert updated_digest != initial_digest, "Digest should change after update"


def test_cminhash_jaccard_identical():
    m1 = CMinHash(num_perm=8, seed=100)
    m2 = CMinHash(num_perm=8, seed=100)
    items = ["apple", "banana", "cherry"]
    m1.update(items)
    m2.update(items)
    sim = m1.jaccard(m2)
    # Identical updates should yield jaccard 1.0 (within numerical precision)
    assert abs(sim - 1.0) < 1e-9, "Identical sets should yield jaccard 1"


def test_cminhash_jaccard_different():
    m1 = CMinHash(num_perm=8, seed=999)
    m2 = CMinHash(num_perm=8, seed=999)
    m1.update(["foo"])
    m2.update(["bar"])
    sim = m1.jaccard(m2)
    # For disjoint sets the estimated jaccard should be very low.
    assert sim < 0.2, f"Jaccard similarity for disjoint sets should be low, got {sim}"


def test_cminhash_serialization_roundtrip():
    m = CMinHash(num_perm=5, seed=2023)
    m.update(["serialize", "this"])
    digest_before = m.digest()
    data = pickle.dumps(m)
    m2 = pickle.loads(data)
    assert m2.digest() == digest_before, "Deserialized CMinHash should have the same digest"


def test_cminhash_update_empty():
    m = CMinHash(num_perm=8, seed=123)
    digest_before = m.digest()
    m.update([])
    digest_after = m.digest()
    assert digest_before == digest_after, "Updating with empty list should not change digest"


def test_cminhash_multiple_updates():
    m = CMinHash(num_perm=8, seed=456)
    m.update(["item1", "item2"])
    digest1 = m.digest()
    m.update(["item3"])
    digest2 = m.digest()
    # Subsequent updates should only decrease (or leave unchanged) the stored minhash values.
    assert all(d2 <= d1 for d1, d2 in zip(digest1, digest2)
               ), "Digest should not increase on subsequent updates"

# --- Simulation-Based Tests for Algorithm Correctness ---


def test_cminhash_unbiasedness():
    # For a pair of sets with known true Jaccard similarity, the C-MinHash estimator should be unbiased.
    # Here, use set1 = {"a", "b", "c", "d"} and set2 = {"a", "b"} giving true jaccard = 2/4 = 0.5.
    true_jaccard = 0.5
    estimates = []
    num_trials = 500
    num_perm = 128
    for seed in range(num_trials):
        m1 = CMinHash(num_perm=num_perm, seed=seed)
        m2 = CMinHash(num_perm=num_perm, seed=seed)
        m1.update(["a", "b", "c", "d"])
        m2.update(["a", "b"])
        estimates.append(m1.jaccard(m2))
    avg_est = statistics.mean(estimates)
    # With 500 trials, the average estimate should be near 0.5 within a small tolerance.
    assert abs(
        avg_est - true_jaccard) < 0.05, f"Estimator mean {avg_est} differs from true {true_jaccard}"


def test_cminhash_variance_reduction():
    # According to the paper, C-MinHash should have strictly lower variance than classical MinHash.
    # Compare estimates on two sets: set1 = {"a", "b", "c", "d", "e"} and set2 = {"a", "b", "c"}.
    # True jaccard = 3 / 5 = 0.6.
    true_jaccard = 0.6
    num_trials = 500
    num_perm = 64
    estimates_c = []
    estimates_r = []
    for seed in range(num_trials):
        cm1 = CMinHash(num_perm=num_perm, seed=seed)
        cm2 = CMinHash(num_perm=num_perm, seed=seed)
        rm1 = RMinHash(num_perm=num_perm, seed=seed)
        rm2 = RMinHash(num_perm=num_perm, seed=seed)
        set1 = ["a", "b", "c", "d", "e"]
        set2 = ["a", "b", "c"]
        cm1.update(set1)
        cm2.update(set2)
        rm1.update(set1)
        rm2.update(set2)
        estimates_c.append(cm1.jaccard(cm2))
        estimates_r.append(rm1.jaccard(rm2))
    var_c = statistics.pvariance(estimates_c)
    var_r = statistics.pvariance(estimates_r)
    # C-MinHash variance should be lower than that of RMinHash.
    assert var_c < var_r, f"CMinHash variance {var_c} is not lower than RMinHash variance {var_r}"
