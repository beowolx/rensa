import pytest

from rensa import CMinHash, CMinHashDeduplicator, RMinHash, RMinHashDeduplicator


class TestInlineDeduplication:
    def test_deduplicator_rejects_invalid_threshold(self):
        with pytest.raises(ValueError, match="threshold"):
            RMinHashDeduplicator(threshold=1.1, num_perm=128, use_lsh=False)

        with pytest.raises(ValueError, match="threshold"):
            CMinHashDeduplicator(threshold=-0.1)

    def test_rminhash_deduplicator_basic(self):
        """
        GIVEN: A deduplicator with threshold 0.7
        WHEN: Adding documents with varying similarity
        THEN: Duplicates are detected and unique documents are stored
        """
        dedup = RMinHashDeduplicator(threshold=0.7, num_perm=128, use_lsh=False)

        # GIVEN: First document MinHash
        m1 = RMinHash(num_perm=128, seed=42)
        m1.update(["hello", "world", "this", "is", "a", "test"])

        # WHEN: Adding first document
        # THEN: Should succeed
        assert dedup.add("doc1", m1) is True
        assert dedup.len() == 1

        # WHEN: Adding same document again
        # THEN: Should fail
        assert dedup.add("doc1", m1) is False
        assert dedup.len() == 1

        # GIVEN: Similar document
        m2 = RMinHash(num_perm=128, seed=42)
        m2.update(
            ["hello", "world", "this", "is", "a", "test", "slightly", "different"]
        )

        # WHEN: Checking if it's detected as duplicate
        # THEN: Should be marked as duplicate
        is_dup = dedup.is_duplicate("doc2", m2)
        assert is_dup is True

        # GIVEN: Very different document
        m3 = RMinHash(num_perm=128, seed=42)
        m3.update(["completely", "different", "content", "here"])

        # WHEN: Adding different document
        # THEN: Should not be duplicate and should be added
        assert dedup.add("doc3", m3) is True
        assert dedup.len() == 2

    def test_rminhash_deduplicator_with_lsh(self):
        """
        GIVEN: A deduplicator with LSH enabled and threshold 0.8
        WHEN: Adding multiple documents with varying similarity
        THEN: Only unique documents are stored based on similarity threshold
        """
        dedup = RMinHashDeduplicator(
            threshold=0.8, num_perm=128, use_lsh=True, num_bands=16
        )

        # GIVEN: Multiple documents with varying similarity
        docs = [
            ("doc1", ["hello", "world", "test"]),
            ("doc2", ["hello", "world", "test", "again"]),  # Similar to doc1
            ("doc3", ["completely", "different", "content"]),
            ("doc4", ["hello", "world", "test"]),  # Duplicate of doc1
        ]

        # WHEN: Adding all documents
        added_count = 0
        for key, words in docs:
            m = RMinHash(num_perm=128, seed=42)
            m.update(words)
            if dedup.add(key, m):
                added_count += 1

        # THEN: Should have added 2 or 3 documents (depending on similarity threshold)
        assert added_count >= 2
        assert added_count <= 3

    def test_cminhash_deduplicator(self):
        """
        GIVEN: A CMinHash deduplicator with threshold 0.8
        WHEN: Adding identical documents
        THEN: Duplicates are correctly identified and can be retrieved
        """
        dedup = CMinHashDeduplicator(threshold=0.8)

        # GIVEN: First document MinHash
        c1 = CMinHash(num_perm=128, seed=42)
        c1.update(["hello", "world", "this", "is", "a", "test"])

        # WHEN: Adding first document
        # THEN: Should succeed
        assert dedup.add("doc1", c1) is True

        # GIVEN: Identical document
        c2 = CMinHash(num_perm=128, seed=42)
        c2.update(["hello", "world", "this", "is", "a", "test"])

        # WHEN: Checking for duplicates
        # THEN: Should be detected as duplicate
        assert dedup.is_duplicate("doc2", c2) is True

        # WHEN: Getting duplicates
        # THEN: Should return the original document
        duplicates = dedup.get_duplicates(c2)
        assert "doc1" in duplicates

    def test_deduplicator_operations(self):
        """
        GIVEN: A deduplicator with unique documents
        WHEN: Performing various operations (add, remove, clear)
        THEN: Operations work correctly and maintain proper state
        """
        dedup = RMinHashDeduplicator(threshold=0.8, num_perm=64, use_lsh=False)

        # GIVEN: Five unique documents
        for i in range(5):
            m = RMinHash(num_perm=64, seed=42)
            # Each document has unique words based on its index
            m.update([f"doc{i}_word{j}" for j in range(10)])
            dedup.add(f"doc{i}", m)

        # THEN: All documents should be added
        assert dedup.len() == 5

        # WHEN: Removing a document
        # THEN: Should succeed and update count
        assert dedup.remove("doc2") is True
        assert dedup.len() == 4

        # WHEN: Removing same document again
        # THEN: Should fail
        assert dedup.remove("doc2") is False  # Already removed

        # WHEN: Clearing all documents
        # THEN: Should be empty
        dedup.clear()
        assert dedup.len() == 0

    def test_continuous_stream_simulation(self):
        """
        GIVEN: A stream of documents with some duplicates and similar content
        WHEN: Processing documents through deduplicator
        THEN: Duplicates are filtered out and unique documents are preserved
        """
        dedup = RMinHashDeduplicator(
            threshold=0.7, num_perm=128, use_lsh=True, num_bands=16
        )

        # GIVEN: Document stream with duplicates and similar content
        stream = [
            "The quick brown fox jumps over the lazy dog",
            "The quick brown fox jumps over the lazy cat",  # Similar
            "Lorem ipsum dolor sit amet consectetur",
            "The quick brown fox jumps over the lazy dog",  # Exact duplicate
            "Lorem ipsum dolor sit amet consectetur adipiscing",  # Similar
            "Completely different content here",
        ]

        # WHEN: Processing the document stream
        unique_docs = []
        duplicate_count = 0

        for idx, text in enumerate(stream):
            m = RMinHash(num_perm=128, seed=42)
            m.update(text.split())

            if dedup.add(f"doc{idx}", m):
                unique_docs.append(text)
            else:
                duplicate_count += 1

        # THEN: Should filter out duplicates
        assert duplicate_count >= 1  # At least the exact duplicate
        assert len(unique_docs) < len(stream)

    def test_rminhash_deduplicator_rejects_num_perm_mismatch(self):
        dedup = RMinHashDeduplicator(threshold=0.7, num_perm=128, use_lsh=False)
        m128 = RMinHash(num_perm=128, seed=42)
        m128.update(["alpha", "beta"])
        assert dedup.add("doc1", m128) is True

        m64 = RMinHash(num_perm=64, seed=42)
        m64.update(["alpha", "beta"])

        with pytest.raises(ValueError, match="num_perm mismatch"):
            dedup.is_duplicate("doc2", m64)

        with pytest.raises(ValueError, match="num_perm mismatch"):
            dedup.get_duplicates(m64)

        with pytest.raises(ValueError, match="num_perm mismatch"):
            dedup.add("doc2", m64)

    def test_cminhash_deduplicator_rejects_num_perm_mismatch(self):
        dedup = CMinHashDeduplicator(threshold=0.8)
        c128 = CMinHash(num_perm=128, seed=42)
        c128.update(["alpha", "beta"])
        assert dedup.add("doc1", c128) is True

        c64 = CMinHash(num_perm=64, seed=42)
        c64.update(["alpha", "beta"])

        with pytest.raises(ValueError, match="num_perm mismatch"):
            dedup.is_duplicate("doc2", c64)

        with pytest.raises(ValueError, match="num_perm mismatch"):
            dedup.get_duplicates(c64)

        with pytest.raises(ValueError, match="num_perm mismatch"):
            dedup.add("doc2", c64)

    def test_rminhash_batch_methods_match_scalar_methods(self):
        entries = []
        for idx, words in enumerate(
            (["alpha", "beta"], ["alpha", "beta", "gamma"], ["x", "y"])
        ):
            minhash = RMinHash(num_perm=64, seed=42)
            minhash.update(words)
            entries.append((f"doc{idx}", minhash))

        scalar = RMinHashDeduplicator(threshold=0.8, num_perm=64, use_lsh=False)
        scalar_add = [scalar.add(key, minhash) for key, minhash in entries]
        scalar_dup = [scalar.is_duplicate(key, minhash) for key, minhash in entries]
        scalar_sets = [scalar.get_duplicates(minhash) for _, minhash in entries]

        batch = RMinHashDeduplicator(threshold=0.8, num_perm=64, use_lsh=False)
        batch_add = batch.add_pairs(entries)
        batch_dup = batch.is_duplicate_pairs(entries)
        batch_sets = batch.get_duplicate_sets([minhash for _, minhash in entries])

        assert batch_add == scalar_add
        assert batch_dup == scalar_dup
        assert batch_sets == scalar_sets

    def test_cminhash_batch_methods_match_scalar_methods(self):
        entries = []
        for idx, words in enumerate(
            (["alpha", "beta"], ["alpha", "beta", "gamma"], ["x", "y"])
        ):
            minhash = CMinHash(num_perm=64, seed=42)
            minhash.update(words)
            entries.append((f"doc{idx}", minhash))

        scalar = CMinHashDeduplicator(threshold=0.8)
        scalar_add = [scalar.add(key, minhash) for key, minhash in entries]
        scalar_dup = [scalar.is_duplicate(key, minhash) for key, minhash in entries]
        scalar_sets = [scalar.get_duplicates(minhash) for _, minhash in entries]

        batch = CMinHashDeduplicator(threshold=0.8)
        batch_add = batch.add_pairs(entries)
        batch_dup = batch.is_duplicate_pairs(entries)
        batch_sets = batch.get_duplicate_sets([minhash for _, minhash in entries])

        assert batch_add == scalar_add
        assert batch_dup == scalar_dup
        assert batch_sets == scalar_sets

    def test_rminhash_batch_methods_reject_malformed_entries(self):
        dedup = RMinHashDeduplicator(threshold=0.8, num_perm=64, use_lsh=False)
        minhash = RMinHash(num_perm=64, seed=42)
        minhash.update(["alpha", "beta"])

        with pytest.raises(TypeError):
            dedup.add_pairs([(1, minhash)])

        with pytest.raises(TypeError):
            dedup.is_duplicate_pairs([(1, minhash)])

    def test_rminhash_batch_methods_reject_num_perm_mismatch(self):
        dedup = RMinHashDeduplicator(threshold=0.8, num_perm=64, use_lsh=False)
        good = RMinHash(num_perm=64, seed=42)
        good.update(["alpha", "beta"])
        bad = RMinHash(num_perm=32, seed=42)
        bad.update(["alpha", "beta"])

        assert dedup.add_pairs([("doc-good", good)]) == [True]

        with pytest.raises(ValueError, match="num_perm mismatch"):
            dedup.add_pairs([("doc-bad", bad)])

        with pytest.raises(ValueError, match="num_perm mismatch"):
            dedup.is_duplicate_pairs([("doc-bad", bad)])

        with pytest.raises(ValueError, match="num_perm mismatch"):
            dedup.get_duplicate_sets([bad])

    def test_rminhash_token_set_batch_entries_match_object_batch(self):
        token_entries = [
            ("doc0", ["alpha", "beta"]),
            ("doc1", ["alpha", "beta", "gamma"]),
            ("doc2", ["x", "y"]),
        ]
        minhash_entries = []
        for key, tokens in token_entries:
            minhash = RMinHash(num_perm=64, seed=42)
            minhash.update(tokens)
            minhash_entries.append((key, minhash))

        object_dedup = RMinHashDeduplicator(
            threshold=0.8, num_perm=64, use_lsh=False
        )
        object_add = object_dedup.add_pairs(minhash_entries)
        object_dup = object_dedup.is_duplicate_pairs(minhash_entries)

        token_dedup = RMinHashDeduplicator(
            threshold=0.8, num_perm=64, use_lsh=False
        )
        token_add = token_dedup.add_pairs(token_entries)
        token_dup = token_dedup.is_duplicate_pairs(token_entries)

        assert token_add == object_add
        assert token_dup == object_dup

    def test_rminhash_token_set_batch_entries_reject_malformed_tokens(self):
        dedup = RMinHashDeduplicator(threshold=0.8, num_perm=64, use_lsh=False)

        with pytest.raises(TypeError):
            dedup.add_pairs([("doc0", [123])])

        with pytest.raises(TypeError):
            dedup.is_duplicate_pairs([("doc0", [123])])

    def test_cminhash_batch_methods_reject_malformed_entries(self):
        dedup = CMinHashDeduplicator(threshold=0.8)
        minhash = CMinHash(num_perm=64, seed=42)
        minhash.update(["alpha", "beta"])

        with pytest.raises(TypeError):
            dedup.add_pairs([(1, minhash)])

        with pytest.raises(TypeError):
            dedup.is_duplicate_pairs([(1, minhash)])

    def test_cminhash_batch_methods_reject_num_perm_mismatch(self):
        dedup = CMinHashDeduplicator(threshold=0.8)
        good = CMinHash(num_perm=64, seed=42)
        good.update(["alpha", "beta"])
        bad = CMinHash(num_perm=32, seed=42)
        bad.update(["alpha", "beta"])

        assert dedup.add_pairs([("doc-good", good)]) == [True]

        with pytest.raises(ValueError, match="num_perm mismatch"):
            dedup.add_pairs([("doc-bad", bad)])

        with pytest.raises(ValueError, match="num_perm mismatch"):
            dedup.is_duplicate_pairs([("doc-bad", bad)])

        with pytest.raises(ValueError, match="num_perm mismatch"):
            dedup.get_duplicate_sets([bad])

    def test_cminhash_token_set_batch_entries_match_object_batch(self):
        token_entries = [
            ("doc0", ["alpha", "beta"]),
            ("doc1", ["alpha", "beta", "gamma"]),
            ("doc2", ["x", "y"]),
        ]
        minhash_entries = []
        for key, tokens in token_entries:
            minhash = CMinHash(num_perm=64, seed=42)
            minhash.update(tokens)
            minhash_entries.append((key, minhash))

        object_dedup = CMinHashDeduplicator(threshold=0.8)
        object_add = object_dedup.add_pairs(minhash_entries)
        object_dup = object_dedup.is_duplicate_pairs(minhash_entries)

        token_dedup = CMinHashDeduplicator(threshold=0.8, num_perm=64, seed=42)
        token_add = token_dedup.add_pairs(token_entries)
        token_dup = token_dedup.is_duplicate_pairs(token_entries)

        assert token_add == object_add
        assert token_dup == object_dup

    def test_cminhash_token_set_batch_entries_reject_malformed_tokens(self):
        dedup = CMinHashDeduplicator(threshold=0.8, num_perm=64, seed=42)

        with pytest.raises(TypeError):
            dedup.add_pairs([("doc0", [123])])

        with pytest.raises(TypeError):
            dedup.is_duplicate_pairs([("doc0", [123])])

    def test_cminhash_token_set_batch_entries_require_num_perm(self):
        dedup = CMinHashDeduplicator(threshold=0.8)

        with pytest.raises(ValueError, match="num_perm is not configured"):
            dedup.add_pairs([("doc0", ["alpha", "beta"])])

        with pytest.raises(ValueError, match="num_perm is not configured"):
            dedup.is_duplicate_pairs([("doc0", ["alpha", "beta"])])
