from rensa import CMinHash, CMinHashDeduplicator, RMinHash, RMinHashDeduplicator


class TestInlineDeduplication:
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
