import time
import random
import string
from tqdm import tqdm
from datasketch import MinHash, MinHashLSH
from rensa import CMinHash, CMinHashLSH


def simple_dataset():
    return [
        "The quick brown fox jumps over the lazy dog",
        "The quick brown fox jumps over the lazy dog",  # Exact duplicate
        "The fast brown fox leaps over the sleepy dog",  # Near duplicate
        "The lazy dog is jumped over by the quick brown fox",  # Reordered
        "A completely different sentence about cats",  # Unique
        "Another unique sentence about the weather"  # Unique
    ]

def simple_deduplicate(documents, minhash_func, num_perm=128, threshold=0.5):
    unique_docs = []
    minhashes = []

    for doc in documents:
        if minhash_func == CMinHash:
            mh = minhash_func(num_perm=num_perm, seed=42)  # Add seed for CMinHash
        else:
            mh = minhash_func(num_perm=num_perm)
        
        if isinstance(mh, CMinHash):
            mh.update(doc.split())
        else:  # datasketch MinHash
            for word in doc.split():
                mh.update(word.encode('utf-8'))

        is_duplicate = False
        for i, existing_mh in enumerate(minhashes):
            if mh.jaccard(existing_mh) >= threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            unique_docs.append(doc)
            minhashes.append(mh)

    return unique_docs

def evaluate_simple(dedup_func, documents, minhash_func, num_perm=128, threshold=0.5):
    start_time = time.time()
    unique_docs = dedup_func(documents, minhash_func, num_perm, threshold)
    end_time = time.time()

    print(f"Execution time: {end_time - start_time:.4f} seconds")
    print(f"Number of unique documents: {len(unique_docs)}")
    print("Unique documents:")
    for doc in unique_docs:
        print(f"- {doc}")

def run_simple_benchmark():
    documents = simple_dataset()
    print("Original documents:")
    for doc in documents:
        print(f"- {doc}")
    print("\nCMinHash (rensa) results:")
    evaluate_simple(simple_deduplicate, documents, CMinHash, num_perm=128, threshold=0.6)

    print("\nDatasketch MinHash results:")
    from datasketch import MinHash
    evaluate_simple(simple_deduplicate, documents, MinHash, num_perm=128, threshold=0.6)

# Run the simple benchmark
if __name__ == "__main__":
    run_simple_benchmark()