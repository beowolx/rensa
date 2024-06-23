import time
import random
import string
from tqdm import tqdm
from datasketch import MinHash, MinHashLSH
from rensa import CMinHash, CMinHashLSH

def generate_random_document(word_count=100, vocab_size=1000):
    vocabulary = [''.join(random.choices(string.ascii_lowercase, k=5)) for _ in range(vocab_size)]
    return ' '.join(random.choices(vocabulary, k=word_count))

def generate_dataset(num_docs=100000, duplicate_ratio=0.2):
    original_docs = [generate_random_document() for _ in range(int(num_docs * (1 - duplicate_ratio)))]
    duplicates = random.choices(original_docs, k=int(num_docs * duplicate_ratio))
    all_docs = original_docs + duplicates
    random.shuffle(all_docs)
    return all_docs

def deduplicate_cminhash_lsh(documents, num_perm=128, threshold=0.5, num_bands=64, seed=42):
    start_time = time.time()

    lsh = CMinHashLSH(threshold=threshold, num_perm=num_perm, num_bands=num_bands)
    unique_docs = []

    for i, doc in enumerate(tqdm(documents, desc="CMinHash-LSH")):
        mh = CMinHash(num_perm=num_perm, seed=seed)
        mh.update(doc.split())

        candidates = lsh.query(mh)
        is_duplicate = any(lsh.is_similar(mh, unique_docs[candidate][1]) for candidate in candidates)

        if not is_duplicate:
            lsh.insert(len(unique_docs), mh)
            unique_docs.append((doc, mh))

    end_time = time.time()
    return [doc for doc, _ in unique_docs], end_time - start_time

def deduplicate_datasketch(documents, num_perm=128, threshold=0.5):
    start_time = time.time()

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    unique_docs = []

    for i, doc in enumerate(tqdm(documents, desc="datasketch")):
        mh = MinHash(num_perm=num_perm)
        for word in doc.split():
            mh.update(word.encode('utf8'))

        if not lsh.query(mh):
            lsh.insert(f"doc_{i}", mh)
            unique_docs.append(doc)

    end_time = time.time()
    return unique_docs, end_time - start_time

def run_benchmark(num_docs=100000, duplicate_ratio=0.2, num_perm=128, threshold=0.5, num_bands=64):
    print(f"Generating dataset with {num_docs} documents ({duplicate_ratio*100}% duplicates)...")
    documents = generate_dataset(num_docs, duplicate_ratio)

    print("\nRunning Rust-implemented CMinHash-LSH deduplication...")
    unique_cminhash, time_cminhash = deduplicate_cminhash_lsh(documents, num_perm, threshold, num_bands)

    print("\nRunning datasketch deduplication...")
    unique_datasketch, time_datasketch = deduplicate_datasketch(documents, num_perm, threshold)

    print("\nResults:")
    print(f"Rust CMinHash-LSH: {len(unique_cminhash)} unique documents, {time_cminhash:.2f} seconds")
    print(f"datasketch: {len(unique_datasketch)} unique documents, {time_datasketch:.2f} seconds")
    print(f"Speed-up: {time_datasketch / time_cminhash:.2f}x")

# Run the benchmark
run_benchmark(num_docs=1000, duplicate_ratio=0.2, num_perm=128, threshold=0.5, num_bands=64)