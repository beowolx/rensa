import time

from datasets import load_dataset
from rensa import CMinHash, OptDensMinHash
from tqdm import tqdm


def cminimash(text, num_perm=128):
    m = CMinHash(num_perm=num_perm, seed=42)
    m.update(text.split())
    return m


def optdens_minhash(text, num_perm=128):
    m = OptDensMinHash(num_perm=num_perm, seed=42)
    m.update(text.split())
    return m


def benchmark_deduplication(dataset, minhash_func, num_perm=128, desc="Processing"):
    start_time = time.time()

    unique_hashes = set()
    deduplicated_indices = []

    for idx, example in tqdm(enumerate(dataset), total=len(dataset), desc=desc):
        minhash = minhash_func(example["text"], num_perm)
        hash_tuple = tuple(minhash.digest())

        if hash_tuple not in unique_hashes:
            unique_hashes.add(hash_tuple)
            deduplicated_indices.append(idx)

    end_time = time.time()

    return {
        "time": end_time - start_time,
        "deduplicated_count": len(deduplicated_indices),
        "deduplicated_indices": set(deduplicated_indices),
    }


def run_benchmark():
    print("Loading Wikipedia dataset...")
    wikipedia_dataset = load_dataset(
        "Salesforce/wikitext", "wikitext-103-raw-v1", split="train"
    )

    sketch_sizes = [64, 128, 256, 512, 1024, 2048]

    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON: C-MinHash vs OptDens-MinHash")
    print("=" * 80)

    results_table = []

    for num_perm in sketch_sizes:
        print(f"\n--- Testing with sketch size: {num_perm} ---")

        # Limit dataset size for larger sketch sizes to keep runtime reasonable
        dataset_size = (
            min(len(wikipedia_dataset), 100000)
            if num_perm >= 1024
            else len(wikipedia_dataset)
        )
        test_dataset = wikipedia_dataset.select(range(dataset_size))

        print(f"Using {dataset_size} documents for this test")

        print(f"\nRunning C-MinHash (m={num_perm})...")
        cminhash_results = benchmark_deduplication(
            test_dataset,
            lambda text, num_perm: cminimash(text, num_perm),
            num_perm,
            desc=f"C-MinHash m={num_perm}",
        )

        print(f"\nRunning OptDens-MinHash (m={num_perm})...")
        optdens_results = benchmark_deduplication(
            test_dataset,
            lambda text, num_perm: optdens_minhash(text, num_perm),
            num_perm,
            desc=f"OptDens m={num_perm}",
        )

        speedup = cminhash_results["time"] / optdens_results["time"]

        results_table.append(
            {
                "sketch_size": num_perm,
                "dataset_size": dataset_size,
                "cminhash_time": cminhash_results["time"],
                "optdens_time": optdens_results["time"],
                "speedup": speedup,
                "cminhash_dedup": cminhash_results["deduplicated_count"],
                "optdens_dedup": optdens_results["deduplicated_count"],
            }
        )

        common = cminhash_results["deduplicated_indices"].intersection(
            optdens_results["deduplicated_indices"]
        )
        jaccard = len(common) / len(
            cminhash_results["deduplicated_indices"].union(
                optdens_results["deduplicated_indices"]
            )
        )

        print(f"\nResults for m={num_perm}:")
        print(f"  C-MinHash time: {cminhash_results['time']:.2f}s")
        print(f"  OptDens time: {optdens_results['time']:.2f}s")
        print(
            f"  Speedup: {speedup:.2f}x {'(OptDens faster)' if speedup > 1 else '(C-MinHash faster)'}"
        )
        print(f"  Accuracy (Jaccard): {jaccard:.4f}")

    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(
        f"{'Sketch Size':<12} {'Dataset':<10} {'C-MinHash (s)':<15} {'OptDens (s)':<15} {'Speedup':<12}"
    )
    print("-" * 80)

    for result in results_table:
        speedup_str = f"{result['speedup']:.2f}x"
        if result["speedup"] > 1:
            speedup_str += " OPH"
        else:
            speedup_str = f"{1 / result['speedup']:.2f}x C-Min"

        print(
            f"{result['sketch_size']:<12} {result['dataset_size']:<10} "
            f"{result['cminhash_time']:<15.2f} {result['optdens_time']:<15.2f} {speedup_str:<12}"
        )

    # Find crossover point
    crossover = None
    for i, result in enumerate(results_table):
        if result["speedup"] > 1:
            if i > 0:
                crossover = (
                    results_table[i - 1]["sketch_size"], result["sketch_size"])
            else:
                crossover = (0, result["sketch_size"])
            break

    if crossover:
        print(
            f"\nCrossover point: OptDens becomes faster between m={crossover[0]} and m={crossover[1]}"
        )
    else:
        print(
            f"\nNo crossover found up to m={results_table[-1]['sketch_size']}")


if __name__ == "__main__":
    run_benchmark()
