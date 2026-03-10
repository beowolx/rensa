package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/beowolx/rensa/examples/go-crawltrap/internal/corpusrunner"
	"github.com/beowolx/rensa/examples/go-crawltrap/internal/testutil"
)

type config struct {
	corpusPath string
	outputPath string
	iterations int
	numPerm    int
	seed       uint64
}

func parseFlags(args []string) (config, error) {
	defaultRoot, err := testutil.RepoRootPath()
	if err != nil {
		return config{}, err
	}

	defaultOutput := filepath.Join(
		defaultRoot,
		"benchmarks",
		"baselines",
		"go_crawltrap_corpus_benchmark.json",
	)

	fs := flag.NewFlagSet("corpusbench", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)

	cfg := config{}
	fs.StringVar(&cfg.corpusPath, "corpus", filepath.Join("testdata", "corpus.json"), "Path to the fixed corpus fixture.")
	fs.StringVar(&cfg.outputPath, "output", "", "Optional JSON output path. Use -write-baseline to emit the pinned baseline path.")
	fs.IntVar(&cfg.iterations, "iterations", 10, "Number of end-to-end corpus iterations to run.")
	fs.IntVar(&cfg.numPerm, "num-perm", 128, "Number of R-MinHash permutations.")
	fs.Uint64Var(&cfg.seed, "seed", 42, "R-MinHash seed.")

	writeBaseline := fs.Bool("write-baseline", false, "Write JSON to benchmarks/baselines/go_crawltrap_corpus_benchmark.json.")
	if err := fs.Parse(args); err != nil {
		return config{}, err
	}
	if *writeBaseline && cfg.outputPath == "" {
		cfg.outputPath = defaultOutput
	}
	return cfg, nil
}

func run(args []string) error {
	cfg, err := parseFlags(args)
	if err != nil {
		return err
	}

	ffiLibPath, err := testutil.BuildFFILibPath()
	if err != nil {
		return err
	}

	options := corpusrunner.DefaultOptions(ffiLibPath, cfg.corpusPath)
	options.NumPerm = cfg.numPerm
	options.DetectorConfig.NumPerm = cfg.numPerm
	options.Seed = cfg.seed
	report, err := corpusrunner.RunBenchmark(options, cfg.iterations)
	if err != nil {
		return err
	}

	encoded, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return fmt.Errorf("encode benchmark report: %w", err)
	}
	encoded = append(encoded, '\n')

	if cfg.outputPath != "" {
		if err := os.MkdirAll(filepath.Dir(cfg.outputPath), 0o755); err != nil {
			return fmt.Errorf("create benchmark output directory: %w", err)
		}
		if err := os.WriteFile(cfg.outputPath, encoded, 0o644); err != nil {
			return fmt.Errorf("write benchmark output %q: %w", cfg.outputPath, err)
		}
		fmt.Fprintf(os.Stderr, "wrote benchmark report to %s\n", cfg.outputPath)
	}

	_, err = os.Stdout.Write(encoded)
	return err
}

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
