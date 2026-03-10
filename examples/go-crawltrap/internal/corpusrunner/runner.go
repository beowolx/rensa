package corpusrunner

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"time"

	"github.com/beowolx/rensa/examples/go-crawltrap/htmlfeat"
	"github.com/beowolx/rensa/examples/go-crawltrap/rensaffi"
	"github.com/beowolx/rensa/examples/go-crawltrap/trapdetector"
)

const SchemaVersion = 1

type Fixture struct {
	Docs []Doc `json:"docs"`
}

type Doc struct {
	Name             string               `json:"name"`
	DocID            trapdetector.DocID   `json:"doc_id"`
	Chain            string               `json:"chain"`
	HTML             string               `json:"html"`
	MaxCandidates    int                  `json:"max_candidates"`
	ExpectedVerified []trapdetector.DocID `json:"expected_verified"`
}

type Options struct {
	FFILibPath     string
	CorpusPath     string
	NumPerm        int
	Seed           uint64
	DetectorConfig trapdetector.Config
	HTMLConfig     htmlfeat.Config
}

type DocResult struct {
	Name           string             `json:"name"`
	DocID          trapdetector.DocID `json:"doc_id"`
	Chain          string             `json:"chain"`
	FeatureCount   int                `json:"feature_count"`
	CandidateCount int                `json:"candidate_count"`
	VerifiedCount  int                `json:"verified_count"`
	Inserted       bool               `json:"inserted"`
}

type RunSummary struct {
	SchemaVersion   int         `json:"schema_version"`
	CorpusPath      string      `json:"corpus_path"`
	DocumentCount   int         `json:"document_count"`
	ChainCount      int         `json:"chain_count"`
	TotalFeatures   int         `json:"total_features"`
	TotalCandidates int         `json:"total_candidates"`
	TotalVerified   int         `json:"total_verified"`
	InsertedCount   int         `json:"inserted_count"`
	DuplicateCount  int         `json:"duplicate_count"`
	Docs            []DocResult `json:"docs"`
}

type BenchmarkReport struct {
	SchemaVersion int              `json:"schema_version"`
	BenchmarkName string           `json:"benchmark_name"`
	GeneratedAt   string           `json:"generated_at_utc"`
	GoVersion     string           `json:"go_version"`
	GOOS          string           `json:"goos"`
	GOARCH        string           `json:"goarch"`
	Corpus        BenchmarkCorpus  `json:"corpus"`
	Config        BenchmarkConfig  `json:"config"`
	Metrics       BenchmarkMetrics `json:"metrics"`
}

type BenchmarkCorpus struct {
	Path      string `json:"path"`
	Documents int    `json:"documents"`
	Chains    int    `json:"chains"`
}

type BenchmarkConfig struct {
	Iterations        int     `json:"iterations"`
	NumPerm           int     `json:"num_perm"`
	NumBands          int     `json:"num_bands"`
	Threshold         float64 `json:"threshold"`
	MaxDocsPerChain   int     `json:"max_docs_per_chain"`
	RebuildStaleRatio float64 `json:"rebuild_stale_ratio"`
	Seed              uint64  `json:"seed"`
}

type BenchmarkMetrics struct {
	TotalDocsProcessed int   `json:"total_docs_processed"`
	TotalFeatures      int   `json:"total_features"`
	TotalCandidates    int   `json:"total_candidates"`
	TotalVerified      int   `json:"total_verified"`
	InsertedCount      int   `json:"inserted_count"`
	DuplicateCount     int   `json:"duplicate_count"`
	TotalDurationNS    int64 `json:"total_duration_ns"`
	AverageDurationNS  int64 `json:"average_duration_ns"`
	MinDurationNS      int64 `json:"min_duration_ns"`
	MaxDurationNS      int64 `json:"max_duration_ns"`
}

func DefaultOptions(ffiLibPath string, corpusPath string) Options {
	cfg := trapdetector.DefaultConfig()
	return Options{
		FFILibPath:     ffiLibPath,
		CorpusPath:     corpusPath,
		NumPerm:        cfg.NumPerm,
		Seed:           42,
		DetectorConfig: cfg,
		HTMLConfig:     htmlfeat.DefaultConfig(),
	}
}

func LoadFixture(path string) (Fixture, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Fixture{}, fmt.Errorf("read corpus fixture %q: %w", path, err)
	}

	var fixture Fixture
	if err := json.Unmarshal(data, &fixture); err != nil {
		return Fixture{}, fmt.Errorf("decode corpus fixture %q: %w", path, err)
	}
	if len(fixture.Docs) == 0 {
		return Fixture{}, fmt.Errorf("corpus fixture %q is empty", path)
	}
	return fixture, nil
}

func sortedDocIDs(ids []trapdetector.DocID) []trapdetector.DocID {
	sorted := append([]trapdetector.DocID(nil), ids...)
	slices.Sort(sorted)
	return sorted
}

func normalizeOptions(options Options) (Options, error) {
	if options.FFILibPath == "" {
		return Options{}, fmt.Errorf("FFILibPath must not be empty")
	}
	if options.CorpusPath == "" {
		options.CorpusPath = filepath.Join("testdata", "corpus.json")
	}

	if options.DetectorConfig == (trapdetector.Config{}) {
		return Options{}, fmt.Errorf(
			"DetectorConfig must be explicit; use DefaultOptions() or trapdetector.DefaultConfig()",
		)
	}
	if options.HTMLConfig == (htmlfeat.Config{}) {
		return Options{}, fmt.Errorf(
			"HTMLConfig must be explicit; use DefaultOptions() or htmlfeat.DefaultConfig()",
		)
	}
	if options.NumPerm <= 0 {
		return Options{}, fmt.Errorf("NumPerm must be > 0 (got %d)", options.NumPerm)
	}
	if options.DetectorConfig.NumPerm != options.NumPerm {
		return Options{}, fmt.Errorf(
			"DetectorConfig.NumPerm mismatch: got=%d NumPerm=%d",
			options.DetectorConfig.NumPerm,
			options.NumPerm,
		)
	}
	if err := options.DetectorConfig.Validate(); err != nil {
		return Options{}, fmt.Errorf("validate detector config: %w", err)
	}
	if err := options.HTMLConfig.Validate(); err != nil {
		return Options{}, fmt.Errorf("validate html config: %w", err)
	}
	return options, nil
}

func newRuntime(options Options) (*rensaffi.RMinHashCtx, *htmlfeat.Scratch, Fixture, error) {
	if err := rensaffi.Load(options.FFILibPath); err != nil {
		return nil, nil, Fixture{}, fmt.Errorf("load rensa ffi %q: %w", options.FFILibPath, err)
	}

	ctx, err := rensaffi.NewRMinHashCtx(options.NumPerm, options.Seed)
	if err != nil {
		return nil, nil, Fixture{}, fmt.Errorf("create rensa ffi context: %w", err)
	}

	scratch, err := htmlfeat.NewScratch(options.HTMLConfig)
	if err != nil {
		ctx.Close()
		return nil, nil, Fixture{}, fmt.Errorf("create html feature scratch: %w", err)
	}

	fixture, err := LoadFixture(options.CorpusPath)
	if err != nil {
		ctx.Close()
		return nil, nil, Fixture{}, err
	}

	return ctx, scratch, fixture, nil
}

func runCorpusPass(
	fixture Fixture,
	options Options,
	ctx *rensaffi.RMinHashCtx,
	scratch *htmlfeat.Scratch,
) (RunSummary, error) {
	det, err := trapdetector.New(options.DetectorConfig)
	if err != nil {
		return RunSummary{}, fmt.Errorf("create trap detector: %w", err)
	}

	chainSet := make(map[string]struct{}, len(fixture.Docs))
	summary := RunSummary{
		SchemaVersion: SchemaVersion,
		CorpusPath:    options.CorpusPath,
		DocumentCount: len(fixture.Docs),
		Docs:          make([]DocResult, 0, len(fixture.Docs)),
	}

	for _, doc := range fixture.Docs {
		chainSet[doc.Chain] = struct{}{}

		features := scratch.ExtractInto([]byte(doc.HTML), nil)
		digest := make([]uint32, options.NumPerm)
		if err := ctx.DigestPrehashed(features, digest); err != nil {
			return RunSummary{}, fmt.Errorf("digest %s: %w", doc.Name, err)
		}

		candidates, err := det.QueryCandidates(doc.Chain, digest)
		if err != nil {
			return RunSummary{}, fmt.Errorf("query candidates for %s: %w", doc.Name, err)
		}
		if len(candidates) > doc.MaxCandidates {
			return RunSummary{}, fmt.Errorf(
				"candidate fanout too high for %s: got=%d max=%d candidates=%v",
				doc.Name,
				len(candidates),
				doc.MaxCandidates,
				candidates,
			)
		}

		verified, err := det.QueryVerified(doc.Chain, digest)
		if err != nil {
			return RunSummary{}, fmt.Errorf("query verified for %s: %w", doc.Name, err)
		}

		wantVerified := sortedDocIDs(doc.ExpectedVerified)
		gotVerified := sortedDocIDs(verified)
		if !slices.Equal(gotVerified, wantVerified) {
			return RunSummary{}, fmt.Errorf(
				"verified matches mismatch for %s: got=%v want=%v",
				doc.Name,
				gotVerified,
				wantVerified,
			)
		}

		for _, expected := range wantVerified {
			if !slices.Contains(candidates, expected) {
				return RunSummary{}, fmt.Errorf(
					"expected candidate %d for %s, got=%v",
					expected,
					doc.Name,
					candidates,
				)
			}
		}

		verifiedAgain, inserted, err := det.CheckAndInsert(doc.Chain, doc.DocID, digest)
		if err != nil {
			return RunSummary{}, fmt.Errorf("check-and-insert %s: %w", doc.Name, err)
		}
		if len(wantVerified) == 0 {
			if !inserted {
				return RunSummary{}, fmt.Errorf(
					"expected %s to insert, got verified=%v",
					doc.Name,
					verifiedAgain,
				)
			}
		} else {
			if inserted {
				return RunSummary{}, fmt.Errorf("expected %s to stay duplicate, but it inserted", doc.Name)
			}
			if !slices.Equal(sortedDocIDs(verifiedAgain), wantVerified) {
				return RunSummary{}, fmt.Errorf(
					"check-and-insert verified mismatch for %s: got=%v want=%v",
					doc.Name,
					sortedDocIDs(verifiedAgain),
					wantVerified,
				)
			}
		}

		summary.TotalFeatures += len(features)
		summary.TotalCandidates += len(candidates)
		summary.TotalVerified += len(verified)
		if inserted {
			summary.InsertedCount++
		} else {
			summary.DuplicateCount++
		}
		summary.Docs = append(summary.Docs, DocResult{
			Name:           doc.Name,
			DocID:          doc.DocID,
			Chain:          doc.Chain,
			FeatureCount:   len(features),
			CandidateCount: len(candidates),
			VerifiedCount:  len(verified),
			Inserted:       inserted,
		})
	}

	summary.ChainCount = len(chainSet)
	return summary, nil
}

func RunFixedCorpus(options Options) (RunSummary, error) {
	options, err := normalizeOptions(options)
	if err != nil {
		return RunSummary{}, err
	}

	ctx, scratch, fixture, err := newRuntime(options)
	if err != nil {
		return RunSummary{}, err
	}
	defer ctx.Close()

	return runCorpusPass(fixture, options, ctx, scratch)
}

func RunBenchmark(options Options, iterations int) (BenchmarkReport, error) {
	options, err := normalizeOptions(options)
	if err != nil {
		return BenchmarkReport{}, err
	}
	if iterations <= 0 {
		return BenchmarkReport{}, fmt.Errorf("iterations must be > 0 (got %d)", iterations)
	}

	ctx, scratch, fixture, err := newRuntime(options)
	if err != nil {
		return BenchmarkReport{}, err
	}
	defer ctx.Close()

	minDuration := time.Duration(0)
	maxDuration := time.Duration(0)
	totalDuration := time.Duration(0)
	metrics := BenchmarkMetrics{}

	for iteration := 0; iteration < iterations; iteration++ {
		startedAt := time.Now()
		summary, err := runCorpusPass(fixture, options, ctx, scratch)
		if err != nil {
			return BenchmarkReport{}, fmt.Errorf("benchmark iteration %d: %w", iteration+1, err)
		}
		elapsed := time.Since(startedAt)

		if minDuration == 0 || elapsed < minDuration {
			minDuration = elapsed
		}
		if elapsed > maxDuration {
			maxDuration = elapsed
		}
		totalDuration += elapsed

		metrics.TotalDocsProcessed += summary.DocumentCount
		metrics.TotalFeatures += summary.TotalFeatures
		metrics.TotalCandidates += summary.TotalCandidates
		metrics.TotalVerified += summary.TotalVerified
		metrics.InsertedCount += summary.InsertedCount
		metrics.DuplicateCount += summary.DuplicateCount
	}

	metrics.TotalDurationNS = totalDuration.Nanoseconds()
	metrics.AverageDurationNS = totalDuration.Nanoseconds() / int64(iterations)
	metrics.MinDurationNS = minDuration.Nanoseconds()
	metrics.MaxDurationNS = maxDuration.Nanoseconds()

	return BenchmarkReport{
		SchemaVersion: SchemaVersion,
		BenchmarkName: "go_crawltrap_corpus",
		GeneratedAt:   time.Now().UTC().Format(time.RFC3339Nano),
		GoVersion:     runtime.Version(),
		GOOS:          runtime.GOOS,
		GOARCH:        runtime.GOARCH,
		Corpus: BenchmarkCorpus{
			Path:      options.CorpusPath,
			Documents: len(fixture.Docs),
			Chains:    countChains(fixture),
		},
		Config: BenchmarkConfig{
			Iterations:        iterations,
			NumPerm:           options.NumPerm,
			NumBands:          options.DetectorConfig.NumBands,
			Threshold:         options.DetectorConfig.Threshold,
			MaxDocsPerChain:   options.DetectorConfig.MaxDocsPerChain,
			RebuildStaleRatio: options.DetectorConfig.RebuildStaleRatio,
			Seed:              options.Seed,
		},
		Metrics: metrics,
	}, nil
}

func countChains(fixture Fixture) int {
	chainSet := make(map[string]struct{}, len(fixture.Docs))
	for _, doc := range fixture.Docs {
		chainSet[doc.Chain] = struct{}{}
	}
	return len(chainSet)
}
