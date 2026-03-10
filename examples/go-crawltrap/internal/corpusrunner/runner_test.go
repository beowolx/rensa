package corpusrunner

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/beowolx/rensa/examples/go-crawltrap/htmlfeat"
	"github.com/beowolx/rensa/examples/go-crawltrap/internal/testutil"
	"github.com/beowolx/rensa/examples/go-crawltrap/trapdetector"
)

func TestRunFixedCorpusProducesExpectedSummary(t *testing.T) {
	libPath := testutil.BuildFFILib(t)
	options := DefaultOptions(libPath, filepath.Join("..", "..", "testdata", "corpus.json"))

	summary, err := RunFixedCorpus(options)
	if err != nil {
		t.Fatalf("RunFixedCorpus failed: %v", err)
	}
	if summary.SchemaVersion != SchemaVersion {
		t.Fatalf("unexpected schema version: got=%d want=%d", summary.SchemaVersion, SchemaVersion)
	}
	if summary.DocumentCount != len(summary.Docs) {
		t.Fatalf("document count mismatch: got=%d len(docs)=%d", summary.DocumentCount, len(summary.Docs))
	}
	if summary.ChainCount == 0 {
		t.Fatal("expected at least one chain in the corpus summary")
	}
}

func TestRunBenchmarkAggregatesCorpusMetrics(t *testing.T) {
	libPath := testutil.BuildFFILib(t)
	options := DefaultOptions(libPath, filepath.Join("..", "..", "testdata", "corpus.json"))

	report, err := RunBenchmark(options, 2)
	if err != nil {
		t.Fatalf("RunBenchmark failed: %v", err)
	}
	if report.SchemaVersion != SchemaVersion {
		t.Fatalf("unexpected schema version: got=%d want=%d", report.SchemaVersion, SchemaVersion)
	}
	if report.Config.Iterations != 2 {
		t.Fatalf("unexpected iteration count: got=%d want=2", report.Config.Iterations)
	}
	if report.Metrics.TotalDocsProcessed != report.Corpus.Documents*2 {
		t.Fatalf(
			"unexpected total docs processed: got=%d want=%d",
			report.Metrics.TotalDocsProcessed,
			report.Corpus.Documents*2,
		)
	}
	if report.Metrics.TotalDurationNS <= 0 {
		t.Fatalf("expected positive benchmark duration, got %d", report.Metrics.TotalDurationNS)
	}
}

func TestNormalizeOptionsRejectsImplicitZeroValueConfigs(t *testing.T) {
	_, err := normalizeOptions(Options{
		FFILibPath: "/tmp/librensa_ffi.so",
		CorpusPath: "testdata/corpus.json",
		NumPerm:    128,
	})
	if err == nil || !strings.Contains(err.Error(), "DetectorConfig must be explicit") {
		t.Fatalf("expected DetectorConfig error, got %v", err)
	}

	_, err = normalizeOptions(Options{
		FFILibPath:     "/tmp/librensa_ffi.so",
		CorpusPath:     "testdata/corpus.json",
		NumPerm:        128,
		DetectorConfig: trapdetector.DefaultConfig(),
	})
	if err == nil || !strings.Contains(err.Error(), "HTMLConfig must be explicit") {
		t.Fatalf("expected HTMLConfig error, got %v", err)
	}
}

func TestNormalizeOptionsPreservesExplicitZeroSeed(t *testing.T) {
	options := Options{
		FFILibPath:     "/tmp/librensa_ffi.so",
		CorpusPath:     "testdata/corpus.json",
		NumPerm:        128,
		Seed:           0,
		DetectorConfig: trapdetector.DefaultConfig(),
		HTMLConfig:     htmlfeat.DefaultConfig(),
	}

	normalized, err := normalizeOptions(options)
	if err != nil {
		t.Fatalf("normalizeOptions failed: %v", err)
	}
	if normalized.Seed != 0 {
		t.Fatalf("expected explicit zero seed to be preserved, got %d", normalized.Seed)
	}
}

func TestNormalizeOptionsRejectsNumPermMismatch(t *testing.T) {
	options := DefaultOptions("/tmp/librensa_ffi.so", "testdata/corpus.json")
	options.NumPerm = 64

	_, err := normalizeOptions(options)
	if err == nil || !strings.Contains(err.Error(), "DetectorConfig.NumPerm mismatch") {
		t.Fatalf("expected NumPerm mismatch error, got %v", err)
	}
}
