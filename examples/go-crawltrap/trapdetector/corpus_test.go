package trapdetector_test

import (
	"path/filepath"
	"testing"

	"github.com/beowolx/rensa/examples/go-crawltrap/internal/corpusrunner"
	"github.com/beowolx/rensa/examples/go-crawltrap/internal/testutil"
	"github.com/beowolx/rensa/examples/go-crawltrap/trapdetector"
)

func testCorpusPath() string {
	return filepath.Join("..", "testdata", "corpus.json")
}

func TestDefaultConfigIsValid(t *testing.T) {
	if err := trapdetector.DefaultConfig().Validate(); err != nil {
		t.Fatalf("DefaultConfig should be valid: %v", err)
	}
}

func TestExplicitConfigIsRequired(t *testing.T) {
	if _, err := trapdetector.New(trapdetector.Config{}); err == nil {
		t.Fatal("expected detector construction to reject zero-value config")
	}
}

func TestThresholdZeroIsValid(t *testing.T) {
	cfg := trapdetector.DefaultConfig()
	cfg.Threshold = 0
	if err := cfg.Validate(); err != nil {
		t.Fatalf("Threshold=0 should be valid: %v", err)
	}
	if _, err := trapdetector.New(cfg); err != nil {
		t.Fatalf("New detector with Threshold=0 failed: %v", err)
	}
}

func TestFixedCorpusEndToEnd(t *testing.T) {
	libPath := testutil.BuildFFILib(t)
	options := corpusrunner.DefaultOptions(libPath, testCorpusPath())

	summary, err := corpusrunner.RunFixedCorpus(options)
	if err != nil {
		t.Fatalf("RunFixedCorpus failed: %v", err)
	}
	if summary.DocumentCount == 0 {
		t.Fatal("expected fixed corpus run to process documents")
	}
	if summary.InsertedCount == 0 {
		t.Fatal("expected fixed corpus run to insert at least one document")
	}
	if summary.DuplicateCount == 0 {
		t.Fatal("expected fixed corpus run to find at least one duplicate")
	}
}

func BenchmarkFixedCorpusEndToEnd(b *testing.B) {
	libPath := testutil.BuildFFILib(b)
	options := corpusrunner.DefaultOptions(libPath, testCorpusPath())

	report, err := corpusrunner.RunBenchmark(options, b.N)
	if err != nil {
		b.Fatalf("RunBenchmark failed: %v", err)
	}
	b.ReportMetric(float64(report.Metrics.AverageDurationNS), "ns/op")
}
