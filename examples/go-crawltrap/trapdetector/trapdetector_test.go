package trapdetector

import (
	"math/bits"
	"slices"
	"sync"
	"testing"

	"github.com/beowolx/rensa/examples/go-crawltrap/htmlfeat"
	"github.com/beowolx/rensa/examples/go-crawltrap/internal/testutil"
	"github.com/beowolx/rensa/examples/go-crawltrap/rensaffi"
	"github.com/beowolx/rensa/examples/go-crawltrap/rensahash"
)

const hashMixConstant = uint64(0xf135_7aea_2e62_a9c5)

func invertOdd64(value uint64) uint64 {
	inverse := value
	for i := 0; i < 6; i++ {
		inverse *= 2 - value*inverse
	}
	return inverse
}

func digestForBandHash(target uint64) []uint32 {
	rotated := bits.RotateLeft64(target, -26)
	packed := rotated * invertOdd64(hashMixConstant)
	return []uint32{0, 0, uint32(packed), uint32(packed >> 32)}
}

func TestEndToEndHTMLDuplicateSameChain(t *testing.T) {
	libPath := testutil.BuildFFILib(t)
	if err := rensaffi.Load(libPath); err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	const numPerm = 128
	const seed = uint64(42)

	ctx, err := rensaffi.NewRMinHashCtx(numPerm, seed)
	if err != nil {
		t.Fatalf("NewRMinHashCtx failed: %v", err)
	}
	defer ctx.Close()

	cfg := DefaultConfig()
	det, err := New(cfg)
	if err != nil {
		t.Fatalf("New detector failed: %v", err)
	}

	html1 := []byte("<html><body>Hello world page=123</body></html>")
	html2 := []byte("<html><body>Hello world page=999</body></html>")

	featCfg := htmlfeat.DefaultConfig()
	features1, err := htmlfeat.ExtractFeatures(html1, featCfg)
	if err != nil {
		t.Fatalf("ExtractFeatures #1 failed: %v", err)
	}
	features2, err := htmlfeat.ExtractFeatures(html2, featCfg)
	if err != nil {
		t.Fatalf("ExtractFeatures #2 failed: %v", err)
	}

	digest1 := make([]uint32, numPerm)
	digest2 := make([]uint32, numPerm)

	if err := ctx.DigestPrehashed(features1, digest1); err != nil {
		t.Fatalf("DigestPrehashed #1 failed: %v", err)
	}
	if err := ctx.DigestPrehashed(features2, digest2); err != nil {
		t.Fatalf("DigestPrehashed #2 failed: %v", err)
	}

	const chainA = "example.com/a"
	const chainB = "example.com/b"

	if err := det.Insert(chainA, 1, digest1); err != nil {
		t.Fatalf("Insert failed: %v", err)
	}

	verifiedA, err := det.QueryVerified(chainA, digest2)
	if err != nil {
		t.Fatalf("QueryVerified chainA failed: %v", err)
	}
	if len(verifiedA) == 0 {
		t.Fatalf("expected duplicate candidate in same chain, got none")
	}

	verifiedB, err := det.QueryVerified(chainB, digest2)
	if err != nil {
		t.Fatalf("QueryVerified chainB failed: %v", err)
	}
	if len(verifiedB) != 0 {
		t.Fatalf("expected no candidates across chains, got %v", verifiedB)
	}
}

func TestDocIDReuseAfterEvictionDoesNotAliasStaleBuckets(t *testing.T) {
	cfg := DefaultConfig()
	cfg.NumPerm = 4
	cfg.NumBands = 2
	cfg.MaxDocsPerChain = 1

	det, err := New(cfg)
	if err != nil {
		t.Fatalf("New detector failed: %v", err)
	}

	chain := "example.com/reuse"
	digestA := []uint32{1, 2, 3, 4}
	digestB := []uint32{101, 102, 103, 104}
	digestC := []uint32{201, 202, 203, 204}

	if err := det.Insert(chain, 1, digestA); err != nil {
		t.Fatalf("Insert digestA failed: %v", err)
	}
	if err := det.Insert(chain, 2, digestB); err != nil {
		t.Fatalf("Insert digestB failed: %v", err)
	}
	if err := det.Insert(chain, 1, digestC); err != nil {
		t.Fatalf("Insert digestC with reused DocID failed: %v", err)
	}

	candidates, err := det.QueryCandidates(chain, digestA)
	if err != nil {
		t.Fatalf("QueryCandidates failed: %v", err)
	}
	if len(candidates) != 0 {
		t.Fatalf("expected stale bucket entries to stay dead after DocID reuse, got %v", candidates)
	}
}

func TestRemoveChainKeepsCrossChainCollisionBucketsAlive(t *testing.T) {
	cfg := DefaultConfig()
	cfg.NumPerm = 4
	cfg.NumBands = 1

	det, err := New(cfg)
	if err != nil {
		t.Fatalf("New detector failed: %v", err)
	}

	const chainA = "example.com/remove-a"
	const chainB = "example.com/remove-b"
	digestA := []uint32{1, 2, 3, 4}
	bandHashA := rensahash.CalculateBandHash(digestA)
	digestB := digestForBandHash(
		bandHashA ^
			det.chainNamespace(chainA) ^
			det.chainNamespace(chainB),
	)
	if got := rensahash.CalculateBandHash(digestB); got != bandHashA^det.chainNamespace(chainA)^det.chainNamespace(chainB) {
		t.Fatalf("failed to build collision digest: got=%d", got)
	}

	if err := det.Insert(chainA, 1, digestA); err != nil {
		t.Fatalf("Insert chainA failed: %v", err)
	}
	if err := det.Insert(chainB, 1, digestB); err != nil {
		t.Fatalf("Insert chainB failed: %v", err)
	}

	beforeRemove, err := det.QueryCandidates(chainB, digestB)
	if err != nil {
		t.Fatalf("QueryCandidates before remove failed: %v", err)
	}
	if !slices.Equal(beforeRemove, []DocID{1}) {
		t.Fatalf("expected live chainB candidates before remove, got %v", beforeRemove)
	}

	if removed := det.RemoveChain(chainA); !removed {
		t.Fatal("expected RemoveChain to remove chainA")
	}

	afterRemove, err := det.QueryCandidates(chainB, digestB)
	if err != nil {
		t.Fatalf("QueryCandidates after remove failed: %v", err)
	}
	if !slices.Equal(afterRemove, []DocID{1}) {
		t.Fatalf("expected chainB buckets to survive chainA removal, got %v", afterRemove)
	}
}

func TestRebuildKeepsCrossChainCollisionBucketsAlive(t *testing.T) {
	cfg := DefaultConfig()
	cfg.NumPerm = 4
	cfg.NumBands = 1
	cfg.MaxDocsPerChain = 1
	cfg.RebuildStaleRatio = 1

	det, err := New(cfg)
	if err != nil {
		t.Fatalf("New detector failed: %v", err)
	}

	const chainA = "example.com/rebuild-a"
	const chainB = "example.com/rebuild-b"
	digestA := []uint32{1, 2, 3, 4}
	digestA2 := []uint32{9, 8, 7, 6}
	bandHashA := rensahash.CalculateBandHash(digestA)
	digestB := digestForBandHash(
		bandHashA ^
			det.chainNamespace(chainA) ^
			det.chainNamespace(chainB),
	)

	if err := det.Insert(chainA, 1, digestA); err != nil {
		t.Fatalf("Insert chainA doc1 failed: %v", err)
	}
	if err := det.Insert(chainB, 1, digestB); err != nil {
		t.Fatalf("Insert chainB failed: %v", err)
	}
	if err := det.Insert(chainA, 2, digestA2); err != nil {
		t.Fatalf("Insert chainA doc2 failed: %v", err)
	}

	candidates, err := det.QueryCandidates(chainB, digestB)
	if err != nil {
		t.Fatalf("QueryCandidates after rebuild failed: %v", err)
	}
	if !slices.Equal(candidates, []DocID{1}) {
		t.Fatalf("expected chainB buckets to survive chainA rebuild, got %v", candidates)
	}
}

func TestCheckAndInsertSerializesSameChainDuplicates(t *testing.T) {
	cfg := DefaultConfig()
	cfg.NumPerm = 4
	cfg.NumBands = 2
	cfg.Threshold = 1.0

	det, err := New(cfg)
	if err != nil {
		t.Fatalf("New detector failed: %v", err)
	}

	const workers = 8
	chain := "example.com/atomic"
	digest := []uint32{7, 7, 7, 7}

	type result struct {
		docID    DocID
		verified []DocID
		inserted bool
		err      error
	}

	results := make([]result, workers)
	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func(index int) {
			defer wg.Done()
			verified, inserted, err := det.CheckAndInsert(chain, DocID(index+1), digest)
			results[index] = result{
				docID:    DocID(index + 1),
				verified: verified,
				inserted: inserted,
				err:      err,
			}
		}(i)
	}
	wg.Wait()

	insertedIDs := make([]DocID, 0, 1)
	for _, result := range results {
		if result.err != nil {
			t.Fatalf("CheckAndInsert(%d) failed: %v", result.docID, result.err)
		}
		if result.inserted {
			insertedIDs = append(insertedIDs, result.docID)
			continue
		}
		if len(result.verified) != 1 {
			t.Fatalf("expected one verified duplicate for doc %d, got %v", result.docID, result.verified)
		}
	}

	if len(insertedIDs) != 1 {
		t.Fatalf("expected exactly one inserted document, got %v", insertedIDs)
	}

	expectedWinner := insertedIDs[0]
	for _, result := range results {
		if result.inserted {
			continue
		}
		if !slices.Equal(result.verified, []DocID{expectedWinner}) {
			t.Fatalf(
				"expected doc %d to report the inserted winner %d, got %v",
				result.docID,
				expectedWinner,
				result.verified,
			)
		}
	}
}

func TestRemoveChainClearsDetectorState(t *testing.T) {
	cfg := DefaultConfig()
	cfg.NumPerm = 4
	cfg.NumBands = 2

	det, err := New(cfg)
	if err != nil {
		t.Fatalf("New detector failed: %v", err)
	}

	chain := "example.com/remove"
	digest := []uint32{5, 5, 5, 5}

	if err := det.Insert(chain, 1, digest); err != nil {
		t.Fatalf("Insert failed: %v", err)
	}
	beforeRemove, err := det.QueryCandidates(chain, digest)
	if err != nil {
		t.Fatalf("QueryCandidates before remove failed: %v", err)
	}
	if !slices.Equal(beforeRemove, []DocID{1}) {
		t.Fatalf("expected candidate before remove, got %v", beforeRemove)
	}

	if removed := det.RemoveChain(chain); !removed {
		t.Fatal("expected RemoveChain to remove the live chain")
	}

	afterRemove, err := det.QueryCandidates(chain, digest)
	if err != nil {
		t.Fatalf("QueryCandidates after remove failed: %v", err)
	}
	if len(afterRemove) != 0 {
		t.Fatalf("expected no candidates after RemoveChain, got %v", afterRemove)
	}

	if err := det.Insert(chain, 2, digest); err != nil {
		t.Fatalf("Insert after RemoveChain failed: %v", err)
	}
	afterRecreate, err := det.QueryCandidates(chain, digest)
	if err != nil {
		t.Fatalf("QueryCandidates after recreate failed: %v", err)
	}
	if !slices.Equal(afterRecreate, []DocID{2}) {
		t.Fatalf("expected recreated chain to be isolated, got %v", afterRecreate)
	}
}
