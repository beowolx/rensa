package rensaffi

import (
	"errors"
	"fmt"
	"path/filepath"
	"slices"
	"sync"
	"testing"

	"github.com/beowolx/rensa/examples/go-crawltrap/internal/testutil"
	"github.com/ebitengine/purego"
)

func resetLoaderForTests(tb testing.TB) {
	tb.Helper()

	loadMu.Lock()
	defer loadMu.Unlock()

	if loadHandle != 0 {
		if err := purego.Dlclose(loadHandle); err != nil {
			tb.Fatalf("Dlclose failed: %v", err)
		}
	}

	loadHandle = 0
	loadPath = ""
	fnABIVersion = nil
	fnCtxNew = nil
	fnCtxFree = nil
	fnDigest = nil
}

func TestDigestPrehashedIsStable(t *testing.T) {
	libPath := testutil.BuildFFILib(t)
	if err := Load(libPath); err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	const numPerm = 16
	const seed = uint64(42)
	tokens := []uint64{1, 2, 3, 10, 20, 30, 999}

	ctx, err := NewRMinHashCtx(numPerm, seed)
	if err != nil {
		t.Fatalf("NewRMinHashCtx failed: %v", err)
	}
	defer ctx.Close()

	out1 := make([]uint32, numPerm)
	out2 := make([]uint32, numPerm)

	if err := ctx.DigestPrehashed(tokens, out1); err != nil {
		t.Fatalf("DigestPrehashed #1 failed: %v", err)
	}
	if err := ctx.DigestPrehashed(tokens, out2); err != nil {
		t.Fatalf("DigestPrehashed #2 failed: %v", err)
	}

	for i := range out1 {
		if out1[i] != out2[i] {
			t.Fatalf("digest mismatch at %d: %d != %d", i, out1[i], out2[i])
		}
	}

	expected := []uint32{
		571_772_611,
		936_133_835,
		752_080_753,
		481_772_021,
		45_178_047,
		175_317_472,
		470_513_164,
		928_706_636,
		766_355_169,
		186_861_125,
		104_841_669,
		219_275_888,
		723_396_917,
		903_182_598,
		455_784_955,
		153_681_280,
	}
	if len(out1) != len(expected) {
		t.Fatalf("unexpected digest length: got %d expected %d", len(out1), len(expected))
	}
	for i := range expected {
		if out1[i] != expected[i] {
			t.Fatalf("golden mismatch at %d: got %d expected %d", i, out1[i], expected[i])
		}
	}
}

func TestDigestPrehashedSharedContextConcurrent(t *testing.T) {
	libPath := testutil.BuildFFILib(t)
	if err := Load(libPath); err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	const numPerm = 16
	const seed = uint64(42)
	tokens := []uint64{1, 2, 3, 10, 20, 30, 999}
	expected := []uint32{
		571_772_611,
		936_133_835,
		752_080_753,
		481_772_021,
		45_178_047,
		175_317_472,
		470_513_164,
		928_706_636,
		766_355_169,
		186_861_125,
		104_841_669,
		219_275_888,
		723_396_917,
		903_182_598,
		455_784_955,
		153_681_280,
	}

	ctx, err := NewRMinHashCtx(numPerm, seed)
	if err != nil {
		t.Fatalf("NewRMinHashCtx failed: %v", err)
	}
	defer ctx.Close()

	const workers = 8
	var wg sync.WaitGroup
	errCh := make(chan error, workers)

	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			out := make([]uint32, numPerm)
			if err := ctx.DigestPrehashed(tokens, out); err != nil {
				errCh <- err
				return
			}
			if !slices.Equal(out, expected) {
				errCh <- testingErrorf("concurrent digest mismatch: got=%v want=%v", out, expected)
			}
		}()
	}

	wg.Wait()
	close(errCh)

	for err := range errCh {
		t.Fatal(err)
	}
}

func TestCloseDoesNotRaceDigestPrehashed(t *testing.T) {
	libPath := testutil.BuildFFILib(t)
	if err := Load(libPath); err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	const numPerm = 16
	const seed = uint64(42)
	tokens := []uint64{1, 2, 3, 10, 20, 30, 999}

	ctx, err := NewRMinHashCtx(numPerm, seed)
	if err != nil {
		t.Fatalf("NewRMinHashCtx failed: %v", err)
	}

	var wg sync.WaitGroup
	errCh := make(chan error, 32)
	stop := make(chan struct{})

	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-stop:
					return
				default:
				}
				out := make([]uint32, numPerm)
				err := ctx.DigestPrehashed(tokens, out)
				if err == nil {
					continue
				}
				if !errors.Is(err, ErrContextClosed) {
					errCh <- err
				}
				return
			}
		}()
	}

	ctx.Close()
	close(stop)
	wg.Wait()
	close(errCh)

	for err := range errCh {
		t.Fatal(err)
	}

	out := make([]uint32, numPerm)
	if err := ctx.DigestPrehashed(tokens, out); err == nil {
		t.Fatal("expected DigestPrehashed after Close to fail")
	}
}

func TestLoadRetriesAfterFailure(t *testing.T) {
	resetLoaderForTests(t)
	t.Cleanup(func() {
		resetLoaderForTests(t)
	})

	missingPath := filepath.Join(t.TempDir(), "missing-librensa_ffi.so")
	if err := Load(missingPath); err == nil {
		t.Fatal("expected Load to fail for a missing library")
	}

	libPath := testutil.BuildFFILib(t)
	if err := Load(libPath); err != nil {
		t.Fatalf("expected Load to succeed after an earlier failure, got %v", err)
	}
	if err := Load(libPath); err != nil {
		t.Fatalf("expected repeated Load with the same path to stay idempotent, got %v", err)
	}
}

func TestMapDigestErrorCodeReturnsNamedErrors(t *testing.T) {
	cases := []struct {
		name string
		code int32
		want error
	}{
		{name: "null_pointer", code: 1, want: ErrDigestNullPointer},
		{name: "len_mismatch", code: 2, want: ErrDigestOutputLenMismatch},
		{name: "invalid_ctx", code: 3, want: ErrDigestInvalidContext},
		{name: "unknown", code: 99, want: ErrDigestUnknown},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := mapDigestErrorCode(tc.code); !errors.Is(got, tc.want) {
				t.Fatalf("mapDigestErrorCode(%d) mismatch: got=%v want=%v", tc.code, got, tc.want)
			}
		})
	}
}

func TestDigestPrehashedWrapsNamedFFIErrors(t *testing.T) {
	resetLoaderForTests(t)
	t.Cleanup(func() {
		resetLoaderForTests(t)
	})

	libPath := testutil.BuildFFILib(t)
	if err := Load(libPath); err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	ctx, err := NewRMinHashCtx(16, 42)
	if err != nil {
		t.Fatalf("NewRMinHashCtx failed: %v", err)
	}
	defer ctx.Close()

	loadMu.Lock()
	originalDigest := fnDigest
	fnDigest = func(
		ctx uintptr,
		tokenHashes *uint64,
		tokenLen uintptr,
		outDigest *uint32,
		outLen uintptr,
	) int32 {
		return 3
	}
	loadMu.Unlock()
	t.Cleanup(func() {
		loadMu.Lock()
		fnDigest = originalDigest
		loadMu.Unlock()
	})

	out := make([]uint32, 16)
	err = ctx.DigestPrehashed([]uint64{1, 2, 3}, out)
	if err == nil {
		t.Fatal("expected DigestPrehashed to fail when the FFI returns an error code")
	}
	if !errors.Is(err, ErrDigestInvalidContext) {
		t.Fatalf("expected named invalid-context error, got %v", err)
	}
	if got := err.Error(); got != "rensa_rminhash_digest_prehashed failed: rensa ffi rejected the context or hit an internal failure (code=3)" {
		t.Fatalf("unexpected wrapped error message: %q", got)
	}
}

type testingError string

func (e testingError) Error() string {
	return string(e)
}

func testingErrorf(format string, args ...any) error {
	return testingError(fmt.Sprintf(format, args...))
}
