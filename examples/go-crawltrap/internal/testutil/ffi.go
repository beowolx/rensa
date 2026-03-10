package testutil

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sync"
	"testing"
)

var (
	buildOnce sync.Once
	buildPath string
	buildErr  error
)

func repoRootFrom(start string) (string, error) {
	dir := start
	for i := 0; i < 20; i++ {
		if _, err := os.Stat(filepath.Join(dir, "Cargo.toml")); err == nil {
			return dir, nil
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	return "", errors.New("could not find repo root (Cargo.toml)")
}

func RepoRoot(tb testing.TB) string {
	tb.Helper()

	root, err := RepoRootPath()
	if err != nil {
		tb.Fatal(err)
	}
	return root
}

func RepoRootPath() (string, error) {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		return "", errors.New("runtime.Caller failed")
	}
	return repoRootFrom(filepath.Dir(file))
}

func LibFilename() (string, error) {
	switch runtime.GOOS {
	case "darwin":
		return "librensa_ffi.dylib", nil
	case "linux":
		return "librensa_ffi.so", nil
	default:
		return "", fmt.Errorf("unsupported GOOS=%s", runtime.GOOS)
	}
}

func BuildFFILib(tb testing.TB) string {
	tb.Helper()

	buildPath, err := BuildFFILibPath()
	if err != nil {
		tb.Fatal(err)
	}
	return buildPath
}

func BuildFFILibPath() (string, error) {
	buildOnce.Do(func() {
		root, err := RepoRootPath()
		if err != nil {
			buildErr = err
			return
		}
		cmd := exec.Command("cargo", "build", "-p", "rensa-ffi", "--profile", "release-ffi")
		cmd.Dir = root
		out, err := cmd.CombinedOutput()
		if err != nil {
			buildErr = fmt.Errorf("cargo build rensa-ffi failed: %w\n%s", err, string(out))
			return
		}

		libName, err := LibFilename()
		if err != nil {
			buildErr = err
			return
		}
		buildPath = filepath.Join(root, "target", "release-ffi", libName)
		if _, err := os.Stat(buildPath); err != nil {
			buildErr = fmt.Errorf("built ffi library not found at %q: %w", buildPath, err)
			return
		}
	})

	if buildErr != nil {
		return "", buildErr
	}
	return buildPath, nil
}
