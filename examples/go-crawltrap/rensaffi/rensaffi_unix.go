//go:build darwin || linux

package rensaffi

import (
	"errors"
	"fmt"
	"runtime"
	"sync"

	"github.com/ebitengine/purego"
)

const expectedABIVersion = uint32(1)

var (
	loadMu     sync.RWMutex
	loadHandle uintptr
	loadPath   string

	fnABIVersion func() uint32
	fnCtxNew     func(numPerm uintptr, seed uint64) uintptr
	fnCtxFree    func(ctx uintptr)
	fnDigest     func(
		ctx uintptr,
		tokenHashes *uint64,
		tokenLen uintptr,
		outDigest *uint32,
		outLen uintptr,
	) int32
)

var (
	ErrNotLoaded               = errors.New("rensa ffi not loaded")
	ErrContextClosed           = errors.New("rensa ffi context is nil or closed")
	ErrDigestNullPointer       = errors.New("rensa ffi digest call received a null pointer")
	ErrDigestOutputLenMismatch = errors.New("rensa ffi digest output length mismatch")
	ErrDigestInvalidContext    = errors.New("rensa ffi rejected the context or hit an internal failure")
	ErrDigestUnknown           = errors.New("rensa ffi digest returned an unknown error code")
)

type digestCallError struct {
	op   string
	code int32
	err  error
}

func (e *digestCallError) Error() string {
	return fmt.Sprintf("%s failed: %v (code=%d)", e.op, e.err, e.code)
}

func (e *digestCallError) Unwrap() error {
	return e.err
}

type ffiSymbols struct {
	handle     uintptr
	abiVersion func() uint32
	ctxNew     func(numPerm uintptr, seed uint64) uintptr
	ctxFree    func(ctx uintptr)
	digest     func(
		ctx uintptr,
		tokenHashes *uint64,
		tokenLen uintptr,
		outDigest *uint32,
		outLen uintptr,
	) int32
}

func loadSymbols(path string) (symbols ffiSymbols, err error) {
	handle, err := purego.Dlopen(path, purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		return ffiSymbols{}, fmt.Errorf("dlopen %q: %w", path, err)
	}

	shouldClose := true
	defer func() {
		if !shouldClose {
			return
		}
		if closeErr := purego.Dlclose(handle); closeErr != nil && err == nil {
			err = fmt.Errorf("dlclose %q after failed load: %w", path, closeErr)
		}
	}()

	defer func() {
		if recovered := recover(); recovered != nil {
			err = fmt.Errorf("purego panic while loading symbols: %v", recovered)
		}
	}()

	purego.RegisterLibFunc(&symbols.abiVersion, handle, "rensa_ffi_abi_version")
	purego.RegisterLibFunc(&symbols.ctxNew, handle, "rensa_rminhash_ctx_new")
	purego.RegisterLibFunc(&symbols.ctxFree, handle, "rensa_rminhash_ctx_free")
	purego.RegisterLibFunc(&symbols.digest, handle, "rensa_rminhash_digest_prehashed")

	if got := symbols.abiVersion(); got != expectedABIVersion {
		return ffiSymbols{}, fmt.Errorf(
			"rensa ffi ABI mismatch for %q: expected=%d got=%d",
			path,
			expectedABIVersion,
			got,
		)
	}

	symbols.handle = handle
	shouldClose = false
	return symbols, nil
}

func Load(path string) error {
	loadMu.Lock()
	defer loadMu.Unlock()

	if path == "" {
		return errors.New("rensa ffi library path is empty")
	}
	if loadHandle != 0 {
		if path == loadPath {
			return nil
		}
		return fmt.Errorf("rensa ffi already loaded from %q, cannot reload from %q", loadPath, path)
	}

	symbols, err := loadSymbols(path)
	if err != nil {
		return err
	}

	loadHandle = symbols.handle
	loadPath = path
	fnABIVersion = symbols.abiVersion
	fnCtxNew = symbols.ctxNew
	fnCtxFree = symbols.ctxFree
	fnDigest = symbols.digest
	return nil
}

func mapDigestErrorCode(code int32) error {
	switch code {
	case 1:
		return ErrDigestNullPointer
	case 2:
		return ErrDigestOutputLenMismatch
	case 3:
		return ErrDigestInvalidContext
	default:
		return ErrDigestUnknown
	}
}

type RMinHashCtx struct {
	mu      sync.RWMutex
	ptr     uintptr
	numPerm int
}

func NewRMinHashCtx(numPerm int, seed uint64) (*RMinHashCtx, error) {
	loadMu.RLock()
	ctxNew := fnCtxNew
	loadMu.RUnlock()
	if ctxNew == nil {
		return nil, fmt.Errorf("%w, call Load() first", ErrNotLoaded)
	}
	if numPerm <= 0 {
		return nil, fmt.Errorf("numPerm must be > 0 (got %d)", numPerm)
	}

	ptr := ctxNew(uintptr(numPerm), seed)
	if ptr == 0 {
		return nil, errors.New("rensa_rminhash_ctx_new returned null, check numPerm and ABI compatibility")
	}

	ctx := &RMinHashCtx{ptr: ptr, numPerm: numPerm}
	runtime.SetFinalizer(ctx, (*RMinHashCtx).finalize)
	return ctx, nil
}

func (c *RMinHashCtx) finalize() {
	c.Close()
}

func (c *RMinHashCtx) Close() {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.ptr == 0 {
		return
	}
	fnCtxFree(c.ptr)
	c.ptr = 0
	runtime.SetFinalizer(c, nil)
}

func (c *RMinHashCtx) DigestPrehashed(tokens []uint64, out []uint32) error {
	if c == nil {
		return ErrContextClosed
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	if c.ptr == 0 {
		return ErrContextClosed
	}
	if len(out) != c.numPerm {
		return fmt.Errorf("out digest length mismatch: expected=%d got=%d", c.numPerm, len(out))
	}

	var tokensPtr *uint64
	if len(tokens) != 0 {
		tokensPtr = &tokens[0]
	}

	loadMu.RLock()
	digestFn := fnDigest
	loadMu.RUnlock()
	if digestFn == nil {
		return fmt.Errorf("%w, call Load() first", ErrNotLoaded)
	}

	code := digestFn(
		c.ptr,
		tokensPtr,
		uintptr(len(tokens)),
		&out[0],
		uintptr(len(out)),
	)

	// Ensure Go memory stays alive until after the FFI call.
	runtime.KeepAlive(tokens)
	runtime.KeepAlive(out)
	runtime.KeepAlive(c)

	if code != 0 {
		return &digestCallError{
			op:   "rensa_rminhash_digest_prehashed",
			code: code,
			err:  mapDigestErrorCode(code),
		}
	}
	return nil
}
