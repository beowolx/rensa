package trapdetector

import (
	"errors"
	"sync"

	"github.com/beowolx/rensa/examples/go-crawltrap/rensahash"
)

type DocID uint64
type entryID uint64

const shardCount = 64
const defaultInitialChainCapacity = 16

type bandFingerprint struct {
	bandIndex int
	hash      uint64
}

type bucketKey struct {
	namespace       uint64
	bandFingerprint bandFingerprint
}

type bucketShard struct {
	mu sync.RWMutex
	m  map[bucketKey][]entryID
}

type bucketShards struct {
	shards [shardCount]bucketShard
}

func newBucketShards() bucketShards {
	var b bucketShards
	for i := 0; i < shardCount; i++ {
		b.shards[i] = bucketShard{m: make(map[bucketKey][]entryID)}
	}
	return b
}

func (k bucketKey) shardIndex() uint64 {
	mixed := rensahash.SplitMix64(
		k.namespace ^
			k.bandFingerprint.hash ^
			uint64(k.bandFingerprint.bandIndex+1),
	)
	return mixed & (shardCount - 1)
}

func (b *bucketShards) shard(key bucketKey) *bucketShard {
	return &b.shards[key.shardIndex()]
}

func (b *bucketShards) insert(key bucketKey, id entryID) {
	sh := b.shard(key)
	sh.mu.Lock()
	sh.m[key] = append(sh.m[key], id)
	sh.mu.Unlock()
}

func (b *bucketShards) delete(key bucketKey) {
	sh := b.shard(key)
	sh.mu.Lock()
	delete(sh.m, key)
	sh.mu.Unlock()
}

func (b *bucketShards) forEach(key bucketKey, fn func(entryID)) {
	sh := b.shard(key)
	sh.mu.RLock()
	ids := sh.m[key]
	for _, id := range ids {
		fn(id)
	}
	sh.mu.RUnlock()
}

type ring struct {
	buf   []entryID
	start int
	size  int
	max   int
}

func newRing(maxCapacity int) ring {
	initialCapacity := maxCapacity
	if initialCapacity > defaultInitialChainCapacity {
		initialCapacity = defaultInitialChainCapacity
	}
	if initialCapacity < 1 {
		initialCapacity = 1
	}
	return ring{
		buf: make([]entryID, initialCapacity),
		max: maxCapacity,
	}
}

func (r *ring) grow() {
	nextCapacity := len(r.buf) * 2
	if nextCapacity < 1 {
		nextCapacity = 1
	}
	if nextCapacity > r.max {
		nextCapacity = r.max
	}
	if nextCapacity <= len(r.buf) {
		return
	}

	buf := make([]entryID, nextCapacity)
	for i := 0; i < r.size; i++ {
		buf[i] = r.buf[(r.start+i)%len(r.buf)]
	}
	r.buf = buf
	r.start = 0
}

func (r *ring) push(id entryID) (evicted entryID, didEvict bool) {
	if r.max == 0 {
		return 0, false
	}
	if r.size < len(r.buf) {
		idx := (r.start + r.size) % len(r.buf)
		r.buf[idx] = id
		r.size++
		return 0, false
	}
	if len(r.buf) < r.max {
		r.grow()
		idx := (r.start + r.size) % len(r.buf)
		r.buf[idx] = id
		r.size++
		return 0, false
	}
	evicted = r.buf[r.start]
	r.buf[r.start] = id
	r.start = (r.start + 1) % len(r.buf)
	return evicted, true
}

func (r *ring) items(dst []entryID) []entryID {
	dst = dst[:0]
	if r.size == 0 {
		return dst
	}
	for i := 0; i < r.size; i++ {
		dst = append(dst, r.buf[(r.start+i)%len(r.buf)])
	}
	return dst
}

type chainEntry struct {
	docID  DocID
	digest []uint32
}

type chainState struct {
	mu                  sync.RWMutex
	ring                ring
	entries             map[entryID]chainEntry
	liveDocs            map[DocID]entryID
	bucketKeys          map[bucketKey]struct{}
	evictedSinceRebuild int
	nextEntryID         entryID
	removed             bool
}

// Detector stores per-chain digests for near-duplicate checks.
//
// All digests compared by a single detector must be produced with the same
// NumPerm and the same seed provenance. The detector validates digest length,
// but it does not encode or enforce seed metadata on its own.
type Detector struct {
	cfg      Config
	bandSize int

	mu     sync.RWMutex
	chains map[string]*chainState

	buckets bucketShards
}

func newChainState(maxDocsPerChain int) *chainState {
	initialCapacity := maxDocsPerChain
	if initialCapacity > defaultInitialChainCapacity {
		initialCapacity = defaultInitialChainCapacity
	}
	if initialCapacity < 1 {
		initialCapacity = 1
	}
	return &chainState{
		ring:        newRing(maxDocsPerChain),
		entries:     make(map[entryID]chainEntry, initialCapacity),
		liveDocs:    make(map[DocID]entryID, initialCapacity),
		bucketKeys:  make(map[bucketKey]struct{}, initialCapacity),
		nextEntryID: 1,
	}
}

func New(cfg Config) (*Detector, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}

	d := &Detector{
		cfg:      cfg,
		bandSize: cfg.NumPerm / cfg.NumBands,
		chains:   make(map[string]*chainState),
		buckets:  newBucketShards(),
	}
	return d, nil
}

func (d *Detector) chain(chainKey string) *chainState {
	d.mu.RLock()
	chain := d.chains[chainKey]
	d.mu.RUnlock()
	return chain
}

func (d *Detector) chainOrCreate(chainKey string) *chainState {
	d.mu.Lock()
	chain := d.chains[chainKey]
	if chain == nil {
		chain = newChainState(d.cfg.MaxDocsPerChain)
		d.chains[chainKey] = chain
	}
	d.mu.Unlock()
	return chain
}

func (d *Detector) chainNamespace(chainKey string) uint64 {
	// Note: this allocates. For production, replace with a non-allocating string hash.
	h := rensahash.CalculateHashFast([]byte(chainKey))
	return rensahash.SplitMix64(h)
}

func (d *Detector) digestBandKeys(
	digest []uint32,
	namespace uint64,
	out []bucketKey,
) ([]bucketKey, error) {
	if len(digest) != d.cfg.NumPerm {
		return nil, errors.New("digest length mismatch")
	}
	if cap(out) < d.cfg.NumBands {
		out = make([]bucketKey, 0, d.cfg.NumBands)
	} else {
		out = out[:0]
	}
	for band := 0; band < d.cfg.NumBands; band++ {
		start := band * d.bandSize
		end := start + d.bandSize
		out = append(out, bucketKey{
			namespace: namespace,
			bandFingerprint: bandFingerprint{
				bandIndex: band,
				hash:      rensahash.CalculateBandHash(digest[start:end]),
			},
		})
	}
	return out, nil
}

func jaccardEstimate(a []uint32, b []uint32) float64 {
	if len(a) == 0 || len(a) != len(b) {
		return 0
	}
	eq := 0
	for i := 0; i < len(a); i++ {
		if a[i] == b[i] {
			eq++
		}
	}
	return float64(eq) / float64(len(a))
}

func collectLiveCandidatesLocked(
	chain *chainState,
	buckets *bucketShards,
	bucketKeys []bucketKey,
) []DocID {
	seen := make(map[DocID]struct{})
	candidates := make([]DocID, 0, 32)
	for _, bucketKey := range bucketKeys {
		buckets.forEach(bucketKey, func(id entryID) {
			entry, ok := chain.entries[id]
			if !ok {
				return
			}
			if _, ok := seen[entry.docID]; ok {
				return
			}
			seen[entry.docID] = struct{}{}
			candidates = append(candidates, entry.docID)
		})
	}
	return candidates
}

func verifyCandidatesLocked(
	chain *chainState,
	buckets *bucketShards,
	bucketKeys []bucketKey,
	digest []uint32,
	threshold float64,
) []DocID {
	candidates := collectLiveCandidatesLocked(chain, buckets, bucketKeys)
	verified := make([]DocID, 0, len(candidates))
	for _, id := range candidates {
		candidateEntryID, ok := chain.liveDocs[id]
		if !ok {
			continue
		}
		entry, ok := chain.entries[candidateEntryID]
		if !ok {
			continue
		}
		if jaccardEstimate(digest, entry.digest) >= threshold {
			verified = append(verified, id)
		}
	}
	return verified
}

func insertDigestLocked(
	d *Detector,
	chain *chainState,
	ns uint64,
	id DocID,
	digest []uint32,
	bucketKeys []bucketKey,
) error {
	if chain.removed {
		return errors.New("chain was removed")
	}
	if _, exists := chain.liveDocs[id]; exists {
		return errors.New("duplicate doc id in chain")
	}

	copied := make([]uint32, len(digest))
	copy(copied, digest)

	newEntryID := chain.nextEntryID
	chain.nextEntryID++
	evictedEntryID, didEvict := chain.ring.push(newEntryID)
	if didEvict {
		chain.evictedSinceRebuild++
	}

	chain.entries[newEntryID] = chainEntry{
		docID:  id,
		digest: copied,
	}
	chain.liveDocs[id] = newEntryID
	if didEvict {
		if evictedEntry, ok := chain.entries[evictedEntryID]; ok {
			delete(chain.liveDocs, evictedEntry.docID)
		}
		delete(chain.entries, evictedEntryID)
	}

	for _, bucketKey := range bucketKeys {
		d.buckets.insert(bucketKey, newEntryID)
		chain.bucketKeys[bucketKey] = struct{}{}
	}

	rebuildAt := int(float64(d.cfg.MaxDocsPerChain) * d.cfg.RebuildStaleRatio)
	shouldRebuild := rebuildAt > 0 && chain.evictedSinceRebuild >= rebuildAt
	if shouldRebuild {
		d.rebuildChainLocked(ns, chain)
	}
	return nil
}

// QueryCandidates returns the unique candidate doc IDs (LSH-only, no verification).
func (d *Detector) QueryCandidates(chainKey string, digest []uint32) ([]DocID, error) {
	ns := d.chainNamespace(chainKey)
	chain := d.chain(chainKey)
	bucketKeys, err := d.digestBandKeys(digest, ns, nil)
	if err != nil {
		return nil, err
	}
	if chain == nil {
		return nil, nil
	}
	chain.mu.RLock()
	defer chain.mu.RUnlock()
	if chain.removed {
		return nil, nil
	}

	return collectLiveCandidatesLocked(chain, &d.buckets, bucketKeys), nil
}

// QueryVerified returns candidates whose digest similarity meets the configured threshold.
// For concurrent same-chain callers, prefer CheckAndInsert to avoid a query/insert race.
func (d *Detector) QueryVerified(chainKey string, digest []uint32) ([]DocID, error) {
	ns := d.chainNamespace(chainKey)
	chain := d.chain(chainKey)
	bucketKeys, err := d.digestBandKeys(digest, ns, nil)
	if err != nil {
		return nil, err
	}
	if chain == nil {
		return nil, nil
	}
	chain.mu.RLock()
	defer chain.mu.RUnlock()
	if chain.removed {
		return nil, nil
	}

	return verifyCandidatesLocked(
		chain,
		&d.buckets,
		bucketKeys,
		digest,
		d.cfg.Threshold,
	), nil
}

// Insert adds a digest for a docID within a chain, applying lazy eviction and optional rebuild.
// For concurrent same-chain callers, prefer CheckAndInsert to keep verification and insertion atomic.
func (d *Detector) Insert(chainKey string, id DocID, digest []uint32) error {
	if len(digest) != d.cfg.NumPerm {
		return errors.New("digest length mismatch")
	}

	ns := d.chainNamespace(chainKey)
	bucketKeys, err := d.digestBandKeys(digest, ns, nil)
	if err != nil {
		return err
	}

	chain := d.chainOrCreate(chainKey)
	chain.mu.Lock()
	defer chain.mu.Unlock()
	return insertDigestLocked(d, chain, ns, id, digest, bucketKeys)
}

// CheckAndInsert verifies duplicates and inserts atomically under a single chain lock.
func (d *Detector) CheckAndInsert(
	chainKey string,
	id DocID,
	digest []uint32,
) ([]DocID, bool, error) {
	if len(digest) != d.cfg.NumPerm {
		return nil, false, errors.New("digest length mismatch")
	}

	ns := d.chainNamespace(chainKey)
	bucketKeys, err := d.digestBandKeys(digest, ns, nil)
	if err != nil {
		return nil, false, err
	}

	chain := d.chainOrCreate(chainKey)
	chain.mu.Lock()
	defer chain.mu.Unlock()
	if chain.removed {
		return nil, false, errors.New("chain was removed")
	}
	if _, exists := chain.liveDocs[id]; exists {
		return nil, false, errors.New("duplicate doc id in chain")
	}

	verified := verifyCandidatesLocked(
		chain,
		&d.buckets,
		bucketKeys,
		digest,
		d.cfg.Threshold,
	)
	if len(verified) != 0 {
		return verified, false, nil
	}

	if err := insertDigestLocked(d, chain, ns, id, digest, bucketKeys); err != nil {
		return nil, false, err
	}
	return nil, true, nil
}

// RemoveChain drops all detector state for a chain and clears its namespaced buckets.
func (d *Detector) RemoveChain(chainKey string) bool {
	d.mu.Lock()
	defer d.mu.Unlock()

	chain := d.chains[chainKey]
	if chain == nil {
		return false
	}

	chain.mu.Lock()
	defer chain.mu.Unlock()

	delete(d.chains, chainKey)
	chain.removed = true

	oldKeys := make([]bucketKey, 0, len(chain.bucketKeys))
	for key := range chain.bucketKeys {
		oldKeys = append(oldKeys, key)
	}
	for _, key := range oldKeys {
		d.buckets.delete(key)
	}

	chain.entries = nil
	chain.liveDocs = nil
	chain.bucketKeys = nil
	chain.ring = ring{}
	chain.evictedSinceRebuild = 0

	return true
}

func (d *Detector) rebuildChainLocked(ns uint64, chain *chainState) {
	// Caller holds chain.mu (write).
	oldKeys := make([]bucketKey, 0, len(chain.bucketKeys))
	for k := range chain.bucketKeys {
		oldKeys = append(oldKeys, k)
	}
	kept := chain.ring.items(nil)
	chain.bucketKeys = make(map[bucketKey]struct{})
	chain.evictedSinceRebuild = 0

	for _, k := range oldKeys {
		d.buckets.delete(k)
	}

	// Reinsert kept docs.
	for _, keptEntryID := range kept {
		entry, ok := chain.entries[keptEntryID]
		if !ok {
			continue
		}
		bucketKeys, err := d.digestBandKeys(entry.digest, ns, nil)
		if err != nil {
			continue
		}
		for _, bucketKey := range bucketKeys {
			d.buckets.insert(bucketKey, keptEntryID)
			chain.bucketKeys[bucketKey] = struct{}{}
		}
	}
}
