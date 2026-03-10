//go:build amd64 || arm64

package rensahash

import "math/bits"

const (
	kU64    = uint64(0xf135_7aea_2e62_a9c5)
	rotate  = 26
	seed1   = uint64(0x243f_6a88_85a3_08d3)
	seed2   = uint64(0x1319_8a2e_0370_7344)
	prevent = uint64(0xa409_3822_299f_31d0)
)

func readU64LE(b []byte, offset int) uint64 {
	_ = b[offset+7]
	return uint64(b[offset]) |
		(uint64(b[offset+1]) << 8) |
		(uint64(b[offset+2]) << 16) |
		(uint64(b[offset+3]) << 24) |
		(uint64(b[offset+4]) << 32) |
		(uint64(b[offset+5]) << 40) |
		(uint64(b[offset+6]) << 48) |
		(uint64(b[offset+7]) << 56)
}

func readU32LE(b []byte, offset int) uint32 {
	_ = b[offset+3]
	return uint32(b[offset]) |
		(uint32(b[offset+1]) << 8) |
		(uint32(b[offset+2]) << 16) |
		(uint32(b[offset+3]) << 24)
}

func multiplyMix(x uint64, y uint64) uint64 {
	lo, hi := bits.Mul64(x, y)
	return lo ^ hi
}

func hashBytes(data []byte) uint64 {
	n := len(data)
	s0 := seed1
	s1 := seed2

	if n <= 16 {
		switch {
		case n >= 8:
			s0 ^= readU64LE(data, 0)
			s1 ^= readU64LE(data, n-8)
		case n >= 4:
			s0 ^= uint64(readU32LE(data, 0))
			s1 ^= uint64(readU32LE(data, n-4))
		case n > 0:
			lo := data[0]
			mid := data[n/2]
			hi := data[n-1]
			s0 ^= uint64(lo)
			s1 ^= (uint64(hi) << 8) | uint64(mid)
		}
	} else {
		off := 0
		for off < n-16 {
			x := readU64LE(data, off)
			y := readU64LE(data, off+8)
			t := multiplyMix(s0^x, prevent^y)
			s0 = s1
			s1 = t
			off += 16
		}
		suffixOff := n - 16
		s0 ^= readU64LE(data, suffixOff)
		s1 ^= readU64LE(data, suffixOff+8)
	}

	return multiplyMix(s0, s1) ^ uint64(n)
}

// CalculateHashFast is a direct port of Rensa's Rust `calculate_hash_fast` (64-bit path).
func CalculateHashFast(data []byte) uint64 {
	compressed := hashBytes(data)
	return bits.RotateLeft64(compressed*kU64, rotate)
}

func hashAddU64(hash uint64, value uint64) uint64 {
	return (hash + value) * kU64
}

func hashAddU32(hash uint64, value uint32) uint64 {
	return (hash + uint64(value)) * kU64
}

// CalculateBandHash is a direct port of Rensa's Rust `calculate_band_hash` (64-bit path).
func CalculateBandHash(band []uint32) uint64 {
	var hash uint64
	i := 0
	for i+4 <= len(band) {
		val1 := uint64(band[i]) | (uint64(band[i+1]) << 32)
		val2 := uint64(band[i+2]) | (uint64(band[i+3]) << 32)
		hash = hashAddU64(hash, val1)
		hash = hashAddU64(hash, val2)
		i += 4
	}
	for ; i < len(band); i++ {
		hash = hashAddU32(hash, band[i])
	}
	return bits.RotateLeft64(hash, rotate)
}

// SplitMix64 is a cheap mixer (used for n-gram / namespace folding).
func SplitMix64(value uint64) uint64 {
	value += 0x9e37_79b9_7f4a_7c15
	value = (value ^ (value >> 30)) * 0xbf58_476d_1ce4_e5b9
	value = (value ^ (value >> 27)) * 0x94d0_49bb_1331_11eb
	return value ^ (value >> 31)
}

func Rotl64(value uint64, shift int) uint64 {
	return bits.RotateLeft64(value, shift)
}

