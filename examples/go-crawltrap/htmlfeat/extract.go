package htmlfeat

import "github.com/beowolx/rensa/examples/go-crawltrap/rensahash"

type scanMode uint8

const (
	modeText scanMode = iota
	modeComment
	modeScript
	modeStyle
)

const (
	prefixTxt  = uint64(0x7478745f_9f6a_2c11) // "txt_" + noise
	prefixTag  = uint64(0x7461675f_d1b5_4a32) // "tag_" + noise
	prefixAttr = uint64(0x61747472_0c42_94d1) // "attr" + noise
)

var (
	hexPlaceholder  = []byte("__hex__")
	emptyHTMLMarker = []byte("__empty_html__")
	scriptTagName   = []byte("script")
	styleTagName    = []byte("style")
	classAttrName   = []byte("class")
	idAttrName      = []byte("id")
)

func isSpace(b byte) bool {
	return b == ' ' || b == '\n' || b == '\r' || b == '\t' || b == '\f'
}

func isAlphaNum(b byte) bool {
	return (b >= 'A' && b <= 'Z') || (b >= 'a' && b <= 'z') || (b >= '0' && b <= '9')
}

func isWordChar(b byte) bool {
	return isAlphaNum(b)
}

func isAttrFragChar(b byte) bool {
	return isAlphaNum(b) || b == '_' || b == '-'
}

func isTagNameChar(b byte) bool {
	return isAlphaNum(b) || b == ':' || b == '_' || b == '-'
}

func isAttrNameTerminator(b byte) bool {
	return isSpace(b) || b == '=' || b == '>' || b == '/'
}

func toLowerASCII(b byte) byte {
	if b >= 'A' && b <= 'Z' {
		return b + ('a' - 'A')
	}
	return b
}

func isHexDigitLower(b byte) bool {
	return (b >= '0' && b <= '9') || (b >= 'a' && b <= 'f')
}

type ngramWindow struct {
	buf  []uint64
	size int
}

func newWindow(size int) ngramWindow {
	return ngramWindow{
		buf:  make([]uint64, 0, size),
		size: size,
	}
}

func (w *ngramWindow) push(value uint64) bool {
	if w.size <= 1 {
		w.buf = w.buf[:0]
		w.buf = append(w.buf, value)
		return true
	}
	if len(w.buf) < w.size {
		w.buf = append(w.buf, value)
		return len(w.buf) == w.size
	}
	copy(w.buf, w.buf[1:])
	w.buf[w.size-1] = value
	return true
}

func (w *ngramWindow) ngramHash(prefix uint64) uint64 {
	if len(w.buf) == 0 {
		return rensahash.SplitMix64(prefix)
	}
	mixed := prefix
	for i, h := range w.buf {
		shift := 1 + (i * 6)
		mixed ^= rensahash.Rotl64(h, shift)
	}
	return rensahash.SplitMix64(mixed)
}

func normalizeInto(
	dst []byte,
	src []byte,
	maxTokenLen int,
	normalizeNumbers bool,
) (normalized []byte, tooLong bool) {
	dst = dst[:0]

	prevWasDigit := false
	for _, raw := range src {
		b := toLowerASCII(raw)
		if normalizeNumbers && b >= '0' && b <= '9' {
			if prevWasDigit {
				continue
			}
			b = '0'
			prevWasDigit = true
		} else {
			prevWasDigit = false
		}

		if len(dst) >= maxTokenLen {
			return nil, true
		}
		dst = append(dst, b)
	}

	return dst, false
}

func hashFragment(
	scratch []byte,
	raw []byte,
	cfg Config,
) (uint64, bool) {
	scratch = scratch[:0]

	prevWasDigit := false
	rawLen := 0
	hexLike := true

	for _, rawByte := range raw {
		bLower := toLowerASCII(rawByte)
		rawLen++
		if !isHexDigitLower(bLower) {
			hexLike = false
		}

		bNorm := bLower
		if cfg.NormalizeNumbers && bLower >= '0' && bLower <= '9' {
			if prevWasDigit {
				continue
			}
			bNorm = '0'
			prevWasDigit = true
		} else {
			prevWasDigit = false
		}

		if len(scratch) >= cfg.MaxTokenLen {
			return 0, false
		}
		scratch = append(scratch, bNorm)
	}

	if rawLen >= 16 && cfg.NormalizeHex && hexLike {
		return rensahash.CalculateHashFast(hexPlaceholder), true
	}
	if len(scratch) == 0 {
		return 0, false
	}
	return rensahash.CalculateHashFast(scratch), true
}

func consumeClosingTag(html []byte, i int, tagLower string) (advance int, found bool) {
	// Expect '<' at html[i].
	if i+2 >= len(html) || html[i] != '<' || html[i+1] != '/' {
		return i, false
	}
	j := i + 2
	for j < len(html) && isSpace(html[j]) {
		j++
	}
	nameStart := j
	for j < len(html) && isTagNameChar(html[j]) {
		j++
	}
	name := html[nameStart:j]
	if len(name) != len(tagLower) {
		return i, false
	}
	for idx := 0; idx < len(tagLower); idx++ {
		if toLowerASCII(name[idx]) != tagLower[idx] {
			return i, false
		}
	}
	for j < len(html) && html[j] != '>' {
		j++
	}
	if j < len(html) && html[j] == '>' {
		j++
	}
	return j, true
}

func consumeTagLike(html []byte, i int) int {
	if i >= len(html) || html[i] != '<' {
		return i + 1
	}
	j := i + 1
	for j < len(html) && html[j] != '>' {
		j++
	}
	if j < len(html) && html[j] == '>' {
		j++
	}
	return j
}
