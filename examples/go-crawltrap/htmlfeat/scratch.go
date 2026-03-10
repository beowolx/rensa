package htmlfeat

import (
	"bytes"

	"github.com/beowolx/rensa/examples/go-crawltrap/rensahash"
)

// Scratch holds reusable buffers for HTML feature extraction.
//
// It is not safe for concurrent use. Create one Scratch per worker goroutine.
type Scratch struct {
	cfg        Config
	featureCap int

	seen     map[uint64]struct{}
	inserted []uint64

	tagWindow  ngramWindow
	wordWindow ngramWindow
}

func NewScratch(cfg Config) (*Scratch, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	featureCap := cfg.MaxTextNGrams + cfg.MaxTagNGrams + cfg.MaxAttrFrags
	return &Scratch{
		cfg:        cfg,
		featureCap: featureCap,
		seen:       make(map[uint64]struct{}, featureCap),
		inserted:   make([]uint64, 0, featureCap),
		tagWindow:  newWindow(cfg.TagNGramSize),
		wordWindow: newWindow(cfg.TextNGramSize),
	}, nil
}

// ExtractInto does a single-pass scan over raw HTML bytes and returns a bounded set
// of pre-hashed u64 features, using dst as the output buffer.
func (s *Scratch) ExtractInto(html []byte, dst []uint64) []uint64 {
	cfg := s.cfg
	if cfg.MaxScanBytes > 0 && len(html) > cfg.MaxScanBytes {
		html = html[:cfg.MaxScanBytes]
	}

	// Clear previous call's seen set without reallocating.
	for _, h := range s.inserted {
		delete(s.seen, h)
	}
	s.inserted = s.inserted[:0]
	s.tagWindow.buf = s.tagWindow.buf[:0]
	s.wordWindow.buf = s.wordWindow.buf[:0]

	featureCap := s.featureCap
	if cap(dst) < featureCap {
		dst = make([]uint64, 0, featureCap)
	} else {
		dst = dst[:0]
	}
	features := dst

	addUnique := func(h uint64) (inserted bool, ok bool) {
		if len(features) >= featureCap {
			return false, false
		}
		if _, exists := s.seen[h]; exists {
			return false, true
		}
		s.seen[h] = struct{}{}
		s.inserted = append(s.inserted, h)
		features = append(features, h)
		return true, true
	}

	tagCount := 0
	textCount := 0
	attrCount := 0

	addTagFeature := func(h uint64) bool {
		if tagCount >= cfg.MaxTagNGrams {
			return true
		}
		inserted, ok := addUnique(h)
		if !ok {
			return false
		}
		if inserted {
			tagCount++
		}
		return true
	}
	addTextFeature := func(h uint64) bool {
		if textCount >= cfg.MaxTextNGrams {
			return true
		}
		inserted, ok := addUnique(h)
		if !ok {
			return false
		}
		if inserted {
			textCount++
		}
		return true
	}
	addAttrFeature := func(h uint64) bool {
		if attrCount >= cfg.MaxAttrFrags {
			return true
		}
		inserted, ok := addUnique(h)
		if !ok {
			return false
		}
		if inserted {
			attrCount++
		}
		return true
	}

	allDone := func() bool {
		return tagCount >= cfg.MaxTagNGrams && textCount >= cfg.MaxTextNGrams && attrCount >= cfg.MaxAttrFrags
	}

	mode := modeText
	tagWindow := &s.tagWindow
	wordWindow := &s.wordWindow

	var scratch [256]byte
	var scratch2 [256]byte

	i := 0
	for i < len(html) {
		if allDone() {
			break
		}
		switch mode {
		case modeText:
			b := html[i]
			if b == '<' {
				if i+3 < len(html) && html[i+1] == '!' && html[i+2] == '-' && html[i+3] == '-' {
					mode = modeComment
					i += 4
					continue
				}

				// Parse tag.
				j := i + 1
				if j >= len(html) {
					i = len(html)
					continue
				}

				closing := false
				for j < len(html) && isSpace(html[j]) {
					j++
				}
				if j < len(html) && html[j] == '/' {
					closing = true
					j++
				}
				for j < len(html) && isSpace(html[j]) {
					j++
				}
				if j < len(html) && (html[j] == '!' || html[j] == '?') {
					i = consumeTagLike(html, i)
					continue
				}

				nameStart := j
				for j < len(html) && isTagNameChar(html[j]) {
					j++
				}
				if nameStart == j {
					i++
					continue
				}
				tagNameRaw := html[nameStart:j]
				tagNameLower, tooLong := normalizeInto(scratch[:], tagNameRaw, cfg.MaxTokenLen, false)
				if tooLong {
					// Skip extremely long tag names.
					tagNameLower = nil
				}

				isScript := bytes.Equal(tagNameLower, scriptTagName)
				isStyle := bytes.Equal(tagNameLower, styleTagName)
				emitTag := len(tagNameLower) != 0 && !isScript && !isStyle
				tagHash := uint64(0)
				if emitTag {
					tagHash = rensahash.CalculateHashFast(tagNameLower)
				}

				// Scan attributes until '>'.
				for j < len(html) && html[j] != '>' {
					if allDone() {
						break
					}
					if isScript || isStyle {
						j++
						continue
					}

					if isSpace(html[j]) || html[j] == '/' {
						j++
						continue
					}

					attrStart := j
					for j < len(html) && !isAttrNameTerminator(html[j]) {
						j++
					}
					if attrStart == j {
						j++
						continue
					}
					attrNameRaw := html[attrStart:j]
					attrNameLower, _ := normalizeInto(scratch2[:], attrNameRaw, cfg.MaxTokenLen, false)

					for j < len(html) && isSpace(html[j]) {
						j++
					}
					if j >= len(html) || html[j] != '=' {
						continue
					}
					j++ // '='
					for j < len(html) && isSpace(html[j]) {
						j++
					}
					if j >= len(html) {
						break
					}

					quote := byte(0)
					if html[j] == '"' || html[j] == '\'' {
						quote = html[j]
						j++
					}
					valStart := j
					if quote != 0 {
						for j < len(html) && html[j] != quote {
							j++
						}
					} else {
						for j < len(html) && !isSpace(html[j]) && html[j] != '>' {
							j++
						}
					}
					val := html[valStart:j]
					if quote != 0 && j < len(html) && html[j] == quote {
						j++
					}

					isClass := bytes.Equal(attrNameLower, classAttrName)
					isID := bytes.Equal(attrNameLower, idAttrName)
					if !isClass && !isID {
						continue
					}

					// Split class/id values into fragments.
					k := 0
					for k < len(val) {
						for k < len(val) && !isAttrFragChar(val[k]) {
							k++
						}
						fragStart := k
						for k < len(val) && isAttrFragChar(val[k]) {
							k++
						}
						frag := val[fragStart:k]
						if len(frag) == 0 {
							continue
						}
						fragHash, ok := hashFragment(scratch[:], frag, cfg)
						if !ok {
							continue
						}
						feat := rensahash.SplitMix64(prefixAttr ^ fragHash)
						if !addAttrFeature(feat) {
							break
						}
					}
				}

				// Advance past '>'.
				for j < len(html) && html[j] != '>' {
					j++
				}
				if j < len(html) && html[j] == '>' {
					j++
				}

				if !closing && (isScript || isStyle) {
					if isScript {
						mode = modeScript
					} else {
						mode = modeStyle
					}
					i = j
					continue
				}

				if emitTag && !closing {
					if tagWindow.push(tagHash) {
						feat := tagWindow.ngramHash(prefixTag)
						if !addTagFeature(feat) {
							i = j
							continue
						}
					}
				}

				i = j
				continue
			}

			if isWordChar(b) {
				start := i
				i++
				for i < len(html) && isWordChar(html[i]) {
					i++
				}
				raw := html[start:i]
				wordHash, ok := hashFragment(scratch[:], raw, cfg)
				if !ok {
					continue
				}
				if wordWindow.push(wordHash) {
					feat := wordWindow.ngramHash(prefixTxt)
					addTextFeature(feat)
				}
				continue
			}

			i++

		case modeComment:
			// Skip until "-->".
			if i+2 < len(html) && html[i] == '-' && html[i+1] == '-' && html[i+2] == '>' {
				mode = modeText
				i += 3
			} else {
				i++
			}

		case modeScript:
			// Skip until "</script".
			if i < len(html) && html[i] == '<' {
				advance, found := consumeClosingTag(html, i, "script")
				if found {
					mode = modeText
					i = advance
					continue
				}
			}
			i++

		case modeStyle:
			// Skip until "</style".
			if i < len(html) && html[i] == '<' {
				advance, found := consumeClosingTag(html, i, "style")
				if found {
					mode = modeText
					i = advance
					continue
				}
			}
			i++
		}
	}

	if len(features) == 0 {
		feat := rensahash.SplitMix64(prefixTxt ^ rensahash.CalculateHashFast(emptyHTMLMarker))
		features = append(features, feat)
		s.seen[feat] = struct{}{}
		s.inserted = append(s.inserted, feat)
	}

	return features
}

func ExtractFeatures(html []byte, cfg Config) ([]uint64, error) {
	scratch, err := NewScratch(cfg)
	if err != nil {
		return nil, err
	}
	return scratch.ExtractInto(html, nil), nil
}
