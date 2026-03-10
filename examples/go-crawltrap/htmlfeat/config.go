package htmlfeat

import "fmt"

type Config struct {
	TextNGramSize int
	TagNGramSize  int

	MaxTextNGrams int
	MaxTagNGrams  int
	MaxAttrFrags  int

	MaxScanBytes int
	MaxTokenLen  int

	NormalizeNumbers bool
	NormalizeHex     bool
}

func DefaultConfig() Config {
	return Config{
		TextNGramSize:    3,
		TagNGramSize:     2,
		MaxTextNGrams:    700,
		MaxTagNGrams:     400,
		MaxAttrFrags:     200,
		MaxScanBytes:     512 * 1024,
		MaxTokenLen:      128,
		NormalizeNumbers: true,
		NormalizeHex:     true,
	}
}

func (c Config) Validate() error {
	if c.TextNGramSize <= 0 {
		return fmt.Errorf("TextNGramSize must be > 0 (got %d)", c.TextNGramSize)
	}
	if c.TagNGramSize <= 0 {
		return fmt.Errorf("TagNGramSize must be > 0 (got %d)", c.TagNGramSize)
	}
	if c.MaxTextNGrams <= 0 {
		return fmt.Errorf("MaxTextNGrams must be > 0 (got %d)", c.MaxTextNGrams)
	}
	if c.MaxTagNGrams <= 0 {
		return fmt.Errorf("MaxTagNGrams must be > 0 (got %d)", c.MaxTagNGrams)
	}
	if c.MaxAttrFrags <= 0 {
		return fmt.Errorf("MaxAttrFrags must be > 0 (got %d)", c.MaxAttrFrags)
	}
	if c.MaxScanBytes <= 0 {
		return fmt.Errorf("MaxScanBytes must be > 0 (got %d)", c.MaxScanBytes)
	}
	if c.MaxTokenLen <= 0 {
		return fmt.Errorf("MaxTokenLen must be > 0 (got %d)", c.MaxTokenLen)
	}
	return nil
}
