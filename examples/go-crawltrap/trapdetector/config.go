package trapdetector

import "fmt"

type Config struct {
	NumPerm   int
	NumBands  int
	Threshold float64

	MaxDocsPerChain   int
	RebuildStaleRatio float64
}

func DefaultConfig() Config {
	return Config{
		NumPerm:           128,
		NumBands:          16,
		Threshold:         0.9,
		MaxDocsPerChain:   2000,
		RebuildStaleRatio: 0.5,
	}
}

func (c Config) Validate() error {
	if c.NumPerm <= 0 {
		return fmt.Errorf("NumPerm must be > 0 (got %d)", c.NumPerm)
	}
	if c.NumBands <= 0 {
		return fmt.Errorf("NumBands must be > 0 (got %d)", c.NumBands)
	}
	if c.NumPerm%c.NumBands != 0 {
		return fmt.Errorf("NumBands must divide NumPerm (NumPerm=%d NumBands=%d)", c.NumPerm, c.NumBands)
	}
	if c.Threshold < 0 || c.Threshold > 1 {
		return fmt.Errorf("Threshold must be in [0,1] (got %f)", c.Threshold)
	}
	if c.MaxDocsPerChain <= 0 {
		return fmt.Errorf("MaxDocsPerChain must be > 0 (got %d)", c.MaxDocsPerChain)
	}
	if c.RebuildStaleRatio <= 0 || c.RebuildStaleRatio > 1 {
		return fmt.Errorf("RebuildStaleRatio must be in (0,1] (got %f)", c.RebuildStaleRatio)
	}
	return nil
}
