package rensahash

import "testing"

func TestCalculateHashFastGoldenVectors(t *testing.T) {
	cases := []struct {
		name string
		data []byte
		want uint64
	}{
		{name: "empty", data: []byte{}, want: 17_606_491_139_363_777_937},
		{name: "word", data: []byte("rensa"), want: 8_318_455_212_453_993_772},
		{name: "html_text", data: []byte("crawler trap html"), want: 18_382_577_365_044_734_984},
		{
			name: "chunk_boundary",
			data: []byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17},
			want: 14_785_003_262_498_004_209,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := CalculateHashFast(tc.data); got != tc.want {
				t.Fatalf("CalculateHashFast mismatch: got=%d want=%d", got, tc.want)
			}
		})
	}
}

func TestCalculateBandHashGoldenVectors(t *testing.T) {
	cases := []struct {
		name string
		band []uint32
		want uint64
	}{
		{name: "packed_4", band: []uint32{1, 2, 3, 4}, want: 14_342_353_734_976_182_488},
		{
			name: "remainder",
			band: []uint32{
				571_772_611,
				936_133_835,
				752_080_753,
				481_772_021,
				45_178_047,
			},
			want: 14_655_236_563_294_719_321,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := CalculateBandHash(tc.band); got != tc.want {
				t.Fatalf("CalculateBandHash mismatch: got=%d want=%d", got, tc.want)
			}
		})
	}
}
