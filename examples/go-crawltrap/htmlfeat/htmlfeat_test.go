package htmlfeat

import (
	"reflect"
	"testing"
	"time"
)

func TestDefaultConfigIsValid(t *testing.T) {
	if err := DefaultConfig().Validate(); err != nil {
		t.Fatalf("DefaultConfig should be valid: %v", err)
	}
}

func TestExplicitConfigIsRequired(t *testing.T) {
	if _, err := NewScratch(Config{}); err == nil {
		t.Fatal("expected NewScratch to reject zero-value config")
	}
	if _, err := ExtractFeatures([]byte("<html><body>hello</body></html>"), Config{}); err == nil {
		t.Fatal("expected ExtractFeatures to reject zero-value config")
	}
}

func TestExtractFeaturesMakesProgressOnTagEdgeCases(t *testing.T) {
	cfg := DefaultConfig()
	cases := []struct {
		name string
		html string
	}{
		{
			name: "doctype",
			html: "<!DOCTYPE html><html><body>hello world</body></html>",
		},
		{
			name: "processing_instruction",
			html: "<?xml version=\"1.0\"?><html><body>hello world</body></html>",
		},
		{
			name: "framework_attrs",
			html: "<div :class=\"state\" @click=\"open()\" [foo]=\"bar\" class=\"page-123\" id=\"deadbeefdeadbeef\">hello world</div>",
		},
		{
			name: "malformed_attr_prefix",
			html: "<div =\"oops\" class=\"page-123\">hello world</div>",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			done := make(chan []uint64, 1)
			errCh := make(chan error, 1)
			go func() {
				features, err := ExtractFeatures([]byte(tc.html), cfg)
				if err != nil {
					errCh <- err
					return
				}
				done <- features
			}()

			select {
			case err := <-errCh:
				t.Fatalf("ExtractFeatures failed: %v", err)
			case features := <-done:
				if len(features) == 0 {
					t.Fatal("expected non-empty feature set")
				}
			case <-time.After(time.Second):
				t.Fatal("ExtractFeatures did not make progress")
			}
		})
	}
}

func TestExtractFeaturesIgnoresClosingTags(t *testing.T) {
	cfg := DefaultConfig()

	withClosing, err := ExtractFeatures([]byte("<div><span>hello world</span></div>"), cfg)
	if err != nil {
		t.Fatalf("ExtractFeatures with closing tags failed: %v", err)
	}
	withoutClosing, err := ExtractFeatures([]byte("<div><span>hello world"), cfg)
	if err != nil {
		t.Fatalf("ExtractFeatures without closing tags failed: %v", err)
	}

	if !reflect.DeepEqual(withClosing, withoutClosing) {
		t.Fatalf("expected closing tags to be ignored\nwith    = %v\nwithout = %v", withClosing, withoutClosing)
	}
}

func TestExtractIntoReturnedSliceMutationDoesNotCorruptNextCall(t *testing.T) {
	cfg := DefaultConfig()
	scratch, err := NewScratch(cfg)
	if err != nil {
		t.Fatalf("NewScratch failed: %v", err)
	}

	html := []byte("<div class=\"page-123\" id=\"deadbeefdeadbeef\">hello world</div>")
	first := scratch.ExtractInto(html, nil)
	expected := append([]uint64(nil), first...)

	for i := range first {
		first[i] = 0
	}

	second := scratch.ExtractInto(html, nil)
	if !reflect.DeepEqual(expected, second) {
		t.Fatalf("expected feature extraction to be stable after caller mutation\nexpected = %v\ngot      = %v", expected, second)
	}
}
