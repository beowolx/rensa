# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "datasets>=2.20",
#   "rensa",
# ]
# ///
"""Exploratory HTML similarity benchmark.

This script is useful for quick MinHash experiments on HTML datasets, but it is
not the production-shaped crawler benchmark. The canonical crawler-trap path
now lives in the Go reference stack (`htmlfeat` + FFI + `trapdetector`).
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable
from urllib.parse import urlsplit

from datasets import load_dataset

from rensa import RMinHash, RMinHashLSH

DEFAULT_DATASET = "philschmid/crawl-dataset"
DEFAULT_SPLIT = "train"
DEFAULT_MAX_ROWS = 2_500
DEFAULT_NUM_PERM = 128
DEFAULT_NUM_BANDS = 16
DEFAULT_THRESHOLD = 0.9
DEFAULT_SEED = 42
DEFAULT_TEXT_NGRAM_SIZE = 3
DEFAULT_TAG_NGRAM_SIZE = 2
DEFAULT_MAX_TEXT_TOKENS = 700
DEFAULT_MAX_TAG_TOKENS = 400
DEFAULT_MAX_ATTR_TOKENS = 200
DEFAULT_MIN_CHAIN_SIZE = 2
DEFAULT_CHAIN_PATH_DEPTH = 2
DEFAULT_SHOW_EXAMPLES = 15
DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / ".bench"
    / "profiling"
    / "html_crawl_similarity.json"
)

HTML_FIELD_CANDIDATES = ("html", "raw_html", "content")
URL_FIELD_CANDIDATES = ("url", "uri", "page_url", "target_url")
CHAIN_FIELD_CANDIDATES = (
    "source_url",
    "seed_url",
    "referrer",
    "parent_url",
    "crawl_id",
)

SCRIPT_STYLE_RE = re.compile(r"(?is)<(script|style)\b[^>]*>.*?</\1>")
COMMENT_RE = re.compile(r"(?s)<!--.*?-->")
TAG_NAME_RE = re.compile(r"(?is)<\s*/?\s*([a-z][a-z0-9:_-]*)")
TAG_RE = re.compile(r"(?is)<[^>]+>")
WORD_RE = re.compile(r"[a-z0-9]+")
CLASS_ID_RE = re.compile(
    r"""(?is)\b(?:class|id)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))"""
)
ATTR_SPLIT_RE = re.compile(r"[^a-z0-9_-]+")
DIGIT_RUN_RE = re.compile(r"\d+")
HEX_LIKE_RE = re.compile(r"^[0-9a-f]{16,}$")


@dataclass(frozen=True)
class PageSketch:
    doc_id: int
    row_index: int
    url: str
    chain_key: str
    token_count: int
    minhash: RMinHash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Exploratory HTML near-duplicate benchmark using a Python-side "
            "regex/downsampling extractor. For production-shaped crawler-trap "
            "benchmarking, prefer the Go corpus benchmark."
        )
    )
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT)
    parser.add_argument(
        "--streaming",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load the dataset in streaming mode (default: true).",
    )
    parser.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS)
    parser.add_argument(
        "--html-field",
        type=str,
        default="",
        help="HTML column name. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--url-field",
        type=str,
        default="",
        help="URL column name. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--chain-field",
        type=str,
        default="",
        help=(
            "Request-chain column name (for example source_url). "
            "Auto-detected when omitted."
        ),
    )
    parser.add_argument(
        "--chain-path-depth",
        type=int,
        default=DEFAULT_CHAIN_PATH_DEPTH,
        help=(
            "If no chain field is present, group URLs by host + this many "
            "path segments."
        ),
    )
    parser.add_argument("--num-perm", type=int, default=DEFAULT_NUM_PERM)
    parser.add_argument("--num-bands", type=int, default=DEFAULT_NUM_BANDS)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--text-ngram-size", type=int, default=DEFAULT_TEXT_NGRAM_SIZE
    )
    parser.add_argument(
        "--tag-ngram-size", type=int, default=DEFAULT_TAG_NGRAM_SIZE
    )
    parser.add_argument("--max-text-tokens", type=int, default=DEFAULT_MAX_TEXT_TOKENS)
    parser.add_argument("--max-tag-tokens", type=int, default=DEFAULT_MAX_TAG_TOKENS)
    parser.add_argument("--max-attr-tokens", type=int, default=DEFAULT_MAX_ATTR_TOKENS)
    parser.add_argument("--min-chain-size", type=int, default=DEFAULT_MIN_CHAIN_SIZE)
    parser.add_argument("--show-examples", type=int, default=DEFAULT_SHOW_EXAMPLES)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--normalize-numbers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize digit runs in tokens (default: true).",
    )
    parser.add_argument(
        "--normalize-hex",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Collapse long hex-like tokens (default: true).",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    strictly_positive = (
        ("max_rows", args.max_rows),
        ("num_perm", args.num_perm),
        ("num_bands", args.num_bands),
        ("text_ngram_size", args.text_ngram_size),
        ("tag_ngram_size", args.tag_ngram_size),
        ("max_text_tokens", args.max_text_tokens),
        ("max_tag_tokens", args.max_tag_tokens),
        ("max_attr_tokens", args.max_attr_tokens),
    )
    for name, value in strictly_positive:
        if value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be > 0")

    if args.min_chain_size <= 1:
        raise ValueError("--min-chain-size must be > 1")

    non_negative = (
        ("chain_path_depth", args.chain_path_depth),
        ("show_examples", args.show_examples),
    )
    for name, value in non_negative:
        if value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 0")

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be in [0, 1]")
    if args.num_bands > args.num_perm:
        raise ValueError("--num-bands must be <= --num-perm")
    if args.num_perm % args.num_bands != 0:
        raise ValueError("--num-bands must divide --num-perm")


def choose_field(
    available_fields: Iterable[str],
    preferred: str,
    candidates: tuple[str, ...],
) -> str:
    available = set(available_fields)
    if preferred:
        if preferred not in available:
            raise ValueError(
                f"Requested field '{preferred}' was not found. Available: "
                f"{sorted(available)}"
            )
        return preferred
    for candidate in candidates:
        if candidate in available:
            return candidate
    return ""


def downsample_tokens(tokens: list[str], limit: int) -> list[str]:
    if len(tokens) <= limit:
        return tokens
    if limit == 1:
        return [tokens[0]]
    stride = (len(tokens) - 1) / (limit - 1)
    return [tokens[round(index * stride)] for index in range(limit)]


def make_ngrams(tokens: list[str], ngram_size: int, prefix: str) -> list[str]:
    if not tokens:
        return []
    if ngram_size <= 1 or len(tokens) < ngram_size:
        return [f"{prefix}:{token}" for token in tokens]
    return [
        f"{prefix}:{'|'.join(tokens[index : index + ngram_size])}"
        for index in range(len(tokens) - ngram_size + 1)
    ]


def normalize_chain_key(url_or_key: str, path_depth: int) -> str:
    value = url_or_key.strip()
    if not value:
        return "__unknown_chain__"
    parsed = urlsplit(value)
    if parsed.scheme and parsed.netloc:
        host = parsed.netloc.lower()
        path_parts = [part for part in parsed.path.split("/") if part]
        if path_depth > 0:
            path_parts = path_parts[:path_depth]
        suffix = "/" + "/".join(path_parts) if path_parts else ""
        return f"{host}{suffix}"
    return value.lower()


def infer_chain_key(
    row: dict[str, Any],
    chain_field: str,
    url_field: str,
    path_depth: int,
) -> str:
    if chain_field:
        raw = row.get(chain_field)
        if raw is not None:
            return normalize_chain_key(str(raw), path_depth)
    if url_field:
        raw_url = row.get(url_field)
        if raw_url is not None:
            return normalize_chain_key(str(raw_url), path_depth)
    return "__unknown_chain__"


def tokenize_html(
    html: str,
    text_ngram_size: int,
    tag_ngram_size: int,
    max_text_tokens: int,
    max_tag_tokens: int,
    max_attr_tokens: int,
    normalize_numbers: bool,
    normalize_hex: bool,
) -> list[str]:
    lowered = html.lower()
    stripped = COMMENT_RE.sub(" ", SCRIPT_STYLE_RE.sub(" ", lowered))

    tag_names = TAG_NAME_RE.findall(stripped)
    if len(tag_names) > max_tag_tokens:
        tag_names = downsample_tokens(tag_names, max_tag_tokens)
    tag_features = make_ngrams(tag_names, tag_ngram_size, "tag")

    attr_tokens: list[str] = []
    for quoted_double, quoted_single, bare in CLASS_ID_RE.findall(stripped):
        raw_value = quoted_double or quoted_single or bare
        for part in ATTR_SPLIT_RE.split(raw_value):
            if part:
                part = normalize_fragment(part, normalize_numbers, normalize_hex)
                attr_tokens.append(f"attr:{part}")
    if len(attr_tokens) > max_attr_tokens:
        attr_tokens = downsample_tokens(attr_tokens, max_attr_tokens)

    text_only = TAG_RE.sub(" ", stripped)
    words = WORD_RE.findall(text_only)
    if len(words) > max_text_tokens:
        words = downsample_tokens(words, max_text_tokens)
    if normalize_numbers or normalize_hex:
        words = [
            normalize_fragment(word, normalize_numbers, normalize_hex)
            for word in words
        ]
    text_features = make_ngrams(words, text_ngram_size, "txt")

    feature_tokens = list(dict.fromkeys(tag_features + attr_tokens + text_features))
    if not feature_tokens:
        return ["txt:__empty_html__"]
    return feature_tokens


def summarize_chain_sizes(chain_sizes: list[int]) -> dict[str, float | int]:
    if not chain_sizes:
        return {
            "count": 0,
            "min": 0,
            "max": 0,
            "median": 0.0,
            "mean": 0.0,
        }
    return {
        "count": len(chain_sizes),
        "min": min(chain_sizes),
        "max": max(chain_sizes),
        "median": statistics.median(chain_sizes),
        "mean": statistics.fmean(chain_sizes),
    }


def normalize_fragment(
    fragment: str,
    normalize_numbers: bool,
    normalize_hex: bool,
) -> str:
    if normalize_hex and HEX_LIKE_RE.match(fragment):
        return "__hex__"
    if normalize_numbers and DIGIT_RUN_RE.search(fragment):
        return DIGIT_RUN_RE.sub("0", fragment)
    return fragment


def main() -> None:
    args = parse_args()
    validate_args(args)

    start_total = perf_counter()
    dataset = load_dataset(args.dataset, split=args.split, streaming=args.streaming)
    iterator = iter(dataset)
    try:
        first_row = next(iterator)
    except StopIteration as error:
        raise ValueError("dataset returned no rows") from error

    html_field = choose_field(first_row.keys(), args.html_field, HTML_FIELD_CANDIDATES)
    if not html_field:
        raise ValueError(
            "Could not auto-detect an HTML field. "
            f"Available fields: {sorted(first_row.keys())}"
        )
    url_field = choose_field(first_row.keys(), args.url_field, URL_FIELD_CANDIDATES)
    chain_field = choose_field(first_row.keys(), args.chain_field, CHAIN_FIELD_CANDIDATES)

    rows = itertools.chain([first_row], iterator)

    build_start = perf_counter()
    pages: list[PageSketch] = []
    rows_seen = 0
    rows_with_html = 0
    skipped_missing_html = 0

    for rows_seen, row in enumerate(itertools.islice(rows, args.max_rows), start=1):
        html_value = row.get(html_field)
        if not isinstance(html_value, str) or not html_value.strip():
            skipped_missing_html += 1
            continue

        rows_with_html += 1
        tokens = tokenize_html(
            html=html_value,
            text_ngram_size=args.text_ngram_size,
            tag_ngram_size=args.tag_ngram_size,
            max_text_tokens=args.max_text_tokens,
            max_tag_tokens=args.max_tag_tokens,
            max_attr_tokens=args.max_attr_tokens,
            normalize_numbers=args.normalize_numbers,
            normalize_hex=args.normalize_hex,
        )
        minhash = RMinHash(num_perm=args.num_perm, seed=args.seed)
        minhash.update(tokens)

        raw_url = row.get(url_field) if url_field else ""
        url = str(raw_url) if raw_url is not None else ""
        chain_key = infer_chain_key(
            row=row,
            chain_field=chain_field,
            url_field=url_field,
            path_depth=args.chain_path_depth,
        )
        pages.append(
            PageSketch(
                doc_id=len(pages),
                row_index=rows_seen,
                url=url,
                chain_key=chain_key,
                token_count=len(tokens),
                minhash=minhash,
            )
        )

    build_seconds = perf_counter() - build_start

    chains: dict[str, list[PageSketch]] = defaultdict(list)
    for page in pages:
        chains[page.chain_key].append(page)

    compare_start = perf_counter()
    chain_count_evaluated = 0
    candidate_pairs = 0
    matched_pairs: list[dict[str, Any]] = []

    for chain_key, chain_pages in chains.items():
        if len(chain_pages) < args.min_chain_size:
            continue
        chain_count_evaluated += 1

        lsh = RMinHashLSH(
            threshold=args.threshold,
            num_perm=args.num_perm,
            num_bands=args.num_bands,
            seed=args.seed,
        )
        by_id = {page.doc_id: page for page in chain_pages}

        lsh.insert_pairs((page.doc_id, page.minhash) for page in chain_pages)
        candidates_per_page = lsh.query_all([page.minhash for page in chain_pages])
        candidate_iter = zip(chain_pages, candidates_per_page)

        seen_pairs: set[tuple[int, int]] = set()
        for left_page, candidates in candidate_iter:
            for candidate in candidates:
                right_id = int(candidate)
                if right_id == left_page.doc_id:
                    continue
                left_id, right_id = sorted((left_page.doc_id, right_id))
                pair_key = (left_id, right_id)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                candidate_pairs += 1

                left = by_id[left_id]
                right = by_id[right_id]
                score = left.minhash.jaccard(right.minhash)
                if score >= args.threshold:
                    matched_pairs.append(
                        {
                            "chain_key": chain_key,
                            "left_doc_id": left.doc_id,
                            "right_doc_id": right.doc_id,
                            "left_row_index": left.row_index,
                            "right_row_index": right.row_index,
                            "left_url": left.url,
                            "right_url": right.url,
                            "left_token_count": left.token_count,
                            "right_token_count": right.token_count,
                            "jaccard_estimate": score,
                        }
                    )

    compare_seconds = perf_counter() - compare_start
    total_seconds = perf_counter() - start_total

    matched_pairs.sort(key=lambda item: item["jaccard_estimate"], reverse=True)
    if args.show_examples > 0:
        examples = matched_pairs[: args.show_examples]
    else:
        examples = []

    token_counts = [page.token_count for page in pages]
    chain_sizes = [len(chain_pages) for chain_pages in chains.values()]
    summary = {
        "rows_seen": rows_seen,
        "rows_with_html": rows_with_html,
        "rows_skipped_missing_html": skipped_missing_html,
        "sketched_pages": len(pages),
        "chains_total": len(chains),
        "chains_evaluated": chain_count_evaluated,
        "candidate_pairs": candidate_pairs,
        "matched_pairs": len(matched_pairs),
        "matched_pair_rate_over_pages": (
            len(matched_pairs) / len(pages) if pages else 0.0
        ),
        "token_count_mean": statistics.fmean(token_counts) if token_counts else 0.0,
        "token_count_median": statistics.median(token_counts) if token_counts else 0.0,
        "chain_sizes": summarize_chain_sizes(chain_sizes),
        "timing_seconds": {
            "build_signatures": build_seconds,
            "compare_chains": compare_seconds,
            "total": total_seconds,
        },
    }

    payload = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        "config": {
            "dataset": args.dataset,
            "split": args.split,
            "streaming": args.streaming,
            "max_rows": args.max_rows,
            "html_field": html_field,
            "url_field": url_field,
            "chain_field": chain_field,
            "chain_path_depth": args.chain_path_depth,
            "num_perm": args.num_perm,
            "num_bands": args.num_bands,
            "threshold": args.threshold,
            "seed": args.seed,
            "text_ngram_size": args.text_ngram_size,
            "tag_ngram_size": args.tag_ngram_size,
            "max_text_tokens": args.max_text_tokens,
            "max_tag_tokens": args.max_tag_tokens,
            "max_attr_tokens": args.max_attr_tokens,
            "min_chain_size": args.min_chain_size,
            "normalize_numbers": args.normalize_numbers,
            "normalize_hex": args.normalize_hex,
        },
        "summary": summary,
        "examples": examples,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print("=" * 80)
    print("Exploratory HTML Crawl Similarity Check")
    print("=" * 80)
    print(f"Dataset: {args.dataset} (split={args.split}, streaming={args.streaming})")
    print(
        f"Fields: html={html_field} url={url_field or '<none>'} "
        f"chain={chain_field or '<inferred from url>'}"
    )
    print(
        f"Rows: seen={summary['rows_seen']} with_html={summary['rows_with_html']} "
        f"sketched={summary['sketched_pages']}"
    )
    print(
        f"Chains: total={summary['chains_total']} "
        f"evaluated(min_size={args.min_chain_size})={summary['chains_evaluated']}"
    )
    print(
        f"Pairs: candidates={summary['candidate_pairs']} "
        f"matched@{args.threshold:.2f}={summary['matched_pairs']}"
    )
    print(
        "Timing (s): "
        f"build={summary['timing_seconds']['build_signatures']:.3f} "
        f"compare={summary['timing_seconds']['compare_chains']:.3f} "
        f"total={summary['timing_seconds']['total']:.3f}"
    )
    print(
        f"Normalization: numbers={args.normalize_numbers} "
        f"hex={args.normalize_hex}"
    )
    print(f"Wrote JSON report to {args.output_json}")
    print(
        "Note: this uses the exploratory Python extractor, not the production "
        "Go crawler-trap benchmark path."
    )
    if examples:
        print("\nTop matches:")
        for item in examples[: min(5, len(examples))]:
            print(
                f"- score={item['jaccard_estimate']:.4f} chain={item['chain_key']} "
                f"left={item['left_url']} right={item['right_url']}"
            )


if __name__ == "__main__":
    main()
