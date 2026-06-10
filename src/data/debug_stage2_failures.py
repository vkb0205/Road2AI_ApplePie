import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    load_dataset = None
    HF_AVAILABLE = False

RE_DIEU = re.compile(r"^\s*Điều\s+(\d+)([a-zđ]?)\s*(?:[\.:\-–—)]\s*)?(.*)$", re.MULTILINE | re.IGNORECASE)


def clean_html_to_text(html: str) -> str:
    soup = BeautifulSoup(html or "", "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text("\n", strip=True)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    lines = text.split("\n")
    deduped_lines = []
    prev = None
    for line in lines:
        if line != prev:
            deduped_lines.append(line)
        prev = line
    return "\n".join(deduped_lines)


def locate_dieu_boundaries(text: str):
    boundaries = []
    for match in RE_DIEU.finditer(text or ""):
        boundaries.append(
            {
                "start": match.start(),
                "end": match.end(),
                "dieu_num": match.group(1),
                "dieu_suffix": match.group(2) or "",
                "dieu_title": match.group(3).strip(),
                "line": match.group(0).strip(),
            }
        )
    return boundaries


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_content(content_path: Path):
    if not content_path.exists():
        raise FileNotFoundError(
            f"content_path does not exist: {content_path}. "
            "Please provide the actual path to the raw content dataset file, not a placeholder."
        )
    path = str(content_path)
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    elif path.endswith(".jsonl") or path.endswith(".ndjson"):
        df = pd.read_json(path, lines=True)
    else:
        raise ValueError("content_path must be .parquet or .jsonl")
    df["id"] = df["id"].astype(str)
    return df.set_index("id")


def load_content_from_hf(doc_ids):
    if not HF_AVAILABLE:
        raise ImportError(
            "The Hugging Face datasets library is not installed. "
            "Install it with: pip install datasets"
        )
    hf_ds = load_dataset(
        "th1nhng0/vietnamese-legal-documents",
        "content",
        split="data",
        streaming=True,
    )
    rows = []
    doc_ids = {str(doc_id) for doc_id in doc_ids}
    for record in hf_ds:
        rec_id = str(record["id"])
        if rec_id in doc_ids:
            record["id"] = rec_id
            rows.append(record)
            if len(rows) == len(doc_ids):
                break
    if not rows:
        return pd.DataFrame(columns=["id"]).set_index("id")
    df = pd.DataFrame(rows)
    df["id"] = df["id"].astype(str)
    return df.set_index("id")


def sample_doc_ids(failures, sample_size):
    # Pick a spread over failure reasons, then fill with random remaining items.
    by_reason = defaultdict(list)
    for rec in failures:
        by_reason[rec["reason"]].append(rec["doc_id"])

    selected = []
    reasons = sorted(by_reason.keys(), key=lambda r: (-len(by_reason[r]), r))
    for reason in reasons:
        if len(selected) >= sample_size:
            break
        if by_reason[reason]:
            selected.append(by_reason[reason][0])
    remaining = [rec["doc_id"] for rec in failures if rec["doc_id"] not in selected]
    random.shuffle(remaining)
    for doc_id in remaining:
        if len(selected) >= sample_size:
            break
        selected.append(doc_id)
    return selected


def inspect_document(doc_id, failure, content_record):
    print("=" * 120)
    print(f"doc_id: {doc_id}")
    print(f"title : {failure.get('title', '')}")
    print(f"reason: {failure.get('reason')}")
    print(f"num_dieu: {failure.get('num_dieu')}")
    print(f"text_preview length: {len(failure.get('text_preview', ''))}")
    print()

    if content_record is None:
        print("[WARN] content record not found for this doc_id")
        return

    html = content_record.get("content_html") or content_record.get("content") or ""
    text = clean_html_to_text(html)
    print(f"clean_html_to_text length: {len(text)}")
    print(f"clean_text first 500 chars:\n{text[:500]!r}\n")

    boundaries = locate_dieu_boundaries(text)
    print(f"detected Điều boundaries: {len(boundaries)}")
    for idx, b in enumerate(boundaries[:10], start=1):
        print(f" {idx}. {b['line']} (start={b['start']}, end={b['end']})")

    if not boundaries and text:
        # show candidate first 10 lines too
        print("\n[INFO] no Điều matches found; showing first 20 non-empty lines from cleaned text")
        lines = [ln for ln in text.split("\n") if ln.strip()]
        for idx, line in enumerate(lines[:20], start=1):
            print(f" {idx}. {line}")
    print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect Stage 2 parse failures in detail.")
    repo_root = Path(__file__).resolve().parent.parent.parent
    parser.add_argument(
        "--failure-file",
        default=repo_root / "data" / "stage2_parse_failures.jsonl",
        type=Path,
        help="Path to stage2_parse_failures.jsonl",
    )
    parser.add_argument(
        "--content-source",
        choices=["local", "huggingface"],
        default="local",
        help="Choose where to load content from: local file or Hugging Face dataset.",
    )
    parser.add_argument(
        "--content-path",
        default=None,
        type=Path,
        help="Path to local content parquet/jsonl file",
    )
    parser.add_argument(
        "--article-file",
        default=repo_root / "data" / "stage2_articles.parquet",
        type=Path,
        help="Optional stage2_articles.parquet for cross-checking whether failure docs still have articles",
    )
    parser.add_argument("--sample-size", type=int, default=12, help="Number of failure doc_ids to inspect")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    random.seed(args.seed)

    failures = list(load_jsonl(args.failure_file))
    print(f"Loaded {len(failures)} failure records from {args.failure_file}")
    print("Reasons summary:")
    reason_counts = defaultdict(int)
    for rec in failures:
        reason_counts[rec.get("reason", "")] += 1
    for reason, count in sorted(reason_counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {reason}: {count}")
    print()

    content_df = None
    failure_ids = {rec["doc_id"] for rec in failures}
    if args.content_source == "huggingface":
        try:
            content_df = load_content_from_hf(failure_ids)
            print(
                f"Loaded content records for {len(content_df)} failure doc_ids from Hugging Face"
            )
        except ImportError as exc:
            print(f"[ERROR] {exc}")
            raise
    elif args.content_path:
        try:
            content_df = load_content(args.content_path)
            print(f"Loaded content records from {args.content_path} ({len(content_df)} rows)")
        except FileNotFoundError as exc:
            print(f"[ERROR] {exc}")
            print("Run again with the correct local dataset file path, e.g. --content-path D:/path/to/content.parquet")
            raise
    else:
        print("[WARN] --content-path not provided; content lookup will be skipped")
    print()

    article_doc_ids = set()
    if args.article_file and args.article_file.exists():
        articles = pd.read_parquet(args.article_file)
        article_doc_ids = set(articles["doc_id"].astype(str).tolist())
        print(f"Loaded article doc_ids from {args.article_file}: {len(article_doc_ids)} unique docs")
    elif args.article_file:
        print(f"[WARN] article file {args.article_file} does not exist")
    print()

    selected_ids = sample_doc_ids(failures, args.sample_size)
    print(f"Selected {len(selected_ids)} failure doc_ids for detailed inspection:\n  {', '.join(selected_ids)}")
    print()

    failure_by_doc = {rec["doc_id"]: rec for rec in failures}
    for doc_id in selected_ids:
        failure = failure_by_doc[doc_id]
        content_record = content_df.loc[doc_id].to_dict() if content_df is not None and doc_id in content_df.index else None
        inspect_document(doc_id, failure, content_record)
        if content_record is not None:
            print(f"article present in output: {doc_id in article_doc_ids}")
        print()
    print("Done.")
