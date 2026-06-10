"""
Stage 2: HTML Parsing
Parse legal document HTML into article-level records.

Input: content config + stage1_sme_docs.parquet
Output: stage2_articles.parquet, stage2_parse_failures.jsonl
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

# Optional: only needed for loading from HuggingFace
try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


# Regex patterns for Vietnamese legal document structure.
# Legal texts are not perfectly standardized: headings may use roman numerals,
# Vietnamese ordinals, dots, colons, dashes, or no separator at all.
RE_PHAN = re.compile(r"^\s*(Phần(?:\s+thứ)?\s+[\dIVXLCDM\w]+.*)$", re.MULTILINE | re.IGNORECASE)
RE_CHUONG = re.compile(r"^\s*(Chương\s+[\dIVXLCDM\w]+.*)$", re.MULTILINE | re.IGNORECASE)
RE_MUC = re.compile(r"^\s*(Mục\s+[\dIVXLCDM\w]+.*)$", re.MULTILINE | re.IGNORECASE)
RE_DIEU = re.compile(
    r"^\s*Điều\s+(\d+)([a-zđ]?)\s*(?:[\.:\-–—)]\s*)?(.*)$",
    re.MULTILINE | re.IGNORECASE
)
RE_KHOAN = re.compile(r"^\s*(\d+)\.\s+(.*)$", re.MULTILINE)
RE_DIEM = re.compile(r"^\s*([a-zđ])\)\s+(.*)$", re.MULTILINE)
LAW_LIKE_TITLE_RE = re.compile(r"\b(Luật|Bộ luật|Nghị định)\b", re.IGNORECASE)


def clean_html_to_text(html: str) -> str:
    """
    Parse HTML and extract clean text.
    
    Args:
        html: Raw HTML content
        
    Returns:
        Cleaned text with normalized whitespace
    """
    # Parse HTML with built-in html.parser (no external dependencies)
    soup = BeautifulSoup(html, "html.parser")
    
    # Remove script and style tags
    for tag in soup(["script", "style"]):
        tag.decompose()
    
    # Extract text with newline separators
    text = soup.get_text("\n", strip=True)
    
    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    
    # Deduplicate consecutive identical lines (nested-tag artifact)
    lines = text.split("\n")
    deduped_lines = []
    prev_line = None
    for line in lines:
        if line != prev_line:
            deduped_lines.append(line)
        prev_line = line
    
    return "\n".join(deduped_lines)


def locate_dieu_boundaries(text: str) -> List[Tuple[int, str, str, str, int]]:
    """
    Locate all Điều boundaries in text.
    
    Returns:
        List of (start_pos, dieu_num, dieu_suffix, dieu_title, title_end_pos)
    """
    boundaries = []
    
    for match in RE_DIEU.finditer(text):
        start_pos = match.start()
        dieu_num = match.group(1)
        dieu_suffix = match.group(2) or ""
        dieu_title = match.group(3).strip()
        title_end_pos = match.end()
        
        boundaries.append((start_pos, dieu_num, dieu_suffix, dieu_title, title_end_pos))
    
    return boundaries


def assign_hierarchy_context(text: str, boundaries: List[Tuple]) -> List[Dict]:
    """
    Assign hierarchy context (Phần, Chương, Mục) to each Điều.
    
    Args:
        text: Full document text
        boundaries: List of Điều boundaries from locate_dieu_boundaries
    
    Returns:
        List of dicts with hierarchy context for each Điều
    """
    # Find all hierarchy markers
    phan_matches = [(m.start(), m.group(1).strip()) for m in RE_PHAN.finditer(text)]
    chuong_matches = [(m.start(), m.group(1).strip()) for m in RE_CHUONG.finditer(text)]
    muc_matches = [(m.start(), m.group(1).strip()) for m in RE_MUC.finditer(text)]
    
    articles = []
    
    for i, (start_pos, dieu_num, dieu_suffix, dieu_title, title_end_pos) in enumerate(boundaries):
        # Determine end position (start of next Điều or end of text)
        if i + 1 < len(boundaries):
            end_pos = boundaries[i + 1][0]
        else:
            end_pos = len(text)
        
        # Find most recent hierarchy markers before this Điều
        phan = ""
        chuong = ""
        muc = ""
        
        for pos, label in phan_matches:
            if pos < start_pos:
                phan = label
            else:
                break
        
        for pos, label in chuong_matches:
            if pos < start_pos:
                chuong = label
            else:
                break
        
        for pos, label in muc_matches:
            if pos < start_pos:
                muc = label
            else:
                break
        
        # Extract content (after title line, before next Điều)
        noi_dung = text[title_end_pos:end_pos].strip()
        
        articles.append({
            "dieu_num": dieu_num,
            "dieu_suffix": dieu_suffix,
            "dieu_title": dieu_title,
            "phan": phan,
            "chuong": chuong,
            "muc": muc,
            "noi_dung": noi_dung,
            "start_char": start_pos,
            "end_char": end_pos
        })
    
    return articles


def is_law_like_title(title: str) -> bool:
    return bool(LAW_LIKE_TITLE_RE.search(title or ""))


def build_article_records(doc_id, articles: List[Dict], metadata: Dict) -> List[Dict]:
    """Build normalized output rows from parsed article dictionaries."""
    records = []
    for art in articles:
        noi_dung = (art.get("noi_dung") or "").strip()
        if len(noi_dung) < 30:
            continue

        dieu_num = str(art["dieu_num"]).strip()
        dieu_suffix = str(art.get("dieu_suffix") or "").strip()
        dieu_so = f"Điều {dieu_num}{dieu_suffix}".strip()
        doc_uid = f"{metadata['law_id']}|{metadata['ten_van_ban']}|{dieu_so}"

        records.append({
            "doc_id": doc_id,
            "law_id": metadata["law_id"],
            "ten_van_ban": metadata["ten_van_ban"],
            "loai_van_ban": metadata["loai_van_ban"],
            "ngay_ban_hanh": metadata["ngay_ban_hanh"],
            "nganh": metadata["nganh"],
            "linh_vuc": metadata["linh_vuc"],
            "phan": art.get("phan", ""),
            "chuong": art.get("chuong", ""),
            "muc": art.get("muc", ""),
            "dieu_so": dieu_so,
            "dieu_ten": art.get("dieu_title", ""),
            "noi_dung": noi_dung,
            "start_char": art.get("start_char", 0),
            "end_char": art.get("end_char", len(noi_dung)),
            "doc_uid": doc_uid
        })

    return records


def build_fallback_article(doc_id, text: str, metadata: Dict) -> Tuple[List[Dict], Optional[Dict]]:
    """
    Preserve non-standard legal documents as one virtual article.

    The thesis notes that legal document indexes are not guaranteed to exist or
    follow one standard format. For RAG, dropping such documents loses valuable
    regulatory content, so we keep the cleaned body as a document-level article.
    """
    text = text.strip()
    if len(text) < 30:
        failure = {
            "doc_id": doc_id,
            "title": metadata.get("title", ""),
            "num_dieu": 0,
            "reason": "text_too_short_after_cleaning",
            "text_preview": text[:500]
        }
        return [], failure

    article = {
        "dieu_num": "VB",
        "dieu_suffix": "",
        "dieu_title": metadata.get("title") or metadata.get("ten_van_ban") or "Toàn văn văn bản",
        "phan": "",
        "chuong": "",
        "muc": "",
        "noi_dung": text,
        "start_char": 0,
        "end_char": len(text)
    }
    return build_article_records(doc_id, [article], metadata), None


def parse_document(doc_id, html: str, metadata: Dict) -> Tuple[List[Dict], Optional[Dict]]:
    """
    Parse a single legal document HTML into articles.
    
    Args:
        doc_id: Document ID
        html: Raw HTML content
        metadata: Document metadata dict
        Returns:
            (list of article dicts, failure record or None)
    """
    # Clean HTML to text
    text = clean_html_to_text(html)
    
    # Locate Điều boundaries
    boundaries = locate_dieu_boundaries(text)
    
    # Many decisions/directives do not contain formal Điều markers. Keep them
    # as document-level fallback records instead of losing almost half the corpus.
    if len(boundaries) == 0:
        records, failure = build_fallback_article(doc_id, text, metadata)
        if failure:
            return [], failure
        if is_law_like_title(metadata.get("title", "")):
            failure = {
                "doc_id": doc_id,
                "title": metadata.get("title", ""),
                "num_dieu": 0,
                "reason": "zero_dieu_law_like_doc",
                "text_preview": text[:500]
            }
            return records, failure
        return records, None
    
    # Assign hierarchy and build article records
    articles = assign_hierarchy_context(text, boundaries)
    records = build_article_records(doc_id, articles, metadata)

    # If every detected article was too short, preserve the whole document.
    if not records:
        records, failure = build_fallback_article(doc_id, text, metadata)
        if failure:
            return [], failure
        # Policy: drop zero-parsable law-like documents (they're law-like
        # but parsing produced no article rows). We log the failure so
        # reviewers can inspect, but do not emit fallback rows for this
        # category to avoid polluting the article corpus with noisy whole-
        # document records. Fallback rows for non-law-like docs are kept.
        if is_law_like_title(metadata.get("title", "")):
            failure = {
                "doc_id": doc_id,
                "title": metadata.get("title", ""),
                "num_dieu": len(boundaries),
                "reason": "zero_parsable_dieu_law_like_doc",
                "text_preview": text[:500]
            }
            # Drop records in this case (return no article rows)
            return [], failure
        return records, None

    if len(records) == 1 and is_law_like_title(metadata.get("title", "")):
        failure = {
            "doc_id": doc_id,
            "title": metadata.get("title", ""),
            "num_dieu": 1,
            "reason": "single_dieu_law_like_doc",
            "text_preview": text[:500]
        }
        return records, failure

    return records, None


def main(
    stage1_path: Optional[str] = None,
    output_articles_path: Optional[str] = None,
    output_failures_path: Optional[str] = None,
    content_source: str = "local",
    content_path: Optional[str] = None
):
    """
    Main Stage 2 pipeline: parse HTML documents into articles.
    
    Args:
        stage1_path: Path to stage1_sme_docs.parquet
        output_articles_path: Output path for parsed articles
        output_failures_path: Output path for parse failures log
        content_source: "huggingface" or "local"
        content_path: Path to local content file (if content_source="local")
    """
    # Resolve project root (src/data -> src -> Road2AI_ApplePie)
    project_root = Path(__file__).resolve().parent.parent.parent

    def resolve_path(path_value: Optional[str], default_path: Path):
        if not path_value:
            return default_path
        if "://" in str(path_value):
            return str(path_value)
        path_obj = Path(path_value)
        if path_obj.is_absolute():
            return path_obj
        return project_root / path_obj

    stage1_path = resolve_path(stage1_path, project_root / "data" / "stage1_sme_docs.parquet")
    output_articles_path = resolve_path(output_articles_path, project_root / "data" / "stage2_articles.parquet")
    output_failures_path = resolve_path(output_failures_path, project_root / "data" / "stage2_parse_failures.jsonl")
    # Resolve content path. If user did not provide --content-path and
    # requested local content, attempt to auto-discover a common local
    # filename in the repository data folder. Otherwise fall back to the
    # default HuggingFace parquet URL.
    default_hf = "hf://datasets/th1nhng0/vietnamese-legal-documents/data/content.parquet"
    if content_path:
        content_path_resolved = resolve_path(content_path, default_hf)
    else:
        # Try common local filenames
        candidate_names = [
            project_root / "data" / "content.parquet",
            project_root / "data" / "content.jsonl",
            project_root / "data" / "content.json",
            project_root / "data" / "content.jsonl.gz",
        ]
        found = None
        for cand in candidate_names:
            if isinstance(cand, Path) and cand.exists():
                found = cand
                break

        if found and content_source == "local":
            content_path_resolved = found
            print(f"  Auto-discovered local content file: {content_path_resolved}")
        else:
            # No local content found — default to HF URL (may raise later)
            content_path_resolved = default_hf

    print("Stage 2: HTML Parsing")
    print("=" * 50)
    
    # Load Stage 1 documents
    print(f"\n[1/4] Loading Stage 1 documents from {stage1_path}")
    sme_docs = pd.read_parquet(stage1_path)
    sme_docs["id"] = sme_docs["id"].astype(str)
    before_dedup = len(sme_docs)
    sme_docs = sme_docs.drop_duplicates(subset=["id"], keep="first")
    print(f"  Loaded {before_dedup} SME documents ({len(sme_docs)} unique ids)")
    
    # Create metadata lookup
    metadata_lookup = {}
    for _, row in sme_docs.iterrows():
        metadata_lookup[row["id"]] = {
            "law_id": row["law_id"],
            "ten_van_ban": row["ten_van_ban"],
            "loai_van_ban": row["loai_van_ban"],
            "ngay_ban_hanh": row["ngay_ban_hanh"],
            "nganh": row["nganh"],
            "linh_vuc": row["linh_vuc"],
            "title": row["title"]
        }
    
    # Load content dataset
    print(f"\n[2/4] Loading content dataset from {content_source}")
    if content_source == "huggingface":
        if not HF_AVAILABLE:
            raise ImportError(
                "The 'datasets' library is required for HuggingFace data loading. "
                "Install it with: pip install datasets\n"
                "Alternatively, use --content-source=local with a local content file."
            )
        content_ds = load_dataset(
            "th1nhng0/vietnamese-legal-documents",
            "content",
            split="data",
            streaming=True
        )
        print("  Loaded content dataset (streaming)")
    else:
        # Load from local or hf:// parquet/jsonl file
        content_path_str = str(content_path_resolved)
        if content_path_str.endswith(".parquet"):
            content_df = pd.read_parquet(content_path_resolved)
        else:
            content_df = pd.read_json(content_path_resolved, lines=True)
        content_df["id"] = content_df["id"].astype(str)
        before_content_dedup = len(content_df)
        content_df = content_df.drop_duplicates(subset=["id"], keep="first")
        content_ds = content_df.to_dict("records")
        print(
            f"  Loaded {before_content_dedup} content records from local file "
            f"({len(content_ds)} unique ids)"
        )
    
    # Parse documents
    print(f"\n[3/4] Parsing HTML documents")
    all_articles = []
    all_failures = []
    
    sme_ids = set(sme_docs["id"].astype(str).tolist())
    processed_count = 0
    
    for record in tqdm(content_ds, desc="Parsing documents"):
        doc_id = str(record["id"])
        
        # Skip if not in SME scope
        if doc_id not in sme_ids:
            continue
        
        html = record["content_html"]
        metadata = metadata_lookup[doc_id]
        
        # Parse document
        articles, failure = parse_document(doc_id, html, metadata)
        
        all_articles.extend(articles)
        if failure:
            all_failures.append(failure)
        
        processed_count += 1
        
        # Progress check
        if processed_count % 100 == 0:
            print(f"  Processed {processed_count}/{len(sme_ids)} documents, "
                  f"{len(all_articles)} articles extracted")
    
    print(f"\n  Total parsed: {processed_count} documents")
    print(f"  Total articles: {len(all_articles)}")
    print(f"  Parse failures: {len(all_failures)}")
    
    # Save articles
    print(f"\n[4/4] Saving outputs")
    articles_df = pd.DataFrame(all_articles)
    before_article_dedup = len(articles_df)

    if not articles_df.empty:
        articles_df = articles_df.sort_values(
            by="noi_dung",
            key=lambda s: s.str.len(),
            ascending=False
        )
        articles_df = articles_df.drop_duplicates(subset=["doc_id", "dieu_so"], keep="first")
        if len(articles_df) != before_article_dedup:
            print(
                f"  Removed {before_article_dedup - len(articles_df)} duplicate "
                "article rows by (doc_id, dieu_so)"
            )
    
    # Quality checks
    print("  Running quality checks...")
    if not articles_df.empty:
        duplicate_uid_count = int(articles_df.duplicated(subset=["doc_uid"]).sum())
        if duplicate_uid_count > 0:
            raise ValueError(
                f"Found {duplicate_uid_count} duplicate doc_uid values after deduplication. "
                "The output must contain unique doc_uid rows."
            )

    # Check for duplicate (doc_id, dieu_so)
    dup_check = articles_df.groupby(["doc_id", "dieu_so"]).size() if not articles_df.empty else pd.Series(dtype=int)
    duplicates = dup_check[dup_check > 1]
    if len(duplicates) > 0:
        print(f"  WARNING: Found {len(duplicates)} duplicate (doc_id, dieu_so) pairs")
    
    # Save to parquet
    Path(output_articles_path).parent.mkdir(parents=True, exist_ok=True)
    articles_df.to_parquet(output_articles_path, index=False)
    print(f"  Saved articles to {output_articles_path}")
    
    # Save failures
    if all_failures:
        Path(output_failures_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_failures_path, "w", encoding="utf-8") as f:
            for failure in all_failures:
                f.write(json.dumps(failure, ensure_ascii=False) + "\n")
        print(f"  Saved parse failures to {output_failures_path}")
    
    # Final statistics
    print(f"\n{'=' * 50}")
    print("Stage 2 Complete!")
    print(f"  Articles: {len(articles_df)}")
    print(f"  Unique documents: {articles_df['doc_id'].nunique() if not articles_df.empty else 0}")
    print(f"  Parse failures: {len(all_failures)}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Stage 2: Parse HTML to articles")
    parser.add_argument(
        "--stage1",
        default=None,
        help="Path to Stage 1 output"
    )
    parser.add_argument(
        "--output-articles",
        default=None,
        help="Output path for articles"
    )
    parser.add_argument(
        "--output-failures",
        default=None,
        help="Output path for parse failures"
    )
    parser.add_argument(
        "--content-source",
        choices=["huggingface", "local"],
        default="local",
        help="Content source: huggingface or local"
    )
    parser.add_argument(
        "--content-path",
        default=None,
        help="Path to content parquet/jsonl file. Defaults to the HuggingFace parquet URL."
    )
    
    args = parser.parse_args()
    
    main(
        stage1_path=args.stage1,
        output_articles_path=args.output_articles,
        output_failures_path=args.output_failures,
        content_source=args.content_source,
        content_path=args.content_path
    )