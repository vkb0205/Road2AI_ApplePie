"""
Stage 3: Chunking
Split Stage 2 article-level records into overlapping chunks for retrieval and summarization.

Input: `stage2_articles.parquet` (optionally merged with `stage2_manual_fixes.json`/`.jsonl`).
Output: `stage3_chunks.parquet`.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from transformers import AutoTokenizer
except ImportError as exc:
    raise ImportError(
        "transformers is required to run src/data/stage3_chunking.py. "
        "Install it with `pip install transformers` or use the repo requirements file.`"
    ) from exc

KHOAN_SPLIT_RE = re.compile(r"(?=^\s*\d+\.\s)", re.MULTILINE)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
CHUNK_EXTRA_COLUMNS = ["breadcrumb", "chunk_id", "part_idx", "chunk_text"]


def resolve_path(path: Optional[str], default_path: Path) -> Path:
    if path:
        return Path(path)
    return default_path


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def load_manual_fixes(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    if path.suffix.lower() == ".jsonl":
        data = load_jsonl(path)
    else:
        data = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(data, dict):
        data = [data]

    if not isinstance(data, list):
        raise ValueError(
            f"Unsupported manual fixes format: {path}. Expected JSON array, object, or JSONL lines."
        )

    return pd.DataFrame(data)


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and value and os.environ.get(key) is None:
                os.environ[key] = value


def build_breadcrumb(record: Dict) -> str:
    parts = [record.get("loai_van_ban", "").strip(), record.get("ten_van_ban", "").strip()]
    for field in ("phan", "chuong", "muc"):
        value = str(record.get(field, "") or "").strip()
        if value:
            parts.append(value)

    dieu_so = str(record.get("dieu_so", "") or "").strip()
    dieu_ten = str(record.get("dieu_ten", "") or "").strip()
    if dieu_so:
        parts.append(f"{dieu_so}. {dieu_ten}" if dieu_ten else dieu_so)
    return " > ".join([p for p in parts if p])


def count_tokens(text: str, tokenizer: AutoTokenizer) -> int:
    if not text:
        return 0
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def split_text_to_token_blocks(text: str, tokenizer: AutoTokenizer, max_tokens: int) -> List[str]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return []

    blocks = []
    for start in range(0, len(token_ids), max_tokens):
        block_ids = token_ids[start : start + max_tokens]
        block_text = tokenizer.decode(block_ids, clean_up_tokenization_spaces=True, skip_special_tokens=True).strip()
        if block_text:
            blocks.append(block_text)
    return blocks


def split_long_piece(piece: str, tokenizer: AutoTokenizer, max_tokens: int) -> List[str]:
    piece = piece.strip()
    if not piece:
        return []

    if count_tokens(piece, tokenizer) <= max_tokens:
        return [piece]

    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(piece) if s.strip()]
    if not sentences:
        return split_text_to_token_blocks(piece, tokenizer, max_tokens)

    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = count_tokens(sentence, tokenizer)
        if sentence_tokens > max_tokens:
            if current:
                chunks.append("\n".join(current).strip())
                current = []
                current_tokens = 0
            chunks.extend(split_text_to_token_blocks(sentence, tokenizer, max_tokens))
            continue

        if current_tokens + sentence_tokens <= max_tokens:
            current.append(sentence)
            current_tokens += sentence_tokens
            continue

        if current:
            chunks.append("\n".join(current).strip())
        current = [sentence]
        current_tokens = sentence_tokens

    if current:
        chunks.append("\n".join(current).strip())

    return chunks


def extract_khoan_pieces(noi_dung: str) -> List[str]:
    pieces = [piece.strip() for piece in KHOAN_SPLIT_RE.split(noi_dung) if piece.strip()]
    if not pieces:
        return [noi_dung.strip()] if noi_dung.strip() else []
    return pieces


def join_chunk_pieces(pieces: List[str]) -> str:
    return "\n\n".join([piece.strip() for piece in pieces if piece.strip()]).strip()


def load_tokenizer(tokenizer_name: str, hf_token: Optional[str] = None):
    try:
        if hf_token:
            return AutoTokenizer.from_pretrained(
                tokenizer_name,
                token=hf_token,
                trust_remote_code=True,
            )
        return AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as err:
        print(f"WARNING: Failed to load tokenizer '{tokenizer_name}': {err}")
        fallback_name = "gpt2"
        print(f"WARNING: Falling back to tokenizer '{fallback_name}' for token counting.")
        try:
            return AutoTokenizer.from_pretrained(fallback_name)
        except Exception as fallback_err:
            raise RuntimeError(
                f"Unable to load tokenizer '{tokenizer_name}' or fallback '{fallback_name}'. "
                "If you have a local tokenizer, pass --tokenizer <path> or install a compatible tokenizer package."
            ) from fallback_err


def make_chunks_for_article(record: Dict, tokenizer: AutoTokenizer, max_tokens: int) -> List[Dict]:

    doc_uid = str(record.get("doc_uid", "") or "").strip()
    if not doc_uid:
        raise ValueError("Each Stage 2 record must contain a non-empty doc_uid.")

    breadcrumb = build_breadcrumb(record)
    article_text = str(record.get("noi_dung", "") or "").strip()
    if not article_text:
        return []

    pieces: List[str] = []
    for piece in extract_khoan_pieces(article_text):
        if not piece:
            continue
        if count_tokens(piece, tokenizer) > max_tokens:
            pieces.extend(split_long_piece(piece, tokenizer, max_tokens))
        else:
            pieces.append(piece)

    if not pieces:
        return []

    chunks: List[List[str]] = []
    current: List[str] = []
    current_tokens = 0

    for piece in pieces:
        piece_tokens = count_tokens(piece, tokenizer)
        if not current:
            current = [piece]
            current_tokens = piece_tokens
            continue

        if current_tokens + piece_tokens <= max_tokens:
            current.append(piece)
            current_tokens += piece_tokens
            continue

        chunks.append(current)
        current = [piece]
        current_tokens = piece_tokens

    if current:
        chunks.append(current)

    for idx in range(1, len(chunks)):
        overlap_piece = chunks[idx - 1][-1]
        candidate = [overlap_piece] + chunks[idx]
        candidate_text = join_chunk_pieces(candidate)
        if count_tokens(candidate_text, tokenizer) <= max_tokens:
            chunks[idx] = candidate

    chunk_records: List[Dict] = []
    for part_idx, chunk_pieces in enumerate(chunks):
        chunk_text = join_chunk_pieces(chunk_pieces)
        if not chunk_text:
            continue

        row = dict(record)
        row["breadcrumb"] = breadcrumb
        row["chunk_id"] = f"{doc_uid}#{part_idx}"
        row["part_idx"] = part_idx
        row["chunk_text"] = f"{breadcrumb}\n{chunk_text}" if breadcrumb else chunk_text
        chunk_records.append(row)

    return chunk_records


def load_stage2_articles(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Stage 2 input not found: {path}")
    return pd.read_parquet(path)


def merge_manual_fixes(stage2_df: pd.DataFrame, manual_df: pd.DataFrame) -> pd.DataFrame:
    if manual_df.empty:
        return stage2_df

    merged = pd.concat([stage2_df, manual_df], ignore_index=True)
    if "doc_uid" in merged.columns:
        merged = merged.drop_duplicates(subset=["doc_uid"], keep="last")
    return merged


def get_output_columns(stage2_df: pd.DataFrame) -> List[str]:
    output_columns = [col for col in stage2_df.columns]
    output_columns.extend([col for col in CHUNK_EXTRA_COLUMNS if col not in output_columns])
    return output_columns


def write_chunks_batched(
    stage2_df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    max_tokens: int,
    output_path: Path,
    batch_size: int,
) -> Tuple[int, int]:
    if batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0.")

    output_columns = get_output_columns(stage2_df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output_path = output_path.with_name(f".{output_path.name}.tmp")
    if temp_output_path.exists():
        temp_output_path.unlink()

    writer: Optional[pq.ParquetWriter] = None
    schema: Optional[pa.Schema] = None
    total_chunks = 0
    doc_ids_with_chunks: Set[str] = set()

    try:
        total_articles = len(stage2_df)
        for start in range(0, total_articles, batch_size):
            end = min(start + batch_size, total_articles)
            article_records = stage2_df.iloc[start:end].to_dict(orient="records")

            chunk_rows: List[Dict] = []
            for record in article_records:
                chunk_rows.extend(make_chunks_for_article(record, tokenizer, max_tokens))

            if not chunk_rows:
                print(f"  Processed {end:,}/{total_articles:,} articles, no chunks in this batch.")
                continue

            batch_df = pd.DataFrame(chunk_rows)
            batch_df = batch_df.reindex(columns=output_columns)
            table = pa.Table.from_pandas(batch_df, schema=schema, preserve_index=False)

            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(temp_output_path, schema=schema, compression="snappy")

            writer.write_table(table)
            total_chunks += len(batch_df)
            if "doc_id" in batch_df.columns:
                doc_ids_with_chunks.update(batch_df["doc_id"].dropna().astype(str).unique())

            print(
                f"  Processed {end:,}/{total_articles:,} articles, "
                f"wrote {total_chunks:,} chunks so far."
            )
    except Exception:
        if writer is not None:
            writer.close()
        if temp_output_path.exists():
            temp_output_path.unlink()
        raise

    if writer is not None:
        writer.close()

    if total_chunks == 0:
        if temp_output_path.exists():
            temp_output_path.unlink()
        raise ValueError("No stage3 chunks were generated. Check Stage 2 input and article text fields.")

    temp_output_path.replace(output_path)
    return total_chunks, len(doc_ids_with_chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3: Chunk Stage 2 articles into overlapping chunks")
    parser.add_argument("--stage2-path", default=None, help="Path to stage2_articles.parquet")
    parser.add_argument("--manual-fixes-path", default=None, help="Optional path to stage2_manual_fixes.json or .jsonl")
    parser.add_argument("--output-path", default=None, help="Output path for stage3_chunks.parquet")
    parser.add_argument("--tokenizer", default="google/gemma-3-12b-it", help="Tokenizer model for token counting")
    parser.add_argument("--hf-token", default=None, help="Hugging Face token for gated repo access")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum tokens per chunk")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Number of Stage 2 article rows to chunk and write per parquet batch",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")

    stage2_path = resolve_path(args.stage2_path, project_root / "data" / "stage2_articles.parquet")
    output_path = resolve_path(args.output_path, project_root / "data" / "stage3_chunks.parquet")
    manual_fixes_path = Path(args.manual_fixes_path) if args.manual_fixes_path else None

    print(f"Loading Stage 2 articles from {stage2_path}")
    stage2_df = load_stage2_articles(stage2_path)
    print(f"Stage 2 rows loaded: {len(stage2_df):,}")

    if manual_fixes_path:
        print(f"Loading manual fixes from {manual_fixes_path}")
        manual_df = load_manual_fixes(manual_fixes_path)
        print(f"Manual fix records loaded: {len(manual_df):,}")
        stage2_df = merge_manual_fixes(stage2_df, manual_df)
        print(f"Merged Stage 2 rows after manual fixes: {len(stage2_df):,}")

    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if hf_token:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
        os.environ["HF_TOKEN"] = hf_token
        if args.hf_token:
            print("Using Hugging Face token from --hf-token for gated repo access.")
        else:
            print("Using existing Hugging Face token from environment.")
    else:
        print("No Hugging Face token found in --hf-token, HF_TOKEN, or HUGGINGFACE_HUB_TOKEN.")

    tokenizer = load_tokenizer(args.tokenizer, hf_token=hf_token)

    total_chunks, unique_documents = write_chunks_batched(
        stage2_df=stage2_df,
        tokenizer=tokenizer,
        max_tokens=args.max_tokens,
        output_path=output_path,
        batch_size=args.batch_size,
    )

    print(f"Wrote {total_chunks:,} chunks to {output_path}")
    print(f"Unique documents: {unique_documents:,}")
    print(f"Chunk IDs generated: {total_chunks:,}")


if __name__ == "__main__":
    main()
