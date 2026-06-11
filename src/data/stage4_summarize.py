"""
Stage 4: Summary Injection
Generate two-field JSON summaries for each chunk using an OpenAI-compatible API.

Input: `stage3_chunks.parquet`.
Output: `stage4_enriched.parquet` (with short, key, enriched_text columns)
        + `summary_cache.jsonl` (resumable cache).

Model: any chat-completions model exposed by an OpenAI-compatible provider.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
from tqdm import tqdm


# Prompt template for Vietnamese legal summarization
SYSTEM_PROMPT = """Bạn là chuyên gia tóm tắt văn bản pháp luật Việt Nam."""

USER_PROMPT_TEMPLATE = """Hãy tạo HAI tóm tắt cho điều luật dưới đây:

[ĐIỀU LUẬT]
{chunk_text_truncated}

[YÊU CẦU]
1) SUM_SHORT: 1 câu duy nhất (≤30 từ) nêu CHỦ ĐỀ chính của điều.
2) SUM_KEY: 3-5 gạch đầu dòng nêu các Ý PHÁP LÝ then chốt
   (đối tượng áp dụng, nghĩa vụ, quyền, điều kiện, chế tài).

Trả về JSON: {{"short": "...", "key": ["...", "..."]}}"""

# Regex to extract JSON from model response
JSON_EXTRACT_RE = re.compile(r'\{[^{}]*"short"\s*:\s*"[^"]*"(?:,\s*"key"\s*:\s*\[[^\]]*\])?[^{}]*\}', re.DOTALL)
JSON_EXTRACT_RE_FALLBACK = re.compile(r'\{.*\}', re.DOTALL)


def load_dotenv(path: Path) -> None:
    """Load environment variables from .env file."""
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


def resolve_path(path: Optional[str], default_path: Path) -> Path:
    """Resolve path from argument or default."""
    if path:
        return Path(path)
    return default_path


def load_cache(cache_path: Path) -> Dict[str, Dict]:
    """Load existing cache, return dict mapping chunk_id to summary dict."""
    if not cache_path.exists():
        return {}
    
    cache = {}
    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                chunk_id = record.get("chunk_id", "")
                if chunk_id:
                    cache[chunk_id] = record
            except json.JSONDecodeError:
                continue
    return cache


def append_to_cache(cache_path: Path, chunk_id: str, short: str, key: List[str]) -> None:
    """Append a single cache entry to the JSONL file."""
    with cache_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"chunk_id": chunk_id, "short": short, "key": key}, ensure_ascii=False) + "\n")


def extract_json_from_response(text: str) -> Optional[Dict[str, Any]]:
    """Extract and parse JSON from model response."""
    # Try to find JSON object with short and key fields
    match = JSON_EXTRACT_RE.search(text)
    if match:
        try:
            result = json.loads(match.group())
            if "short" in result and "key" in result:
                return result
        except json.JSONDecodeError:
            pass
    
    # Fallback: try to find any JSON object
    match = JSON_EXTRACT_RE_FALLBACK.search(text)
    if match:
        try:
            result = json.loads(match.group())
            if "short" in result and "key" in result:
                return result
        except json.JSONDecodeError:
            pass
    
    return None


def build_enriched_text(short: str, key: List[str], chunk_text: str) -> str:
    """Build enriched text from summary and chunk body."""
    key_bullets = "\n".join([f"• {k}" for k in key]) if key else ""
    
    parts = []
    if short:
        parts.append(f"[TÓM TẮT] {short}")
    if key_bullets:
        parts.append(f"[Ý CHÍNH]\n{key_bullets}")
    parts.append(f"[NỘI DUNG ĐẦY ĐỦ]\n{chunk_text}")
    
    return "\n\n".join(parts)


class SummaryGenerator:
    """Wrapper for OpenAI-compatible chat-completions summarization."""
    
    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str,
        max_input_chars: int = 3000,
        max_new_tokens: int = 256,
        batch_size: int = 2,
        timeout: int = 120,
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.max_input_chars = max_input_chars
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.timeout = timeout
        self.chat_completions_url = f"{self.base_url}/chat/completions"
        print(f"Using OpenAI-compatible model: {model_name}")
        print(f"Using OpenAI-compatible base URL: {self.base_url}")
    
    def truncate_chunk_text(self, text: str) -> str:
        """Truncate chunk text to max_input_chars."""
        if len(text) <= self.max_input_chars:
            return text
        return text[:self.max_input_chars]
    
    def generate_summary(self, chunk_text: str) -> Tuple[str, List[str]]:
        """Generate summary for a single chunk. Returns (short, key_list)."""
        truncated = self.truncate_chunk_text(chunk_text)
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(chunk_text_truncated=truncated)},
        ]
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_new_tokens,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }
        request = Request(
            self.chat_completions_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        
        try:
            with urlopen(request, timeout=self.timeout) as response_handle:
                response_payload = json.loads(response_handle.read().decode("utf-8"))
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            print(f"OpenAI-compatible API HTTP error {exc.code}: {error_body}")
            return "", []
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            print(f"OpenAI-compatible API request failed: {exc}")
            return "", []
        
        try:
            response = response_payload["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            print(f"Unexpected OpenAI-compatible API response: {response_payload}")
            return "", []
        
        # Extract JSON
        parsed = extract_json_from_response(response)
        if parsed:
            short = parsed.get("short", "")
            key = parsed.get("key", [])
            if isinstance(key, str):
                key = [key]
            return short, key
        
        # Fallback: return empty values
        return "", []
    
    def generate_batch(self, chunk_texts: List[str]) -> List[Tuple[str, List[str]]]:
        """Generate summaries for a batch of chunks."""
        results = []
        for text in chunk_texts:
            short, key = self.generate_summary(text)
            results.append((short, key))
        return results
    
    def unload(self) -> None:
        """No-op for remote API compatibility with the previous local model interface."""
        return None


def normalize_key_value(key: Any) -> List[str]:
    """Normalize the key field from cache/output into a list of strings."""
    if isinstance(key, list):
        return key
    if isinstance(key, str):
        key = key.strip()
        if not key:
            return []
        try:
            parsed = json.loads(key)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
        return [key]
    return []


def load_existing_output(output_path: Path) -> pd.DataFrame:
    """Load an existing enriched output parquet file for resumable processing."""
    if not output_path.exists():
        return pd.DataFrame()
    try:
        output_df = pd.read_parquet(output_path)
    except Exception as exc:
        print(f"Could not read existing output at {output_path}; starting from scratch. Error: {exc}")
        return pd.DataFrame()
    if "chunk_id" not in output_df.columns:
        print(f"Existing output at {output_path} has no chunk_id column; starting from scratch.")
        return pd.DataFrame()
    return output_df


def get_resume_chunks(chunks_df: pd.DataFrame, existing_output_df: pd.DataFrame) -> pd.DataFrame:
    """Return chunks after the last chunk already present in the output file."""
    if existing_output_df.empty:
        return chunks_df

    processed_chunk_ids: Set[str] = set(existing_output_df["chunk_id"].dropna().astype(str))
    if not processed_chunk_ids:
        return chunks_df

    stage3_chunk_ids = chunks_df["chunk_id"].astype(str).tolist()
    last_processed_positions = [idx for idx, chunk_id in enumerate(stage3_chunk_ids) if chunk_id in processed_chunk_ids]
    if not last_processed_positions:
        return chunks_df

    resume_from_idx = max(last_processed_positions) + 1
    skipped_count = min(resume_from_idx, len(chunks_df))
    print(f"Resuming from existing output: {len(existing_output_df):,} rows found; "
          f"skipping first {skipped_count:,} Stage 3 chunks.")
    return chunks_df.iloc[resume_from_idx:]


def process_chunks(
    chunks_df: pd.DataFrame,
    generator: SummaryGenerator,
    cache: Dict[str, Dict],
    cache_path: Path,
    batch_size: int = 2,
) -> List[Dict]:
    """Process chunks, using cache for already-processed ones and showing progress."""
    results = []
    total = len(chunks_df)
    cached = 0
    failed = 0

    progress = tqdm(total=total, desc="Summarizing chunks", unit="chunk")
    for idx in range(0, total, batch_size):
        batch = chunks_df.iloc[idx : idx + batch_size]
        batch_results = []

        for row in batch.to_dict(orient="records"):
            chunk_id = row.get("chunk_id", "")
            chunk_text = row.get("chunk_text", "")

            if chunk_id in cache:
                cached += 1
            else:
                short, key = generator.generate_summary(chunk_text)
                append_to_cache(cache_path, chunk_id, short, key)
                cache[chunk_id] = {"chunk_id": chunk_id, "short": short, "key": key}

                if not short and not key:
                    failed += 1

        for row in batch.to_dict(orient="records"):
            chunk_id = row.get("chunk_id", "")
            chunk_text = row.get("chunk_text", "")

            cached_record = cache.get(chunk_id, {})
            short = cached_record.get("short", "")
            key = normalize_key_value(cached_record.get("key", []))

            enriched_text = build_enriched_text(short, key, chunk_text)

            result = dict(row)
            result["short"] = short
            result["key"] = key
            result["enriched_text"] = enriched_text
            batch_results.append(result)

        results.extend(batch_results)
        progress.update(len(batch))
        progress.set_postfix(cached=cached, failed=failed)

    progress.close()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 4: Generate summaries for chunks using an OpenAI-compatible API"
    )
    parser.add_argument("--stage3-path", default=None, help="Path to stage3_chunks.parquet")
    parser.add_argument("--output-path", default=None, help="Output path for stage4_enriched.parquet")
    parser.add_argument("--cache-path", default=None, help="Path for summary_cache.jsonl")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"), help="Provider model name")
    parser.add_argument("--max-input-chars", type=int, default=3000, help="Max input characters")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max new tokens for generation")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for processing")
    parser.add_argument("--api-key", default=None, help="OpenAI-compatible provider API key. Defaults to OPENAI_API_KEY or API_KEY.")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible provider base URL, e.g. https://api.provider.com/v1. Defaults to OPENAI_BASE_URL.")
    parser.add_argument("--timeout", type=int, default=120, help="HTTP request timeout in seconds")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of chunks to process (for testing)")
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parents[2]
    workspace_root = project_root.parent
    load_dotenv(workspace_root / ".env")
    load_dotenv(project_root / ".env")
    
    # Resolve paths
    stage3_path = resolve_path(args.stage3_path, project_root / "data" / "stage3_chunks.parquet")
    output_path = resolve_path(args.output_path, project_root / "data" / "stage4_enriched.parquet")
    cache_path = resolve_path(args.cache_path, project_root / "data" / "summary_cache.jsonl")
    
    # Resolve OpenAI-compatible provider credentials
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY")
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL")
    if not api_key:
        raise ValueError("Missing API key. Pass --api-key or set OPENAI_API_KEY/API_KEY in .env or environment.")
    if not base_url:
        raise ValueError("Missing base URL. Pass --base-url or set OPENAI_BASE_URL in .env or environment.")
    
    # Load chunks
    print(f"Loading Stage 3 chunks from {stage3_path}")
    chunks_df = pd.read_parquet(stage3_path)
    print(f"Stage 3 chunks loaded: {len(chunks_df):,}")
    
    # Apply limit if specified
    if args.limit:
        chunks_df = chunks_df.head(args.limit)
        print(f"Limited to first {args.limit} chunks for testing.")
    
    # Load existing output first so reruns continue from the last written row.
    existing_output_df = load_existing_output(output_path)
    chunks_to_process = get_resume_chunks(chunks_df, existing_output_df)

    # Load existing cache
    print(f"Loading existing cache from {cache_path}")
    cache = load_cache(cache_path)
    print(f"Cache entries loaded: {len(cache):,}")
    print(f"Chunks to process: {len(chunks_to_process):,}")
    
    if len(chunks_to_process) == 0:
        print("All chunks already present in the output path. Reusing existing output.")
        results = existing_output_df.to_dict(orient="records")
    else:
        # Initialize generator
        generator = SummaryGenerator(
            model_name=args.model,
            max_input_chars=args.max_input_chars,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            api_key=api_key,
            base_url=base_url,
            timeout=args.timeout,
        )
        
        # Process chunks
        results = process_chunks(
            chunks_df=chunks_to_process,
            generator=generator,
            cache=cache,
            cache_path=cache_path,
            batch_size=args.batch_size,
        )
        
        # Unload model to free memory
        generator.unload()
        
        if not existing_output_df.empty:
            previous_results = existing_output_df.to_dict(orient="records")
            results = previous_results + results
    
    # Create output DataFrame
    output_df = pd.DataFrame(results)
    
    # Ensure key column is stored as JSON string for Parquet compatibility
    if "key" in output_df.columns:
        output_df["key"] = output_df["key"].apply(lambda x: json.dumps(normalize_key_value(x), ensure_ascii=False))
    
    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_parquet(output_path, index=False)
    
    print(f"Wrote {len(output_df):,} enriched chunks to {output_path}")
    
    # Summary statistics
    non_empty_short = output_df["short"].apply(lambda x: bool(x and str(x).strip())).sum()
    non_empty_key = output_df["key"].apply(lambda x: bool(x and str(x).strip() and str(x) != "[]")).sum()
    print(f"Non-empty short summaries: {non_empty_short:,} ({100*non_empty_short/len(output_df):.1f}%)")
    print(f"Non-empty key points: {non_empty_key:,} ({100*non_empty_key/len(output_df):.1f}%)")


if __name__ == "__main__":
    main()