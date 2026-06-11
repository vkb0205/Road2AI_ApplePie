"""
Stage 4: Summary Injection
Generate two-field JSON summaries for each chunk using gemma-3-12b-it.

Input: `stage3_chunks.parquet`.
Output: `stage4_enriched.parquet` (with short, key, enriched_text columns)
        + `summary_cache.jsonl` (resumable cache).

Model: google/gemma-3-12b-it with 4-bit NF4 quantization.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
except ImportError as exc:
    raise ImportError(
        "transformers and torch are required to run src/data/stage4_summarize.py. "
        "Install with `pip install transformers torch` or use the repo requirements file."
    ) from exc

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
    """Wrapper for gemma-3-12b-it summarization with 4-bit NF4 quantization."""
    
    def __init__(
        self,
        model_name: str = "google/gemma-3-12b-it",
        max_input_chars: int = 3000,
        max_new_tokens: int = 256,
        batch_size: int = 2,
        device: str = "auto",
    ):
        self.model_name = model_name
        self.max_input_chars = max_input_chars
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        
        # Load tokenizer
        print(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
            trust_remote_code=True,
        )
        
        # Configure 4-bit NF4 quantization
        print("Configuring 4-bit NF4 quantization...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        
        # Load model
        print(f"Loading model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map=device,
            trust_remote_code=True,
            attn_implementation="eager",
        )
        self.model.eval()
        print("Model loaded successfully.")
    
    def truncate_chunk_text(self, text: str) -> str:
        """Truncate chunk text to max_input_chars."""
        if len(text) <= self.max_input_chars:
            return text
        return text[:self.max_input_chars]
    
    def generate_summary(self, chunk_text: str) -> Tuple[str, List[str]]:
        """Generate summary for a single chunk. Returns (short, key_list)."""
        truncated = self.truncate_chunk_text(chunk_text)
        
        # Build messages for chat template
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(chunk_text_truncated=truncated)},
        ]
        
        # Apply chat template
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=0.1,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
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
        """Unload model to free memory."""
        del self.model
        del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def process_chunks(
    chunks_df: pd.DataFrame,
    generator: SummaryGenerator,
    cache: Dict[str, Dict],
    cache_path: Path,
    batch_size: int = 2,
) -> List[Dict]:
    """Process all chunks, using cache for already-processed ones."""
    results = []
    total = len(chunks_df)
    processed = 0
    cached = 0
    failed = 0
    
    for idx in range(0, total, batch_size):
        batch = chunks_df.iloc[idx : idx + batch_size]
        batch_results = []
        
        for row in batch.to_dict(orient="records"):
            chunk_id = row.get("chunk_id", "")
            chunk_text = row.get("chunk_text", "")
            
            # Check cache
            if chunk_id in cache:
                cached_record = cache[chunk_id]
                short = cached_record.get("short", "")
                key = cached_record.get("key", [])
            else:
                # Generate summary
                short, key = generator.generate_summary(chunk_text)
                
                # Save to cache
                append_to_cache(cache_path, chunk_id, short, key)
                cache[chunk_id] = {"chunk_id": chunk_id, "short": short, "key": key}
                
                if not short and not key:
                    failed += 1
        
        # Build enriched records for this batch
        for row in batch.to_dict(orient="records"):
            chunk_id = row.get("chunk_id", "")
            chunk_text = row.get("chunk_text", "")
            
            cached_record = cache.get(chunk_id, {})
            short = cached_record.get("short", "")
            key = cached_record.get("key", [])
            
            enriched_text = build_enriched_text(short, key, chunk_text)
            
            result = dict(row)
            result["short"] = short
            result["key"] = key
            result["enriched_text"] = enriched_text
            batch_results.append(result)
            processed += 1
        
        results.extend(batch_results)
        
        # Progress logging
        if (idx + batch_size) % 100 == 0 or idx + batch_size >= total:
            print(f"Processed {min(idx + batch_size, total):,} / {total:,} chunks "
                  f"(cached: {cached}, failed: {failed})")
    
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 4: Generate summaries for chunks using gemma-3-12b-it"
    )
    parser.add_argument("--stage3-path", default=None, help="Path to stage3_chunks.parquet")
    parser.add_argument("--output-path", default=None, help="Output path for stage4_enriched.parquet")
    parser.add_argument("--cache-path", default=None, help="Path for summary_cache.jsonl")
    parser.add_argument("--model", default="google/gemma-3-12b-it", help="Model name")
    parser.add_argument("--max-input-chars", type=int, default=3000, help="Max input characters")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max new tokens for generation")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for processing")
    parser.add_argument("--hf-token", default=None, help="Hugging Face token for gated repo access")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of chunks to process (for testing)")
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")
    
    # Resolve paths
    stage3_path = resolve_path(args.stage3_path, project_root / "data" / "stage3_chunks.parquet")
    output_path = resolve_path(args.output_path, project_root / "data" / "stage4_enriched.parquet")
    cache_path = resolve_path(args.cache_path, project_root / "data" / "summary_cache.jsonl")
    
    # Set HF token
    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
        print("Using Hugging Face token for gated repo access.")
    else:
        print("WARNING: No HF token found. May fail on gated models.")
    
    # Load chunks
    print(f"Loading Stage 3 chunks from {stage3_path}")
    chunks_df = pd.read_parquet(stage3_path)
    print(f"Stage 3 chunks loaded: {len(chunks_df):,}")
    
    # Apply limit if specified
    if args.limit:
        chunks_df = chunks_df.head(args.limit)
        print(f"Limited to first {args.limit} chunks for testing.")
    
    # Load existing cache
    print(f"Loading existing cache from {cache_path}")
    cache = load_cache(cache_path)
    print(f"Cache entries loaded: {len(cache):,}")
    
    # Filter out already-cached chunks
    chunks_to_process = chunks_df[~chunks_df["chunk_id"].isin(cache.keys())]
    print(f"Chunks to process: {len(chunks_to_process):,}")
    
    if len(chunks_to_process) == 0:
        print("All chunks already cached. Building enriched parquet from cache.")
        # Build results from cache only
        results = []
        for row in chunks_df.to_dict(orient="records"):
            chunk_id = row.get("chunk_id", "")
            chunk_text = row.get("chunk_text", "")
            cached_record = cache.get(chunk_id, {})
            short = cached_record.get("short", "")
            key = cached_record.get("key", [])
            enriched_text = build_enriched_text(short, key, chunk_text)
            result = dict(row)
            result["short"] = short
            result["key"] = key
            result["enriched_text"] = enriched_text
            results.append(result)
    else:
        # Initialize generator
        generator = SummaryGenerator(
            model_name=args.model,
            max_input_chars=args.max_input_chars,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
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
        
        # Add cached chunks to results
        cached_results = []
        for row in chunks_df[chunks_df["chunk_id"].isin(cache.keys())].to_dict(orient="records"):
            chunk_id = row.get("chunk_id", "")
            chunk_text = row.get("chunk_text", "")
            cached_record = cache.get(chunk_id, {})
            short = cached_record.get("short", "")
            key = cached_record.get("key", [])
            enriched_text = build_enriched_text(short, key, chunk_text)
            result = dict(row)
            result["short"] = short
            result["key"] = key
            result["enriched_text"] = enriched_text
            cached_results.append(result)
        
        results.extend(cached_results)
    
    # Create output DataFrame
    output_df = pd.DataFrame(results)
    
    # Ensure key column is stored as JSON string for Parquet compatibility
    if "key" in output_df.columns:
        output_df["key"] = output_df["key"].apply(lambda x: json.dumps(x) if isinstance(x, list) else x)
    
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