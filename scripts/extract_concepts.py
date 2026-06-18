#!/usr/bin/env python3
"""
Stage 3 Chunk Concept Extraction Script

This script reads a chunk parquet file (e.g. stage3_chunks.parquet),
sends batches of 10 chunks to the Gemini API to extract legal concepts,
and maps candidate concepts to a standard registry using similarity matching.

It supports:
1. Custom base URL for Gemini API proxy/endpoint
2. Custom Gemini API key
3. Resume mechanism (resume from checkpoint vs restart from scratch)
"""

import os
import re
import sys
import json
import time
import argparse
import pandas as pd

# Default configuration matching the original notebook
DEFAULT_INPUT_PATH = '/Users/mac/Downloads/stage3_chunks.parquet'
DEFAULT_OUTPUT_DIR = '/Users/mac/.gemini/antigravity-ide/scratch/legal_concept_extractor/output'
DEFAULT_BATCH_SIZE = 10
DEFAULT_SIM_THRESHOLD = 0.88
DEFAULT_MODEL_NAME = 'gemini-2.0-flash'
DEFAULT_EMBED_MODEL = 'BAAI/bge-m3'

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract legal concepts from document chunks using Gemini API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Core requirements
    parser.add_argument(
        "--api-key", "-k",
        type=str,
        help="Gemini API key. Overrides the GEMINI_API_KEY environment variable."
    )
    parser.add_argument(
        "--base-url", "-b",
        type=str,
        help="Custom base URL / API endpoint for Gemini client configuration."
    )
    parser.add_argument(
        "--resume-mechanism", "-r",
        choices=["resume", "restart"],
        default="resume",
        help="Resume mechanism: 'resume' starts from the last saved checkpoint and preserves existing output data. 'restart' wipes existing output and starts from index 0."
    )
    
    # Additional configurations
    parser.add_argument(
        "--input-path", "-i",
        type=str,
        default=DEFAULT_INPUT_PATH,
        help="Path to the input stage3_chunks parquet file."
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save output files and checkpoints."
    )
    parser.add_argument(
        "--batch-size", "-s",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of chunks to send per Gemini request."
    )
    parser.add_argument(
        "--sim-threshold", "-t",
        type=float,
        default=DEFAULT_SIM_THRESHOLD,
        help="Fuzzy matching similarity threshold for mapping concepts."
    )
    parser.add_argument(
        "--model-name", "-m",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Gemini model name to use."
    )
    parser.add_argument(
        "--embed-model", "-e",
        type=str,
        default=DEFAULT_EMBED_MODEL,
        help="Sentence transformer model (defined for compatibility, lazy-loaded if needed)."
    )
    
    return parser.parse_args()

def norm_text(s):
    s = '' if s is None else str(s)
    s = s.strip().lower()
    s = re.sub(r'\s+', ' ', s)
    return s

def canonical_key(name):
    s = norm_text(name)
    s = s.replace('đ', 'd')
    s = re.sub(r'[^a-z0-9\s]+', '', s)
    s = re.sub(r'\s+', '_', s).strip('_')
    return s

def load_jsonl(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def append_jsonl(path, obj):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False) + '\n')

def build_chunk_text(row):
    parts = []
    for c in ['breadcrumb', 'chunk_text']:
        if c in row and pd.notna(row.get(c, None)) and str(row.get(c)).strip():
            parts.append(str(row[c]))
    if not parts:
        return ''
    return '\n'.join(parts)

def tokenize_for_prompt(text, max_chars=3000):
    text = re.sub(r'\s+', ' ', str(text)).strip()
    return text[:max_chars]

SYSTEM_PROMPT = """Bạn là bộ trích xuất concept pháp lý.

Quy tắc:
- Chỉ xuất concept quan trọng, ngắn gọn, có tính pháp lý.
- Tên concept phải là tiếng Việt có dấu.
- Không bịa concept nếu có thể quy về concept hiện có.
- Mỗi chunk tối đa 3 concept.
- Trả về JSON hợp lệ đúng schema.

Schema bắt buộc:
{
  "items": [
    {
      "source_id": "string",
      "concepts": ["string", "string"]
    }
  ]
}
"""

def _clean_json_string(s: str) -> str:
    """Helper to remove comments and trailing commas before parsing JSON."""
    # Remove single line comments
    s = re.sub(r'//.*$', '', s, flags=re.MULTILINE)
    # Remove multi-line comments
    s = re.sub(r'/\*.*?\*/', '', s, flags=re.DOTALL)
    # Remove trailing commas before closing braces/brackets
    s = re.sub(r',\s*([\]}])', r'\1', s)
    return s.strip()


def _parse_json_object_text(txt):
    """Parse JSON from LLM response text, tolerating markdown fences and extra text.

    Strategies in order:
    1. Try direct json.loads (fast path for pure JSON).
    2. Strip markdown code fences (```json ... ```) and retry.
    3. Extract first {...} or [...] block with regex.
    4. Raise ValueError with the raw text on total failure.
    """
    raw = (txt or '').strip()
    if not raw:
        raise ValueError("Empty response from LLM")

    # Strategy 1: direct parse (handles response_format='json_object')
    try:
        return json.loads(_clean_json_string(raw))
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: strip markdown code fences
    text = raw
    # Remove ```json ... ``` fences
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'```\s*$', '', text, flags=re.MULTILINE)
    text = text.strip()
    try:
        return json.loads(_clean_json_string(text))
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 3: extract first { ... } block with regex (non-greedy for nested braces)
    m = re.search(r'\{.*\}', text, re.S)
    if m:
        candidate = m.group(0)
        try:
            return json.loads(_clean_json_string(candidate))
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 4: extract first [ ... ] block (some models return arrays)
    m = re.search(r'\[.*\]', text, re.S)
    if m:
        candidate = m.group(0)
        try:
            wrapped = json.loads(_clean_json_string(candidate))
            return {"items": wrapped}
        except (json.JSONDecodeError, ValueError):
            pass

    raise ValueError(
        f"Could not parse JSON from LLM response. "
        f"First 500 chars: {raw[:500]!r}"
    )


def _parse_gemini_json_response(resp):
    return _parse_json_object_text(resp.text or '')


def gemini_extract(llm, batch_items):
    prompt = SYSTEM_PROMPT + '\n\nDữ liệu đầu vào:\n' + json.dumps(batch_items, ensure_ascii=False)
    resp = llm.generate_content(prompt)
    return _parse_gemini_json_response(resp)


def build_mentions_prompt(batch_items, allowed_concepts):
    """Build the shared prompt for Stage 5.6 LLM MENTIONS extraction."""
    return """Bạn là bộ phát hiện quan hệ MENTIONS cho đồ thị tri thức pháp lý.

Quy tắc:
- Với mỗi chunk, chọn tối đa 3 concept được nhắc đến rõ ràng hoặc diễn đạt tương đương.
- Chỉ được chọn concept từ danh sách allowed_concepts.
- Không tạo concept mới, không đổi tên concept, không bịa concept.
- Nếu không có concept phù hợp, trả về mảng rỗng.
- Trả về JSON hợp lệ đúng schema.

Schema bắt buộc:
{
  "items": [
    {
      "source_id": "string",
      "concepts": ["string", "string"]
    }
  ]
}

allowed_concepts:
""" + json.dumps(allowed_concepts, ensure_ascii=False) + "\n\nDữ liệu đầu vào:\n" + json.dumps(batch_items, ensure_ascii=False)


def gemini_extract_mentions(llm, batch_items, allowed_concepts):
    """Use Gemini to select curated concepts mentioned by each chunk."""
    resp = llm.generate_content(build_mentions_prompt(batch_items, allowed_concepts))
    return _parse_gemini_json_response(resp)


def openai_compatible_extract_mentions(client, model_name, batch_items, allowed_concepts):
    """Use an OpenAI-compatible chat completions client for MENTIONS extraction.

    Returns:
        dict: Parsed JSON response with an "items" key (possibly empty if parsing fails).

    Raises:
        ValueError: Only after all retries and parse strategies are exhausted.
    """
    messages = [
        {
            "role": "system",
            "content": "Bạn chỉ trả về JSON hợp lệ theo schema được yêu cầu. "
                       "Phải bắt đầu bằng { và kết thúc bằng }, không được dùng markdown fence. "
                       "Không viết suy nghĩ, không suy luận, không dùng thẻ <think>.",
        },
        {
            "role": "user",
            "content": build_mentions_prompt(batch_items, allowed_concepts),
        },
    ]

    import time
    import random

    # Some providers advertise json_object support but don't enforce it well;
    # try without the response_format parameter as a fallback.
    for attempt, use_format in enumerate([True, False]):
        kwargs = dict(
            model=model_name,
            messages=messages,
            temperature=0,
            stream=True,  # Always use stream=True to adapt to deepseek-v4-flash proxy bugs
        )
        if use_format:
            kwargs["response_format"] = {"type": "json_object"}

        # Inner retry loop with exponential backoff for transient provider/upstream issues
        max_retries = 5
        base_delay = 2.0
        success = False
        content = ""

        for retry in range(max_retries):
            try:
                response = client.chat.completions.create(**kwargs)
                full_content = []
                for chunk in response:
                    val = chunk.choices[0].delta.content
                    if val:
                        full_content.append(val)
                content = "".join(full_content).strip()
                success = True
                break
            except Exception as exc:
                err_msg = str(exc)
                # Check if this looks like a transient error (e.g. rate limit, upstream busy, etc.)
                is_transient = any(
                    x in err_msg.lower()
                    for x in ["bận", "busy", "upstream", "rate limit", "429", "500", "502", "503", "504", "400"]
                )
                if not is_transient or retry == max_retries - 1:
                    # If not transient, or we exhausted retries, fail this attempt (moves to next format attempt or raises)
                    if attempt == 1 and retry == max_retries - 1:
                        raise ValueError(
                            f"OpenAI-compatible API call failed after {max_retries} retries: {exc}"
                        ) from exc
                    break

                # Sleep with exponential backoff + jitter
                delay = base_delay * (2 ** retry) + random.uniform(0, 1)
                print(
                    f"\n  [WARN] API error: {err_msg}. "
                    f"Retrying in {delay:.2f}s (attempt {retry + 1}/{max_retries})...",
                    file=sys.stderr,
                    flush=True
                )
                time.sleep(delay)

        if not success:
            continue

        if not content and attempt < 1:
            continue

        # Strip <think> reasoning tags if they are generated by Deepseek
        clean_content = re.sub(r'<think>.*?</think>', '', content, flags=re.S).strip()

        try:
            return _parse_json_object_text(clean_content)
        except ValueError:
            if attempt == 1:
                raise
            # Fall through to retry without response_format
            continue

    # Should not reach here, but belt-and-suspenders:
    return {"items": []}

def similarity_score(a, b):
    from rapidfuzz import fuzz

    a = norm_text(a)
    b = norm_text(b)
    if not a or not b:
        return 0.0
    return fuzz.ratio(a, b) / 100.0

def best_match(candidate, registry):
    best = (None, 0.0)
    cand_norm = norm_text(candidate)
    cand_key = canonical_key(candidate)
    for c in registry:
        score = max(
            similarity_score(candidate, c.get('display_name', c.get('name', ''))),
            similarity_score(cand_norm, c.get('name_lower', '')),
            similarity_score(cand_key, c.get('name_lower', ''))
        )
        if score > best[1]:
            best = (c, score)
    return best

def upsert_concept(candidate, source_id, registry, reg_path, threshold):
    match, score = best_match(candidate, registry)
    if match and score >= threshold:
        return {
            'source_id': source_id,
            'input_concept': candidate,
            'matched_concept_id': match['concept_id'],
            'matched_name': match.get('display_name', match.get('name')),
            'matched_score': score,
            'action': 'map_existing'
        }, registry
    
    rec = {
        'concept_id': f"CONCEPT:{canonical_key(candidate)}",
        'display_name': candidate,
        'name': candidate,
        'name_lower': canonical_key(candidate),
        'aliases': [],
        'created_at': int(time.time())
    }
    registry.append(rec)
    append_jsonl(reg_path, rec)
    
    return {
        'source_id': source_id,
        'input_concept': candidate,
        'matched_concept_id': rec['concept_id'],
        'matched_name': rec['display_name'],
        'matched_score': 1.0,
        'action': 'add_new'
    }, registry

def main():
    args = parse_args()
    
    # 1. Setup output paths
    os.makedirs(args.output_dir, exist_ok=True)
    cand_path = os.path.join(args.output_dir, 'chunk_concept_candidates.jsonl')
    reg_path = os.path.join(args.output_dir, 'concept_registry.jsonl')
    map_path = os.path.join(args.output_dir, 'chunk_concept_mapping.parquet')
    edge_path = os.path.join(args.output_dir, 'chunk_concept_edges.parquet')
    checkpoint_path = os.path.join(args.output_dir, 'chunk_concept_checkpoint.json')
    
    # 2. Determine Resume vs Restart state
    start_idx = 0
    all_mappings = []
    all_edges = []
    
    if args.resume_mechanism == "restart":
        print("Restart mode enabled. Wiping existing outputs and checkpoint files...")
        for file_path in [cand_path, reg_path, map_path, edge_path, checkpoint_path]:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Warning: could not remove {file_path}: {e}")
    else:
        # Load from checkpoint
        if os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, 'r', encoding='utf-8') as f:
                    ckpt = json.load(f)
                    start_idx = int(ckpt.get('next_idx', 0))
                print(f"Resuming from checkpoint. Next index to process: {start_idx}")
            except Exception as e:
                print(f"Warning: Failed to load checkpoint. Starting from 0. Error: {e}")
                start_idx = 0
        
        # Load existing mappings and edges to prevent data loss
        if start_idx > 0:
            if os.path.exists(map_path):
                try:
                    all_mappings = pd.read_parquet(map_path).to_dict(orient='records')
                    print(f"Loaded {len(all_mappings)} existing mapping records.")
                except Exception as e:
                    print(f"Warning: could not load existing mapping parquet file: {e}")
            if os.path.exists(edge_path):
                try:
                    all_edges = pd.read_parquet(edge_path).to_dict(orient='records')
                    print(f"Loaded {len(all_edges)} existing edge records.")
                except Exception as e:
                    print(f"Warning: could not load existing edge parquet file: {e}")
                    
    # 3. Configure Gemini Client
    import google.generativeai as genai

    api_key = args.api_key or os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("Error: Gemini API key must be provided via --api-key / -k or the GEMINI_API_KEY environment variable.", file=sys.stderr)
        sys.exit(1)
        
    client_options = {}
    if args.base_url:
        client_options["api_endpoint"] = args.base_url
        print(f"Configuring Gemini API with custom base URL: {args.base_url}")
        
    genai.configure(api_key=api_key, client_options=client_options)
    llm = genai.GenerativeModel(args.model_name)

    # SentenceTransformer warning (not initialized since it's unused in matching)
    print(f"Registry matching threshold set to: {args.sim_threshold}")
    
    # 4. Load input data
    if not os.path.exists(args.input_path):
        print(f"Error: Input parquet file not found at {args.input_path}", file=sys.stderr)
        sys.exit(1)
        
    print(f"Loading input parquet file: {args.input_path}")
    df = pd.read_parquet(args.input_path)
    
    # Ensure necessary columns are present in DataFrame
    required_cols = [
        'chunk_id', 'doc_id', 'doc_uid', 'breadcrumb', 'chunk_text',
        'law_id', 'dieu_so', 'dieu_ten', 'part_idx', 'rowidx'
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = None
            
    df['source_text'] = df.apply(build_chunk_text, axis=1)
    
    registry = load_jsonl(reg_path)
    print(f"Loaded registry size: {len(registry)} concepts.")
    print(f"Total rows to process: {len(df)}")
    
    if start_idx >= len(df):
        print("All rows have already been processed.")
        sys.exit(0)
        
    # 5. Extraction loop
    try:
        from tqdm import tqdm

        # Progress bar starting from start_idx
        for batch_start in tqdm(range(start_idx, len(df), args.batch_size), desc="Extracting concepts"):
            batch_end = min(batch_start + args.batch_size, len(df))
            batch_df = df.iloc[batch_start:batch_end].copy()
            batch_items = []
            
            for _, row in batch_df.iterrows():
                source_id = str(row['chunk_id'])
                batch_items.append({
                    'source_id': source_id,
                    'text': tokenize_for_prompt(row['source_text'])
                })
                
            try:
                result = gemini_extract(llm, batch_items)
            except Exception as e:
                print(f"\nWarning: Gemini extraction failed for batch {batch_start}-{batch_end}. Falling back to empty concepts. Error: {e}")
                result = {
                    'items': [{'source_id': x['source_id'], 'concepts': []} for x in batch_items],
                    'error': str(e)
                }
                
            append_jsonl(cand_path, {'batch_start': batch_start, 'batch_end': batch_end, 'result': result})
            
            items = result.get('items', [])
            by_source = {x['source_id']: x for x in items}
            
            for x in batch_items:
                source_id = x['source_id']
                concepts = by_source.get(source_id, {}).get('concepts', [])[:3]
                for cand in concepts:
                    mapping, registry = upsert_concept(cand, source_id, registry, reg_path, args.sim_threshold)
                    all_mappings.append(mapping)
                    all_edges.append({
                        'source_id': source_id,
                        'concept_id': mapping['matched_concept_id'],
                        'concept_name': mapping['matched_name'],
                        'action': mapping['action'],
                        'score': mapping['matched_score']
                    })
                    
            # Update checkpoint
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump({'next_idx': batch_end}, f, ensure_ascii=False, indent=2)
                
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user. Saving current progress...")
        
    # 6. Save final mapping and edge Parquet files
    if all_mappings:
        mapping_df = pd.DataFrame(all_mappings)
        mapping_df.to_parquet(map_path, index=False)
        print(f"Saved {len(mapping_df)} mapping entries to {map_path}")
        
    if all_edges:
        edge_df = pd.DataFrame(all_edges)
        edge_df.to_parquet(edge_path, index=False)
        print(f"Saved {len(edge_df)} edge entries to {edge_path}")
        
    print(f"Execution finished. Registry size: {len(registry)} concepts.")

if __name__ == '__main__':
    main()
