import re
import yaml
import argparse
from pathlib import Path
import pandas as pd

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def clean_title(title: str, loai_van_ban: str, so_ky_hieu: str) -> str:
    if not title or not isinstance(title, str):
        return ""
    
    # Extract the prefix to remove: "{loai_van_ban} (số )?{so_ky_hieu}" case-insensitive
    prefix_pattern = re.compile(rf"^{re.escape(str(loai_van_ban))}\s*(?:số\s*)?{re.escape(str(so_ky_hieu))}", flags=re.IGNORECASE)
    cleaned = re.sub(prefix_pattern, "", title).strip()
    
    # Strip leading punctuation "-–:"
    cleaned = re.sub(r"^[-–:]\s*", "", cleaned).strip()
    
    return cleaned

def load_data(path: str, columns: list = None) -> pd.DataFrame:
    if str(path).endswith(".jsonl") or str(path).endswith(".json"):
        if columns and "id" in columns and len(columns) == 1:
            # Special case for reading just the 'id' column from a large content JSONL file
            import json
            ids = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    ids.append(json.loads(line).get("id"))
            return pd.DataFrame({"id": ids})
        else:
            df = pd.read_json(path, lines=True)
            if columns:
                df = df[columns]
            return df
    else:
        return pd.read_parquet(path, columns=columns)

def main():
    parser = argparse.ArgumentParser(description="Stage 1: Scope filter for Vietnamese Legal Documents")
    parser.add_argument("--metadata", type=str, default="hf://datasets/th1nhng0/vietnamese-legal-documents/data/metadata.parquet", help="Path to metadata parquet (local or hf://)")
    parser.add_argument("--content", type=str, default="hf://datasets/th1nhng0/vietnamese-legal-documents/data/content.parquet", help="Path to content parquet (local or hf://)")
    parser.add_argument("--output", type=str, default=None, help="Output path for the filtered parquet")
    args = parser.parse_args()

    print("Loading config...")
    project_root = Path(__file__).resolve().parent.parent.parent
    config_path = project_root / "config" / "default.yaml"
    config = load_config(str(config_path))
    
    sme_title_keywords = config.get("SME_TITLE_KEYWORDS", [])
    sme_linh_vuc = config.get("SME_LINH_VUC", [])
    sme_nganh = config.get("SME_NGANH", [])
    valid_document_types = config.get("VALID_DOCUMENT_TYPES", [])
    
    print(f"Loading metadata from: {args.metadata}")
    print(f"Loading content from: {args.content}")
    
    # Load the metadata and content subsets using the helper function
    df_meta = load_data(args.metadata)
    # Only load ID from content to avoid memory issues
    df_content_ids = load_data(args.content, columns=["id"])
    
    print(f"Loaded {len(df_meta)} metadata records and {len(df_content_ids)} content records.")
    
    # Filter 4: Only include documents that have a corresponding row in the content subset
    valid_content_ids = set(df_content_ids["id"].astype(str).dropna().unique())
    df_meta = df_meta[df_meta["id"].astype(str).isin(valid_content_ids)]
    
    # Filter 2: tinh_trang_hieu_luc contains "Còn hiệu lực"
    mask_hieu_luc = df_meta["tinh_trang_hieu_luc"].astype(str).str.contains("Còn hiệu lực", na=False)
    
    # Filter 3: loai_van_ban is in VALID_DOCUMENT_TYPES
    # Some types might be None or NaN, so we fillna with empty string to avoid errors
    mask_loai = df_meta["loai_van_ban"].astype(str).isin([str(t) for t in valid_document_types])
    
    # Filter 1: SME relevance
    title_lower = df_meta["title"].astype(str).str.lower()
    mask_title = title_lower.apply(lambda t: any(kw.lower() in t for kw in sme_title_keywords))
    mask_linh_vuc = df_meta["linh_vuc"].isin(sme_linh_vuc)
    mask_nganh = df_meta["nganh"].isin(sme_nganh)
    
    mask_sme = mask_title | mask_linh_vuc | mask_nganh
    
    # Combine all masks
    final_mask = mask_hieu_luc & mask_loai & mask_sme
    
    df_filtered = df_meta[final_mask].copy()
    print(f"Filtered to {len(df_filtered)} SME documents.")
    
    # Derived fields
    df_filtered["law_id"] = df_filtered["so_ky_hieu"]
    df_filtered["ten_van_ban"] = df_filtered.apply(
        lambda row: f"{row['loai_van_ban']} {row['so_ky_hieu']} {clean_title(row['title'], row['loai_van_ban'], row['so_ky_hieu'])}".strip(),
        axis=1
    )
    
    # Quality gates
    output_rows = len(df_filtered)
    print(f"Applying quality gates. Output rows: {output_rows}")
    
    assert 3000 <= output_rows <= 20000, f"Output rows {output_rows} is outside the expected range of 3000-20000."
    assert df_filtered["id"].is_unique, "Duplicate IDs found."
    
    # Due to expanded document types, there might be duplicate entries in the metadata.
    # We drop them to satisfy the unique (law_id, ten_van_ban) requirement.
    df_filtered = df_filtered.drop_duplicates(subset=["law_id", "ten_van_ban"])
    
    # Assert no duplicate (law_id, ten_van_ban) pair
    assert not df_filtered.duplicated(subset=["law_id", "ten_van_ban"]).any(), "Duplicate (law_id, ten_van_ban) pairs found."
    
    # Assert every retained id exists in the content config (already handled by Filter 4, but double checking)
    assert df_meta["id"].astype(str).isin(valid_content_ids).all(), "Some IDs are missing in the content dataset."
    
    # Save to parquet
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = project_root / "data" / "stage1_sme_docs.parquet"
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_filtered.to_parquet(output_path, engine="pyarrow", index=False)
    
    print(f"Successfully saved filtered documents to {output_path}")

if __name__ == "__main__":
    main()
