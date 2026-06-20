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
    prefix_pattern = re.compile(
        rf"^{re.escape(str(loai_van_ban))}\s*(?:số\s*)?{re.escape(str(so_ky_hieu))}",
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(prefix_pattern, "", title).strip()
    cleaned = re.sub(r"^[-–:]\s*", "", cleaned).strip()
    return cleaned


def load_data(path: str, columns: list = None) -> pd.DataFrame:
    if str(path).endswith((".jsonl", ".json")):
        if columns and "id" in columns and len(columns) == 1:
            import json
            ids = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        ids.append(json.loads(line).get("id"))
            return pd.DataFrame({"id": ids})
        df = pd.read_json(path, lines=True)
        if columns:
            df = df[columns]
        return df
    return pd.read_parquet(path, columns=columns)


def main():
    parser = argparse.ArgumentParser(description="Stage 1: Scope filter for Vietnamese Legal Documents")
    parser.add_argument("--metadata", type=str, default="hf://datasets/th1nhng0/vietnamese-legal-documents/data/metadata.parquet")
    parser.add_argument("--content", type=str, default="hf://datasets/th1nhng0/vietnamese-legal-documents/data/content.parquet")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    config_path = project_root / "config" / "default.yaml"
    config = load_config(str(config_path))

    def normalize_text(value) -> str:
        if pd.isna(value):
            return ""
        return str(value).strip()

    keep_types = {normalize_text(value) for value in config.get("VALID_DOCUMENT_TYPES", [])}

    print(f"Loading metadata from: {args.metadata}")
    print(f"Loading content from: {args.content}")

    df_meta = load_data(args.metadata)
    df_content_ids = load_data(args.content, columns=["id"])

    print(f"Loaded {len(df_meta)} metadata records and {len(df_content_ids)} content records.")

    ngay_het_hieu_luc = df_meta["ngay_het_hieu_luc"]
    ngay_het_hieu_luc_text = ngay_het_hieu_luc.astype(str).str.strip().str.lower()
    mask_not_expired = (
        ngay_het_hieu_luc.isna()
        | ngay_het_hieu_luc_text.isin({"", "nan", "nat", "none", "null"})
    )

    loai_van_ban = df_meta["loai_van_ban"].apply(normalize_text)
    mask_loai = loai_van_ban.isin(keep_types)

    print(f"Total metadata rows: {len(df_meta)}")
    print(f"Rows not expired: {int(mask_not_expired.sum())}")
    print(f"Rows with valid document type: {int(mask_loai.sum())}")
    print(f"Rows after all filters: {int((mask_not_expired & mask_loai).sum())}")

    final_mask = mask_not_expired & mask_loai
    df_filtered = df_meta[final_mask].copy()

    df_filtered["law_id"] = df_filtered["so_ky_hieu"]
    df_filtered["ten_van_ban"] = df_filtered.apply(
        lambda row: f"{row['loai_van_ban']} {row['so_ky_hieu']} {clean_title(row['title'], row['loai_van_ban'], row['so_ky_hieu'])}".strip(),
        axis=1,
    )

    df_filtered = df_filtered.drop_duplicates(subset=["law_id", "ten_van_ban"]).copy()

    output_rows = len(df_filtered)
    print(f"Filtered to {output_rows} SME documents.")

    assert output_rows > 0, "No documents after filtering."
    assert df_filtered["id"].is_unique, "Duplicate IDs found."
    assert not df_filtered.duplicated(subset=["law_id", "ten_van_ban"]).any(), "Duplicate (law_id, ten_van_ban) pairs found."

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = project_root / "data" / "stage1_sme_docs.parquet"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_filtered.to_parquet(output_path, engine="pyarrow", index=False)

    print(f"Successfully saved filtered documents to {output_path}")


if __name__ == "__main__":
    main()
