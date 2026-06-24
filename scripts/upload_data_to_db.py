#!/usr/bin/env python3
"""
Upload pipeline data from parquet/CSV files to Supabase PostgreSQL.

Usage:
    export DATABASE_URL="postgresql://postgres:...@db....supabase.co:5432/postgres"
    python upload_data_to_db.py

Or:
    python upload_data_to_db.py --db-url "postgresql://..."
"""

import os
import sys
import re
import argparse
from pathlib import Path
from datetime import datetime

try:
    import psycopg2
    from psycopg2.extras import execute_values
except ImportError:
    print("ERROR: psycopg2 is required. Install with: pip install psycopg2-binary")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas is required. Install with: pip install pandas")
    sys.exit(1)


DATA_DIR = Path(__file__).resolve().parent / "data"
BATCH_SIZE = 1000


def get_db_connection(db_url):
    """Create and return a database connection with autocommit."""
    conn = psycopg2.connect(db_url)
    return conn


def batch_insert(conn, table_name, columns, rows, on_conflict=None):
    """
    Bulk-insert rows using execute_values with optional ON CONFLICT.
    
    Args:
        conn: psycopg2 connection
        table_name: target table
        columns: list of column names
        rows: iterable of tuples
        on_conflict: optional ON CONFLICT clause, e.g. "(id) DO NOTHING"
    """
    if not rows:
        print(f"  No rows to insert into {table_name}")
        return 0

    col_list = ", ".join(columns)
    conflict = f" ON CONFLICT {on_conflict}" if on_conflict else ""
    sql = f"INSERT INTO {table_name} ({col_list}) VALUES %s{conflict}"

    cur = conn.cursor()
    try:
        execute_values(cur, sql, rows, page_size=BATCH_SIZE)
        conn.commit()
        inserted = len(rows)
        cur.close()
        return inserted
    except Exception as e:
        conn.rollback()
        cur.close()
        raise e


def clean_nan(value):
    """Convert NaN/NaT to None for PostgreSQL compatibility."""
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    return value


def parse_date(val):
    """Parse date string to yyyy-mm-dd or None."""
    if val is None or pd.isna(val):
        return None
    val = str(val).strip()
    if not val or val == "...":
        return None
    # Handle dd/mm/yyyy format
    if re.match(r"^\d{1,2}/\d{1,2}/\d{4}$", val):
        parts = val.split("/")
        return f"{parts[2]}-{parts[1].zfill(2)}-{parts[0].zfill(2)}"
    # Handle yyyy-mm-dd format
    if re.match(r"^\d{4}-\d{2}-\d{2}$", val):
        return val
    # Fallback: try pandas parsing
    try:
        ts = pd.Timestamp(val)
        return ts.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return None


# ──────────────────────────────────────────────
# Upload Functions
# ──────────────────────────────────────────────

def upload_documents(conn):
    """Upload stage1_sme_docs.parquet → documents table."""
    path = DATA_DIR / "stage1_sme_docs.parquet"
    if not path.exists():
        print(f"  SKIP: {path} not found")
        return 0

    print(f"\n{'='*60}")
    print(f"Uploading: stage1_sme_docs.parquet → documents")
    print(f"{'='*60}")

    df = pd.read_parquet(path)
    print(f"  Loaded {len(df):,} rows from parquet")

    columns = [
        "id", "law_id", "ten_van_ban", "loai_van_ban", "nganh", "linh_vuc",
        "ngay_ban_hanh", "tinh_trang_hieu_luc", "ngay_co_hieu_luc", "ngay_het_hieu_luc"
    ]

    rows = []
    for _, r in df.iterrows():
        rows.append((
            int(r["id"]),
            str(r["law_id"]) if pd.notna(r["law_id"]) else "",
            str(r["ten_van_ban"]) if pd.notna(r["ten_van_ban"]) else "",
            clean_nan(r.get("loai_van_ban")),
            clean_nan(r.get("nganh")),
            clean_nan(r.get("linh_vuc")),
            parse_date(r.get("ngay_ban_hanh")),
            clean_nan(r.get("tinh_trang_hieu_luc")),
            parse_date(r.get("ngay_co_hieu_luc")),
            parse_date(r.get("ngay_het_hieu_luc")),
        ))

    count = batch_insert(conn, "documents", columns, rows,
                         on_conflict="(id) DO NOTHING")
    print(f"  Inserted {count:,} rows into documents")
    return count


def upload_articles(conn):
    """Upload stage2_articles.parquet → articles table."""
    path = DATA_DIR / "stage2_articles.parquet"
    if not path.exists():
        print(f"  SKIP: {path} not found")
        return 0

    print(f"\n{'='*60}")
    print(f"Uploading: stage2_articles.parquet → articles")
    print(f"{'='*60}")

    df = pd.read_parquet(path)
    print(f"  Loaded {len(df):,} rows from parquet")

    columns = [
        "doc_uid", "doc_id", "law_id", "ten_van_ban", "loai_van_ban",
        "ngay_ban_hanh", "nganh", "linh_vuc", "phan", "chuong", "muc",
        "dieu_so", "dieu_ten", "noi_dung", "start_char", "end_char"
    ]

    rows = []
    for _, r in df.iterrows():
        rows.append((
            str(r["doc_uid"]),
            int(r["doc_id"]),
            str(r["law_id"]) if pd.notna(r["law_id"]) else "",
            str(r["ten_van_ban"]) if pd.notna(r["ten_van_ban"]) else "",
            clean_nan(r.get("loai_van_ban")),
            parse_date(r.get("ngay_ban_hanh")),
            clean_nan(r.get("nganh")),
            clean_nan(r.get("linh_vuc")),
            clean_nan(r.get("phan")),
            clean_nan(r.get("chuong")),
            clean_nan(r.get("muc")),
            str(r["dieu_so"]),
            clean_nan(r.get("dieu_ten")),
            str(r["noi_dung"]),
            int(r["start_char"]) if pd.notna(r.get("start_char")) else None,
            int(r["end_char"]) if pd.notna(r.get("end_char")) else None,
        ))

    count = batch_insert(conn, "articles", columns, rows,
                         on_conflict="(doc_uid) DO NOTHING")
    print(f"  Inserted {count:,} rows into articles")
    return count


def upload_chunks(conn):
    """Upload stage3_chunks.parquet → chunks table."""
    path = DATA_DIR / "stage3_chunks.parquet"
    if not path.exists():
        print(f"  SKIP: {path} not found")
        return 0

    print(f"\n{'='*60}")
    print(f"Uploading: stage3_chunks.parquet → chunks")
    print(f"{'='*60}")

    df = pd.read_parquet(path)
    print(f"  Loaded {len(df):,} rows from parquet")

    columns = [
        "chunk_id", "doc_uid", "doc_id", "part_idx", "breadcrumb", "chunk_text"
    ]

    rows = []
    for _, r in df.iterrows():
        rows.append((
            str(r["chunk_id"]),
            str(r["doc_uid"]),
            int(r["doc_id"]),
            int(r["part_idx"]),
            str(r["breadcrumb"]) if pd.notna(r.get("breadcrumb")) else None,
            str(r["chunk_text"]),
        ))

    count = batch_insert(conn, "chunks", columns, rows,
                         on_conflict="(chunk_id) DO NOTHING")
    print(f"  Inserted {count:,} rows into chunks")
    return count


def upload_concepts(conn):
    """Upload concepts.csv → concepts table."""
    path = DATA_DIR / "concepts.csv"
    if not path.exists():
        print(f"  SKIP: {path} not found")
        return 0

    print(f"\n{'='*60}")
    print(f"Uploading: concepts.csv → concepts")
    print(f"{'='*60}")

    df = pd.read_csv(path)
    print(f"  Loaded {len(df):,} rows from CSV")

    columns = ["name", "name_lower"]
    rows = [(str(r["name"]), str(r["name_lower"])) for _, r in df.iterrows()]

    count = batch_insert(conn, "concepts", columns, rows,
                         on_conflict="(name) DO NOTHING")
    print(f"  Inserted {count:,} rows into concepts")
    return count


def upload_chunk_concept_mentions(conn):
    """Upload chunk_concept_mentions.csv → chunk_concept_mentions table."""
    path = DATA_DIR / "chunk_concept_mentions.csv"
    if not path.exists():
        print(f"  SKIP: {path} not found")
        return 0

    print(f"\n{'='*60}")
    print(f"Uploading: chunk_concept_mentions.csv → chunk_concept_mentions")
    print(f"{'='*60}")

    df = pd.read_csv(path)
    print(f"  Loaded {len(df):,} rows from CSV")

    columns = ["chunk_id", "doc_uid", "doc_id", "concept_name", "mentions_source"]
    rows = []
    for _, r in df.iterrows():
        rows.append((
            str(r["chunk_id"]),
            str(r["doc_uid"]),
            int(r["doc_id"]),
            str(r["concept_name"]),
            str(r["mentions_source"]),
        ))

    count = batch_insert(conn, "chunk_concept_mentions", columns, rows,
                         on_conflict="(chunk_id, concept_name) DO NOTHING")
    print(f"  Inserted {count:,} rows into chunk_concept_mentions")
    return count


def upload_chunk_processing(conn):
    """Upload chunk_processing.csv → chunk_processing table."""
    path = DATA_DIR / "chunk_processing.csv"
    if not path.exists():
        print(f"  SKIP: {path} not found")
        return 0

    print(f"\n{'='*60}")
    print(f"Uploading: chunk_processing.csv → chunk_processing")
    print(f"{'='*60}")

    df = pd.read_csv(path)
    print(f"  Loaded {len(df):,} rows from CSV")

    columns = [
        "chunk_id", "doc_uid", "doc_id", "mentions_source",
        "concepts_found", "concepts_count"
    ]
    rows = []
    for _, r in df.iterrows():
        concepts_raw = str(r["concepts_found"]) if pd.notna(r.get("concepts_found")) else "{}"
        # Convert {a,b,c} format or bare text to PostgreSQL array literal
        pg_array = concepts_raw.strip()
        if pg_array and not pg_array.startswith("{"):
            pg_array = "{" + pg_array + "}"
        rows.append((
            str(r["chunk_id"]),
            str(r["doc_uid"]),
            int(r["doc_id"]),
            clean_nan(r.get("mentions_source")),
            pg_array,
            int(r["concepts_count"]) if pd.notna(r.get("concepts_count")) else 0,
        ))

    count = batch_insert(conn, "chunk_processing", columns, rows,
                         on_conflict="(chunk_id) DO NOTHING")
    print(f"  Inserted {count:,} rows into chunk_processing")
    return count


def verify_counts(conn):
    """Print row counts for all tables."""
    print(f"\n{'='*60}")
    print(f"Verification: table row counts")
    print(f"{'='*60}")

    cur = conn.cursor()
    tables = [
        "documents", "articles", "chunks", "concepts",
        "chunk_concept_mentions", "chunk_processing", "pipeline_metadata"
    ]
    total = 0
    for t in tables:
        cur.execute(f"SELECT COUNT(*) FROM {t}")
        cnt = cur.fetchone()[0]
        print(f"  {t:30s} {cnt:>8,} rows")
        total += cnt
    print(f"  {'─'*40}")
    print(f"  {'TOTAL':30s} {total:>8,} rows")

    # Check view
    cur.execute("SELECT COUNT(*) FROM v_article_details")
    cnt = cur.fetchone()[0]
    print(f"  {'v_article_details (view)':30s} {cnt:>8,} rows")

    cur.close()


def main():
    parser = argparse.ArgumentParser(description="Upload pipeline data to Supabase")
    parser.add_argument("--db-url", default=None,
                        help="PostgreSQL connection string (default: DATABASE_URL env var)")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip final verification")
    args = parser.parse_args()

    db_url = args.db_url or os.environ.get("DATABASE_URL")
    if not db_url:
        print("ERROR: No database URL. Provide --db-url or set DATABASE_URL env var.")
        sys.exit(1)

    conn = get_db_connection(db_url)

    counts = {}
    try:
        counts["documents"] = upload_documents(conn)
        counts["articles"] = upload_articles(conn)
        counts["chunks"] = upload_chunks(conn)
        counts["concepts"] = upload_concepts(conn)
        counts["chunk_concept_mentions"] = upload_chunk_concept_mentions(conn)
        counts["chunk_processing"] = upload_chunk_processing(conn)
    except Exception as e:
        print(f"\nERROR during upload: {e}")
        conn.close()
        sys.exit(1)

    if not args.no_verify:
        verify_counts(conn)

    conn.close()

    print(f"\n{'='*60}")
    print(f"Upload complete!")
    print(f"{'='*60}")
    total = sum(counts.values())
    for table, cnt in counts.items():
        print(f"  {table:30s} {cnt:>8,} rows")
    print(f"  {'─'*40}")
    print(f"  {'TOTAL':30s} {total:>8,} rows")


if __name__ == "__main__":
    main()
