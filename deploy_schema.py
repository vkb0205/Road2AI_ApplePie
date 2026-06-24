#!/usr/bin/env python3
"""Deploy SQL schema files to Supabase PostgreSQL database."""

import os
import sys
import argparse
from pathlib import Path

try:
    import psycopg2
except ImportError:
    print("ERROR: psycopg2 is required. Install with: pip install psycopg2-binary")
    sys.exit(1)

def load_sql(filepath: str) -> str:
    """Load a SQL file, stripping comment-only lines for cleaner execution."""
    path = Path(filepath)
    if not path.exists():
        print(f"ERROR: SQL file not found: {filepath}")
        sys.exit(1)
    
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    print(f"  Loaded {path.name} ({len(content)} bytes)")
    return content


def main():
    parser = argparse.ArgumentParser(description="Deploy SQL schema to Supabase")
    parser.add_argument(
        "--sql-dir",
        default=None,
        help="Directory containing SQL files (default: Road2AI_ApplePie/sql)",
    )
    parser.add_argument(
        "--db-url",
        default=None,
        help="PostgreSQL connection string (default: DATABASE_URL env var)",
    )
    args = parser.parse_args()

    # Determine SQL directory
    base_dir = Path(__file__).resolve().parent
    sql_dir = Path(args.sql_dir) if args.sql_dir else base_dir / "sql"
    
    if not sql_dir.exists():
        print(f"ERROR: SQL directory not found: {sql_dir}")
        sys.exit(1)

    # Get database URL
    db_url = args.db_url or os.environ.get("DATABASE_URL")
    if not db_url:
        print("ERROR: No database URL. Provide --db-url or set DATABASE_URL env var.")
        sys.exit(1)

    # Define SQL files in order
    sql_files = [
        sql_dir / "00_schema.sql",
        sql_dir / "01_chunk_concept_mentions.sql",
        sql_dir / "02_chunk_fts.sql",
    ]

    print(f"Connecting to Supabase PostgreSQL...")
    conn = psycopg2.connect(db_url)
    conn.autocommit = True
    cur = conn.cursor()

    for sql_file in sql_files:
        if not sql_file.exists():
            print(f"  SKIP: {sql_file.name} (not found)")
            continue

        print(f"\n{'='*60}")
        print(f"Executing: {sql_file.name}")
        print(f"{'='*60}")
        
        sql = load_sql(str(sql_file))
        
        try:
            cur.execute(sql)
            print(f"  SUCCESS: {sql_file.name} executed without errors.")
        except Exception as e:
            print(f"  ERROR: {e}")
            # Don't exit — try the next file
            print(f"  CONTINUING to next file...")

    cur.close()
    conn.close()
    print(f"\n{'='*60}")
    print("Schema deployment complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
