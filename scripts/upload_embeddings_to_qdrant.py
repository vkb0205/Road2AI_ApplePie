"""Upload the Stage 6 BGE-M3 embeddings into Qdrant (user-requested add-on).

The Stage 6 artifact bundle stores embeddings only inside the single FAISS
``IndexFlatIP`` (no standalone ``.npy``). This script reconstructs those
vectors via :meth:`retrieval.faiss_index.FAISSIndex.reconstruct_rows` and
upserts them into a Qdrant collection as points, keyed by the canonical
Stage 6 ``row_idx`` so the retrieval pipeline needs no changes.

Each point payload carries the ``chunk_meta_slim`` fields
(``row_idx, chunk_id, doc_uid, law_id, ten_van_ban, dieu_so``). Optionally
the full ``chunk_text`` can be included (pulled from the SQLite bundle) so
Qdrant becomes a self-contained retrieval+context store.

Usage::

    # Upload vectors + slim metadata (recommended; text stays in SQLite)
    python scripts/upload_embeddings_to_qdrant.py \\
        --collection glrag_bge_m3 \\
        --batch-size 1000

    # Also embed the full chunk_text in each Qdrant payload
    python scripts/upload_embeddings_to_qdrant.py --include-text

    # Dry-run: validate the bundle + print the plan without contacting Qdrant
    python scripts/upload_embeddings_to_qdrant.py --dry-run

Environment
-----------
``QDRANT_URL`` (required) and ``QDRANT_API_KEY`` (required for Qdrant Cloud)
are read from the environment / ``.env`` (loaded via python-dotenv when
available). They can also be passed with ``--url`` / ``--api-key``.

Notes
-----
* ``faiss`` and ``qdrant-client`` are imported lazily inside the script body,
  so ``--dry-run`` works without them installed (it only validates the
  parquet + model-meta).
* Vectors are reconstructed in batches to bound peak memory (~1000 vectors
  × 1024 dims × 4 bytes ≈ 4 MB per batch).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

STAGE6 = ROOT / "data" / "stage6_data"
DB = STAGE6 / "chunk_store.sqlite"
META = STAGE6 / "chunk_meta_slim.parquet"
FAISS_IDX = STAGE6 / "faiss_index__BAAI_bge-m3.index"
MODEL_META = STAGE6 / "embed_model_meta__BAAI_bge-m3.json"


def _load_env() -> None:
    """Populate os.environ from .env if python-dotenv is available."""
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except Exception:
        pass


def main(argv=None) -> int:
    _load_env()
    p = argparse.ArgumentParser(
        description="Upload Stage 6 BGE-M3 embeddings into Qdrant"
    )
    p.add_argument("--faiss-index", default=str(FAISS_IDX))
    p.add_argument("--meta", default=str(META))
    p.add_argument("--model-meta", default=str(MODEL_META))
    p.add_argument("--sqlite", default=str(DB),
                   help="chunk_store.sqlite (used only with --include-text)")
    p.add_argument("--collection", default="glrag_bge_m3")
    p.add_argument("--vector-name", default="dense")
    p.add_argument("--url", default=os.environ.get("QDRANT_URL"),
                   help="Qdrant endpoint (default: $QDRANT_URL)")
    p.add_argument("--api-key", default=os.environ.get("QDRANT_API_KEY"),
                   help="Qdrant API key (default: $QDRANT_API_KEY)")
    p.add_argument("--batch-size", type=int, default=1000)
    p.add_argument("--include-text", action="store_true",
                   help="store full chunk_text in each Qdrant payload")
    p.add_argument("--dry-run", action="store_true",
                   help="validate the bundle + print the plan; no Qdrant calls")
    p.add_argument("--limit", type=int, default=0,
                   help="upload only the first N points (0 = all)")
    args = p.parse_args(argv)

    # ---- validate bundle (no heavy deps needed for meta/model-meta) ----- #
    print("[qdrant-upload] validating Stage 6 bundle ...")
    if not Path(args.faiss_index).exists():
        print(f"  FAISS index missing: {args.faiss_index}")
        return 2
    if not Path(args.meta).exists():
        print(f"  metadata sidecar missing: {args.meta}")
        return 2
    model_meta = json.loads(Path(args.model_meta).read_text(encoding="utf-8"))
    print(f"  embed_model : {model_meta.get('embed_model')}")
    print(f"  dim         : {model_meta.get('dim')}")
    print(f"  count       : {model_meta.get('count')}")

    if args.dry_run:
        print("[qdrant-upload] DRY RUN — no Qdrant calls will be made.")
        print(f"  collection  : {args.collection}")
        print(f"  vector_name : {args.vector_name}")
        print(f"  batch_size  : {args.batch_size}")
        print(f"  include_text: {args.include_text}")
        print(f"  limit       : {args.limit or 'all'}")
        return 0

    # ---- real run: needs faiss + qdrant-client --------------------------- #
    if not args.url:
        print("[qdrant-upload] ERROR: no Qdrant URL. Set QDRANT_URL in the "
              "environment or pass --url.")
        return 2

    from retrieval.faiss_index import FAISSIndex
    from retrieval.qdrant_index import QdrantUploader

    print("[qdrant-upload] loading FAISS index + metadata ...")
    t0 = time.time()
    faiss_index = FAISSIndex(
        args.faiss_index, args.meta, args.model_meta
    ).load_index()
    print(f"  loaded {faiss_index.ntotal:,} vectors (dim {faiss_index.dim}) "
          f"in {time.time()-t0:.1f}s")

    text_lookup = None
    if args.include_text:
        from retrieval.bm25_index import FTSIndex

        print("[qdrant-upload] opening SQLite for chunk_text lookup ...")
        text_lookup = FTSIndex(args.sqlite, mode="fts_fast").open()

    uploader = QdrantUploader(
        collection=args.collection,
        url=args.url,
        api_key=args.api_key,
        vector_name=args.vector_name,
        dim=faiss_index.dim,
        batch_size=args.batch_size,
    )

    n_total = faiss_index.ntotal
    if args.limit and args.limit < n_total:
        print(f"[qdrant-upload] LIMIT: uploading only the first {args.limit} "
              f"of {n_total:,} points")
        # Truncate by wrapping reconstruct_rows to only the requested slice.
        faiss_index = _SlicedFAISS(faiss_index, args.limit)

    t0 = time.time()
    uploaded = uploader.upload_from_faiss(
        faiss_index, text_lookup=text_lookup, include_text=args.include_text
    )
    elapsed = time.time() - t0
    rate = uploaded / elapsed if elapsed else 0.0
    print(f"[qdrant-upload] done: {uploaded:,} points in {elapsed:.1f}s "
          f"({rate:.0f} pts/s)")
    if text_lookup is not None:
        text_lookup.close()
    return 0


class _SlicedFAISS:
    """Adapter that limits a FAISSIndex to its first ``limit`` rows.

    Wraps the real index so :meth:`QdrantUploader.upload_from_faiss` only
    uploads ``limit`` points while reusing the same metadata sidecar.
    """

    def __init__(self, inner, limit: int) -> None:
        self._inner = inner
        self._limit = int(limit)

    @property
    def ntotal(self) -> int:
        return self._limit

    @property
    def dim(self) -> int:
        return self._inner.dim

    def meta_for_row(self, row_idx: int):
        return self._inner.meta_for_row(row_idx)

    def reconstruct_rows(self, row_idxs):
        return self._inner.reconstruct_rows(row_idxs)


if __name__ == "__main__":
    sys.exit(main())
