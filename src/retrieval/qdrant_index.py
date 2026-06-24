"""Qdrant dense-retrieval backend for G-LRAG (user-requested Stage 7 add-on).

The user asked to upload the Stage 6 embeddings into **Qdrant** (in addition
to the SQLite/FAISS bundle) for easy retrieval. This module provides a
:class:`QdrantIndex` with the **same ``search`` contract as**
:class:`retrieval.faiss_index.FAISSIndex`, so the :class:`HybridRetriever`
can use either backend interchangeably for the dense leg.

Design notes
------------
* ``qdrant-client`` is imported **lazily**, so this module imports & unit-
  tests on machines without it (matching the lazy-import discipline of the
  rest of the retrieval package).
* Point ids are the canonical Stage 6 ``row_idx`` integers (Qdrant supports
  integer point ids). This preserves the bundle invariant::

      Qdrant point id i == chunk_meta_slim.row_idx i == chunks.row_idx i

  so the rest of the pipeline (RRF, graph expansion, ``make_relevant_lists``)
  needs no changes.
* Payloads mirror the ``chunk_meta_slim`` columns: ``row_idx, chunk_id,
  doc_uid, law_id, ten_van_ban, dieu_so`` — enough to build
  ``relevant_docs`` / ``relevant_articles`` and to fetch full ``chunk_text``
  from SQLite.
* Distance: Qdrant ``Cosine`` with pre-normalised vectors reproduces the
  Stage 6 ``IndexFlatIP`` (cosine-via-inner-product) ranking exactly.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

__all__ = ["QdrantIndex", "QdrantUploader", "DEFAULT_COLLECTION"]

DEFAULT_COLLECTION = "glrag_bge_m3"
DEFAULT_VECTOR_NAME = "dense"
DEFAULT_DIM = 1024

# Payload field names (must match chunk_meta_slim schema).
_PAYLOAD_FIELDS = ("row_idx", "chunk_id", "doc_uid", "law_id",
                   "ten_van_ban", "dieu_so")


class QdrantIndex:
    """Read-only Qdrant dense backend exposing the FAISSIndex.search contract.

    Parameters
    ----------
    collection:
        Qdrant collection name (default ``glrag_bge_m3``).
    url / api_key:
        Qdrant endpoint. If omitted, reads ``QDRANT_URL`` / ``QDRANT_API_KEY``
        from the environment (loaded via ``python-dotenv`` if available).
    vector_name:
        Named vector field in the collection (default ``dense``).
    prefer_grpc:
        Use the gRPC transport when ``url`` is a plain host:port (faster for
        large batched searches). Defaults to False for cloud endpoints.

    Call :meth:`ensure_ready` (returns ``self``) before :meth:`search`.
    """

    def __init__(
        self,
        collection: str = DEFAULT_COLLECTION,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        vector_name: str = DEFAULT_VECTOR_NAME,
        prefer_grpc: bool = False,
        timeout: int = 60,
    ) -> None:
        self.collection = collection
        self.vector_name = vector_name
        self.prefer_grpc = prefer_grpc
        self.timeout = timeout
        self.url = url or os.environ.get("QDRANT_URL")
        self.api_key = api_key or os.environ.get("QDRANT_API_KEY")
        if self.url is None:
            # Try loading from a .env file if python-dotenv is around.
            try:
                from dotenv import load_dotenv

                load_dotenv()
                self.url = os.environ.get("QDRANT_URL")
                self.api_key = self.api_key or os.environ.get("QDRANT_API_KEY")
            except Exception:
                pass
        self._client: Any = None
        self._dim: int = DEFAULT_DIM
        self._n_points: int = 0

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def ensure_ready(self) -> "QdrantIndex":
        """Connect to Qdrant and verify the collection exists & is non-empty."""
        from qdrant_client import QdrantClient  # lazy

        if self.url is None:
            raise RuntimeError(
                "QdrantIndex: no QDRANT_URL configured. Set QDRANT_URL (and "
                "QDRANT_API_KEY for cloud) in the environment or .env."
            )
        kwargs: Dict[str, Any] = {"url": self.url, "timeout": self.timeout}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.prefer_grpc:
            kwargs["prefer_grpc"] = True
        self._client = QdrantClient(**kwargs)

        info = self._client.get_collection(self.collection)
        cfg = getattr(info, "config", None)
        vcfg = getattr(cfg, "params", None)
        vsize = getattr(vcfg, "vectors", None)
        # Named-vector config → vsize is a dict of VectorParams.
        if isinstance(vsize, dict) and self.vector_name in vsize:
            self._dim = int(vsize[self.vector_name].size)
        elif hasattr(vsize, "size"):
            self._dim = int(vsize.size)
        self._n_points = int(getattr(info, "points_count", 0) or 0)
        return self

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    @property
    def ntotal(self) -> int:
        return self._n_points

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def embed_model(self) -> str:
        return "BAAI/bge-m3"

    # ------------------------------------------------------------------ #
    # Search (same contract as FAISSIndex.search)
    # ------------------------------------------------------------------ #
    def search(self, query_vec: Any, top_k: int = 50) -> List[Dict[str, Any]]:
        """Dense search with a pre-encoded, L2-normalised query vector.

        ``query_vec`` is a 1-D or 2-D float32 vector. Returns hit dicts
        with ``row_idx`` (the Qdrant point id), ``dense_score``,
        ``dense_rank`` and the carried metadata payload.
        """
        if self._client is None:
            raise RuntimeError("QdrantIndex not ready; call .ensure_ready()")

        import numpy as np

        q = np.asarray(query_vec, dtype="float32")
        if q.ndim == 2:
            q = q.reshape(-1)
        if q.shape[0] != self._dim:
            raise ValueError(
                f"QdrantIndex: query dim {q.shape[0]} != collection dim "
                f"{self._dim}"
            )

        try:
            from qdrant_client.models import (
                QueryRequest,
                NamedQuery,
            )
        except Exception:
            QueryRequest = None  # type: ignore

        if QueryRequest is not None:
            # Newer qdrant-client: query_points with named vector.
            query = NamedQuery(
                name=self.vector_name,
                query=q.tolist(),
            )
            res = self._client.query_points(
                collection_name=self.collection,
                query=query,
                limit=int(top_k),
                with_payload=True,
                with_vector=False,
            )
            points = res.points
        else:
            # Older API fallback: search with vector= kwarg.
            res = self._client.search(
                collection_name=self.collection,
                query_vector=(self.vector_name, q.tolist()),
                limit=int(top_k),
                with_payload=True,
                with_vector=False,
            )
            points = res

        out: List[Dict[str, Any]] = []
        for rank, point in enumerate(points, start=1):
            payload = getattr(point, "payload", {}) or {}
            row_idx = int(getattr(point, "id", payload.get("row_idx", -1)))
            out.append({
                "row_idx": row_idx,
                "dense_rank": rank,
                "dense_score": float(getattr(point, "score", 0.0)),
                "chunk_id": str(payload.get("chunk_id", "")),
                "doc_uid": str(payload.get("doc_uid", "")),
                "law_id": str(payload.get("law_id", "")),
                "ten_van_ban": str(payload.get("ten_van_ban", "")),
                "dieu_so": str(payload.get("dieu_so", "")),
            })
        return out


class QdrantUploader:
    """Upload the Stage 6 embeddings + metadata into a Qdrant collection.

    Vectors are reconstructed from the Stage 6 ``IndexFlatIP`` via
    :meth:`retrieval.faiss_index.FAISSIndex.reconstruct_all`, so no raw
    ``.npy`` is required. Metadata payloads come from
    ``chunk_meta_slim.parquet``.

    Example
    -------
    >>> from retrieval.faiss_index import FAISSIndex
    >>> from retrieval.qdrant_index import QdrantUploader
    >>> faiss_idx = FAISSIndex(idx_path, meta_path, model_meta_path).load_index()
    >>> QdrantUploader(collection="glrag_bge_m3").upload_from_faiss(faiss_idx)
    """

    def __init__(
        self,
        collection: str = DEFAULT_COLLECTION,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        vector_name: str = DEFAULT_VECTOR_NAME,
        dim: int = DEFAULT_DIM,
        on_disk_payload: bool = True,
        batch_size: int = 1000,
    ) -> None:
        self.collection = collection
        self.vector_name = vector_name
        self.dim = dim
        self.on_disk_payload = on_disk_payload
        self.batch_size = batch_size
        self.url = url or os.environ.get("QDRANT_URL")
        self.api_key = api_key or os.environ.get("QDRANT_API_KEY")
        self._client: Any = None

    def _connect(self) -> Any:
        from qdrant_client import QdrantClient  # lazy
        from qdrant_client.models import (
            VectorParams,
            Distance,
            HnswConfigDiff,
            OptimizersConfigDiff,
        )

        if self.url is None:
            raise RuntimeError(
                "QdrantUploader: no QDRANT_URL configured. Set QDRANT_URL "
                "(and QDRANT_API_KEY for cloud) in the environment or .env."
            )
        kwargs: Dict[str, Any] = {"url": self.url, "timeout": 120}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        self._client = QdrantClient(**kwargs)
        return self._client

    def _ensure_collection(self) -> None:
        from qdrant_client.models import (
            VectorParams,
            Distance,
            HnswConfigDiff,
            OptimizersConfigDiff,
        )

        collections = {c.name for c in self._client.get_collections().collections}
        if self.collection in collections:
            return
        self._client.create_collection(
            collection_name=self.collection,
            vectors_config={
                self.vector_name: VectorParams(
                    size=self.dim,
                    distance=Distance.COSINE,
                    on_disk=True,
                )
            },
            hnsw_config=HnswConfigDiff(m=16, ef_construct=100, on_disk=True),
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=20000,
                flush_interval_sec=15,
            ),
            on_disk_payload=self.on_disk_payload,
            shard_number=1,
        )

    def upload_from_faiss(
        self,
        faiss_index: Any,
        text_lookup: Optional[Any] = None,
        include_text: bool = False,
    ) -> int:
        """Reconstruct vectors from ``faiss_index`` and upsert into Qdrant.

        Parameters
        ----------
        faiss_index:
            A loaded :class:`retrieval.faiss_index.FAISSIndex`.
        text_lookup:
            Optional callable ``row_idx -> chunk_text`` (e.g. an
            :class:`FTSIndex`) used only when ``include_text=True``.
        include_text:
            If True, add a ``chunk_text`` field to each payload. Off by
            default to keep payloads small (text lives in SQLite).

        Returns
        -------
        int
            Number of points upserted.
        """
        import numpy as np
        from qdrant_client.models import PointStruct, VectorStruct

        client = self._connect()
        self._ensure_collection()

        n = faiss_index.ntotal
        dim = faiss_index.dim
        self.dim = dim
        total = 0
        upload_batch = self.batch_size
        for start in range(0, n, upload_batch):
            stop = min(start + upload_batch, n)
            vectors = faiss_index.reconstruct_rows(range(start, stop))
            points: List[Any] = []
            for offset in range(stop - start):
                row_idx = start + offset
                meta = faiss_index.meta_for_row(row_idx)
                payload: Dict[str, Any] = {f: meta[f] for f in _PAYLOAD_FIELDS}
                if include_text and text_lookup is not None:
                    rows = text_lookup.fetch_chunks([row_idx])
                    if rows:
                        payload["chunk_text"] = rows[0].get("chunk_text", "")
                points.append(
                    PointStruct(
                        id=int(row_idx),
                        vector={self.vector_name: vectors[offset].tolist()},
                        payload=payload,
                    )
                )
            client.upsert(
                collection_name=self.collection,
                points=points,
                wait=False,
            )
            total += len(points)
            print(f"  [qdrant] upserted {total:,}/{n:,} points")
        # Flush so count() reflects the upload.
        client.update_collection(
            collection_name=self.collection,
            optimizers_config=None,
        )
        return total
