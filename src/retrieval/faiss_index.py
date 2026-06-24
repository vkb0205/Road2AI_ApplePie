"""FAISS dense-retrieval backend for G-LRAG (Stage 6 bundle).

Wraps the single Stage 6 ``IndexFlatIP`` (dim 1024, L2-normalised = cosine)
and the ``chunk_meta_slim.parquet`` sidecar. The Stage 6 bundle invariant::

    FAISS index position i
    == chunk_meta_slim.row_idx i
    == chunks.row_idx i
    == chunks_fts.rowid i

is asserted on load (PLAN.md task 6.4).

Heavy deps (``faiss``, ``numpy``, ``FlagEmbedding``, ``torch``) are imported
**lazily** so this module imports & unit-tests on a CPU-only machine that
has none of them installed. The dense leg only activates when the caller
explicitly constructs and loads this object.

This module also provides :meth:`FAISSIndex.reconstruct_all` which rebuilds
the raw embedding matrix from the flat index — used by the Qdrant uploader
(``scripts/upload_embeddings_to_qdrant.py``) since the Stage 6 bundle stores
no standalone ``.npy`` file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["FAISSIndex", "BGEQueryEncoder"]


class FAISSIndex:
    """Lazy FAISS wrapper over the Stage 6 ``IndexFlatIP`` + metadata sidecar.

    Parameters
    ----------
    index_path:
        Path to ``faiss_index__BAAI_bge-m3.index``.
    meta_path:
        Path to ``chunk_meta_slim.parquet``.
    model_meta_path:
        Path to ``embed_model_meta__BAAI_bge-m3.json``.

    Call :meth:`load_index` (returns ``self``) before :meth:`search`.
    """

    def __init__(
        self,
        index_path: str,
        meta_path: str,
        model_meta_path: str,
    ) -> None:
        self.index_path = index_path
        self.meta_path = meta_path
        self.model_meta_path = model_meta_path
        self._index: Any = None  # faiss.Index
        self._meta: Any = None   # pandas.DataFrame
        self._model_meta: Dict[str, Any] = {}
        self._dim: int = 0
        self._ntotal: int = 0

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def load_index(self) -> "FAISSIndex":
        """Load & validate the FAISS index, metadata sidecar and model meta.

        Asserts the Stage 6 bundle contract (row_idx contiguous 0..N-1 and
        aligned with ``index.ntotal``). Returns ``self`` for chaining.
        """
        import numpy as np  # noqa: F401  (validate availability)
        import pandas as pd
        import faiss

        if not Path(self.index_path).exists():
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")
        self._index = faiss.read_index(self.index_path)
        self._ntotal = int(self._index.ntotal)
        self._dim = int(self._index.d)

        self._meta = pd.read_parquet(self.meta_path)
        self._model_meta = json.loads(
            Path(self.model_meta_path).read_text(encoding="utf-8")
        )

        # ---- bundle-contract assertions (artifacts_guide.md §6.4) ----
        if int(self._model_meta.get("dim", -1)) != self._dim:
            raise ValueError(
                f"FAISS dim {self._dim} != model_meta dim "
                f"{self._model_meta.get('dim')}"
            )
        if int(self._model_meta.get("count", -1)) != self._ntotal:
            raise ValueError(
                f"FAISS ntotal {self._ntotal} != model_meta count "
                f"{self._model_meta.get('count')}"
            )
        row_idx = self._meta["row_idx"].to_numpy()
        if len(self._meta) != self._ntotal:
            raise ValueError(
                f"meta rows {len(self._meta)} != faiss ntotal {self._ntotal}"
            )
        import numpy as _np
        if not _np.array_equal(row_idx, _np.arange(self._ntotal)):
            raise ValueError(
                "chunk_meta_slim.row_idx is not contiguous 0..N-1; the FAISS "
                "index position would not align. Refusing to use meta.iloc."
            )
        # Row-locate index for safe O(1) lookup by row_idx even if a future
        # build reorders metadata.
        self._meta_by_row = self._meta.set_index("row_idx", drop=False)
        return self

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    @property
    def ntotal(self) -> int:
        return self._ntotal

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def embed_model(self) -> str:
        return str(self._model_meta.get("embed_model", ""))

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #
    def search(self, query_vec: Any, top_k: int = 50) -> List[Dict[str, Any]]:
        """Dense search with a pre-encoded, L2-normalised query vector.

        ``query_vec`` is a 2-D float32 array of shape ``(1, dim)``. Returns
        hit dicts with ``row_idx``, ``dense_score``, ``dense_rank`` and the
        carried metadata (``chunk_id``, ``doc_uid``, ``law_id``,
        ``ten_van_ban``, ``dieu_so``).
        """
        if self._index is None:
            raise RuntimeError("FAISSIndex not loaded; call .load_index()")
        import numpy as np

        q = np.ascontiguousarray(query_vec, dtype="float32")
        if q.ndim == 1:
            q = q.reshape(1, -1)
        scores, idx = self._index.search(q, int(top_k))
        out: List[Dict[str, Any]] = []
        for rank, (row_idx, score) in enumerate(zip(idx[0], scores[0]), start=1):
            if int(row_idx) < 0:
                continue
            row_idx = int(row_idx)
            meta_row = self._meta_by_row.loc[row_idx]
            out.append({
                "row_idx": row_idx,
                "dense_rank": rank,
                "dense_score": float(score),
                "chunk_id": str(meta_row["chunk_id"]),
                "doc_uid": str(meta_row["doc_uid"]),
                "law_id": str(meta_row["law_id"]),
                "ten_van_ban": str(meta_row["ten_van_ban"]),
                "dieu_so": str(meta_row["dieu_so"]),
            })
        return out

    # ------------------------------------------------------------------ #
    # Vector reconstruction (for Qdrant upload)
    # ------------------------------------------------------------------ #
    def reconstruct_all(self, batch: int = 4096) -> Any:
        """Reconstruct the full embedding matrix from the flat index.

        ``IndexFlatIP`` stores raw vectors, so ``reconstruct_n`` works. This
        is the bridge between the Stage 6 FAISS bundle and the Qdrant upload
        (which needs per-point vectors). Returns a ``(N, dim)`` float32 numpy
        array. ``batch`` controls the per-call reconstruction chunk size to
        bound peak memory.
        """
        if self._index is None:
            raise RuntimeError("FAISSIndex not loaded; call .load_index()")
        import numpy as np

        n, d = self._ntotal, self._dim
        out = np.empty((n, d), dtype="float32")
        for start in range(0, n, batch):
            stop = min(start + batch, n)
            out[start:stop] = self._index.reconstruct_n(start, stop - start)
        return out

    def reconstruct_rows(self, row_idxs: Sequence[int]) -> Any:
        """Reconstruct vectors for a subset of row indices."""
        if self._index is None:
            raise RuntimeError("FAISSIndex not loaded; call .load_index()")
        import numpy as np

        idxs = [int(x) for x in row_idxs]
        out = np.empty((len(idxs), self._dim), dtype="float32")
        for i, ridx in enumerate(idxs):
            out[i] = self._index.reconstruct(int(ridx))
        return out

    # ------------------------------------------------------------------ #
    # Metadata helpers
    # ------------------------------------------------------------------ #
    def meta_for_row(self, row_idx: int) -> Dict[str, Any]:
        if self._index is None:
            raise RuntimeError("FAISSIndex not loaded; call .load_index()")
        row = self._meta_by_row.loc[int(row_idx)]
        return {
            "row_idx": int(row["row_idx"]),
            "chunk_id": str(row["chunk_id"]),
            "doc_uid": str(row["doc_uid"]),
            "law_id": str(row["law_id"]),
            "ten_van_ban": str(row["ten_van_ban"]),
            "dieu_so": str(row["dieu_so"]),
        }

    def iter_meta_rows(self, batch: int = 4096):
        """Yield ``(row_idx, meta_dict)`` over the whole sidecar in batches."""
        if self._index is None:
            raise RuntimeError("FAISSIndex not loaded; call .load_index()")
        import pandas as pd

        n = len(self._meta)
        for start in range(0, n, batch):
            chunk = self._meta.iloc[start:start + batch]
            for _, row in chunk.iterrows():
                yield int(row["row_idx"]), {
                    "row_idx": int(row["row_idx"]),
                    "chunk_id": str(row["chunk_id"]),
                    "doc_uid": str(row["doc_uid"]),
                    "law_id": str(row["law_id"]),
                    "ten_van_ban": str(row["ten_van_ban"]),
                    "dieu_so": str(row["dieu_so"]),
                }


class BGEQueryEncoder:
    """Lazily-loaded ``BAAI/bge-m3`` query encoder.

    Encodes a query string into an L2-normalised float32 dense vector ready
    for :meth:`FAISSIndex.search`. Heavy deps (``FlagEmbedding``, ``torch``)
    are imported on first :meth:`encode`.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        max_length: int = 512,
        use_fp16: bool = True,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.use_fp16 = use_fp16
        self._encoder: Any = None

    def _ensure_loaded(self) -> None:
        if self._encoder is not None:
            return
        from FlagEmbedding import BGEM3FlagModel  # lazy

        self._encoder = BGEM3FlagModel(
            self.model_name, use_fp16=self.use_fp16
        )

    def encode(self, query: str) -> Any:
        """Encode a single query → ``(1, dim)`` L2-normalised float32 array."""
        import numpy as np

        self._ensure_loaded()
        out = self._encoder.encode(
            [query],
            batch_size=1,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        qemb = out["dense_vecs"].astype("float32")
        qemb = qemb / np.maximum(
            np.linalg.norm(qemb, axis=1, keepdims=True), 1e-12
        )
        return np.ascontiguousarray(qemb)

    def encode_batch(self, queries: Sequence[str]) -> Any:
        """Encode a batch of queries → ``(N, dim)`` L2-normalised float32."""
        import numpy as np

        self._ensure_loaded()
        out = self._encoder.encode(
            list(queries),
            batch_size=24,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        qemb = out["dense_vecs"].astype("float32")
        qemb = qemb / np.maximum(
            np.linalg.norm(qemb, axis=1, keepdims=True), 1e-12
        )
        return np.ascontiguousarray(qemb)
