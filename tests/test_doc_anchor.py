"""Unit tests for document-anchored retrieval (no GPU deps).

Exercises the pure-Python orchestration with injected fake search / text /
rerank callables:
  - chunk->document ranking collapse (best chunk wins the doc rank);
  - RRF fusion + authority prior at the document stage;
  - within-document article harvesting (anchored filter, best chunk per Điều,
    per-document cap);
  - DocAnchoredRetriever end-to-end in lexical-only, dense, and reranked modes;
  - pool_record serialisation to the tune_selector schema.

No torch/faiss/FlagEmbedding imports are triggered.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from retrieval.doc_anchor import (  # noqa: E402
    AnchoredDoc,
    DocAnchorConfig,
    DocAnchoredRetriever,
    anchor_documents,
    harvest_articles,
    pool_record,
)
from retrieval.article_select import ArticleCandidate  # noqa: E402


def _hit(row_idx, law_id, dieu, ten="", doc_uid=None, **scores):
    h = {
        "row_idx": row_idx,
        "law_id": law_id,
        "ten_van_ban": ten or f"VB {law_id}",
        "dieu_so": dieu,
        "doc_uid": doc_uid if doc_uid is not None else law_id,
    }
    h.update(scores)
    return h


# --------------------------------------------------------------------------- #
# Stage A: document anchoring
# --------------------------------------------------------------------------- #
def test_anchor_fuses_and_dedupes_documents():
    lex = [
        _hit(1, "01/2021/NĐ-CP", "Điều 1", bm25_score=5.0),
        _hit(2, "01/2021/NĐ-CP", "Điều 12", bm25_score=4.0),
        _hit(3, "1934/2007/QĐ-UBND", "Điều 4", bm25_score=3.0),
    ]
    den = [
        _hit(4, "01/2021/NĐ-CP", "Điều 1", dense_score=0.8),
        _hit(5, "65/2023/NĐ-CP", "Điều 10", dense_score=0.7),
    ]
    anchored = anchor_documents(lex, den, DocAnchorConfig(top_docs=10))
    keys = [d.doc_uid for d in anchored]
    # Three distinct documents, deduped.
    assert set(keys) == {"01/2021/NĐ-CP", "1934/2007/QĐ-UBND", "65/2023/NĐ-CP"}
    # The central decree appearing in both legs ranks first.
    assert keys[0] == "01/2021/NĐ-CP"


def test_anchor_authority_demotes_provincial():
    # Provincial doc appears earlier in both legs but the prior must push the
    # central decree above it.
    lex = [
        _hit(1, "1934/2007/QĐ-UBND", "Điều 4", bm25_score=5.0),
        _hit(2, "01/2021/NĐ-CP", "Điều 1", bm25_score=4.0),
    ]
    den = [
        _hit(3, "1934/2007/QĐ-UBND", "Điều 4", dense_score=0.9),
        _hit(4, "01/2021/NĐ-CP", "Điều 1", dense_score=0.8),
    ]
    anchored = anchor_documents(lex, den, DocAnchorConfig(doc_authority_weight=1.0))
    assert anchored[0].doc_uid == "01/2021/NĐ-CP"


def test_anchor_respects_top_docs():
    lex = [_hit(i, f"{i}/2020/NĐ-CP", "Điều 1", bm25_score=float(10 - i))
           for i in range(1, 8)]
    anchored = anchor_documents(lex, [], DocAnchorConfig(top_docs=3))
    assert len(anchored) == 3


# --------------------------------------------------------------------------- #
# Stage B: harvesting
# --------------------------------------------------------------------------- #
def test_harvest_only_anchored_docs():
    lex = [
        _hit(1, "01/2021/NĐ-CP", "Điều 1", bm25_score=5.0),
        _hit(2, "99/1999/QĐ-UB", "Điều 7", bm25_score=4.9),
    ]
    anchored = [AnchoredDoc("01/2021/NĐ-CP", "01/2021/NĐ-CP", "VB", 1.0, 1.0, 0.0)]
    harvested = harvest_articles(lex, [], anchored, DocAnchorConfig())
    dieus = {h.dieu_so for h in harvested}
    assert dieus == {"Điều 1"}


def test_harvest_best_chunk_per_dieu():
    lex = [
        _hit(1, "01/2021/NĐ-CP", "Điều 1", bm25_score=2.0),
        _hit(2, "01/2021/NĐ-CP", "Điều 1", bm25_score=9.0),  # better chunk
    ]
    anchored = [AnchoredDoc("01/2021/NĐ-CP", "01/2021/NĐ-CP", "VB", 1.0, 1.0, 0.0)]
    harvested = harvest_articles(lex, [], anchored, DocAnchorConfig())
    assert len(harvested) == 1
    assert harvested[0].base_score == 9.0
    assert harvested[0].row_idx == 2


def test_harvest_caps_per_document():
    lex = [
        _hit(i, "01/2021/NĐ-CP", f"Điều {i}", bm25_score=float(10 - i))
        for i in range(1, 9)
    ]
    anchored = [AnchoredDoc("01/2021/NĐ-CP", "01/2021/NĐ-CP", "VB", 1.0, 1.0, 0.0)]
    harvested = harvest_articles(lex, [], anchored, DocAnchorConfig(per_doc_articles=3))
    assert len(harvested) == 3
    # Highest-scoring articles retained.
    assert {h.dieu_so for h in harvested} == {"Điều 1", "Điều 2", "Điều 3"}


# --------------------------------------------------------------------------- #
# Stage C: orchestration
# --------------------------------------------------------------------------- #
def _lex_fn(hits):
    def fn(query, top_k):
        return hits[:top_k]
    return fn


def test_retriever_lexical_only_no_rerank():
    lex = [
        _hit(1, "01/2021/NĐ-CP", "Điều 1", bm25_score=5.0),
        _hit(2, "01/2021/NĐ-CP", "Điều 12", bm25_score=4.0),
        _hit(3, "1934/2007/QĐ-UBND", "Điều 4", bm25_score=6.0),
    ]
    r = DocAnchoredRetriever(
        lexical_search=_lex_fn(lex), dense_search=None,
        fetch_text=None, rerank=None, cfg=DocAnchorConfig(),
    )
    pool = r.retrieve_pool("đăng ký doanh nghiệp")
    dieus = {c.dieu_so for c in pool}
    # Provincial article is still harvested here (suppression happens in the
    # selector, not the retriever) but the central articles are present.
    assert "Điều 1" in dieus and "Điều 12" in dieus
    assert all(isinstance(c, ArticleCandidate) for c in pool)


def test_retriever_applies_injected_rerank():
    lex = [
        _hit(1, "01/2021/NĐ-CP", "Điều 1", bm25_score=2.0),
        _hit(2, "01/2021/NĐ-CP", "Điều 12", bm25_score=1.0),
    ]

    def fetch_text(row_idxs):
        return {i: f"text-{i}" for i in row_idxs}

    def rerank(query, passages):
        # Reverse the base order: later passages score higher.
        return [0.1 * (i + 1) for i in range(len(passages))]

    r = DocAnchoredRetriever(
        lexical_search=_lex_fn(lex), dense_search=None,
        fetch_text=fetch_text, rerank=rerank, cfg=DocAnchorConfig(),
    )
    pool = r.retrieve_pool("q")
    # Scores come from the reranker, not bm25.
    assert all(0.0 < c.score <= 1.0 for c in pool)


def test_retriever_rerank_shape_mismatch_falls_back():
    lex = [_hit(1, "01/2021/NĐ-CP", "Điều 1", bm25_score=3.0)]

    def fetch_text(row_idxs):
        return {i: f"t{i}" for i in row_idxs}

    def bad_rerank(query, passages):
        return [0.5, 0.6]  # wrong length

    r = DocAnchoredRetriever(
        lexical_search=_lex_fn(lex), dense_search=None,
        fetch_text=fetch_text, rerank=bad_rerank, cfg=DocAnchorConfig(),
    )
    pool = r.retrieve_pool("q")
    assert pool[0].score == 3.0  # fell back to base score


def test_retriever_empty_pool_on_no_hits():
    r = DocAnchoredRetriever(
        lexical_search=_lex_fn([]), dense_search=None, cfg=DocAnchorConfig()
    )
    assert r.retrieve_pool("q") == []


def test_pool_record_schema():
    cands = [ArticleCandidate("01/2021/NĐ-CP", "Nghị định 01", "Điều 12", 0.83)]
    rec = pool_record(7, "câu hỏi", cands)
    assert rec["id"] == 7
    assert rec["candidates"][0]["dieu_so"] == "Điều 12"
    assert rec["candidates"][0]["score"] == pytest.approx(0.83)
