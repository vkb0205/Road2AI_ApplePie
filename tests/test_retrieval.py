"""Unit tests for the G-LRAG retrieval modules (no heavy GPU deps).

These tests exercise:
  - RRF fusion (determinism, formula, payload carry-through).
  - Pure-Python BM25 fallback + FTS5 query-string builders.
  - Graph expansion over a tiny synthetic MultiDiGraph (DOC→ART + DOC→DOC),
    including graceful degradation when CHUNK/CONCEPT nodes are absent.
  - The HybridRetriever in lexical-only + graph-expansion mode against the
    real Stage 6 ``chunk_store.sqlite`` bundle (FTS5 is available locally),
    and ``make_relevant_lists`` article collapse.
  - F2 macro metric (eval.py) edge cases per SPEC §14.3.

No torch/faiss/FlagEmbedding imports are triggered.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import networkx as nx
import pytest

# Make `src` importable (pyproject sets pythonpath=src for pytest, but be safe).
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from retrieval.rrf import rrf_fuse, rrf_score_only  # noqa: E402
from retrieval.bm25_index import (  # noqa: E402
    PureBM25,
    build_fts_match_expr,
    tokenize_query,
)
from retrieval.graph_expand import GraphExpander, EXPANSION_DOC_RELS  # noqa: E402
from retrieval.retriever import (  # noqa: E402
    HybridRetriever,
    RetrievalConfig,
    Hit,
    make_relevant_lists,
)
from dev_set.eval import f2_macro, f2_single, f2_from_answers  # noqa: E402

DEV_SET = ROOT / "dev_set"
STAGE6 = ROOT / "data" / "stage6_data"
HAVE_FTS_BUNDLE = (STAGE6 / "chunk_store.sqlite").exists()


# ============================ RRF ============================ #


class TestRRF:
    def test_basic_fusion(self):
        a = [0, 1, 2]
        b = [2, 3, 0]
        fused = rrf_fuse([a, b], k=60, fused_top=30)
        # row 0 and 2 appear in both lists; row 2 is rank-3 in a and rank-1 in b
        ids = [r["row_idx"] for r in fused]
        assert 0 in ids and 2 in ids and 1 in ids and 3 in ids
        # row 2 (rank1 in b + rank3 in a) > row 0 (rank1 a + rank3 b) by symmetry?
        # both have symmetric ranks → equal score; tie-break by row_idx asc → 0 first.
        sc = {r["row_idx"]: r["rrf_score"] for r in fused}
        assert sc[2] == pytest.approx(1 / (60 + 3) + 1 / (60 + 1))
        assert sc[0] == pytest.approx(1 / (60 + 1) + 1 / (60 + 3))
        # row 1 and 3 appear in one list each
        assert sc[1] == pytest.approx(1 / (60 + 2))
        assert sc[3] == pytest.approx(1 / (60 + 2))

    def test_determinism(self):
        a = [5, 3, 1, 9, 7]
        b = [9, 1, 3, 5, 11]
        r1 = rrf_fuse([a, b], k=60, fused_top=10, seed=42)
        r2 = rrf_fuse([a, b], k=60, fused_top=10, seed=42)
        assert r1 == r2  # exact equality across re-runs (acceptance: seed 42)

    def test_fused_top_truncation(self):
        a = list(range(100))
        fused = rrf_fuse([a], k=60, fused_top=5)
        assert len(fused) == 5
        assert [r["row_idx"] for r in fused] == [0, 1, 2, 3, 4]

    def test_mapping_payload_carry_through(self):
        a = [{"row_idx": 7, "law_id": "L1"}, {"row_idx": 3, "law_id": "L2"}]
        fused = rrf_fuse([a], k=60, fused_top=10)
        payloads = {r["row_idx"]: r["payload"] for r in fused}
        assert payloads[7]["law_id"] == "L1"
        assert payloads[3]["law_id"] == "L2"

    def test_invalid_k(self):
        with pytest.raises(ValueError):
            rrf_fuse([[1]], k=0)

    def test_missing_row_idx_key(self):
        with pytest.raises(KeyError):
            rrf_fuse([[{"law_id": "X"}]])

    def test_score_only_no_truncation(self):
        a = [0, 1, 2]
        b = [2, 3]
        sc = rrf_score_only([a, b], k=60)
        assert set(sc.keys()) == {0, 1, 2, 3}
        assert sc[2] > sc[0]  # 2 appears in both; 0 only in a


# ============================ BM25 ============================ #


class TestPureBM25:
    def test_retrieves_relevant_doc(self):
        corpus = [
            "doanh nghiệp đăng ký kinh doanh",
            "thuế xuất nhập khẩu hải quan",
            "doanh nghiệp nhỏ và vừa hỗ trợ",
        ]
        bm = PureBM25(corpus, k1=1.5, b=0.75)
        hits = bm.search("doanh nghiệp", top_k=2)
        assert hits[0]["row_idx"] in (0, 2)  # both mention doanh nghiệp
        assert hits[0]["bm25_score"] > 0.0
        # doc 1 has no "doanh nghiệp" → not returned (score<=0 break)
        assert all(h["row_idx"] != 1 for h in hits)

    def test_invalid_k1(self):
        with pytest.raises(ValueError):
            PureBM25(["a"], k1=0)

    def test_invalid_b(self):
        with pytest.raises(ValueError):
            PureBM25(["a"], b=1.5)

    def test_empty_query(self):
        bm = PureBM25(["doanh nghiệp kinh doanh"])
        assert bm.search("", top_k=5) == []


class TestFTSQueryBuilders:
    def test_tokenize_simple(self):
        assert tokenize_query("doanh nghiệp nhỏ và vừa") == [
            "doanh", "nghiệp", "nhỏ", "và", "vừa"
        ]

    def test_tokenize_empty(self):
        assert tokenize_query("") == []
        assert tokenize_query("   ") == []

    def test_build_match_expr_quotes_and_ors(self):
        expr = build_fts_match_expr(["doanh nghiệp", "thuế"])
        assert '"doanh nghiệp"' in expr
        assert '"thuế"' in expr
        assert " OR " in expr

    def test_build_match_expr_empty(self):
        assert build_fts_match_expr([]) == ""


# ============================ Graph expansion ============================ #


def _build_tiny_graph():
    """DOC1 -DETAILS-> DOC2; DOC1 -HAS_ARTICLE-> ART1a; DOC2 -HAS_ARTICLE-> ART2a."""
    G = nx.MultiDiGraph()
    G.add_node("DOC:1", type="Document", ten="Nghị định 01")
    G.add_node("DOC:2", type="Document", ten="Nghị định 02")
    G.add_node("ART:law|ten|Điều 5", type="Article", doc_uid="law|ten|Điều 5")
    G.add_node("ART:law2|ten2|Điều 7", type="Article", doc_uid="law2|ten2|Điều 7")
    G.add_edge("DOC:1", "ART:law|ten|Điều 5", key="HAS_ARTICLE", relation="HAS_ARTICLE")
    G.add_edge("DOC:2", "ART:law2|ten2|Điều 7", key="HAS_ARTICLE", relation="HAS_ARTICLE")
    G.add_edge("DOC:1", "DOC:2", key="DETAILS", relation="DETAILS")
    return G


class TestGraphExpand:
    def test_doc_expansion_adds_neighbour_article_chunks(self):
        G = _build_tiny_graph()
        # row_idx 0 → doc_uid of ART1a; row_idx 1 → doc_uid of ART2a
        row_to_uid = {0: "law|ten|Điều 5", 1: "law2|ten2|Điều 7"}
        exp = GraphExpander(G, row_to_uid)
        expanded = expander_expand(exp, [(0, 1.0)], top_n=10)
        ids = [e["row_idx"] for e in expanded]
        assert 0 in ids  # seed retained
        assert 1 in ids  # neighbour article's chunk added
        # neighbour score = seed * discount_doc
        nb = next(e for e in expanded if e["row_idx"] == 1)
        assert nb["score"] == pytest.approx(1.0 * 0.6)
        assert nb["source"] == "doc_expand"

    def test_no_expansion_when_seed_doc_has_no_neighbours(self):
        G = _build_tiny_graph()
        # Use a doc_uid that maps to a non-existent ART node → no expansion.
        exp = GraphExpander(G, {0: "nonexistent|uid|Điều 9"})
        expanded = expander_expand(exp, [(0, 1.0)], top_n=10)
        assert len(expanded) == 1
        assert expanded[0]["row_idx"] == 0
        assert expanded[0]["source"] == "candidate"

    def test_graceful_degrade_without_chunk_concept_nodes(self):
        G = _build_tiny_graph()  # no CHUNK/CONCEPT nodes
        exp = GraphExpander(G, {0: "law|ten|Điều 5", 1: "law2|ten2|Điều 7"})
        assert exp._has_chunks is False
        assert exp._has_concepts is False
        # concept co-mention path must not crash; only doc-expand runs.
        expanded = expander_expand(exp, [(0, 1.0)], top_n=10)
        assert all(e["source"] in ("candidate", "doc_expand") for e in expanded)

    def test_graph_context_with_neighbours(self):
        G = _build_tiny_graph()
        exp = GraphExpander(G, {0: "law|ten|Điều 5"})
        ctx = exp.build_graph_context(["DOC:1"])
        assert "DETAILS: Nghị định 02" in ctx

    def test_graph_context_no_neighbours_placeholder(self):
        G = _build_tiny_graph()
        exp = GraphExpander(G, {0: "law|ten|Điều 5"})
        ctx = exp.build_graph_context(["DOC:2"])
        # DOC:2 has no out-edges in INTERESTING_RELS and no in-edges of those rels
        # (DOC:1->DOC:2 is DETAILS out of DOC:1, so DOC:2 has an in-edge DETAILS).
        # Therefore DOC:2 yields "DETAILS_BY: Nghị định 01".
        assert "DETAILS_BY" in ctx

    def test_graph_context_empty_placeholder(self):
        G = nx.MultiDiGraph()
        G.add_node("DOC:9", type="Document", ten="Lonely")
        exp = GraphExpander(G, {0: "x|y|Điều 1"})
        ctx = exp.build_graph_context(["DOC:9"])
        assert ctx == "(Không có quan hệ chéo đáng chú ý)"


def expander_expand(exp, candidates, top_n=50):
    """Wrapper to call the private-ish expand method (kept public)."""
    return exp.expand(candidates, top_n=top_n)


# ============================ Retriever (lexical-only) ============================ #


@pytest.mark.skipif(not HAVE_FTS_BUNDLE, reason="Stage 6 chunk_store.sqlite not present")
class TestHybridRetrieverLexical:
    def test_retrieve_returns_hits(self):
        from retrieval.bm25_index import FTSIndex

        with FTSIndex(str(STAGE6 / "chunk_store.sqlite"), mode="fts_fast") as fts:
            cfg = RetrievalConfig(use_dense=False, use_reranker=False, top_bm25=50)
            r = HybridRetriever(fts, faiss_index=None, graph_expander=None, config=cfg)
            hits = r.retrieve("đăng ký doanh nghiệp", fetch_text=True)
            assert len(hits) >= 1
            assert all(h.row_idx >= 0 for h in hits)
            assert all(h.chunk_id for h in hits)  # metadata attached
            assert all(h.law_id for h in hits)

    def test_retrieve_smoke_all_devset_questions(self):
        """Acceptance 7.1: ≥ 1 hit for all devset questions (lexical-only)."""
        from retrieval.bm25_index import FTSIndex

        questions = json.loads((DEV_SET / "questions.json").read_text(encoding="utf-8"))
        with FTSIndex(str(STAGE6 / "chunk_store.sqlite"), mode="fts_fast") as fts:
            cfg = RetrievalConfig(use_dense=False, use_reranker=False, top_bm25=50)
            r = HybridRetriever(fts, faiss_index=None, graph_expander=None, config=cfg)
            for q in questions:
                hits = r.retrieve(q["question"], fetch_text=False)
                assert len(hits) >= 1, f"no hits for q{q['id']}: {q['question'][:50]}"

    def test_make_relevant_lists_dedup_and_dieu_filter(self):
        hits = [
            Hit(0, 1.0, "candidate", law_id="L1", ten_van_ban="T1", dieu_so="Điều 5"),
            Hit(1, 0.9, "candidate", law_id="L1", ten_van_ban="T1", dieu_so="Điều 5"),
            Hit(2, 0.8, "candidate", law_id="L2", ten_van_ban="T2", dieu_so="Điều 9"),
            Hit(3, 0.7, "candidate", law_id="L3", ten_van_ban="T3", dieu_so="Văn bản"),
        ]
        docs, articles = make_relevant_lists(hits)
        # docs: any hit with valid law+ten is kept (deduped, order preserved)
        assert docs == ["L1|T1", "L2|T2", "L3|T3"]
        # articles: only dieu_so starting with "Điều", deduped
        assert articles == ["L1|T1|Điều 5", "L2|T2|Điều 9"]


# ============================ F2 eval ============================ #


class TestF2:
    def test_perfect_match(self):
        gt = [{"id": 1, "relevant_articles": ["L|T|Điều 1", "L|T|Điều 2"]}]
        pred = [{"id": 1, "relevant_articles": ["L|T|Điều 1", "L|T|Điều 2"]}]
        assert f2_macro(pred, gt) == pytest.approx(1.0)

    def test_both_empty_is_one(self):
        gt = [{"id": 1, "relevant_articles": []}]
        pred = [{"id": 1, "relevant_articles": []}]
        assert f2_macro(pred, gt) == pytest.approx(1.0)

    def test_one_empty_is_zero(self):
        gt = [{"id": 1, "relevant_articles": ["L|T|Điều 1"]}]
        pred = [{"id": 1, "relevant_articles": []}]
        assert f2_macro(pred, gt) == pytest.approx(0.0)

    def test_partial_recall(self):
        gt = [{"id": 1, "relevant_articles": ["L|T|Điều 1", "L|T|Điều 2"]}]
        pred = [{"id": 1, "relevant_articles": ["L|T|Điều 1"]}]
        # p=1, r=0.5 → F2 = 5*1*0.5/(4*1+0.5)=2.5/4.5
        assert f2_macro(pred, gt) == pytest.approx(5 * 1 * 0.5 / (4 * 1 + 0.5))

    def test_normalisation_to_dieu_only(self):
        # Different law/ten but same dieu → counted as match (grader normalises to Điều X)
        gt = [{"id": 1, "relevant_articles": ["L1|T1|Điều 1"]}]
        pred = [{"id": 1, "relevant_articles": ["L2|T2|Điều 1"]}]
        assert f2_macro(pred, gt) == pytest.approx(1.0)

    def test_f2_from_answers(self):
        gt = [{"id": 1, "relevant_articles": ["L|T|Điều 5"]}]
        pred = [{"id": 1, "answer": "Theo Điều 5 của văn bản, doanh nghiệp phải..."}]
        assert f2_from_answers(pred, gt) == pytest.approx(1.0)

    def test_f2_macro_missing_id_raises(self):
        with pytest.raises(KeyError):
            f2_macro([{"id": 99, "relevant_articles": []}], [{"id": 1, "relevant_articles": []}])


# ============================ FTS5 real-bundle smoke ============================ #


@pytest.mark.skipif(not HAVE_FTS_BUNDLE, reason="Stage 6 chunk_store.sqlite not present")
class TestFTSIndexRealBundle:
    def test_open_and_search(self):
        from retrieval.bm25_index import FTSIndex

        with FTSIndex(str(STAGE6 / "chunk_store.sqlite"), mode="fts_fast") as fts:
            assert fts.n_rows > 600000
            hits = fts.search("đăng ký doanh nghiệp", top_k=10)
            assert len(hits) >= 1
            assert all("row_idx" in h for h in hits)
            assert all(h["law_id"] for h in hits)

    def test_fetch_chunks(self):
        from retrieval.bm25_index import FTSIndex

        with FTSIndex(str(STAGE6 / "chunk_store.sqlite")) as fts:
            hits = fts.search("thuế", top_k=3)
            idxs = [h["row_idx"] for h in hits]
            rows = fts.fetch_chunks(idxs)
            assert len(rows) == len(idxs)
            assert all("chunk_text" in r for r in rows)
            assert [r["row_idx"] for r in rows] == idxs  # order preserved
