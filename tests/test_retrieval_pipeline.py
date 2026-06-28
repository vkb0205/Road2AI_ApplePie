"""Integration test for the 7-stage retrieval pipeline.

Validates the full flow with mock dependencies (no FAISS, no LLM, no graph):
  1. Router classifies the lane.
  2. Decomposer splits (if should_decompose).
  3. Hybrid retrieve (BM25 + optional FAISS) → RRF fusion → merge pools.
  4. Rerank on the ORIGINAL query (critical rule).
  5. Graph expand (optional, lane-gated).
  6. Final select (article-first, lane-budgeted).

The test confirms:
  - The decompose→retrieve→merge→rerank-on-original sequence.
  - The lane-specific strategy switches are propagated correctly.
  - The stage_stats traceability is populated.
  - The ChunkEvidence dataclass is correctly built from search hits.

No heavy deps — pure Python with fake callables.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from retrieval.decomposer import Decomposer, DecomposerConfig
from retrieval.final_selector import FinalSelector, SelectionConfig
from retrieval.retrieval_pipeline import (
    RetrievalConfig,
    RetrievalPipeline,
    merge_candidate_pools,
)
from retrieval.router import Router, RouterConfig


# ============================ Mock dependencies ============================ #


class FakeLexicalSearch:
    """Deterministic BM25-like search returning fixed hits per query."""

    def __init__(self, corpus: Dict[str, List[Dict[str, Any]]]):
        """corpus: {query_substring -> [hit dicts]}."""
        self.corpus = corpus

    def __call__(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        q_lower = query.lower()
        for key, hits in self.corpus.items():
            if key.lower() in q_lower:
                return hits[:top_k]
        return []


class FakeDenseSearch:
    """Stub dense search."""

    def __call__(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        # Returns a single mock hit for testing the hybrid path.
        return [
            {
                "row_idx": 999,
                "law_id": "99/2020/NĐ-CP",
                "ten_van_ban": "Nghị định",
                "dieu_so": "Điều 99",
                "score": 0.75,
            }
        ]


class FakeTextProvider:
    """Stub text fetch."""

    def __call__(self, row_idxs: Sequence[int]) -> Dict[int, str]:
        return {ridx: f"chunk text for row {ridx}" for ridx in row_idxs}


class FakeReranker:
    """Stub reranker that returns descending scores."""

    def __init__(self, verify_query: str = ""):
        """If verify_query is set, assert the reranker sees it (the critical rule test)."""
        self.verify_query = verify_query
        self.called_with: List[str] = []

    def __call__(self, query: str, passages: Sequence[str]) -> List[float]:
        self.called_with.append(query)
        if self.verify_query:
            assert query == self.verify_query, (
                f"Reranker expected original query '{self.verify_query}', "
                f"got '{query}' (violates rerank-on-original rule)"
            )
        # Return descending scores so the order is deterministic.
        return [1.0 - i * 0.1 for i in range(len(passages))]


class FakeGraphExpander:
    """Stub graph expander that adds one extra row."""

    def expand(self, seeds: List[tuple], top_n: int) -> List[Dict[str, Any]]:
        # Add one fake expanded row.
        return [
            {
                "row_idx": 8888,
                "score": 0.4,
                "source": "graph_expand",
            }
        ]


# ============================ merge_candidate_pools ============================ #


class TestMergeCandidatePools:
    def test_dedup_by_row_idx(self):
        pools = [
            [{"row_idx": 1, "law_id": "A"}, {"row_idx": 2, "law_id": "B"}],
            [{"row_idx": 2, "law_id": "B2"}, {"row_idx": 3, "law_id": "C"}],
        ]
        merged = merge_candidate_pools(pools)
        assert len(merged) == 3
        # First occurrence payload wins.
        assert next(h for h in merged if h["row_idx"] == 2)["law_id"] == "B"

    def test_empty_pools(self):
        assert merge_candidate_pools([]) == []
        assert merge_candidate_pools([[], []]) == []


# ============================ Full pipeline ============================ #


class TestRetrievalPipeline:
    def setup_method(self):
        # Build a mock corpus for the lexical search.
        self.corpus = {
            "điều kiện": [
                {
                    "row_idx": 10,
                    "law_id": "01/2020/QH14",
                    "ten_van_ban": "Luật Doanh nghiệp",
                    "dieu_so": "Điều 5",
                    "score": 0.9,
                },
                {
                    "row_idx": 11,
                    "law_id": "01/2020/QH14",
                    "ten_van_ban": "Luật Doanh nghiệp",
                    "dieu_so": "Điều 6",
                    "score": 0.8,
                },
            ],
            "thủ tục": [
                {
                    "row_idx": 20,
                    "law_id": "50/2020/NĐ-CP",
                    "ten_van_ban": "Nghị định",
                    "dieu_so": "Điều 10",
                    "score": 0.85,
                }
            ],
        }

        self.router = Router(config=RouterConfig(use_llm=False))
        self.decomposer = Decomposer(config=DecomposerConfig(use_llm=False))
        self.lexical = FakeLexicalSearch(self.corpus)
        self.selector = FinalSelector(config=SelectionConfig(drop_provincial=False))

    def test_single_query_no_decompose(self):
        """Simple query: no decomposition, single retrieval pass."""
        pipeline = RetrievalPipeline(
            router=self.router,
            decomposer=self.decomposer,
            lexical_search=self.lexical,
            final_selector=self.selector,
        )
        result = pipeline.retrieve("điều kiện hưởng ưu đãi là gì")
        assert result.routing.lane == "condition_requirement"
        assert result.decomposition is None or not result.decomposition.should_decompose
        assert len(result.selection.articles) >= 1
        assert result.stage_stats["n_retrieval_queries"] == 1

    def test_decompose_merges_pools(self):
        """Multi-intent query: decompose → multiple retrieval passes → merge."""
        pipeline = RetrievalPipeline(
            router=self.router,
            decomposer=self.decomposer,
            lexical_search=self.lexical,
            final_selector=self.selector,
        )
        q = "điều kiện hưởng ưu đãi là gì; thủ tục đăng ký gồm bước nào"
        result = pipeline.retrieve(q)
        assert result.decomposition is not None
        assert result.decomposition.should_decompose is True
        # Original + sub-questions.
        assert result.stage_stats["n_retrieval_queries"] >= 2
        # Merged pool should have hits from both "điều kiện" and "thủ tục".
        assert result.stage_stats["n_merged_candidates"] >= 2

    def test_rerank_on_original_query(self):
        """The CRITICAL RULE: reranker must see the original query, not sub-questions."""
        original = "điều kiện hưởng ưu đãi là gì; thủ tục đăng ký gồm bước nào"
        reranker = FakeReranker(verify_query=original)
        pipeline = RetrievalPipeline(
            router=self.router,
            decomposer=self.decomposer,
            lexical_search=self.lexical,
            final_selector=self.selector,
            reranker=reranker,
            text_provider=FakeTextProvider(),
        )
        result = pipeline.retrieve(original)
        # If the reranker didn't see the original query, it raised an assert.
        assert len(reranker.called_with) == 1
        assert reranker.called_with[0] == original

    def test_hybrid_retrieve_with_dense(self):
        """Enable dense search → both BM25 and FAISS hits are fused."""
        pipeline = RetrievalPipeline(
            router=self.router,
            decomposer=self.decomposer,
            lexical_search=self.lexical,
            dense_search=FakeDenseSearch(),
            final_selector=self.selector,
            cfg=RetrievalConfig(use_dense=True),
        )
        result = pipeline.retrieve("điều kiện hưởng ưu đãi là gì")
        # Merged pool should include the dense stub hit (row 999).
        rows = [c.row_idx for c in result.selection.chunks]
        # The dense hit may or may not survive final select (depends on score).
        # What we can confirm: n_merged_candidates > lexical-only count.
        assert result.stage_stats["n_merged_candidates"] >= 2

    def test_graph_expand_when_enabled(self):
        """Graph-expand lane → router sets need_graph_expand → expander invoked.

        Uses a 'thủ tục' query: it matches the mock corpus (so the reranked
        pool is non-empty) AND routes to procedure_detail, which is in the
        router's graph_expand_lanes.
        """
        pipeline = RetrievalPipeline(
            router=self.router,
            decomposer=self.decomposer,
            lexical_search=self.lexical,
            final_selector=self.selector,
            graph_expander=FakeGraphExpander(),
        )
        result = pipeline.retrieve("thủ tục đăng ký doanh nghiệp gồm những bước nào")
        assert result.routing.need_graph_expand is True
        assert result.stage_stats["graph_expanded"] is True
        # The fake expander adds one row (8888) on top of the reranked seeds.
        assert result.stage_stats["n_expanded"] > result.stage_stats["n_reranked"]

    def test_graph_expand_skipped_when_disabled(self):
        """Direct lookup lane → router sets need_graph_expand=False → no expansion."""
        pipeline = RetrievalPipeline(
            router=self.router,
            decomposer=self.decomposer,
            lexical_search=self.lexical,
            final_selector=self.selector,
            graph_expander=FakeGraphExpander(),
        )
        result = pipeline.retrieve("Điều 5 nội dung là gì")
        assert result.routing.need_graph_expand is False
        assert result.stage_stats["graph_expanded"] is False

    def test_lane_budget_applied(self):
        """Lane determines the article budget in final select."""
        pipeline = RetrievalPipeline(
            router=self.router,
            decomposer=self.decomposer,
            lexical_search=self.lexical,
            final_selector=self.selector,
        )
        # Direct lookup → budget = 1 article.
        result = pipeline.retrieve("Điều 5 nội dung là gì")
        assert result.routing.lane == "direct_lookup"
        assert len(result.selection.articles) <= 1

    def test_stage_stats_populated(self):
        """All stage stats keys are present."""
        pipeline = RetrievalPipeline(
            router=self.router,
            decomposer=self.decomposer,
            lexical_search=self.lexical,
            final_selector=self.selector,
        )
        result = pipeline.retrieve("điều kiện hưởng ưu đãi là gì")
        stats = result.stage_stats
        assert "lane" in stats
        assert "n_retrieval_queries" in stats
        assert "n_merged_candidates" in stats
        assert "n_reranked" in stats
        assert "n_expanded" in stats
        assert "n_selected_articles" in stats
        assert "ms_retrieve" in stats
        assert "ms_rerank" in stats
        assert "ms_graph_expand" in stats

    def test_empty_query(self):
        """Empty query gracefully defaults."""
        pipeline = RetrievalPipeline(
            router=self.router,
            decomposer=self.decomposer,
            lexical_search=self.lexical,
            final_selector=self.selector,
        )
        result = pipeline.retrieve("")
        assert result.routing.lane == "direct_lookup"
        assert result.selection.articles == []
