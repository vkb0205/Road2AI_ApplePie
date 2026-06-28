"""Unit tests for the article-first final selector (final_selection.md).

Covers:
- Article-first grouping (chunks → articles by canonical Điều).
- Per-lane budgets (direct=1, scenario=6, etc.).
- Scoring: relevance + support + graph_gain + authority - redundancy.
- Provincial suppression + recall floor.
- SelectionResult grader-string output.

No heavy deps — pure Python.
"""

from __future__ import annotations

from retrieval.final_selector import (
    LANE_BUDGETS,
    ChunkEvidence,
    FinalSelector,
    SelectionConfig,
)


def _chunk(row, law, dieu, score, ten="Luật X", from_graph=False, chunk_id=None):
    return ChunkEvidence(
        row_idx=row,
        law_id=law,
        ten_van_ban=ten,
        dieu_so=dieu,
        score=score,
        chunk_id=chunk_id,
        from_graph=from_graph,
    )


# ============================ Article grouping ============================ #


class TestArticleGrouping:
    def setup_method(self):
        self.sel = FinalSelector()

    def test_chunks_grouped_by_dieu(self):
        pool = [
            _chunk(1, "01/2020/QH14", "Điều 5", 0.9),
            _chunk(2, "01/2020/QH14", "Điều 5", 0.7),  # same article
            _chunk(3, "01/2020/QH14", "Điều 6", 0.8),
        ]
        result = self.sel.select(pool, lane="condition_requirement")
        dieus = {a.dieu for a in result.articles}
        assert dieus == {"Điều 5", "Điều 6"}

    def test_best_chunk_score_is_relevance(self):
        pool = [
            _chunk(1, "01/2020/QH14", "Điều 5", 0.6),
            _chunk(2, "01/2020/QH14", "Điều 5", 0.95),
        ]
        result = self.sel.select(pool, lane="direct_lookup")
        assert result.articles[0].relevance == 0.95

    def test_empty_pool(self):
        result = self.sel.select([], lane="direct_lookup")
        assert result.articles == []
        assert result.selection_metadata["reason"] == "empty pool"


# ============================ Lane budgets ============================ #


class TestLaneBudgets:
    def setup_method(self):
        # Many distinct articles, all with similar scores so the margin gate
        # doesn't trim before the budget does.
        self.pool = [
            _chunk(i, f"{i:02d}/2020/QH14", f"Điều {i}", 0.9 - i * 0.01, ten=f"Luật {i}")
            for i in range(1, 11)
        ]

    def test_direct_lookup_keeps_one(self):
        sel = FinalSelector()
        result = sel.select(self.pool, lane="direct_lookup")
        assert len(result.articles) == LANE_BUDGETS["direct_lookup"][0] == 1

    def test_scenario_keeps_up_to_six(self):
        sel = FinalSelector()
        result = sel.select(self.pool, lane="scenario")
        assert len(result.articles) <= LANE_BUDGETS["scenario"][0]
        assert len(result.articles) == 6

    def test_unknown_lane_uses_default(self):
        sel = FinalSelector()
        result = sel.select(self.pool, lane="made_up_lane")
        assert len(result.articles) <= 3  # _DEFAULT_BUDGET = (3, 2)

    def test_chunks_per_article_capped(self):
        pool = [
            _chunk(i, "01/2020/QH14", "Điều 5", 0.9 - i * 0.01)
            for i in range(5)
        ]
        sel = FinalSelector()
        result = sel.select(pool, lane="direct_lookup")
        # direct_lookup → max 2 chunks per article.
        assert len(result.articles[0].chunks) == 2


# ============================ Scoring components ============================ #


class TestScoring:
    def test_graph_gain_applied(self):
        cfg = SelectionConfig()
        sel = FinalSelector(cfg)
        pool = [_chunk(1, "01/2020/QH14", "Điều 5", 0.8, from_graph=True)]
        result = sel.select(pool, lane="cross_doc")
        assert result.articles[0].graph_gain == cfg.graph_gain_bonus

    def test_support_bonus_for_multiple_chunks(self):
        cfg = SelectionConfig()
        sel = FinalSelector(cfg)
        pool = [
            _chunk(1, "01/2020/QH14", "Điều 5", 0.8),
            _chunk(2, "01/2020/QH14", "Điều 5", 0.7),
            _chunk(3, "01/2020/QH14", "Điều 5", 0.6),
        ]
        result = sel.select(pool, lane="direct_lookup")
        assert result.articles[0].support > 0

    def test_authority_prior_prefers_central_over_provincial_tie(self):
        # A central decree and a (non-provincial) agency decision, same score.
        # The central one should win the representative / top slot.
        sel = FinalSelector(SelectionConfig(drop_provincial=False))
        pool = [
            _chunk(1, "50/2020/NĐ-CP", "Điều 5", 0.8, ten="Nghị định"),
            _chunk(2, "99/2005/QĐ-TCHQ", "Điều 9", 0.8, ten="Quyết định"),
        ]
        result = sel.select(pool, lane="condition_requirement")
        assert result.articles[0].law_id == "50/2020/NĐ-CP"


# ============================ Provincial suppression ============================ #


class TestProvincialSuppression:
    def test_provincial_dropped_by_default(self):
        sel = FinalSelector()
        pool = [
            _chunk(1, "1934/2007/QĐ-UBND", "Điều 4", 0.95, ten="QĐ tỉnh"),
            _chunk(2, "50/2020/NĐ-CP", "Điều 5", 0.6, ten="Nghị định"),
        ]
        result = sel.select(pool, lane="procedure_detail")
        laws = {a.law_id for a in result.articles}
        assert "1934/2007/QĐ-UBND" not in laws
        assert "50/2020/NĐ-CP" in laws

    def test_provincial_kept_when_disabled(self):
        sel = FinalSelector(SelectionConfig(drop_provincial=False))
        pool = [_chunk(1, "1934/2007/QĐ-UBND", "Điều 4", 0.95, ten="QĐ tỉnh")]
        result = sel.select(pool, lane="direct_lookup")
        assert len(result.articles) == 1


# ============================ Recall floor + output ============================ #


class TestRecallAndOutput:
    def test_recall_floor_keeps_one_even_below_margin(self):
        sel = FinalSelector()
        # Single article → always returned.
        pool = [_chunk(1, "01/2020/QH14", "Điều 5", 0.3)]
        result = sel.select(pool, lane="direct_lookup")
        assert len(result.articles) == 1

    def test_relevant_articles_grader_string(self):
        sel = FinalSelector()
        pool = [_chunk(1, "01/2020/QH14", "Điều 5", 0.9, ten="Luật DN")]
        result = sel.select(pool, lane="direct_lookup")
        assert result.relevant_articles() == ["01/2020/QH14|Luật DN|Điều 5"]

    def test_relevant_docs_dedup(self):
        sel = FinalSelector()
        pool = [
            _chunk(1, "01/2020/QH14", "Điều 5", 0.9, ten="Luật DN"),
            _chunk(2, "01/2020/QH14", "Điều 6", 0.85, ten="Luật DN"),
        ]
        result = sel.select(pool, lane="condition_requirement")
        # Two articles, one document.
        assert result.relevant_docs() == ["01/2020/QH14|Luật DN"]
