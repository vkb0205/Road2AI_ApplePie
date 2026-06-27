"""Unit tests for the F2-optimal article selector (no GPU deps).

Exercises:
  - Grader-faithful ``Điều X`` canonicalisation.
  - Document-type / year parsing and the soft authority prior.
  - Provincial / superseded suppression in aggregation.
  - Article-number aggregation across documents (grader is number-only).
  - The F2-optimal K policy (min_k recall floor, margin admission, max_k cap).
  - An end-to-end demonstration that the selector recovers the gold article
    from a pool where provincial/superseded noise outnumbers it — the exact
    failure mode seen in the dev-set baseline.

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

from retrieval.article_select import (  # noqa: E402
    ArticleCandidate,
    AuthorityConfig,
    SelectConfig,
    aggregate_articles,
    authority_prior,
    canonical_dieu,
    is_provincial,
    parse_law_type,
    parse_year,
    select_articles,
    select_relevant_strings,
)


# --------------------------------------------------------------------------- #
# Canonicalisation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Điều 12", "Điều 12"),
        ("Điều  12", "Điều 12"),
        ("Điều12", "Điều 12"),
        ("Điều 12a", "Điều 12a"),
        ("Điều 14.3", "Điều 14.3"),
        ("  Điều 5 ", "Điều 5"),
    ],
)
def test_canonical_dieu(raw, expected):
    assert canonical_dieu(raw) == expected


# --------------------------------------------------------------------------- #
# Document metadata parsing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "law_id,expected",
    [
        ("01/2021/NĐ-CP", "NĐ-CP"),
        ("156/2013/TT-BTC", "TT"),
        ("38/2015/TT-BTC", "TT"),
        ("68/2014/QH13", "QH"),
        ("59/2020/QH14", "QH"),
        ("1934/2007/QĐ-UBND", "QĐ-UBND"),
        ("72/2002/QĐ-UB", "QĐ-UB"),
        ("2396/QĐ-TCHQ", "QĐ-TCHQ"),
    ],
)
def test_parse_law_type(law_id, expected):
    assert parse_law_type(law_id) == expected


@pytest.mark.parametrize(
    "law_id,expected",
    [
        ("01/2021/NĐ-CP", 2021),
        ("156/2013/TT-BTC", 2013),
        ("72/2002/QĐ-UB", 2002),
        ("2396/QĐ-TCHQ", None),
    ],
)
def test_parse_year(law_id, expected):
    assert parse_year(law_id) == expected


def test_authority_prior_orders_central_above_provincial():
    cfg = AuthorityConfig()
    central = authority_prior("01/2021/NĐ-CP", "Nghị định ...", cfg)
    provincial = authority_prior("1934/2007/QĐ-UBND", "Quyết định ...", cfg)
    assert central > provincial


def test_authority_prior_rewards_recency():
    cfg = AuthorityConfig()
    new = authority_prior("65/2023/NĐ-CP", "Nghị định ...", cfg)
    old = authority_prior("02/2000/NĐ-CP", "Nghị định ...", cfg)
    assert new > old


def test_is_provincial():
    cfg = AuthorityConfig()
    assert is_provincial("1934/2007/QĐ-UBND", cfg)
    assert is_provincial("72/2002/QĐ-UB", cfg)
    assert not is_provincial("01/2021/NĐ-CP", cfg)


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def test_aggregate_drops_provincial_when_configured():
    cands = [
        ArticleCandidate("1934/2007/QĐ-UBND", "QĐ ...", "Điều 4", 0.9),
        ArticleCandidate("01/2021/NĐ-CP", "Nghị định ...", "Điều 12", 0.5),
    ]
    agg = aggregate_articles(cands, SelectConfig(drop_provincial=True))
    dieus = {a.dieu for a in agg}
    assert dieus == {"Điều 12"}


def test_aggregate_groups_same_number_across_docs():
    # Same article number from two central docs -> single aggregated entry
    # with multi-document agreement support.
    cands = [
        ArticleCandidate("01/2021/NĐ-CP", "Nghị định A", "Điều 1", 0.6),
        ArticleCandidate("65/2023/NĐ-CP", "Nghị định B", "Điều 1", 0.55),
    ]
    agg = aggregate_articles(cands, SelectConfig())
    assert len(agg) == 1
    assert agg[0].dieu == "Điều 1"
    assert agg[0].n_support == 2


def test_aggregate_keeps_most_authoritative_representative():
    cands = [
        ArticleCandidate("02/2000/NĐ-CP", "Nghị định cũ", "Điều 8", 0.61),
        ArticleCandidate("01/2021/NĐ-CP", "Nghị định mới", "Điều 8", 0.60),
    ]
    agg = aggregate_articles(cands, SelectConfig())
    assert len(agg) == 1
    # Newer decree wins the representative slot via the recency prior despite a
    # marginally lower base score.
    assert agg[0].law_id == "01/2021/NĐ-CP"


# --------------------------------------------------------------------------- #
# K-selection policy
# --------------------------------------------------------------------------- #
def test_select_returns_min_k_even_without_close_second():
    cands = [
        ArticleCandidate("01/2021/NĐ-CP", "ND", "Điều 12", 0.90),
        ArticleCandidate("01/2021/NĐ-CP", "ND", "Điều 99", 0.20),
    ]
    chosen = select_articles(cands, SelectConfig(min_k=1, max_k=3))
    assert [a.dieu for a in chosen] == ["Điều 12"]


def test_select_admits_close_second():
    cands = [
        ArticleCandidate("45/2022/NĐ-CP", "ND", "Điều 23", 0.80),
        ArticleCandidate("45/2022/NĐ-CP", "ND", "Điều 33", 0.78),
    ]
    chosen = select_articles(
        cands, SelectConfig(rel_margin=0.18, abs_margin=0.12, max_k=3)
    )
    assert {a.dieu for a in chosen} == {"Điều 23", "Điều 33"}


def test_select_caps_at_max_k():
    cands = [
        ArticleCandidate("01/2021/NĐ-CP", "ND", f"Điều {i}", 0.80 - i * 0.001)
        for i in range(1, 8)
    ]
    chosen = select_articles(cands, SelectConfig(max_k=3))
    assert len(chosen) == 3


def test_select_relevant_strings_format():
    cands = [ArticleCandidate("01/2021/NĐ-CP", "Nghị định 01", "Điều 12", 0.9)]
    out = select_relevant_strings(cands, SelectConfig())
    assert out == ["01/2021/NĐ-CP|Nghị định 01|Điều 12"]


# --------------------------------------------------------------------------- #
# End-to-end: recovery from the dev-set baseline failure mode
# --------------------------------------------------------------------------- #
def test_recovers_gold_from_provincial_noise():
    """Mirror dev-set id=1 *after* the cross-encoder rerank stage.

    The selector runs on post-rerank scores, where semantic relevance already
    ranks the on-topic central-decree articles competitively. An off-topic
    customs decision that scored highly under raw BM25 has been demoted by the
    reranker. The authority prior + provincial suppression then break the
    remaining ties so both gold articles (``Điều 1``/``Điều 12`` from
    01/2021/NĐ-CP) are selected.
    """
    cands = [
        # Provincial decision: dropped outright regardless of score.
        ArticleCandidate("1934/2007/QĐ-UBND", "QĐ tỉnh", "Điều 4", 0.55),
        # Off-topic customs decision: low post-rerank score.
        ArticleCandidate("2396/QĐ-TCHQ", "QĐ hải quan", "Điều 3", 0.40),
        # Superseded but on-topic registration decree: mid score.
        ArticleCandidate("02/2000/NĐ-CP", "Nghị định cũ", "Điều 8", 0.62),
        # Authoritative current decree: top semantic scores.
        ArticleCandidate("01/2021/NĐ-CP", "Nghị định 01/2021", "Điều 1", 0.70),
        ArticleCandidate("01/2021/NĐ-CP", "Nghị định 01/2021", "Điều 12", 0.68),
    ]
    chosen = select_relevant_strings(
        cands, SelectConfig(drop_provincial=True, rel_margin=0.18, abs_margin=0.12)
    )
    chosen_dieus = {s.split("|")[2] for s in chosen}
    # Both gold articles recovered; provincial Điều 4 dropped entirely.
    assert "Điều 1" in chosen_dieus
    assert "Điều 12" in chosen_dieus
    assert "Điều 4" not in chosen_dieus
