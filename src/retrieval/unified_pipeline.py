"""Unified G-LRAG pipeline: retrieve → (optional decompose) → select → generate.

A single, maintainable orchestrator that supersedes the ~7700-line decomposition
notebook by composing the unit-tested building blocks instead of duplicating
them, and **without any hardcoded law-id / domain / facet tables** (the main
generalisation risk in the legacy notebook):

    query
      → [optional] SubQueryRouter.route  (rule-based decomposition)
      → DocAnchoredRetriever.retrieve_pool  (per sub-query: FTS+FAISS→anchor→
        harvest→cross-encoder rerank)
      → fuse sub-query candidate pools (MAX reranker score per article — keeps
        the reranker [0,1] scale the selector's margins/authority expect)
      → select_articles  (authority prior by document TYPE + adaptive K)
      → [optional] IRACGenerator  (grounded answer with in-list Điều X cite)
      → grader record {id, question, answer, relevant_docs, relevant_articles}

Every heavy dependency (torch / faiss / transformers) is reached only through
the injected retriever / generator, so this module imports and unit-tests on a
CPU-only box with fake callables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from retrieval.article_select import (
    AggregatedArticle,
    ArticleCandidate,
    SelectConfig,
    aggregate_articles,
    canonical_dieu,
    select_articles,
    select_config_for_complexity,
)
from retrieval.doc_anchor import DocAnchoredRetriever

__all__ = [
    "UnifiedConfig",
    "UnifiedPipeline",
    "fuse_candidate_pools",
    "build_query_variants",
    "summarize_timings",
    "build_record",
    "run_dev_set",
    "validate_submission",
    "dump_json",
]

# (law_id, ten_van_ban, dieu_so) -> chunk text, for generation context.
TextProvider = Callable[[str, str, str], str]


@dataclass
class UnifiedConfig:
    """Top-level switches for the unified pipeline."""

    use_decomposition: bool = False   # rule-based SubQueryRouter pre-split
    max_sub_queries: int = 4
    gen_context_topk: int = 6         # passages handed to the generator
    # --- multi-variant recall layer (ported from the teammate, de-hardcoded) -
    use_multi_variant: bool = False   # gather across variants, rerank ONCE
    max_variants: int = 5             # hard cap on dense searches per question
    use_complexity_k: bool = False    # let plan.complexity set per-query K
    coverage_quota: bool = True       # admit each sub-query's top article


def build_query_variants(
    query: str,
    sub_queries: Optional[Sequence[str]] = None,
    facets: Optional[Sequence[str]] = None,
    max_variants: int = 5,
) -> List[str]:
    """Assemble the retrieval variants: original + atomic + facet, deduped.

    The original question always comes first (highest-trust signal). Atomic
    sub-queries and facet phrases follow, each contributing a distinct search
    angle. Exact-duplicate strings (case/space-insensitive) are dropped, and
    the list is hard-capped at ``max_variants`` to bound the dense-search count
    (the only per-variant cost — the cross-encoder still runs once downstream).
    """
    out: List[str] = []
    seen: set = set()

    def _add(text: str) -> None:
        t = (text or "").strip()
        if not t:
            return
        key = " ".join(t.lower().split())
        if key in seen:
            return
        seen.add(key)
        out.append(t)

    _add(query)
    for sq in sub_queries or []:
        _add(sq)
    for f in facets or []:
        _add(f)
    return out[:max_variants]


def fuse_candidate_pools(
    pools: Sequence[Sequence[ArticleCandidate]],
) -> List[ArticleCandidate]:
    """Fuse per-sub-query candidate pools by canonical ``(law_id, Điều)``.

    The fused score is the **MAX** reranker score for that article across
    sub-queries. Max (not summed RRF) deliberately preserves the reranker's
    ``[0, 1]`` scale, which is exactly the scale
    :func:`retrieval.article_select.select_articles` expects for its relative /
    absolute admission margins and additive authority prior — summing or RRF
    would distort that scale and silently break K-selection.
    """
    best: Dict[Tuple[str, str], ArticleCandidate] = {}
    for pool in pools:
        for c in pool:
            key = (str(c.law_id), canonical_dieu(c.dieu_so))
            cur = best.get(key)
            if cur is None or c.score > cur.score:
                best[key] = c
    out = list(best.values())
    out.sort(key=lambda c: c.score, reverse=True)
    return out


class UnifiedPipeline:
    """Compose retrieval, optional decomposition, selection and generation.

    Parameters
    ----------
    retriever:
        A built :class:`retrieval.doc_anchor.DocAnchoredRetriever`.
    select_cfg:
        :class:`retrieval.article_select.SelectConfig` for K-selection.
    router:
        Optional :class:`retrieval.sub_query_router.SubQueryRouter`. Only used
        when ``cfg.use_decomposition`` is True.
    generator:
        Optional object with ``generate(question, contexts, relevant_articles)``
        (an :class:`generation.generator.IRACGenerator`). ``None`` → answer="".
    text_provider:
        Optional ``(law_id, ten_van_ban, dieu_so) -> str`` used to attach chunk
        text to the generation contexts. ``None`` → contexts carry metadata only
        (the answer still cites correctly via the allowed list).
    cfg:
        :class:`UnifiedConfig` switches.
    """

    def __init__(
        self,
        retriever: DocAnchoredRetriever,
        select_cfg: Optional[SelectConfig] = None,
        router: Any = None,
        generator: Any = None,
        text_provider: Optional[TextProvider] = None,
        cfg: Optional[UnifiedConfig] = None,
    ) -> None:
        self.retriever = retriever
        self.select_cfg = select_cfg or SelectConfig()
        self.router = router
        self.generator = generator
        self.text_provider = text_provider
        self.cfg = cfg or UnifiedConfig()

    # ------------------------------------------------------------------ #
    def _plan(self, query: str) -> Any:
        """Run the planner/router once and return its decision.

        Uses the richer :meth:`SubQueryRouter.plan` (decomposition + complexity
        + facets) when the multi-variant or complexity-K layers are active, so
        the single shared Qwen call yields every signal at once. Falls back to
        the lighter :meth:`route` otherwise, and to ``None`` when no router is
        wired (the caller then treats the query as a single, medium-complexity
        variant — i.e. exactly the legacy behaviour).
        """
        if self.router is None:
            return None
        wants_plan = self.cfg.use_multi_variant or self.cfg.use_complexity_k
        if wants_plan and hasattr(self.router, "plan"):
            return self.router.plan(query)
        if self.cfg.use_decomposition:
            return self.router.route(query)
        return None

    def _sub_queries(self, query: str) -> List[str]:
        """Return the sub-queries to retrieve for (>=1; the original if no split)."""
        if not (self.cfg.use_decomposition and self.router is not None):
            return [query]
        decision = self.router.route(query)
        subs = list(getattr(decision, "sub_queries", []) or [])
        if getattr(decision, "should_decompose", False) and len(subs) > 1:
            return subs[: self.cfg.max_sub_queries]
        return [query]

    def _select_cfg_for(self, plan: Any) -> SelectConfig:
        """Per-query SelectConfig: widen/narrow K by plan complexity if enabled."""
        if self.cfg.use_complexity_k and plan is not None:
            complexity = getattr(plan, "complexity", "medium")
            return select_config_for_complexity(complexity, self.select_cfg)
        return self.select_cfg

    def retrieve_candidates(self, query: str) -> List[ArticleCandidate]:
        """Retrieve the article-candidate pool (legacy single/decomp path)."""
        subs = self._sub_queries(query)
        if len(subs) == 1:
            return list(self.retriever.retrieve_pool(subs[0]))
        pools = [self.retriever.retrieve_pool(sq) for sq in subs]
        return fuse_candidate_pools(pools)

    def _retrieve_candidates_planned(
        self, query: str, plan: Any
    ) -> Tuple[List[ArticleCandidate], int]:
        """Retrieve candidates using the plan; return (candidates, n_sub).

        When multi-variant is enabled, builds ``original + atomic + facet``
        variants and calls the gather-then-rerank-ONCE retriever so the
        cross-encoder runs a single time. Otherwise reuses the legacy
        single/decomposition path. ``n_sub`` is the number of atomic
        sub-queries (drives the coverage-quota floor).
        """
        if plan is not None:
            subs = list(getattr(plan, "sub_queries", []) or [])
            if not (getattr(plan, "should_decompose", False) and len(subs) > 1):
                subs = [query]
        else:
            subs = [query]
        n_sub = len(subs)

        if self.cfg.use_multi_variant and hasattr(
            self.retriever, "retrieve_pool_multi"
        ):
            facets = list(getattr(plan, "facets", []) or []) if plan else []
            variants = build_query_variants(
                query,
                sub_queries=subs if n_sub > 1 else None,
                facets=facets,
                max_variants=self.cfg.max_variants,
            )
            cands = list(
                self.retriever.retrieve_pool_multi(variants, rerank_query=query)
            )
            return cands, n_sub

        # Legacy retrieval (single or per-sub-query fusion).
        if n_sub == 1:
            return list(self.retriever.retrieve_pool(subs[0])), 1
        pools = [self.retriever.retrieve_pool(sq) for sq in subs]
        return fuse_candidate_pools(pools), n_sub

    def _select_from(
        self, candidates: Sequence[ArticleCandidate], cfg: SelectConfig, n_sub: int
    ) -> List[AggregatedArticle]:
        """Margin-select with an optional coverage-quota recall floor.

        The coverage quota is the budget-safe, de-hardcoded port of the
        teammate's role/facet-completion passes: when the query decomposed into
        ``n_sub`` distinct angles, raise the effective ``min_k`` so at least one
        article per angle survives (bounded by ``max_k``). It reuses the scores
        already computed by the single shared rerank — no extra rerank pass — so
        it costs nothing on the GPU. With ``n_sub==1`` it is a no-op.
        """
        if self.cfg.coverage_quota and n_sub > 1:
            from dataclasses import replace

            floor = max(cfg.min_k, min(n_sub, cfg.max_k))
            cfg = replace(cfg, min_k=floor)
        return select_articles(candidates, cfg)

    def select(self, query: str) -> List[AggregatedArticle]:
        """Retrieve + select the final articles for a query (plan-aware)."""
        plan = self._plan(query)
        candidates, n_sub = self._retrieve_candidates_planned(query, plan)
        cfg = self._select_cfg_for(plan)
        return self._select_from(candidates, cfg, n_sub)

    def _gen_contexts(
        self, chosen: Sequence[AggregatedArticle]
    ) -> List[Dict[str, Any]]:
        """Build generation contexts (metadata + optional chunk text)."""
        contexts: List[Dict[str, Any]] = []
        for a in chosen[: self.cfg.gen_context_topk]:
            text = ""
            if self.text_provider is not None:
                try:
                    text = self.text_provider(a.law_id, a.ten_van_ban, a.dieu) or ""
                except Exception:
                    text = ""
            contexts.append(
                {
                    "law_id": a.law_id,
                    "ten_van_ban": a.ten_van_ban,
                    "dieu_so": a.dieu,
                    "chunk_text": text,
                }
            )
        return contexts

    def answer_record(self, qid: Any, question: str) -> Dict[str, Any]:
        """Produce one grader-format record for a question."""
        chosen = self.select(question)
        relevant_articles = [a.to_relevant_string() for a in chosen]
        # relevant_docs: distinct source documents of the chosen articles.
        seen: List[str] = []
        for a in chosen:
            doc = f"{a.law_id}|{a.ten_van_ban}"
            if doc not in seen:
                seen.append(doc)
        answer = ""
        if self.generator is not None:
            contexts = self._gen_contexts(chosen)
            answer = self.generator.generate(question, contexts, relevant_articles)
        return {
            "id": qid,
            "question": question,
            "answer": answer,
            "relevant_docs": seen,
            "relevant_articles": relevant_articles,
        }

    def answer_record_timed(
        self, qid: Any, question: str
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Like :meth:`answer_record`, but also returns per-stage wall-clock.

        The timing dict separates the cost centres so the recall layers'
        overhead is measurable and the single-rerank invariant is verifiable:

        ``decompose``  seconds in the planner/router decision (one LLM call).
        ``retrieve``   seconds in retrieval+rerank. With multi-variant this
                       runs N cheap dense searches but the cross-encoder ONCE.
        ``select``     seconds in F2 article selection (cheap, CPU).
        ``generate``   seconds in the single IRAC generation pass (fixed).
        ``n_sub``      number of atomic sub-queries from the plan.
        ``n_variants`` number of retrieval variants actually searched.
        ``complexity`` plan complexity label (drives K when enabled).
        ``multi_variant`` True when the gather-then-rerank-once path ran.
        ``decomposed`` True when the plan split into >1 sub-query.
        ``total``      sum of the above stage times.
        """
        import time

        t0 = time.perf_counter()
        plan = self._plan(question)
        t_decompose = time.perf_counter() - t0

        # Reconstruct the variants count for reporting (mirrors the retrieval
        # path) without paying for retrieval twice.
        subs = list(getattr(plan, "sub_queries", []) or []) if plan else []
        if not (plan and getattr(plan, "should_decompose", False) and len(subs) > 1):
            subs = [question]
        n_sub = len(subs)
        multi_variant = self.cfg.use_multi_variant and hasattr(
            self.retriever, "retrieve_pool_multi"
        )
        if multi_variant:
            facets = list(getattr(plan, "facets", []) or []) if plan else []
            variants = build_query_variants(
                question,
                sub_queries=subs if n_sub > 1 else None,
                facets=facets,
                max_variants=self.cfg.max_variants,
            )
            n_variants = len(variants)
        else:
            n_variants = n_sub

        t0 = time.perf_counter()
        candidates, n_sub = self._retrieve_candidates_planned(question, plan)
        t_retrieve = time.perf_counter() - t0

        cfg = self._select_cfg_for(plan)
        t0 = time.perf_counter()
        chosen = self._select_from(candidates, cfg, n_sub)
        t_select = time.perf_counter() - t0

        relevant_articles = [a.to_relevant_string() for a in chosen]
        seen: List[str] = []
        for a in chosen:
            doc = f"{a.law_id}|{a.ten_van_ban}"
            if doc not in seen:
                seen.append(doc)

        answer = ""
        t_generate = 0.0
        if self.generator is not None:
            contexts = self._gen_contexts(chosen)
            t0 = time.perf_counter()
            answer = self.generator.generate(question, contexts, relevant_articles)
            t_generate = time.perf_counter() - t0

        record = {
            "id": qid,
            "question": question,
            "answer": answer,
            "relevant_docs": seen,
            "relevant_articles": relevant_articles,
        }
        timing = {
            "decompose": t_decompose,
            "retrieve": t_retrieve,
            "select": t_select,
            "generate": t_generate,
            "n_sub": n_sub,
            "n_variants": n_variants,
            "complexity": getattr(plan, "complexity", "medium") if plan else "medium",
            "multi_variant": multi_variant,
            "decomposed": n_sub > 1,
            "total": t_decompose + t_retrieve + t_select + t_generate,
        }
        return record, timing


def summarize_timings(timings: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate a list of per-query timing dicts into a report.

    Returns mean/total seconds per stage, the decomposition rate, the mean
    sub-query / variant counts, and the multi-variant rate — so both the
    retrieval multiplier AND the single-rerank invariant are visible (retrieve
    time should stay flat in N_variants because the cross-encoder runs once).
    """
    n = len(timings)
    if n == 0:
        return {"n": 0}
    stages = ("decompose", "retrieve", "select", "generate", "total")
    out: Dict[str, Any] = {"n": n}
    for s in stages:
        vals = [float(t.get(s, 0.0)) for t in timings]
        out[f"{s}_mean"] = sum(vals) / n
        out[f"{s}_total"] = sum(vals)
    n_decomposed = sum(1 for t in timings if t.get("decomposed"))
    out["decomposed_count"] = n_decomposed
    out["decomposed_rate"] = n_decomposed / n
    out["mean_n_sub"] = sum(float(t.get("n_sub", 1)) for t in timings) / n
    out["mean_n_variants"] = sum(float(t.get("n_variants", 1)) for t in timings) / n
    out["max_n_variants"] = max(int(t.get("n_variants", 1)) for t in timings)
    n_mv = sum(1 for t in timings if t.get("multi_variant"))
    out["multi_variant_count"] = n_mv
    out["multi_variant_rate"] = n_mv / n
    return out


# --------------------------------------------------------------------------- #
# Batch driver + submission helpers
# --------------------------------------------------------------------------- #
def run_dev_set(
    pipe: UnifiedPipeline,
    questions_path: str,
    output_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run the pipeline over a ``[{id, question}, ...]`` JSON file.

    When ``output_path`` is provided, the partial result list is flushed to disk
    immediately after every processed question. This makes long Kaggle runs
    crash-inspectable instead of losing all completed records if the kernel dies
    before the full batch finishes.
    """
    import json

    records = json.loads(Path(questions_path).read_text(encoding="utf-8"))
    out: List[Dict[str, Any]] = []
    out_path = Path(output_path) if output_path else None

    def _flush() -> None:
        if out_path is None:
            return
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_name(out_path.name + ".tmp")
        tmp_path.write_text(
            json.dumps(out, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(out_path)

    for rec in records:
        qid = rec.get("id")
        question = rec.get("question") or rec.get("query") or ""
        out.append(pipe.answer_record(qid, question))
        _flush()
    return out


def build_record(
    qid: Any,
    question: str,
    chosen: Sequence[AggregatedArticle],
    answer: str = "",
) -> Dict[str, Any]:
    """Assemble a grader record from already-selected articles (test helper)."""
    relevant_articles = [a.to_relevant_string() for a in chosen]
    seen: List[str] = []
    for a in chosen:
        doc = f"{a.law_id}|{a.ten_van_ban}"
        if doc not in seen:
            seen.append(doc)
    return {
        "id": qid,
        "question": question,
        "answer": answer,
        "relevant_docs": seen,
        "relevant_articles": relevant_articles,
    }


def validate_submission(
    records: Sequence[Dict[str, Any]],
    expected_ids: Optional[Sequence[Any]] = None,
    require_answer_citation: bool = False,
) -> Dict[str, Any]:
    """Validate records against the grader contract (SPEC §13.3).

    Asserts: JSON list of dicts; exactly the 5 required fields per record;
    unique ids (matching ``expected_ids`` when given); ``relevant_docs`` items
    have exactly one ``|``; ``relevant_articles`` items have exactly two ``|``
    with a third segment starting ``Điều``. When ``require_answer_citation`` is
    set, also asserts each answer cites a ``Điều X`` present in its
    ``relevant_articles`` (FR-02). Returns a small summary dict on success.
    """
    import re

    required = {"id", "question", "answer", "relevant_docs", "relevant_articles"}
    assert isinstance(records, list), "submission must be a JSON list"

    ids: List[Any] = []
    for r in records:
        assert isinstance(r, dict), "each record must be a dict"
        assert set(r.keys()) == required, f"record {r.get('id')} has wrong keys: {sorted(r.keys())}"
        ids.append(r["id"])
        assert isinstance(r["relevant_docs"], list), f"record {r['id']} relevant_docs not a list"
        assert isinstance(r["relevant_articles"], list), f"record {r['id']} relevant_articles not a list"
        for d in r["relevant_docs"]:
            assert str(d).count("|") == 1, f"record {r['id']} bad relevant_docs item: {d!r}"
        for a in r["relevant_articles"]:
            parts = str(a).split("|")
            assert len(parts) == 3, f"record {r['id']} bad relevant_articles item: {a!r}"
            assert parts[2].strip().startswith("Điều"), f"record {r['id']} 3rd seg not Điều: {a!r}"
        if require_answer_citation and r["relevant_articles"]:
            cited = set(re.findall(r"Điều\s+\d+[A-Za-zÀ-ỹ]*", str(r["answer"]), re.IGNORECASE))
            allowed = set()
            for a in r["relevant_articles"]:
                allowed |= set(re.findall(r"Điều\s+\d+[A-Za-zÀ-ỹ]*", str(a), re.IGNORECASE))
            norm = lambda s: {" ".join(x.lower().split()) for x in s}
            assert norm(cited) & norm(allowed), f"record {r['id']} answer cites no allowed Điều"

    assert len(ids) == len(set(map(str, ids))), "duplicate ids in submission"
    if expected_ids is not None:
        got, exp = set(map(str, ids)), set(map(str, expected_ids))
        assert got == exp, {"missing": sorted(exp - got)[:20], "extra": sorted(got - exp)[:20]}

    return {"records": len(records), "schema_ok": True, "ids_ok": expected_ids is not None}


def dump_json(obj: Any, path: str) -> None:
    import json

    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
