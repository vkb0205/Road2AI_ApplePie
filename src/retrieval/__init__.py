# G-LRAG retrieval package.
# Heavy GPU deps (faiss, torch, FlagEmbedding, qdrant_client) are imported
# lazily inside the methods that need them — importing this package does NOT
# trigger those imports. See RETRIEVAL_WORKFLOW.md for the pipeline contract.

from retrieval.retriever import (
    HybridRetriever,
    RetrievalConfig,
    Hit,
    make_relevant_lists,
    RetrievalTrace,
    StageSnapshot,
)

from retrieval.debug import (
    format_trace,
    snapshot_items,
    SubQueryTrace,
    RouterDebugLog,
    format_sub_query_trace,
)

from retrieval.sub_query_router import (
    SubQueryRouter,
    SubQueryRouterConfig,
    RouterDecision,
    RouterFallbackLog,
    clean_sub_query,
    normalize_sub_queries,
    rule_based_decompose,
    DECOMPOSITION_PROMPT,
)

from retrieval.rrf import rrf_fuse, rrf_score_only

from retrieval.graph_expand import GraphExpander, EXPANSION_DOC_RELS

from retrieval.bm25_index import (
    FTSIndex,
    PureBM25,
    build_fts_match_expr,
    tokenize_query,
)

__all__ = [
    # Core retriever
    "HybridRetriever",
    "RetrievalConfig",
    "Hit",
    "make_relevant_lists",
    "RetrievalTrace",
    "StageSnapshot",
    # Debug / observability
    "format_trace",
    "snapshot_items",
    "SubQueryTrace",
    "RouterDebugLog",
    "format_sub_query_trace",
    # Sub-query routing
    "SubQueryRouter",
    "SubQueryRouterConfig",
    "RouterDecision",
    "RouterFallbackLog",
    "clean_sub_query",
    "normalize_sub_queries",
    "rule_based_decompose",
    "DECOMPOSITION_PROMPT",
    # RRF
    "rrf_fuse",
    "rrf_score_only",
    # Graph expansion
    "GraphExpander",
    "EXPANSION_DOC_RELS",
    # BM25 / FTS
    "FTSIndex",
    "PureBM25",
    "build_fts_match_expr",
    "tokenize_query",
]
