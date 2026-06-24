"""Graph expansion for G-LRAG retrieval (PLAN.md task 7.3).

Adds graph-derived candidate chunks to a fused candidate list, with
discounted scores so they only surface when the cross-encoder reranker
(Stage 7.4) judges them relevant.

Two expansion paths (G-LRAG_SPECIFICATIONS.md §10.4):

1. **1-hop DOC→ART expansion** — for each seed chunk, walk to its parent DOC,
   traverse the cross-document relations in :data:`EXPANSION_DOC_RELS`
   (canonical + reverse) to neighbouring DOCs, and pull in the neighbour
   DOCs' article chunks. Discount = ``DISCOUNT_DOC`` (0.6).

2. **1.5-hop concept co-mention** — for each seed chunk, find the CONCEPT
   nodes it MENTIONS, then find sibling chunks that also MENTION those
   concepts. Discount = ``DISCOUNT_CONCEPT`` (0.3).

Both paths **gracefully degrade** when CHUNK / CONCEPT nodes are absent
(e.g. a DOC→ART-only graph) — verified by the CPU baseline.

The expander also builds a human-readable ``build_graph_context`` block for
the generation prompt (Stage 8.1).

KG node-id conventions (from ``src/data/stage5_build_graph.py``):
  - ``DOC:{doc_id}``  (type "Document")
  - ``ART:{doc_uid}`` (type "Article", attr ``doc_uid``)
  - ``CHUNK:{chunk_id}`` (type "Chunk")
  - ``CONCEPT:{name_lower}`` (type "Concept", attr ``name``)
"""

from __future__ import annotations

import pickle
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx

__all__ = [
    "GraphExpander",
    "EXPANSION_DOC_RELS",
    "DISCOUNT_DOC",
    "DISCOUNT_CONCEPT",
]

# Cross-document relations traversed at retrieval time
# (config/relationship_mapping.yaml EXPANSION_RELATIONS). Both the canonical
# direction (out-edge) and the reverse (in-edge) are followed.
EXPANSION_DOC_RELS = (
    "DETAILS",
    "AMENDS",
    "REPLACES",
    "CITES_REF",
    "BASED_ON",
)

DISCOUNT_DOC = 0.6
DISCOUNT_CONCEPT = 0.3

# A "directed" relation label → human-readable forward/reverse phrasing used
# by build_graph_context. Forward = the canonical edge direction (DOC→neighbour);
# Reverse = the neighbour→DOC direction we reconstruct at query time.
_REL_LABELS: Dict[str, Tuple[str, str]] = {
    "DETAILS": ("DETAILS", "DETAILS_BY"),
    "AMENDS": ("AMENDS", "AMENDED_BY"),
    "REPLACES": ("REPLACES", "REPLACED_BY"),
    "CITES_REF": ("CITES_REF", "CITED_BY_REF"),
    "BASED_ON": ("BASED_ON", "BASE_FOR"),
}


class GraphExpander:
    """Expand a fused candidate list with graph-derived neighbour chunks.

    Parameters
    ----------
    G:
        The ``networkx.MultiDiGraph`` knowledge graph (``kg.gpickle``).
    row_to_uid:
        Mapping ``row_idx -> doc_uid`` for every chunk that can appear in a
        candidate list. Used to locate the seed chunk's parent ART node via
        ``ART:{doc_uid}``.
    discount_doc:
        Score multiplier applied to 1-hop DOC→ART expanded candidates.
    discount_concept:
        Score multiplier applied to 1.5-hop concept-co-mention candidates.
    """

    @staticmethod
    def _art_uid_from_node(node_id: str) -> Optional[str]:
        """Return the ``doc_uid`` portion of an ``ART:{doc_uid}`` node id."""
        node_id = str(node_id)
        prefix = "ART:"
        if node_id.startswith(prefix):
            return node_id[len(prefix):]
        return None

    def __init__(
        self,
        G: nx.MultiDiGraph,
        row_to_uid: Dict[int, str],
        discount_doc: float = DISCOUNT_DOC,
        discount_concept: float = DISCOUNT_CONCEPT,
    ) -> None:
        self.G = G
        self.row_to_uid = {int(k): str(v) for k, v in row_to_uid.items()}
        self.discount_doc = discount_doc
        self.discount_concept = discount_concept

        # uid -> doc_id: resolve the parent DOC of an ART via the HAS_ARTICLE
        # *in-edge* (robust to graphs whose ART nodes carry only `doc_uid` and
        # no `doc_id` attribute, e.g. the unit-test synthetic graph). The
        # `doc_id` attribute is used only as a fallback.
        self._uid_to_doc_id: Dict[str, str] = {}
        for u, v, k, d in G.edges(keys=True, data=True):
            if d.get("relation") != "HAS_ARTICLE":
                continue
            # u is DOC:{doc_id}, v is ART:{doc_uid}
            art_uid = self._art_uid_from_node(v)
            if art_uid is None:
                continue
            doc_id = str(u)[len("DOC:"):]
            self._uid_to_doc_id.setdefault(art_uid, doc_id)
        # Fallback: ART nodes that carry an explicit doc_id attr.
        for n, d in G.nodes(data=True):
            if d.get("type") == "Article":
                uid = d.get("doc_uid")
                doc_id = d.get("doc_id")
                if uid and doc_id and str(uid) not in self._uid_to_doc_id:
                    self._uid_to_doc_id[str(uid)] = str(doc_id)

        # doc_id -> set of doc_uids for its ART nodes (neighbour articles),
        # resolved via HAS_ARTICLE *out-edges* (DOC -> ART) with attr fallback.
        self._doc_id_to_uids: Dict[str, set] = {}
        for u, v, k, d in G.edges(keys=True, data=True):
            if d.get("relation") != "HAS_ARTICLE":
                continue
            doc_id = str(u)[len("DOC:"):]
            art_uid = self._art_uid_from_node(v)
            if art_uid is not None:
                self._doc_id_to_uids.setdefault(doc_id, set()).add(art_uid)
        for n, d in G.nodes(data=True):
            if d.get("type") == "Article":
                uid = str(d.get("doc_uid", ""))
                doc_id = str(d.get("doc_id", ""))
                if uid and doc_id:
                    self._doc_id_to_uids.setdefault(doc_id, set()).add(uid)

        # uid -> set of row_idx (chunks belonging to that article).
        # Built lazily from row_to_uid.
        self._uid_to_rows: Dict[str, set] = {}
        for row_idx, uid in self.row_to_uid.items():
            self._uid_to_rows.setdefault(uid, set()).add(row_idx)

        # Detect CHUNK / CONCEPT nodes for graceful-degradation flags.
        self._has_chunks = any(
            d.get("type") == "Chunk" for _, d in G.nodes(data=True)
        )
        self._has_concepts = any(
            d.get("type") == "Concept" for _, d in G.nodes(data=True)
        )

        # chunk_id -> row_idx and reverse, only if chunks present.
        self._chunk_id_to_row: Dict[str, int] = {}
        self._row_to_chunk_id: Dict[int, str] = {}
        if self._has_chunks:
            for n, d in G.nodes(data=True):
                if d.get("type") == "Chunk":
                    cid = str(d.get("chunk_id", ""))
                    rowidx = d.get("rowidx")
                    if cid and rowidx is not None:
                        self._chunk_id_to_row[cid] = int(rowidx)
                        self._row_to_chunk_id[int(rowidx)] = cid

        # CONCEPT -> set of chunk row_idx that mention it (concept co-mention).
        self._concept_to_rows: Dict[str, set] = {}
        if self._has_chunks and self._has_concepts:
            for u, v, k, d in G.edges(keys=True, data=True):
                if d.get("relation") != "MENTIONS":
                    continue
                # u is CHUNK:{chunk_id}, v is CONCEPT:{name_lower}
                if not str(u).startswith("CHUNK:"):
                    continue
                chunk_id = str(u)[len("CHUNK:"):]
                row_idx = self._chunk_id_to_row.get(chunk_id)
                if row_idx is None:
                    continue
                self._concept_to_rows.setdefault(str(v), set()).add(row_idx)

    # ------------------------------------------------------------------ #
    # Constructors
    # ------------------------------------------------------------------ #
    @classmethod
    def from_graph_and_meta(
        cls,
        kg_pickle_path: str,
        meta_parquet_path: str,
        discount_doc: float = DISCOUNT_DOC,
        discount_concept: float = DISCOUNT_CONCEPT,
    ) -> "GraphExpander":
        """Build from ``kg.gpickle`` + ``chunk_meta_slim.parquet``.

        The parquet must contain at least ``row_idx`` and ``doc_uid`` columns.
        """
        with open(kg_pickle_path, "rb") as f:
            G = pickle.load(f)
        import pandas as pd

        meta = pd.read_parquet(meta_parquet_path)
        row_to_uid = {
            int(r["row_idx"]): str(r["doc_uid"])
            for _, r in meta[["row_idx", "doc_uid"]].iterrows()
        }
        return cls(G, row_to_uid, discount_doc=discount_doc,
                   discount_concept=discount_concept)

    # ------------------------------------------------------------------ #
    # Core expansion
    # ------------------------------------------------------------------ #
    def expand(
        self,
        candidates: Sequence[Tuple[int, float]],
        top_n: int = 50,
    ) -> List[Dict[str, Any]]:
        """Expand a candidate list ``[(row_idx, score), ...]``.

        Returns a list of dicts ``{row_idx, score, source}`` where
        ``source`` is one of ``"candidate"`` (seed), ``"doc_expand"`` or
        ``"concept_expand"``, truncated to ``top_n`` and sorted by score
        descending (row_idx asc tiebreak). Seeds are always retained.
        """
        out: List[Dict[str, Any]] = []
        seen: Dict[int, str] = {}  # row_idx -> source

        # 1) retain seeds
        for row_idx, score in candidates:
            row_idx = int(row_idx)
            out.append({"row_idx": row_idx, "score": float(score),
                        "source": "candidate"})
            seen[row_idx] = "candidate"

        # 2) 1-hop DOC→ART expansion
        self._expand_doc_hop(candidates, out, seen)

        # 3) 1.5-hop concept co-mention (only if chunks+concepts present)
        if self._has_chunks and self._has_concepts:
            self._expand_concept_comention(candidates, out, seen)

        # Sort by score desc, row_idx asc; truncate.
        out.sort(key=lambda r: (-r["score"], r["row_idx"]))
        return out[:top_n]

    def _expand_doc_hop(
        self,
        candidates: Sequence[Tuple[int, float]],
        out: List[Dict[str, Any]],
        seen: Dict[int, str],
    ) -> None:
        """Add neighbour-article chunks via 1-hop DOC→DOC traversal."""
        for row_idx, score in candidates:
            row_idx = int(row_idx)
            uid = self.row_to_uid.get(row_idx)
            if not uid:
                continue
            doc_id = self._uid_to_doc_id.get(uid)
            if not doc_id:
                continue
            doc_node = f"DOC:{doc_id}"
            if doc_node not in self.G.nodes:
                continue
            # Gather neighbour doc_ids (canonical out + reverse in).
            neighbour_doc_ids = self._doc_neighbours(doc_node)
            for nb_doc_id in neighbour_doc_ids:
                for nb_uid in self._doc_id_to_uids.get(nb_doc_id, ()):
                    for nb_row in self._uid_to_rows.get(nb_uid, ()):
                        self._add_expanded(
                            out, seen, nb_row,
                            float(score) * self.discount_doc,
                            "doc_expand",
                        )

    def _doc_neighbours(self, doc_node: str) -> List[str]:
        """Return doc_ids of DOC neighbours via EXPANSION_DOC_RELS (both dirs)."""
        nb_doc_ids: List[str] = []
        # canonical out-edges
        for _, v, d in self.G.out_edges(doc_node, data=True):
            if d.get("relation") in EXPANSION_DOC_RELS:
                nb_doc_ids.append(str(v)[len("DOC:"):])
        # reverse in-edges (reconstruct reverse direction in code)
        for u, _, d in self.G.in_edges(doc_node, data=True):
            if d.get("relation") in EXPANSION_DOC_RELS:
                nb_doc_ids.append(str(u)[len("DOC:"):])
        # dedupe preserving order
        seen = set()
        uniq = []
        for x in nb_doc_ids:
            if x and x not in seen:
                seen.add(x)
                uniq.append(x)
        return uniq

    def _expand_concept_comention(
        self,
        candidates: Sequence[Tuple[int, float]],
        out: List[Dict[str, Any]],
        seen: Dict[int, str],
    ) -> None:
        """Add sibling chunks sharing a MENTIONED concept with a seed chunk."""
        for row_idx, score in candidates:
            row_idx = int(row_idx)
            chunk_id = self._row_to_chunk_id.get(row_idx)
            if not chunk_id:
                continue
            chunk_node = f"CHUNK:{chunk_id}"
            if chunk_node not in self.G.nodes:
                continue
            # CONCEPT nodes this chunk mentions.
            concepts = [
                v for _, v, d in self.G.out_edges(chunk_node, data=True)
                if d.get("relation") == "MENTIONS"
            ]
            for concept_node in concepts:
                for sib_row in self._concept_to_rows.get(concept_node, ()):
                    self._add_expanded(
                        out, seen, sib_row,
                        float(score) * self.discount_concept,
                        "concept_expand",
                    )

    @staticmethod
    def _add_expanded(
        out: List[Dict[str, Any]],
        seen: Dict[int, str],
        row_idx: int,
        score: float,
        source: str,
    ) -> None:
        """Add an expanded row, preferring the higher-priority source.

        Seed ("candidate") always wins over expanded entries; among expanded
        entries the higher score wins.
        """
        existing = seen.get(row_idx)
        if existing is None:
            out.append({"row_idx": row_idx, "score": score, "source": source})
            seen[row_idx] = source
            return
        if existing == "candidate":
            return  # never downgrade a seed
        # Both expanded: keep the better score/source.
        prev = next((r for r in out if r["row_idx"] == row_idx), None)
        if prev is not None and score > prev["score"]:
            prev["score"] = score
            prev["source"] = source
            seen[row_idx] = source

    # ------------------------------------------------------------------ #
    # Prompt-context builder
    # ------------------------------------------------------------------ #
    def build_graph_context(self, doc_nodes: Sequence[str]) -> str:
        """Build a human-readable graph-neighbour context string for a prompt.

        ``doc_nodes`` is a list of ``DOC:{doc_id}`` ids (typically the
        top-K hit doc nodes). Returns a newline-joined block of the form::

            DETAILS: <ten>
            AMENDED_BY: <ten>
            ...

        or the placeholder ``"(Không có quan hệ chéo đáng chú ý)"`` when no
        expansion relation is incident on any of the given docs.
        """
        lines: List[str] = []
        for doc_node in doc_nodes:
            if doc_node not in self.G.nodes:
                continue
            my_ten = str(self.G.nodes[doc_node].get("ten", doc_node))
            # out-edges: canonical direction → forward label
            for _, v, d in self.G.out_edges(doc_node, data=True):
                rel = d.get("relation")
                if rel in EXPANSION_DOC_RELS:
                    fwd, _ = _REL_LABELS.get(rel, (rel, rel + "_BY"))
                    other_ten = str(self.G.nodes[v].get("ten", v))
                    lines.append(f"{fwd}: {other_ten}")
            # in-edges: reverse direction → reverse label
            for u, _, d in self.G.in_edges(doc_node, data=True):
                rel = d.get("relation")
                if rel in EXPANSION_DOC_RELS:
                    _, rev = _REL_LABELS.get(rel, (rel, rel + "_BY"))
                    other_ten = str(self.G.nodes[u].get("ten", u))
                    lines.append(f"{rev}: {other_ten}")
        # Preserve order but dedupe identical lines.
        deduped: List[str] = []
        seen = set()
        for ln in lines:
            if ln not in seen:
                seen.add(ln)
                deduped.append(ln)
        if not deduped:
            return "(Không có quan hệ chéo đáng chú ý)"
        return "\n".join(deduped)
