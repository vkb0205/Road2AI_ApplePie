"""Neo4j-backed graph expansion for G-LRAG retrieval.

A drop-in replacement for :class:`retrieval.graph_expand.GraphExpander` that
queries a **self-hosted Neo4j** instance (populated by
``scripts/upload_graph_to_neo4j.py``) instead of loading ``kg.gpickle`` into
memory. It shares the exact same public contract so :class:`HybridRetriever`
swaps backends with no other change::

    # pickle (in-memory networkx)
    expander = GraphExpander.from_graph_and_meta(kg_pickle, meta_parquet)
    # neo4j (remote, live)
    expander = Neo4jGraphExpander.from_env(meta_parquet)

Neo4j schema (matches ``scripts/upload_graph_to_neo4j.py``)
-----------------------------------------------------------
* Every node carries the base label ``:GraphNode`` **plus** a type label
  (``:Document``, ``:Article``, ``:Chunk``, ``:Concept``).
* The stable graph key (``DOC:{doc_id}``, ``ART:{doc_uid}``,
  ``CHUNK:{chunk_id}``, ``CONCEPT:{name_lower}``) is stored as the node ``id``
  property, with a uniqueness constraint on ``GraphNode.id``.
* Other Stage 5 attributes are node properties:
  ``doc_id``, ``doc_uid``, ``chunk_id``, ``rowidx``, ``ten``, ``name``.
* Every ``relation`` value becomes a Neo4j relationship type:
  ``HAS_ARTICLE``, ``HAS_CHUNK``, ``MENTIONS``, ``DETAILS``, ``AMENDS``,
  ``REPLACES``, ``CITES_REF``, ``BASED_ON``. The networkx edge ``key`` is
  stored as the relationship property ``key`` (only relevant for multi-edges,
  which we don't query by here).

Expansion logic is identical to :class:`GraphExpander`:

1. **1-hop DOC→ART** — seed chunk → parent DOC → cross-document relations in
   ``EXPANSION_DOC_RELS`` (canonical out + reverse in) → neighbour DOCs →
   their article chunks. Discount ``DISCOUNT_DOC`` (0.6).
2. **1.5-hop concept co-mention** — seed chunk → ``MENTIONS`` → CONCEPT
   nodes → sibling chunks that also ``MENTIONS`` those concepts. Discount
   ``DISCOUNT_CONCEPT`` (0.3).

Both paths gracefully degrade when CHUNK/CONCEPT nodes are absent — the Cypher
simply returns no rows, so no special-casing is needed.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "Neo4jGraphExpander",
]

# Re-export the shared constants so callers can import everything from one
# place (and the values stay in sync with the pickle-backed expander).
from retrieval.graph_expand import (  # noqa: E402
    DISCOUNT_CONCEPT,
    DISCOUNT_DOC,
    EXPANSION_DOC_RELS,
    _REL_LABELS,
)

# Relationship types traversed for DOC→DOC hops, as a Cypher alternation.
# EXPANSION_DOC_RELS is a tuple of sanitized identifiers (uppercase ASCII),
# safe to interpolate directly into a Cypher type expression.
_DOC_RELS_CY = "|".join(EXPANSION_DOC_RELS)


class Neo4jGraphExpander:
    """Graph expansion backed by a remote Neo4j instance.

    Parameters
    ----------
    driver:
        An open ``neo4j.GraphDatabase`` driver. The caller (or
        :meth:`from_env`) is responsible for closing it; this class keeps a
        reference and uses sessions per-query.
    row_to_uid:
        Mapping ``row_idx -> doc_uid`` for every chunk that can appear in a
        candidate list. Used to locate the seed chunk's parent ART/DOC node.
        Identical to the ``GraphExpander`` constructor argument.
    discount_doc, discount_concept:
        Score multipliers for the two expansion paths (default 0.6 / 0.3).
    """

    def __init__(
        self,
        driver: Any,
        row_to_uid: Dict[int, str],
        discount_doc: float = DISCOUNT_DOC,
        discount_concept: float = DISCOUNT_CONCEPT,
    ) -> None:
        self.driver = driver
        self.row_to_uid = {int(k): str(v) for k, v in row_to_uid.items()}
        # Reverse mapping uid -> set(row_idx), mirroring GraphExpander._uid_to_rows.
        # This is what makes the DOC→ART expansion work on a graph that has NO
        # Chunk nodes: neighbour article doc_uids are translated back to row_idx
        # via the parquet sidecar, not via HAS_CHUNK edges in the graph.
        self._uid_to_rows: Dict[str, set] = {}
        for ridx, uid in self.row_to_uid.items():
            self._uid_to_rows.setdefault(uid, set()).add(ridx)
        self.discount_doc = float(discount_doc)
        self.discount_concept = float(discount_concept)

    # ------------------------------------------------------------------ #
    # Constructors
    # ------------------------------------------------------------------ #
    @classmethod
    def from_env(
        cls,
        meta_parquet_path: str,
        *,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        discount_doc: float = DISCOUNT_DOC,
        discount_concept: float = DISCOUNT_CONCEPT,
    ) -> "Neo4jGraphExpander":
        """Build from ``chunk_meta_slim.parquet`` + Neo4j env vars.

        Reads (in order) the constructor arguments, then env vars:
        ``NEO4J_URI``, ``NEO4J_USERNAME`` (also accepts ``NEO4J_USER``),
        ``NEO4J_PASSWORD``. ``database`` defaults to Neo4j's default DB
        (``None`` → the driver picks ``neo4j``).

        The ``neo4j`` driver is imported lazily so this module imports on a
        machine without the driver installed.
        """
        uri = uri or os.environ.get("NEO4J_URI")
        username = (
            username
            or os.environ.get("NEO4J_USERNAME")
            or os.environ.get("NEO4J_USER")
            or "neo4j"
        )
        password = password or os.environ.get("NEO4J_PASSWORD")
        if not uri:
            raise RuntimeError(
                "Neo4jGraphExpander.from_env: NEO4J_URI is not set. "
                "Set it in .env (e.g. bolt+s://host:7687) or pass uri=."
            )
        if not password:
            raise RuntimeError(
                "Neo4jGraphExpander.from_env: NEO4J_PASSWORD is not set. "
                "Set it in .env or pass password=."
            )

        # Load python-dotenv if available so .env is picked up when this is
        # called outside a script that already loaded it.
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except Exception:
            pass

        try:
            from neo4j import GraphDatabase
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "neo4j driver is required; install it with `pip install neo4j`"
            ) from exc

        driver = GraphDatabase.driver(uri, auth=(username, password))
        driver.verify_connectivity()

        import pandas as pd

        meta = pd.read_parquet(meta_parquet_path)
        row_to_uid = {
            int(r["row_idx"]): str(r["doc_uid"])
            for _, r in meta[["row_idx", "doc_uid"]].iterrows()
        }
        return cls(
            driver,
            row_to_uid,
            discount_doc=discount_doc,
            discount_concept=discount_concept,
        )

    # ------------------------------------------------------------------ #
    # Core expansion — duck-typed to match GraphExpander.expand()
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
        seen: Dict[int, str] = {}

        # 1) retain seeds (identical to GraphExpander)
        for row_idx, score in candidates:
            row_idx = int(row_idx)
            out.append(
                {"row_idx": row_idx, "score": float(score), "source": "candidate"}
            )
            seen[row_idx] = "candidate"

        # 2) 1-hop DOC→ART expansion via Cypher
        doc_rows = self._expand_doc_hop(candidates)
        for r in doc_rows:
            self._add_expanded(
                out, seen, int(r["row_idx"]),
                float(r["score"]), "doc_expand",
            )

        # 3) 1.5-hop concept co-mention via Cypher
        concept_rows = self._expand_concept_comention(candidates)
        for r in concept_rows:
            self._add_expanded(
                out, seen, int(r["row_idx"]),
                float(r["score"]), "concept_expand",
            )

        out.sort(key=lambda r: (-r["score"], r["row_idx"]))
        return out[:top_n]

    # ------------------------------------------------------------------ #
    # 1-hop DOC→ART expansion
    # ------------------------------------------------------------------ #
    def _expand_doc_hop(
        self,
        candidates: Sequence[Tuple[int, float]],
    ) -> List[Dict[str, Any]]:
        """Neighbour-article chunks via 1-hop DOC→DOC traversal.

        For every seed ``(row_idx, score)`` we resolve its ART node
        (``ART:{doc_uid}``), walk to the parent DOC, traverse the
        ``EXPANSION_DOC_RELS`` relations in **both directions** to
        neighbouring DOCs, and collect the ``doc_uid`` of every Article
        belonging to those neighbour DOCs.

        The neighbour ``doc_uid``s are then translated back to ``row_idx`` in
        Python via the parquet sidecar (``self._uid_to_rows``) — **not** via
        ``HAS_CHUNK`` edges. This mirrors
        :meth:`GraphExpander._expand_doc_hop` and is what makes the expansion
        work on a DOC→ART-only graph (no Chunk nodes present). Each result
        carries the seed score multiplied by ``discount_doc``.
        """
        if not candidates:
            return []
        # Build the per-seed parameter rows. We pass the seed score along so
        # the discount can be applied in Python (Cypher can't easily multiply
        # a per-row constant into an aggregated score while preserving the
        # seed grouping for the dedupe logic — cleaner to do it here).
        seed_rows = []
        for row_idx, score in candidates:
            row_idx = int(row_idx)
            uid = self.row_to_uid.get(row_idx)
            if not uid:
                continue
            seed_rows.append(
                {
                    "row_idx": row_idx,
                    "art_id": f"ART:{uid}",
                    "seed_score": float(score),
                }
            )
        if not seed_rows:
            return []

        # Walk to neighbour DOCs' article doc_uids in Cypher. The seed's own
        # DOC is excluded from the neighbour set (a doc doesn't expand into
        # itself). nbArt.id is "ART:{doc_uid}" — strip the prefix in Python.
        cypher = f"""
        UNWIND $seeds AS seed
        MATCH (art:Article {{id: seed.art_id}})
        MATCH (doc:Document)-[:HAS_ARTICLE]->(art)
        // canonical out-edges: doc -> neighbour doc
        OPTIONAL MATCH (doc)-[:{_DOC_RELS_CY}]->(nbDoc:Document)
        WITH seed, doc, collect(DISTINCT nbDoc) AS outNbs
        // reverse in-edges: neighbour doc -> doc
        OPTIONAL MATCH (inDoc:Document)-[:{_DOC_RELS_CY}]->(doc)
        WITH seed, doc, outNbs, collect(DISTINCT inDoc) AS inNbs
        // Drop nulls (from OPTIONAL MATCH) + the seed's own doc, then UNWIND.
        // NB: a bare WHERE directly after UNWIND is rejected by the server's
        // Cypher 25 parser, so we filter via a list comprehension instead.
        WITH seed, [n IN (outNbs + inNbs) WHERE n IS NOT NULL AND n <> doc] AS nbs
        UNWIND nbs AS nbDoc
        // neighbour doc -> its articles (doc_uid lives on the Article node id)
        MATCH (nbDoc)-[:HAS_ARTICLE]->(nbArt:Article)
        RETURN DISTINCT seed.row_idx     AS seed_row,
                       seed.seed_score  AS seed_score,
                       nbArt.id         AS nb_art_id
        """
        prefix = "ART:"
        out: List[Dict[str, Any]] = []
        with self.driver.session() as sess:
            result = sess.run(cypher, seeds=seed_rows)
            for rec in result:
                nb_art_id = str(rec["nb_art_id"])
                uid = nb_art_id[len(prefix):] if nb_art_id.startswith(prefix) else nb_art_id
                # translate neighbour article doc_uid -> row_idx(es) via parquet
                for nb_row in self._uid_to_rows.get(uid, ()):
                    out.append(
                        {
                            "row_idx": int(nb_row),
                            "score": float(rec["seed_score"]) * self.discount_doc,
                        }
                    )
            return out

    # ------------------------------------------------------------------ #
    # 1.5-hop concept co-mention
    # ------------------------------------------------------------------ #
    def _expand_concept_comention(
        self,
        candidates: Sequence[Tuple[int, float]],
    ) -> List[Dict[str, Any]]:
        """Sibling chunks sharing a MENTIONED concept with a seed chunk.

        seed chunk -> (MENTIONS) -> CONCEPT -> (MENTIONS, reverse) -> sibling
        chunk. Each sibling carries the seed score × ``discount_concept``.
        Returns no rows if the graph has no CHUNK/CONCEPT nodes (graceful
        degradation — the MATCH simply finds nothing).
        """
        if not candidates:
            return []
        seed_rows = []
        for row_idx, score in candidates:
            row_idx = int(row_idx)
            # We don't have chunk_id here (only row_idx -> doc_uid); the
            # seed chunk is located by its rowidx property on :Chunk nodes.
            seed_rows.append({"row_idx": row_idx, "seed_score": float(score)})
        if not seed_rows:
            return []

        cypher = """
        UNWIND $seeds AS seed
        MATCH (chunk:Chunk {rowidx: seed.row_idx})
        MATCH (chunk)-[:MENTIONS]->(concept:Concept)
        // sibling chunks that also MENTION the same concept (incoming MENTIONS)
        MATCH (sib:Chunk)-[:MENTIONS]->(concept)
        WHERE sib.rowidx IS NOT NULL AND sib <> chunk
        RETURN DISTINCT seed.row_idx    AS seed_row,
                       seed.seed_score  AS seed_score,
                       toInteger(sib.rowidx) AS row_idx
        """
        with self.driver.session() as sess:
            result = sess.run(cypher, seeds=seed_rows)
            out: List[Dict[str, Any]] = []
            for rec in result:
                out.append(
                    {
                        "row_idx": int(rec["row_idx"]),
                        "score": float(rec["seed_score"]) * self.discount_concept,
                    }
                )
            return out

    # ------------------------------------------------------------------ #
    # Dedupe helper — identical semantics to GraphExpander._add_expanded
    # ------------------------------------------------------------------ #
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
        prev = next((r for r in out if r["row_idx"] == row_idx), None)
        if prev is not None and score > prev["score"]:
            prev["score"] = score
            prev["source"] = source
            seen[row_idx] = source

    # ------------------------------------------------------------------ #
    # Prompt-context builder — duck-typed to match build_graph_context()
    # ------------------------------------------------------------------ #
    def build_graph_context(self, doc_nodes: Sequence[str]) -> str:
        """Build a human-readable graph-neighbour context string for a prompt.

        ``doc_nodes`` is a list of ``DOC:{doc_id}`` ids. Returns a
        newline-joined block of the form::

            DETAILS: <ten>
            AMENDED_BY: <ten>
            ...

        or the placeholder ``"(Không có quan hệ chéo đáng chú ý)"`` when no
        expansion relation is incident on any of the given docs. Identical
        behaviour to ``GraphExpander.build_graph_context``.
        """
        if not doc_nodes:
            return "(Không có quan hệ chéo đáng chú ý)"

        # Collect (relation, neighbour_ten, is_reverse) rows.
        cypher = f"""
        UNWIND $doc_ids AS did
        MATCH (doc:Document {{id: did}})
        // canonical out-edges → forward label
        OPTIONAL MATCH (doc)-[r:{_DOC_RELS_CY}]->(nb:Document)
        WITH did, doc, collect(DISTINCT [type(r), coalesce(nb.ten, nb.id)]) AS outRows
        // reverse in-edges → reverse label
        OPTIONAL MATCH (in:Document)-[r2:{_DOC_RELS_CY}]->(doc)
        WITH outRows, collect(DISTINCT [type(r2), coalesce(in.ten, in.id)]) AS inRows
        UNWIND (outRows + inRows) AS row
        WITH row[0] AS rel, row[1] AS other_ten
        WHERE rel IS NOT NULL
        RETURN rel, other_ten
        """
        # We can't tell out vs in edge direction from a single RETURN, so run
        # two queries (forward + reverse) to label them correctly. Cheap and
        # explicit.
        lines: List[Tuple[str, str]] = []

        # Forward edges.
        fwd_cypher = f"""
        UNWIND $doc_ids AS did
        MATCH (doc:Document {{id: did}})-[r:{_DOC_RELS_CY}]->(nb:Document)
        RETURN DISTINCT type(r) AS rel, coalesce(nb.ten, nb.id) AS other_ten
        """
        with self.driver.session() as sess:
            for rec in sess.run(fwd_cypher, doc_ids=list(doc_nodes)):
                rel = rec["rel"]
                fwd, _ = _REL_LABELS.get(rel, (rel, rel + "_BY"))
                lines.append((fwd, str(rec["other_ten"])))

        # Reverse edges.
        rev_cypher = f"""
        UNWIND $doc_ids AS did
        MATCH (in:Document)-[r:{_DOC_RELS_CY}]->(doc:Document {{id: did}})
        RETURN DISTINCT type(r) AS rel, coalesce(in.ten, in.id) AS other_ten
        """
        with self.driver.session() as sess:
            for rec in sess.run(rev_cypher, doc_ids=list(doc_nodes)):
                rel = rec["rel"]
                _, rev = _REL_LABELS.get(rel, (rel, rel + "_BY"))
                lines.append((rev, str(rec["other_ten"])))

        # Dedupe identical (label, ten) lines preserving order.
        deduped: List[str] = []
        seen = set()
        for label, ten in lines:
            ln = f"{label}: {ten}"
            if ln not in seen:
                seen.add(ln)
                deduped.append(ln)
        if not deduped:
            return "(Không có quan hệ chéo đáng chú ý)"
        return "\n".join(deduped)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def close(self) -> None:
        """Close the underlying Neo4j driver."""
        try:
            self.driver.close()
        except Exception:
            pass

    def __enter__(self) -> "Neo4jGraphExpander":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
