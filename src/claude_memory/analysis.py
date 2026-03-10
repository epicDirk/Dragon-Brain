"""Analysis & maintenance operations for the Exocortex memory system.

Provides graph analytics, gap detection, stale entity management,
and diagnostic queries. Maintenance ops (archive, prune, consolidate)
live in analysis_maintenance.py.
"""

import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal

from claude_memory.analysis_maintenance import AnalysisMaintenanceMixin
from claude_memory.graph_algorithms import compute_louvain, compute_pagerank

if TYPE_CHECKING:  # pragma: no cover
    from .interfaces import Embedder, VectorStore
    from .ontology import OntologyManager
    from .repository import MemoryRepository
    from .schema import GapDetectionParams

logger = logging.getLogger(__name__)


class AnalysisMixin(AnalysisMaintenanceMixin):
    """Graph health / gaps / stale / diagnostics — mixed into MemoryService.

    Inherits archive, prune, consolidate, and create_memory_type
    from :class:`AnalysisMaintenanceMixin`.
    """

    repo: "MemoryRepository"
    embedder: "Embedder"
    vector_store: "VectorStore"
    ontology: "OntologyManager"

    async def get_graph_health(self) -> dict[str, Any]:
        """Compute graph health metrics including community count.

        Merges repository-level stats (nodes, edges, density, orphans, avg_degree)
        with clustering-based community count.
        """
        from .clustering import ClusteringService  # noqa: PLC0415

        health = self.repo.get_graph_health()

        # Compute community count via clustering
        community_count = 0
        cs = ClusteringService()
        if health["total_nodes"] >= cs.min_samples:
            try:
                nodes = self.repo.get_all_nodes(limit=2000)
                clusters = cs.cluster_nodes(nodes)
                community_count = len(clusters)
            except Exception:
                logger.warning("Clustering failed during health check — community_count=0")

        health["community_count"] = community_count
        return health  # type: ignore[no-any-return]

    async def list_orphans(self, limit: int = 50) -> list[dict[str, Any]]:
        """List graph nodes with zero relationships (orphans).

        Delegates to repository-level Cypher query. Returns id, name,
        node_type, project_id, focus, labels, and created_at.
        """
        return self.repo.list_orphans(limit=limit)  # type: ignore[no-any-return]

    async def system_diagnostics(self) -> dict[str, Any]:
        """E-5: Unified system diagnostics — graph, vector, and split-brain."""
        result: dict[str, Any] = {}

        # Graph section
        result["graph"] = self.repo.get_graph_health()

        # Vector section
        vector_section: dict[str, Any] = {}
        try:
            vector_section["count"] = await self.vector_store.count()
            vector_section["error"] = None
        except Exception as exc:
            vector_section["count"] = None
            vector_section["error"] = str(exc)
        result["vector"] = vector_section

        # Split-brain detection
        split: dict[str, Any] = {}
        if vector_section["error"] is not None:
            split["status"] = "unavailable"
            split["graph_only_count"] = 0
            split["graph_only_ids"] = []
        else:
            try:
                graph_ids = set(self.repo.get_all_node_ids())
                vector_ids = set(await self.vector_store.list_ids())
                graph_only = graph_ids - vector_ids
                split["status"] = "ok" if not graph_only else "drift"
                split["graph_only_count"] = len(graph_only)
                split["graph_only_ids"] = sorted(graph_only)
            except Exception as exc:
                split["status"] = "error"
                split["graph_only_count"] = 0
                split["graph_only_ids"] = []
                logger.error("split_brain_check_failed: %s", exc)
        result["split_brain"] = split

        return result

    async def reconnect(
        self,
        project_id: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """E-4: Session reconnect — structured briefing for a returning agent."""
        now = datetime.now(UTC)
        since = (now - timedelta(days=1)).isoformat()
        until = now.isoformat()

        recent = self.repo.query_timeline(
            start=since,
            end=until,
            limit=limit,
            project_id=project_id,
        )

        health = self.repo.get_graph_health()

        return {
            "recent_entities": recent,
            "health": health,
            "window": {"start": since, "end": until},
        }

    async def detect_structural_gaps(self, params: "GapDetectionParams") -> list[dict[str, Any]]:
        """Detect structural gaps between knowledge clusters.

        Runs clustering, computes cross-cluster connectivity vs similarity,
        and generates research prompts for each identified gap.
        """
        from .clustering import ClusteringService, detect_gaps  # noqa: PLC0415

        # 1. Cluster all nodes
        nodes = self.repo.get_all_nodes(limit=2000)
        cs = ClusteringService()
        clusters = cs.cluster_nodes(nodes)

        if len(clusters) < 2:  # noqa: PLR2004
            return []

        # 2. Get all edges for cross-cluster connectivity
        edges = self.repo.get_all_edges()

        # 3. Detect gaps
        gaps = detect_gaps(
            clusters,
            edges,
            min_similarity=params.min_similarity,
            max_edges=params.max_edges,
        )

        # 4. Build results with research prompts
        cluster_map = {c.id: c for c in clusters}
        return [self._build_gap_result(gap, cluster_map) for gap in gaps[: params.limit]]

    @staticmethod
    def _build_gap_result(gap: Any, cluster_map: dict[str, Any]) -> dict[str, Any]:
        """Convert a single gap into a result dict with research prompt."""
        ca = cluster_map.get(gap.cluster_a_id)
        cb = cluster_map.get(gap.cluster_b_id)
        a_names = [n.get("name", "?") for n in (ca.nodes[:3] if ca else [])]
        b_names = [n.get("name", "?") for n in (cb.nodes[:3] if cb else [])]

        prompt = (
            f"These knowledge areas seem related (similarity: {gap.similarity:.0%}) "
            f"but are poorly connected ({gap.edge_count} edges).\n"
            f"Cluster A: {', '.join(a_names)}\n"
            f"Cluster B: {', '.join(b_names)}\n"
            f"Consider: What connections exist between these topics? "
            f"Are there shared concepts, dependencies, or patterns?"
        )

        return {
            "cluster_a_id": gap.cluster_a_id,
            "cluster_b_id": gap.cluster_b_id,
            "similarity": gap.similarity,
            "edge_count": gap.edge_count,
            "suggested_bridges": gap.suggested_bridges,
            "research_prompt": prompt,
        }

    async def analyze_graph(
        self, algorithm: Literal["pagerank", "louvain"] = "pagerank"
    ) -> list[dict[str, Any]]:
        """Run graph algorithms to find key entities or communities.

        Computes algorithms in Python using adjacency data fetched via Cypher.
        FalkorDB does not provide built-in algo.pageRank/algo.louvain procedures.

        Args:
            algorithm: 'pagerank' for influence, 'louvain' for communities.
        """
        # Fetch all nodes
        node_res = self.repo.execute_cypher("MATCH (n:Entity) RETURN n")
        if not node_res.result_set:
            return []

        nodes = {row[0].properties["name"]: row[0] for row in node_res.result_set}
        node_names = list(nodes.keys())

        # Fetch all edges
        edge_res = self.repo.execute_cypher(
            "MATCH (a:Entity)-[r]->(b:Entity) RETURN a.name, b.name"
        )
        edges = [(row[0], row[1]) for row in edge_res.result_set] if edge_res.result_set else []

        if algorithm == "pagerank":
            return compute_pagerank(nodes, node_names, edges)
        elif algorithm == "louvain":
            return compute_louvain(nodes, node_names, edges)
        return []

    async def get_stale_entities(self, days: int = 30) -> list[dict[str, Any]]:
        """Identify entities not modified/accessed in N days."""

        cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
        query = """
        MATCH (n:Entity)
        WHERE n.updated_at < $cutoff AND (n.status IS NULL OR n.status <> 'archived')
        RETURN n
        ORDER BY n.updated_at ASC
        LIMIT 20
        """
        res = self.repo.execute_cypher(query, {"cutoff": cutoff})
        entities = [row[0].properties for row in res.result_set]
        for e in entities:
            e.pop("embedding", None)
        return entities
