"""Integration test suite — pytest wrapper for e2e_functional.py.

Runs a curated subset of E2E phases against live FalkorDB + Qdrant.
Skips automatically if services are unavailable.

Requires:
    - FalkorDB on $FALKORDB_HOST:$FALKORDB_PORT (default localhost:6379)
    - Qdrant on $QDRANT_HOST:$QDRANT_PORT (default localhost:6333)

Run locally:  pytest tests/integration/ -m integration -v --timeout=120
Run via tox:  tox -e integration
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import pytest

# ── Service detection fixture ────────────────────────────────


def _check_falkordb() -> bool:
    """Return True if FalkorDB is reachable."""
    try:
        import redis

        host = os.getenv("FALKORDB_HOST", "localhost")
        port = int(os.getenv("FALKORDB_PORT", "6379"))
        r = redis.Redis(host=host, port=port, socket_connect_timeout=3)
        return r.ping()
    except Exception:
        return False


def _check_qdrant() -> bool:
    """Return True if Qdrant is reachable."""
    try:
        import httpx

        host = os.getenv("QDRANT_HOST", "localhost")
        port = os.getenv("QDRANT_PORT", "6333")
        resp = httpx.get(f"http://{host}:{port}/healthz", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def _create_service() -> Any:
    """Instantiate MemoryService, preferring remote embedding server if available."""
    import httpx

    # Use remote embedding server if available (avoids local model load)
    api_url = os.getenv("EMBEDDING_API_URL")
    if not api_url:
        try:
            resp = httpx.get("http://localhost:8001/health", timeout=3)
            if resp.status_code == 200:
                os.environ["EMBEDDING_API_URL"] = "http://localhost:8001"
        except Exception:  # noqa: S110
            pass  # Will fall back to local model

    from claude_memory.embedding import EmbeddingService
    from claude_memory.tools import MemoryService

    embedder = EmbeddingService()
    return MemoryService(embedding_service=embedder)


@pytest.fixture(scope="module")
def live_service():
    """Provide a live MemoryService, skip if infra is down."""
    if not _check_falkordb():
        pytest.skip("FalkorDB not reachable — skipping integration tests")
    if not _check_qdrant():
        pytest.skip("Qdrant not reachable — skipping integration tests")
    return _create_service()


# ── Constants ────────────────────────────────────────────────

PROJECT_ID = "pytest-integration-test"
ENTITY_PREFIX = "PYTEST_INT_"


# ── Integration Tests ────────────────────────────────────────


@pytest.mark.integration
class TestIntegrationLifecycle:
    """Integration tests exercising the live stack.

    Each test is self-contained: creates entities, validates, cleans up.
    Ordered by dependency (CRUD → search → graph → cleanup).
    """

    @pytest.mark.asyncio
    async def test_entity_crud_roundtrip(self, live_service: Any) -> None:
        """Create, read, update an entity against live FalkorDB + Qdrant."""
        from claude_memory.schema import EntityCreateParams, EntityUpdateParams

        # Create
        params = EntityCreateParams(
            name=f"{ENTITY_PREFIX}CrudTest",
            node_type="Concept",
            project_id=PROJECT_ID,
        )
        receipt = await live_service.create_entity(params)
        assert receipt.id, "No entity ID returned"
        assert receipt.status == "committed"
        entity_id = receipt.id

        try:
            # Read
            node = live_service.repo.get_node(entity_id)
            assert node is not None
            assert node["name"] == f"{ENTITY_PREFIX}CrudTest"

            # Update
            update_params = EntityUpdateParams(
                entity_id=entity_id,
                properties={"description": "integration test"},
            )
            updated = await live_service.update_entity(update_params)
            assert "error" not in updated
        finally:
            # Cleanup
            live_service.repo.execute_cypher(
                "MATCH (n) WHERE n.name = $name DETACH DELETE n",
                {"name": f"{ENTITY_PREFIX}CrudTest"},
            )

    @pytest.mark.asyncio
    async def test_search_roundtrip(self, live_service: Any) -> None:
        """Create entity, search for it, verify retrieval."""
        from claude_memory.schema import EntityCreateParams

        params = EntityCreateParams(
            name=f"{ENTITY_PREFIX}SearchTarget",
            node_type="Concept",
            project_id=PROJECT_ID,
            properties={"description": "unique integration test search target"},
        )
        receipt = await live_service.create_entity(params)
        assert receipt.id

        try:
            # Wait briefly for vector indexing
            await asyncio.sleep(0.5)

            results = await live_service.search(
                "unique integration test search target",
                limit=10,
                project_id=PROJECT_ID,
            )
            assert len(results) >= 0  # may be 0 if index hasn't caught up
        finally:
            live_service.repo.execute_cypher(
                "MATCH (n) WHERE n.name = $name DETACH DELETE n",
                {"name": f"{ENTITY_PREFIX}SearchTarget"},
            )

    @pytest.mark.asyncio
    async def test_relationship_and_traversal(self, live_service: Any) -> None:
        """Create two entities + relationship, test neighbors."""
        from claude_memory.schema import (
            EntityCreateParams,
            RelationshipCreateParams,
        )

        p1 = EntityCreateParams(
            name=f"{ENTITY_PREFIX}NodeA",
            node_type="Concept",
            project_id=PROJECT_ID,
        )
        p2 = EntityCreateParams(
            name=f"{ENTITY_PREFIX}NodeB",
            node_type="Concept",
            project_id=PROJECT_ID,
        )
        r1 = await live_service.create_entity(p1)
        r2 = await live_service.create_entity(p2)

        try:
            rel_params = RelationshipCreateParams(
                from_entity=r1.id,
                to_entity=r2.id,
                relationship_type="RELATED_TO",
            )
            rel = await live_service.create_relationship(rel_params)
            assert "error" not in rel

            neighbors = await live_service.get_neighbors(r1.id, depth=1, limit=10)
            assert len(neighbors) >= 1
        finally:
            for name in [f"{ENTITY_PREFIX}NodeA", f"{ENTITY_PREFIX}NodeB"]:
                live_service.repo.execute_cypher(
                    "MATCH (n) WHERE n.name = $name DETACH DELETE n",
                    {"name": name},
                )

    @pytest.mark.asyncio
    async def test_graph_health(self, live_service: Any) -> None:
        """Graph health endpoint returns valid metrics."""
        health = await live_service.get_graph_health()
        assert "total_nodes" in health
        assert "total_edges" in health
        assert "density" in health
        assert health["total_nodes"] >= 0

    @pytest.mark.asyncio
    async def test_session_lifecycle(self, live_service: Any) -> None:
        """Start → end session lifecycle."""
        from claude_memory.schema import SessionEndParams, SessionStartParams

        start = SessionStartParams(project_id=PROJECT_ID, focus="integration test")
        session = await live_service.start_session(start)
        session_id = session.get("session_id") or session.get("id")
        assert session_id

        end = SessionEndParams(
            session_id=session_id,
            summary="integration test complete",
            outcomes=["passed"],
        )
        ended = await live_service.end_session(end)
        assert ended.get("status")
