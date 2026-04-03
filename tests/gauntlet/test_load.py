"""Load & stress testing — throughput under realistic concurrent pressure.

Unlike test_performance.py (microbenchmarks on pure functions) and
test_concurrent.py (thread-safety of data structures), these tests
hammer the LIVE system with realistic concurrent operations.

Requires: Live FalkorDB + Qdrant (skips if unavailable).
Run: pytest tests/gauntlet/test_load.py -m "slow and integration" -v
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import pytest

# ── Service detection ────────────────────────────────────────


def _services_available() -> bool:
    """Check if FalkorDB and Qdrant are reachable."""
    try:
        import httpx
        import redis

        fhost = os.getenv("FALKORDB_HOST", "localhost")
        fport = int(os.getenv("FALKORDB_PORT", "6379"))
        r = redis.Redis(host=fhost, port=fport, socket_connect_timeout=3)
        if not r.ping():
            return False

        qhost = os.getenv("QDRANT_HOST", "localhost")
        qport = os.getenv("QDRANT_PORT", "6333")
        resp = httpx.get(f"http://{qhost}:{qport}/healthz", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


@pytest.fixture(scope="module")
def live_service():
    """Provide live MemoryService or skip."""
    if not _services_available():
        pytest.skip("Live services not available — skipping load tests")

    import httpx

    # Use remote embedding server if available (avoids local model load)
    api_url = os.getenv("EMBEDDING_API_URL")
    if not api_url:
        try:
            resp = httpx.get("http://localhost:8001/health", timeout=3)
            if resp.status_code == 200:
                os.environ["EMBEDDING_API_URL"] = "http://localhost:8001"
        except Exception:  # noqa: S110
            pass

    from claude_memory.embedding import EmbeddingService
    from claude_memory.tools import MemoryService

    embedder = EmbeddingService()
    return MemoryService(embedding_service=embedder)


PROJECT_ID = "load-test"
PREFIX = "LOAD_TEST_"


# ── Load Tests ───────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.integration
class TestLoadScenarios:
    """Throughput under realistic concurrent load."""

    @pytest.mark.asyncio
    async def test_search_throughput(self, live_service: Any) -> None:
        """20 sequential searches — measures per-query throughput, <60s total."""
        queries = [f"search query {i} about knowledge graph entities" for i in range(20)]

        start = time.monotonic()
        result_counts: list[int] = []

        for q in queries:
            results = await live_service.search(q, limit=5, project_id=PROJECT_ID)
            result_counts.append(len(results))

        elapsed = time.monotonic() - start

        assert len(result_counts) == 20, f"Expected 20, got {len(result_counts)}"
        assert elapsed < 60, f"20 searches took {elapsed:.1f}s (limit: 60s)"

    @pytest.mark.asyncio
    async def test_bulk_entity_creation(self, live_service: Any) -> None:
        """Create 20 entities in rapid succession — no failures, <60s."""
        from claude_memory.schema import EntityCreateParams

        start = time.monotonic()
        entity_ids: list[str] = []

        try:
            for i in range(20):
                params = EntityCreateParams(
                    name=f"{PREFIX}Bulk_{i}",
                    node_type="Concept",
                    project_id=PROJECT_ID,
                )
                receipt = await live_service.create_entity(params)
                assert receipt.id, f"Entity {i} returned no ID"
                entity_ids.append(receipt.id)

            elapsed = time.monotonic() - start
            assert len(entity_ids) == 20
            assert elapsed < 60, f"20 entity creations took {elapsed:.1f}s (limit: 60s)"
        finally:
            # Cleanup
            live_service.repo.execute_cypher(
                "MATCH (n) WHERE n.name STARTS WITH $prefix DETACH DELETE n",
                {"prefix": PREFIX},
            )

    @pytest.mark.asyncio
    async def test_interleaved_read_write(self, live_service: Any) -> None:
        """20 writes + 20 reads interleaved — no corruption, <60s."""
        from claude_memory.schema import EntityCreateParams

        start = time.monotonic()
        created_ids: list[str] = []

        try:
            for i in range(20):
                # Write
                params = EntityCreateParams(
                    name=f"{PREFIX}Interleave_{i}",
                    node_type="Concept",
                    project_id=PROJECT_ID,
                )
                receipt = await live_service.create_entity(params)
                assert receipt.id
                created_ids.append(receipt.id)

                # Read back
                node = live_service.repo.get_node(receipt.id)
                assert node is not None, f"Entity {i} not immediately readable"
                assert node["name"] == f"{PREFIX}Interleave_{i}"

            elapsed = time.monotonic() - start
            assert len(created_ids) == 20
            assert elapsed < 60, f"20 interleaved R/W took {elapsed:.1f}s (limit: 60s)"
        finally:
            live_service.repo.execute_cypher(
                "MATCH (n) WHERE n.name STARTS WITH $prefix DETACH DELETE n",
                {"prefix": PREFIX},
            )

    @pytest.mark.asyncio
    async def test_concurrent_graph_health(self, live_service: Any) -> None:
        """20 simultaneous graph_health calls — no crashes, <30s."""
        start = time.monotonic()

        results = await asyncio.gather(*[live_service.get_graph_health() for _ in range(20)])
        elapsed = time.monotonic() - start

        assert len(results) == 20
        for r in results:
            assert "total_nodes" in r
        assert elapsed < 30, f"20 concurrent health checks took {elapsed:.1f}s (limit: 30s)"
