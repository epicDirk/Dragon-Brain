"""Tests for find_similar_by_id (Semantic Radar Layer 1).

Covers: RecommendQuery construction, self-exclusion, threshold forwarding,
empty results, and correct result mapping.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ─── Test Constants ─────────────────────────────────────────────────

ENTITY_ID = "entity-001"
ENTITY_ID_2 = "entity-002"
ENTITY_ID_3 = "entity-003"
COLLECTION_NAME = "memory_embeddings"
SCORE_THRESHOLD = 0.6

# ─── Module Import ──────────────────────────────────────────────────

with patch("claude_memory.vector_store.AsyncQdrantClient"):
    from claude_memory.vector_store import QdrantVectorStore


# ─── Fixtures ───────────────────────────────────────────────────────


@pytest.fixture()
def mock_qdrant_client() -> AsyncMock:
    """Creates a fully mocked AsyncQdrantClient."""
    client = AsyncMock()

    # get_collections: collection already exists
    collection_info = MagicMock()
    collection_info.name = COLLECTION_NAME
    collections_response = MagicMock()
    collections_response.collections = [collection_info]
    client.get_collections.return_value = collections_response

    # query_points: default empty response
    query_response = MagicMock()
    query_response.points = []
    client.query_points.return_value = query_response

    return client


@pytest.fixture()
def store(mock_qdrant_client: AsyncMock) -> QdrantVectorStore:
    """Creates a QdrantVectorStore with a mocked client."""
    with patch(
        "claude_memory.vector_store.AsyncQdrantClient",
        return_value=mock_qdrant_client,
    ):
        s = QdrantVectorStore(host="localhost", port=6333)
    return s


# ─── Evil Tests ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_evil1_recommend_query_structure(
    store: QdrantVectorStore, mock_qdrant_client: AsyncMock
) -> None:
    """Evil: verify query_points is called with RecommendQuery wrapping RecommendInput."""
    from qdrant_client.http.models import RecommendInput, RecommendQuery

    await store.find_similar_by_id(ENTITY_ID)

    mock_qdrant_client.query_points.assert_awaited_once()
    call_kwargs = mock_qdrant_client.query_points.call_args[1]

    query = call_kwargs["query"]
    assert isinstance(query, RecommendQuery)
    assert isinstance(query.recommend, RecommendInput)
    assert query.recommend.positive == [ENTITY_ID]


@pytest.mark.asyncio
async def test_evil2_self_exclusion_always_applied(
    store: QdrantVectorStore, mock_qdrant_client: AsyncMock
) -> None:
    """Evil: entity_id is always excluded even when no exclude_ids provided."""
    from qdrant_client.http.models import HasIdCondition

    await store.find_similar_by_id(ENTITY_ID)

    call_kwargs = mock_qdrant_client.query_points.call_args[1]
    q_filter = call_kwargs["query_filter"]

    assert q_filter is not None
    assert q_filter.must_not is not None
    assert len(q_filter.must_not) == 1
    condition = q_filter.must_not[0]
    assert isinstance(condition, HasIdCondition)
    assert ENTITY_ID in condition.has_id


@pytest.mark.asyncio
async def test_evil3_score_threshold_passed(
    store: QdrantVectorStore, mock_qdrant_client: AsyncMock
) -> None:
    """Evil: score_threshold kwarg is forwarded to query_points."""
    custom_threshold = 0.85
    await store.find_similar_by_id(ENTITY_ID, threshold=custom_threshold)

    call_kwargs = mock_qdrant_client.query_points.call_args[1]
    assert call_kwargs["score_threshold"] == custom_threshold


# ─── Sad Tests ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_sad1_no_results_returns_empty(
    store: QdrantVectorStore, mock_qdrant_client: AsyncMock
) -> None:
    """Sad: no similar entities found → returns empty list."""
    mock_qdrant_client.query_points.return_value.points = []

    result = await store.find_similar_by_id(ENTITY_ID)

    assert result == []


# ─── Happy Tests ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_happy_results_mapped_correctly(
    store: QdrantVectorStore, mock_qdrant_client: AsyncMock
) -> None:
    """Happy: multiple scored points are mapped to _id/_score/payload dicts."""
    point1 = MagicMock()
    point1.id = ENTITY_ID_2
    point1.score = 0.92
    point1.payload = {"name": "Alpha", "node_type": "Concept"}

    point2 = MagicMock()
    point2.id = ENTITY_ID_3
    point2.score = 0.78
    point2.payload = {"name": "Beta", "node_type": "Project"}

    mock_qdrant_client.query_points.return_value.points = [point1, point2]

    results = await store.find_similar_by_id(ENTITY_ID, limit=5)

    assert len(results) == 2
    assert results[0] == {
        "_id": ENTITY_ID_2,
        "_score": 0.92,
        "payload": {"name": "Alpha", "node_type": "Concept"},
    }
    assert results[1] == {
        "_id": ENTITY_ID_3,
        "_score": 0.78,
        "payload": {"name": "Beta", "node_type": "Project"},
    }
