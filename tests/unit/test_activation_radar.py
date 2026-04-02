"""Tests for detect_weak_connections (Semantic Radar Layer 3).

Covers: empty inputs, bridge detection, questionable edge detection,
perfect overlap, and mixed results.
"""

from unittest.mock import MagicMock

import pytest

from claude_memory.activation import ActivationEngine

# ─── Test Constants ─────────────────────────────────────────────────

SEED_ID = "seed-001"
ENTITY_A = "entity-a"
ENTITY_B = "entity-b"
ENTITY_C = "entity-c"
ENTITY_D = "entity-d"
ENTITY_E = "entity-e"


# ─── Fixtures ───────────────────────────────────────────────────────


@pytest.fixture()
def engine() -> ActivationEngine:
    return ActivationEngine(repo=MagicMock())


# ─── Evil Tests ─────────────────────────────────────────────────────


def test_evil1_empty_inputs(engine: ActivationEngine) -> None:
    """Evil: all inputs empty — both result lists empty."""
    result = engine.detect_weak_connections(
        seed_ids=[],
        activation_map={},
        vector_scores={},
    )
    assert result["bridge_opportunities"] == []
    assert result["questionable_edges"] == []


def test_evil2_similar_but_not_activated(engine: ActivationEngine) -> None:
    """Evil: entities in vector_scores but not activation_map → bridges."""
    result = engine.detect_weak_connections(
        seed_ids=[SEED_ID],
        activation_map={SEED_ID: 1.0},  # only seed, no spread
        vector_scores={
            ENTITY_A: 0.9,
            ENTITY_B: 0.7,
            ENTITY_C: 0.5,
        },
        similarity_threshold=0.3,
    )
    bridges = result["bridge_opportunities"]
    assert len(bridges) == 3
    # Sorted by vector_score descending
    assert bridges[0]["entity_id"] == ENTITY_A
    assert bridges[0]["vector_score"] == 0.9
    assert bridges[2]["entity_id"] == ENTITY_C

    assert result["questionable_edges"] == []


def test_evil3_activated_but_dissimilar(engine: ActivationEngine) -> None:
    """Evil: entities in activation_map but below vector threshold → questionable."""
    result = engine.detect_weak_connections(
        seed_ids=[SEED_ID],
        activation_map={
            SEED_ID: 1.0,
            ENTITY_A: 0.6,
            ENTITY_B: 0.4,
            ENTITY_C: 0.2,
        },
        vector_scores={
            ENTITY_A: 0.1,  # below threshold
            ENTITY_B: 0.05,  # below threshold
            ENTITY_C: 0.0,  # below threshold
        },
        similarity_threshold=0.3,
    )
    questionable = result["questionable_edges"]
    assert len(questionable) == 3
    # Sorted by activation_energy descending
    assert questionable[0]["entity_id"] == ENTITY_A
    assert questionable[0]["activation_energy"] == 0.6
    assert questionable[2]["entity_id"] == ENTITY_C

    assert result["bridge_opportunities"] == []


# ─── Sad Tests ──────────────────────────────────────────────────────


def test_sad1_perfect_overlap(engine: ActivationEngine) -> None:
    """Sad: all similar entities are also activated — no anomalies."""
    result = engine.detect_weak_connections(
        seed_ids=[SEED_ID],
        activation_map={
            SEED_ID: 1.0,
            ENTITY_A: 0.5,
            ENTITY_B: 0.3,
        },
        vector_scores={
            ENTITY_A: 0.8,
            ENTITY_B: 0.6,
        },
        similarity_threshold=0.3,
    )
    assert result["bridge_opportunities"] == []
    assert result["questionable_edges"] == []


# ─── Happy Tests ────────────────────────────────────────────────────


def test_happy_mixed_results(engine: ActivationEngine) -> None:
    """Happy: some overlap, some bridges, some questionable → correct partition."""
    result = engine.detect_weak_connections(
        seed_ids=[SEED_ID],
        activation_map={
            SEED_ID: 1.0,
            ENTITY_A: 0.5,  # activated + similar → overlap (neither list)
            ENTITY_B: 0.4,  # activated but dissimilar → questionable
        },
        vector_scores={
            ENTITY_A: 0.8,  # similar + activated → overlap
            ENTITY_C: 0.7,  # similar but not activated → bridge
            ENTITY_B: 0.1,  # below threshold → not in similar_set
        },
        similarity_threshold=0.3,
    )
    bridges = result["bridge_opportunities"]
    questionable = result["questionable_edges"]

    assert len(bridges) == 1
    assert bridges[0]["entity_id"] == ENTITY_C

    assert len(questionable) == 1
    assert questionable[0]["entity_id"] == ENTITY_B
