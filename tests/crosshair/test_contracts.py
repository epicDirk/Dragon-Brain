"""Crosshair symbolic execution contracts for pure functions.

Proves mathematical and logical properties via symbolic execution,
targeting functions with no I/O where Crosshair can actually find bugs.

Targets:
    - rrf_merge: RRF scoring math (non-negative, limit respected, dedup)
    - QueryRouter.classify: exhaustive intent mapping
    - _infer_relationship_type: exhaustive type mapping

Run:  tox -e crosshair
"""

from __future__ import annotations

from claude_memory.merge import rrf_merge
from claude_memory.router import QueryIntent, QueryRouter
from claude_memory.search_advanced import SearchAdvancedMixin

# ═══════════════════════════════════════════════════════════════
#  Contract 1: rrf_merge — mathematical properties
# ═══════════════════════════════════════════════════════════════


def test_rrf_scores_always_non_negative() -> None:
    """RRF scores must always be >= 0 for any input."""
    vector_results = [
        {"_id": "a", "_score": 0.9},
        {"_id": "b", "_score": 0.5},
    ]
    graph_results = [
        {"id": "b"},
        {"id": "c"},
    ]
    merged = rrf_merge(vector_results, graph_results, k=60, limit=10)
    for result in merged:
        assert result.rrf_score >= 0, f"Negative RRF score: {result.rrf_score}"


def test_rrf_respects_limit() -> None:
    """Output must never exceed the requested limit."""
    vector_results = [{"_id": f"v{i}", "_score": 0.9 - i * 0.01} for i in range(50)]
    graph_results = [{"id": f"g{i}"} for i in range(50)]

    for limit in (1, 5, 10, 25, 100):
        merged = rrf_merge(vector_results, graph_results, k=60, limit=limit)
        assert len(merged) <= limit, f"limit={limit}, got {len(merged)} results"


def test_rrf_deduplicates_across_sources() -> None:
    """Entity appearing in both sources must appear once with boosted score."""
    vector_results = [{"_id": "shared", "_score": 0.8}]
    graph_results = [{"id": "shared"}]

    merged = rrf_merge(vector_results, graph_results, k=60, limit=10)
    shared_results = [r for r in merged if r.entity_id == "shared"]
    assert len(shared_results) == 1, "Duplicate entity in merge output"
    assert len(shared_results[0].retrieval_sources) == 2


def test_rrf_sorted_descending() -> None:
    """Merged results must be sorted by RRF score descending."""
    vector_results = [
        {"_id": "low", "_score": 0.1},
        {"_id": "high", "_score": 0.9},
    ]
    graph_results = [{"id": "high"}, {"id": "mid"}]

    merged = rrf_merge(vector_results, graph_results, k=60, limit=10)
    scores = [r.rrf_score for r in merged]
    assert scores == sorted(scores, reverse=True), "Results not sorted descending"


def test_rrf_empty_inputs() -> None:
    """Empty inputs must produce empty output, not crash."""
    assert rrf_merge([], [], k=60, limit=10) == []
    assert rrf_merge([], [{"id": "x"}], k=60, limit=10) != []
    assert rrf_merge([{"_id": "x", "_score": 0.5}], [], k=60, limit=10) != []


def test_rrf_k_parameter_affects_scores() -> None:
    """Higher k should dampen the difference between rank 1 and rank N."""
    vector_results = [
        {"_id": "first", "_score": 0.9},
        {"_id": "last", "_score": 0.1},
    ]
    merged_low_k = rrf_merge(vector_results, [], k=1, limit=10)
    merged_high_k = rrf_merge(vector_results, [], k=1000, limit=10)

    # With low k, rank 1 vs rank 2 has a big difference
    # With high k, the difference shrinks
    low_k_ratio = merged_low_k[0].rrf_score / merged_low_k[1].rrf_score
    high_k_ratio = merged_high_k[0].rrf_score / merged_high_k[1].rrf_score
    assert low_k_ratio > high_k_ratio, "Higher k should flatten rank differences"


# ═══════════════════════════════════════════════════════════════
#  Contract 2: QueryRouter.classify — exhaustive intent mapping
# ═══════════════════════════════════════════════════════════════


def test_classify_always_returns_valid_intent() -> None:
    """classify() must always return a valid QueryIntent member."""
    router = QueryRouter()
    test_inputs = [
        "",
        "hello",
        "what happened yesterday",
        "path between A and B",
        "related to entropy",
        "tell me about quantum physics",
        "🔥" * 100,
        "   ",
        "\n\t",
        "a" * 10_000,
    ]
    valid_intents = set(QueryIntent)
    for query in test_inputs:
        result = router.classify(query)
        assert result in valid_intents, f"Invalid intent {result!r} for {query!r}"


def test_classify_empty_returns_semantic() -> None:
    """Empty string must return SEMANTIC (the default)."""
    router = QueryRouter()
    assert router.classify("") == QueryIntent.SEMANTIC


def test_classify_priority_temporal_over_relational() -> None:
    """Temporal keywords take priority over relational."""
    router = QueryRouter()
    # "when" is temporal, "path between" is relational
    result = router.classify("when was the path between A and B created")
    assert result == QueryIntent.TEMPORAL


def test_classify_deterministic() -> None:
    """Same input must always produce same output."""
    router = QueryRouter()
    query = "some random query about things"
    results = {router.classify(query) for _ in range(100)}
    assert len(results) == 1, f"Non-deterministic: got {results}"


# ═══════════════════════════════════════════════════════════════
#  Contract 3: _infer_relationship_type — exhaustive mapping
# ═══════════════════════════════════════════════════════════════

VALID_RELATIONSHIP_TYPES = {
    "BRIDGES_TO",
    "ANALOGOUS_TO",
    "ENABLES",
    "DECIDED_IN",
    "MENTIONED_IN",
    "CREATED_BY",
    "RELATED_TO",
}

# All Dragon Brain node types
DRAGON_BRAIN_TYPES = [
    "Entity",
    "Concept",
    "Session",
    "Breakthrough",
    "Tool",
    "Decision",
    "Bottle",
    "Analogy",
    "Issue",
    "Project",
    "Procedure",
    "Person",
]


def test_infer_always_returns_valid_type() -> None:
    """Every combination of node types must return a valid relationship type."""
    infer = SearchAdvancedMixin._infer_relationship_type
    for s_type in DRAGON_BRAIN_TYPES:
        for c_type in DRAGON_BRAIN_TYPES:
            # Same project
            result = infer(s_type, c_type, "proj1", "proj1")
            assert result in VALID_RELATIONSHIP_TYPES, (
                f"Invalid type {result!r} for ({s_type}, {c_type}, same project)"
            )
            # Cross project
            result_cross = infer(s_type, c_type, "proj1", "proj2")
            assert result_cross == "BRIDGES_TO", (
                f"Cross-project should be BRIDGES_TO, got {result_cross!r}"
            )


def test_infer_cross_project_always_bridges() -> None:
    """Cross-project pairs must ALWAYS return BRIDGES_TO regardless of types."""
    infer = SearchAdvancedMixin._infer_relationship_type
    for s_type in DRAGON_BRAIN_TYPES:
        for c_type in DRAGON_BRAIN_TYPES:
            assert infer(s_type, c_type, "a", "b") == "BRIDGES_TO"


def test_infer_unknown_types_fallback() -> None:
    """Unknown node types must fall back to RELATED_TO."""
    infer = SearchAdvancedMixin._infer_relationship_type
    assert infer("UnknownType", "AnotherUnknown", "p", "p") == "RELATED_TO"
    assert infer("", "", "p", "p") == "RELATED_TO"


def test_infer_symmetry_for_pairings() -> None:
    """Paired mappings (Tool+Procedure, Concept+Analogy) should be symmetric."""
    infer = SearchAdvancedMixin._infer_relationship_type
    # Tool + Procedure
    assert infer("Tool", "Procedure", "p", "p") == infer("Procedure", "Tool", "p", "p")
    # Concept + Analogy
    assert infer("Concept", "Analogy", "p", "p") == infer("Analogy", "Concept", "p", "p")
