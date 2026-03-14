# SPEC: Architectural Invariants — Structural Drift Detection

**Spec ID:** DRIFT-003
**Priority:** P2 — cheapest to build, catches structural violations on every commit
**Depends on:** ADR-007 (Hybrid Search Unification) — shipped
**Location:** `tests/gauntlet/test_invariants.py`

---

## 1. Problem Statement

Golden queries (DRIFT-001) catch behavioral drift. Distribution monitors (DRIFT-002)
catch statistical drift. Architectural invariants catch **structural drift** — violations
of properties that should hold true across ALL versions of the system, regardless of
features added.

These are laws, not tests. They encode the system's design contract.

## 2. What Makes a Good Invariant

An invariant is a property that:
- Holds for **every valid state** of the system, not just current state
- Can be checked **mechanically** without human interpretation
- Violating it means the **architecture shifted**, not just the behavior
- Should **never be loosened** without an ADR — only tightened or replaced

Bad invariant: "search returns at least 5 results" (depends on data)
Good invariant: "score is always between 0.0 and 1.0" (design contract)

## 3. Proposed Invariants

### 3.1 Score Invariants

```python
class TestScoreInvariants:
    """Scores must always be normalized and meaningful."""

    async def test_score_bounded_0_1(self, service, sample_queries):
        """Score must be in [0.0, 1.0] for all results, all queries."""
        for query in sample_queries:
            results = await service.search(query, limit=10)
            for r in results:
                assert 0.0 <= r.score <= 1.0, (
                    f"Score {r.score} out of bounds for query={query!r}, entity={r.name}"
                )

    async def test_vector_score_bounded_when_present(self, service, sample_queries):
        """vector_score, when not None, must be in [0.0, 1.0]."""
        for query in sample_queries:
            results = await service.search(query, limit=10)
            for r in results:
                if r.vector_score is not None:
                    assert 0.0 <= r.vector_score <= 1.0

    async def test_recency_score_bounded(self, service, sample_queries):
        """recency_score must be in [0.0, 1.0]."""
        for query in sample_queries:
            results = await service.search(query, limit=10)
            for r in results:
                assert 0.0 <= r.recency_score <= 1.0

    async def test_distance_complement_of_score(self, service, sample_queries):
        """distance must equal 1.0 - score (within float tolerance)."""
        for query in sample_queries:
            results = await service.search(query, limit=10)
            for r in results:
                if r.retrieval_strategy == "semantic":
                    assert abs(r.distance - (1.0 - r.score)) < 1e-6
```

### 3.2 Schema Invariants

```python
class TestSchemaInvariants:
    """SearchResult schema contract must hold."""

    async def test_retrieval_strategy_never_empty(self, service, sample_queries):
        """Every result must declare its retrieval strategy."""
        for query in sample_queries:
            results = await service.search(query, limit=10)
            for r in results:
                assert r.retrieval_strategy in (
                    "semantic", "hybrid", "temporal", "relational", "associative"
                ), f"Invalid strategy: {r.retrieval_strategy!r}"

    async def test_id_never_empty(self, service, sample_queries):
        """Every result must have a non-empty id."""
        for query in sample_queries:
            results = await service.search(query, limit=10)
            for r in results:
                assert r.id and len(r.id) > 0

    async def test_required_fields_present(self, service, sample_queries):
        """model_dump() must always contain the canonical field set."""
        REQUIRED = {
            "id", "name", "node_type", "project_id", "score", "distance",
            "retrieval_strategy", "recency_score", "vector_score",
            "activation_score", "path_distance", "salience_score",
            "observations", "relationships", "content",
        }
        for query in sample_queries:
            results = await service.search(query, limit=10)
            for r in results:
                dumped = r.model_dump()
                missing = REQUIRED - set(dumped.keys())
                assert not missing, f"Missing fields: {missing}"
```

### 3.3 Router Invariants

```python
class TestRouterInvariants:
    """Router classification must be deterministic and exhaustive."""

    @given(st.text(min_size=0, max_size=500))
    def test_classify_always_returns_valid_intent(self, query):
        """Any string must classify to a valid QueryIntent."""
        router = QueryRouter(repo=None, embedder=None, vector_store=None)
        intent = router.classify(query)
        assert intent in QueryIntent.__members__.values()

    @given(st.text(min_size=1, max_size=200))
    def test_classify_is_deterministic(self, query):
        """Same query must always classify to the same intent."""
        router = QueryRouter(repo=None, embedder=None, vector_store=None)
        assert router.classify(query) == router.classify(query)

    def test_intent_enum_completeness(self):
        """QueryIntent must have exactly 4 values. Adding a new intent requires an ADR."""
        assert set(QueryIntent) == {
            QueryIntent.SEMANTIC,
            QueryIntent.TEMPORAL,
            QueryIntent.RELATIONAL,
            QueryIntent.ASSOCIATIVE,
        }, "QueryIntent enum changed — requires an ADR"
```

### 3.4 RRF Merge Invariants

```python
class TestMergeInvariants:
    """RRF merge must preserve mathematical properties."""

    @given(
        vector_list=st.lists(st.tuples(st.text(min_size=1), st.floats(0, 1)), max_size=50),
        graph_list=st.lists(st.tuples(st.text(min_size=1), st.floats(0, 1)), max_size=50),
        k=st.integers(1, 100),
        limit=st.integers(1, 100),
    )
    def test_rrf_score_always_non_negative(self, vector_list, graph_list, k, limit):
        """RRF scores must be >= 0."""
        vec = [{"_id": t[0], "_score": t[1]} for t in vector_list]
        graph = [{"id": t[0]} for t in graph_list]
        merged = rrf_merge(vec, graph, k=k, limit=limit)
        for m in merged:
            assert m.rrf_score >= 0.0

    @given(
        vector_list=st.lists(st.tuples(st.text(min_size=1), st.floats(0, 1)), max_size=50),
        graph_list=st.lists(st.tuples(st.text(min_size=1), st.floats(0, 1)), max_size=50),
        limit=st.integers(1, 100),
    )
    def test_rrf_output_respects_limit(self, vector_list, graph_list, limit):
        """Output length must be <= limit."""
        vec = [{"_id": t[0], "_score": t[1]} for t in vector_list]
        graph = [{"id": t[0]} for t in graph_list]
        merged = rrf_merge(vec, graph, limit=limit)
        assert len(merged) <= limit

    @given(
        items=st.lists(st.tuples(st.text(min_size=1), st.floats(0, 1)), min_size=1, max_size=30),
    )
    def test_dual_source_always_beats_single_source(self, items):
        """Entity in both sources must score >= entity in only one source."""
        vec = [{"_id": t[0], "_score": t[1]} for t in items]
        graph = [{"id": t[0]} for t in items]
        merged_both = rrf_merge(vec, graph, limit=100)
        merged_vec_only = rrf_merge(vec, [], limit=100)

        both_scores = {m.entity_id: m.rrf_score for m in merged_both}
        vec_scores = {m.entity_id: m.rrf_score for m in merged_vec_only}

        for eid in both_scores:
            if eid in vec_scores:
                assert both_scores[eid] >= vec_scores[eid]
```

### 3.5 Graph Health Invariants

```python
class TestGraphInvariants:
    """Graph structural properties that must hold."""

    async def test_orphan_ratio_below_threshold(self, service):
        """Orphan ratio must stay below 5%. Higher means entity wiring is broken."""
        health = await service.graph_health()
        if health["total_nodes"] > 0:
            ratio = health["orphan_count"] / health["total_nodes"]
            assert ratio < 0.05, (
                f"Orphan ratio {ratio:.1%} exceeds 5% — "
                f"{health['orphan_count']} orphans / {health['total_nodes']} nodes"
            )

    async def test_vector_count_tracks_entity_count(self, service):
        """Vector store should have embeddings for most entities.

        A large gap means vector sync is broken.
        """
        health = await service.graph_health()
        vector_count = await service.vector_store.count()
        entity_count = health["entity_count"]

        if entity_count > 0:
            sync_ratio = vector_count / entity_count
            assert sync_ratio > 0.7, (
                f"Vector sync ratio {sync_ratio:.1%} — "
                f"{vector_count} vectors / {entity_count} entities"
            )
```

## 4. Implementation Details

### Test Fixtures

```python
@pytest.fixture
def sample_queries():
    """Diverse query set for invariant testing."""
    return [
        "what is this",                          # semantic
        "recent changes",                         # temporal
        'connect "A" and "B"',                   # relational
        "things associated with memory",          # associative
        "",                                       # edge case: empty
        "a" * 500,                               # edge case: very long
        "🧠💡🔥",                                # edge case: emoji
    ]
```

### Markers

```python
# All invariant tests get the 'invariant' marker
pytestmark = pytest.mark.invariant

# Run with: pytest -m invariant
# These should run on EVERY commit (they're fast)
```

### CI Integration

Invariant tests are lightweight (no external services for pure-logic invariants,
mocked services for search invariants). They should run in the standard test suite,
not just the gauntlet.

```
# Fast invariants (no services needed): router, merge, schema
pytest tests/gauntlet/test_invariants.py -k "Router or Merge or Schema" -v

# Full invariants (needs services): score, graph health
pytest tests/gauntlet/test_invariants.py -v
```

## 5. Adding New Invariants

When shipping a new ADR:

1. Identify what properties the change must **preserve** (not just what it adds)
2. Encode those as invariant tests
3. The invariant test should fail if the ADR's assumptions are violated by future changes

Example: If ADR-008 adds a new QueryIntent, the `test_intent_enum_completeness`
invariant will fail — forcing the developer to update the invariant explicitly and
acknowledge the architectural change.

## 6. Test Plan (Meta — Testing the Tests)

| Test | Assertion |
|------|-----------|
| Score out of bounds detected | Injected score=1.5 fails invariant |
| Empty retrieval_strategy detected | Injected strategy="" fails invariant |
| RRF negative score detected | Injected negative score fails invariant |
| Router enum change detected | Added fake intent fails completeness check |
| All invariants pass against current codebase | Green baseline |

## 7. Success Criteria

- [ ] 15+ invariant tests covering scores, schema, router, merge, graph health
- [ ] All pass against current codebase
- [ ] Pure-logic invariants run in < 2 seconds
- [ ] `pytest -m invariant` works as a standalone check
- [ ] At least one invariant would have caught the ADR-007 score-0 bug
  (e.g., "score=0.0 requires vector_score=None")
