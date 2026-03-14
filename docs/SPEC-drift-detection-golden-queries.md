# SPEC: Golden Query Set — Behavioral Drift Detection

**Spec ID:** DRIFT-001
**Priority:** P1 — first line of defense against emergent drift
**Depends on:** ADR-007 (Hybrid Search Unification) — shipped
**Location:** `tests/gauntlet/test_golden_queries.py`

---

## 1. Problem Statement

Individual code reviews catch spec violations. CI catches regressions in tested paths.
Neither catches **behavioral drift** — the slow accumulation of individually-correct
changes that compound into search results nobody designed.

Example: ADR-007 changed search routing. A future change tweaks temporal keywords.
Another adjusts activation weights. Another adds 500 new entities. Each change passes
review. But the combined effect is that "recent work on Dragon Brain" no longer returns
Dragon Brain entities in the top 3.

## 2. What It Is

A frozen set of 20-30 queries with **expected behavioral contracts** (not expected exact
results). Run as part of the test suite. Failures indicate something drifted.

These are NOT unit tests. They don't test code paths. They test **outcomes**.

## 3. Query Design Principles

Each golden query specifies:
- The query text (frozen — never changes)
- Behavioral assertions (what MUST be true about the results)
- Soft assertions (what SHOULD be true — logged as warnings, not failures)
- The intent the router should classify it as

Assertions are about **properties of results**, not specific entities:
- "top-3 results have score > 0.3" (not "result #1 is entity X")
- "at least one result has retrieval_strategy='hybrid'" (not "exactly 5 results")
- "recency_score of top result > 0.5" (not "recency_score is 0.91")

This makes them resilient to graph growth while catching behavioral shifts.

## 4. Proposed Golden Query Set

### Semantic Intent Queries

```python
GoldenQuery(
    query="what is spreading activation",
    expected_intent=QueryIntent.SEMANTIC,
    must=[
        "len(results) >= 1",
        "results[0].score > 0.5",
        "results[0].retrieval_strategy == 'semantic'",
        "'activation' in results[0].name.lower() or 'activation' in (results[0].content or '').lower()",
    ],
    should=[
        "results[0].score > 0.7",
        "len(results) >= 3",
    ],
)

GoldenQuery(
    query="how does the knowledge graph store relationships",
    expected_intent=QueryIntent.SEMANTIC,
    must=[
        "len(results) >= 1",
        "results[0].score > 0.4",
        "any(r.node_type == 'Entity' for r in results[:5])",
    ],
)
```

### Temporal Intent Queries

```python
GoldenQuery(
    query="what happened recently",
    expected_intent=QueryIntent.TEMPORAL,
    must=[
        "len(results) >= 1",
        "all(r.retrieval_strategy in ('hybrid', 'temporal') for r in results)",
        "all(r.recency_score > 0.0 for r in results if r.vector_score is not None)",
    ],
    should=[
        "results[0].recency_score > 0.3",
    ],
)

GoldenQuery(
    query="timeline of changes last week",
    expected_intent=QueryIntent.TEMPORAL,
    must=[
        "all(r.retrieval_strategy in ('hybrid', 'temporal') for r in results)",
        "not any(r.score == 0.0 and r.vector_score is not None for r in results)",
    ],
)
```

### Relational Intent Queries

```python
GoldenQuery(
    query='how does "Dragon Brain" connect to "Claude Memory"',
    expected_intent=QueryIntent.RELATIONAL,
    must=[
        "all(r.retrieval_strategy in ('hybrid', 'relational') for r in results)",
    ],
    should=[
        "any(r.path_distance is not None for r in results)",
    ],
)
```

### Associative Intent Queries

```python
GoldenQuery(
    query="things related to memory architecture",
    expected_intent=QueryIntent.ASSOCIATIVE,
    must=[
        "len(results) >= 1",
        "all(r.retrieval_strategy in ('hybrid', 'associative') for r in results)",
    ],
    should=[
        "any(r.activation_score > 0.0 for r in results)",
        "results[0].score > 0.3",
    ],
)
```

### Score Integrity Queries (The Whole Point)

```python
GoldenQuery(
    query="recent work on projects",
    expected_intent=QueryIntent.TEMPORAL,
    must=[
        # THE ORIGINAL BUG — this must NEVER return all-zero scores again
        "not all(r.score == 0.0 for r in results)",
        "any(r.vector_score is not None and r.vector_score > 0.0 for r in results)",
    ],
)

GoldenQuery(
    query="search for any entity",
    expected_intent=QueryIntent.SEMANTIC,
    must=[
        "all(r.score > 0.0 for r in results)",
        "all(r.retrieval_strategy != '' for r in results)",
        "all(r.vector_score is not None for r in results)",
    ],
)
```

## 5. Implementation

```python
# tests/gauntlet/test_golden_queries.py

@dataclass
class GoldenQuery:
    query: str
    expected_intent: QueryIntent
    must: list[str]           # Hard assertions — test fails
    should: list[str] = None  # Soft assertions — warning logged

@pytest.fixture
def golden_queries() -> list[GoldenQuery]:
    return [ ... ]  # The full set from §4

@pytest.mark.slow  # These hit real services
@pytest.mark.parametrize("gq", golden_queries())
async def test_golden_query(gq: GoldenQuery, service: MemoryService):
    # Verify intent classification
    intent = service.router.classify(gq.query)
    assert intent == gq.expected_intent, (
        f"Router drift: {gq.query!r} classified as {intent}, expected {gq.expected_intent}"
    )

    # Run the search
    results = await service.search(gq.query, limit=10)

    # Hard assertions
    for assertion in gq.must:
        assert eval(assertion), (
            f"Golden query FAILED: {gq.query!r}\n"
            f"  Assertion: {assertion}\n"
            f"  Results: {[r.model_dump() for r in results[:3]]}"
        )

    # Soft assertions (warnings only)
    for assertion in (gq.should or []):
        if not eval(assertion):
            warnings.warn(
                f"Golden query SOFT FAIL: {gq.query!r}\n  Assertion: {assertion}",
                UserWarning,
            )
```

## 6. Maintenance Rules

- **Queries are frozen.** Never change the query text. If a query becomes irrelevant
  (e.g., entity was deleted), retire it and add a replacement.
- **Assertions evolve slowly.** Only tighten thresholds, never loosen them without an
  ADR justifying why.
- **Add new queries when new features ship.** Every ADR should add 1-2 golden queries
  testing its behavioral impact.
- **Run on every release, not every commit.** These hit real services and are slow.
  Mark as `@pytest.mark.slow`.
- **Failures are investigated, not auto-fixed.** A golden query failure means something
  drifted — find which of the last N commits caused it.

## 7. Success Criteria

- [ ] 20+ golden queries covering all 4 intents + score integrity
- [ ] All pass against current graph state
- [ ] Integrated into gauntlet test suite
- [ ] At least 2 queries specifically guard against the ADR-007 score-0 bug recurring
