# SPEC: Semantic Radar — Relationship Discovery Layer

**Status:** Draft
**Author:** Claude (Architect) + Tabish
**Date:** 2026-04-02
**Priority:** High — foundational enhancement for graph density

## Problem Statement

Dragon Brain's knowledge graph has ~975 entities averaging ~2 edges each (density 0.001). Entities are wired per the Entity Wiring Rule (minimum 1 relationship beyond `HAS_OBSERVATION`), but most stop at the minimum. This makes the graph structurally sparse:

- **Graph-native algorithms** (PageRank, Louvain, `traverse_path`, `find_cross_domain_patterns`) walk real edges — sparse graph = blind algorithms
- **Holographic retrieval** compensates at query time but is a crutch, not a cure
- **Cross-domain discovery** suffers — conceptual links that aren't textually similar go undetected

## Solution: Semantic Radar

A 4-layer discovery system that surfaces **potential relationships** between entities based on vector similarity vs. graph distance analysis. The core insight: if two entities are semantically close but far apart (or disconnected) in the graph, that's a **bridge opportunity**.

**Design philosophy (stolen from Tesseract):** Never auto-commit relationships. The graph stays clean. Radar discovers, humans/Claude decide.

## Architecture Overview

```
Layer 1: find_similar_by_id()          [vector_store.py]
    |         Pure vector primitive — "who's similar to X?"
    v
Layer 2: semantic_radar()              [search_advanced.py]
    |         Vector sim + graph distance = scored suggestions
    v
Layer 3: detect_weak_connections()     [activation.py]
    |         Energy-based gap detection post-activation
    v
Layer 4: find_semantic_opportunities() [analysis.py]
    |         Batch project-wide gap scanner
    v
MCP Tool: semantic_radar()             [tools_extra.py]
    |         Exposed to Claude via MCP
    v
MCP Tool: find_semantic_opportunities()[tools_extra.py]
              Batch mode exposed to Claude via MCP
```

## Phased Delivery

**Build and test each layer before moving to the next.** If Layer 1's design is wrong, Layers 2-4 inherit the problem.

---

### Layer 1: `find_similar_by_id()` — Vector Primitive

**File:** `src/claude_memory/vector_store.py`
**Also update:** `src/claude_memory/interfaces.py` (VectorStore Protocol)

**What it does:** Given an entity ID, find the N most similar entities by vector cosine similarity. Uses Qdrant's native `recommend` API for efficiency (no need to fetch the entity's vector first).

```python
async def find_similar_by_id(
    self,
    entity_id: str,
    limit: int = 10,
    threshold: float = 0.6,
    exclude_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Find entities similar to a given entity by vector proximity.

    Uses Qdrant's recommend API — looks up the entity's vector internally
    and finds nearest neighbors, filtered by threshold.

    Returns: [{"_id": str, "_score": float, "payload": {...}}, ...]
    """
```

**Implementation notes:**
- Use `client.recommend()` (Qdrant 1.16+) with `positive=[entity_id]` — this does the vector lookup internally
- Apply `score_threshold` parameter for filtering
- `exclude_ids` should always include the entity itself (don't return "X is similar to X")
- Filter support: project_id scoping via Qdrant filter (optional)
- Add to `VectorStore` Protocol in `interfaces.py`

**Tests:**
- Unit test: mock Qdrant, verify recommend call with correct params
- Unit test: verify self-exclusion (entity_id always excluded from results)
- Unit test: threshold filtering
- Integration test (if Docker available): create 5 entities, verify similarity ordering

---

### Layer 2: `semantic_radar()` — The Core Feature

**File:** `src/claude_memory/search_advanced.py` (new method on `SearchAdvancedMixin`)

**What it does:** For a given entity, discovers potential relationships by comparing vector similarity against graph distance. High similarity + high/infinite graph distance = bridge opportunity.

```python
async def semantic_radar(
    self,
    entity_id: str,
    limit: int = 10,
    similarity_threshold: float = 0.6,
    project_id: str | None = None,
) -> dict[str, Any]:
    """Discover potential relationships for an entity.

    Compares vector similarity with graph distance to identify:
    - Bridge opportunities (similar but disconnected/distant in graph)
    - Each suggestion includes: candidate entity, cosine sim, graph distance,
      suggested relationship type, and reasoning.

    Returns suggestions only — does NOT commit edges.
    """
```

**Algorithm:**
1. Call `vector_store.find_similar_by_id(entity_id, limit=limit*2, threshold=similarity_threshold)` — over-fetch for filtering
2. For each candidate, compute graph distance:
   - Use `repo.shortest_path_length(entity_id, candidate_id)` (new repo method, simple Cypher `shortestPath`)
   - If no path exists → distance = infinity (best bridge candidate)
3. Compute a **radar score**: `radar_score = cosine_similarity * (1 / (1 + graph_distance))` inverted — HIGH score = high similarity AND high graph distance
   - Actually: `radar_score = cosine_similarity * log(1 + graph_distance)` — rewards similarity but amplifies when graph distance is high
   - If graph_distance = infinity: `radar_score = cosine_similarity * MAX_DISTANCE_FACTOR` (e.g., 5.0)
   - If graph_distance <= 1: `radar_score = 0` (already directly connected, skip)
4. Sort by radar_score descending, take top `limit`
5. For each suggestion, infer a relationship type:
   - Use existing `EdgeType` literals from `schema.py`
   - Heuristic based on node_types of source and candidate (e.g., both Concept → `ANALOGOUS_TO`, Concept→Project → `PART_OF`, etc.)
   - Fallback: `RELATED_TO`
6. Return structured suggestions

**New repo method needed:** `shortest_path_length(from_id, to_id) -> int | None`
- **File:** `src/claude_memory/repository_traversal.py`
- Cypher: `MATCH p=shortestPath((a:Entity {id: $from})-[*]-(b:Entity {id: $to})) RETURN length(p)`
- Returns `None` if no path exists
- Should handle the case where either node doesn't exist

**Output structure:**
```python
{
    "entity_id": str,
    "entity_name": str,
    "suggestions": [
        {
            "candidate_id": str,
            "candidate_name": str,
            "candidate_type": str,
            "cosine_similarity": float,
            "graph_distance": int | None,  # None = disconnected
            "radar_score": float,
            "suggested_relationship": str,  # EdgeType
            "reasoning": str,  # "High semantic similarity (0.82) but no graph path — potential bridge"
        }
    ],
    "stats": {
        "candidates_scanned": int,
        "already_connected": int,  # skipped because graph_distance <= 1
        "disconnected": int,       # no path at all
    }
}
```

**Tests:**
- Unit test: entity with no similar neighbors → empty suggestions
- Unit test: entity where all similar neighbors are already connected → empty (all filtered)
- Unit test: entity with similar but disconnected neighbors → suggestions returned, sorted by radar_score
- Unit test: verify radar_score formula (known inputs → expected outputs)
- Unit test: verify graph_distance=None handling (disconnected entities)
- Unit test: relationship type inference based on node types
- Integration test: create a mini-graph with known topology, verify radar finds the expected gaps

---

### Layer 3: `detect_weak_connections()` — Energy Gap Detection

**File:** `src/claude_memory/activation.py` (new method on `ActivationEngine`)

**What it does:** After spreading activation, identifies nodes that received activation energy (graph-reachable) but have LOW vector similarity to the seed — suggesting the graph connection might be questionable or the vector embedding needs updating.

Also detects the inverse: nodes that are vector-similar but received NO activation energy (graph-unreachable) — these are the bridge opportunities that Layer 2 finds, but detected through the activation lens.

```python
def detect_weak_connections(
    self,
    seed_ids: list[str],
    activation_map: dict[str, float],
    vector_scores: dict[str, float],
    similarity_threshold: float = 0.3,
) -> dict[str, list[dict[str, Any]]]:
    """Analyze activation results to find structural anomalies.

    Returns:
        {
            "bridge_opportunities": [...],  # vector-similar but no activation (unreachable)
            "questionable_edges": [...],    # activated (graph-close) but low vector similarity
        }
    """
```

**Algorithm:**
1. Partition candidates into:
   - `activated_set`: entities in `activation_map` with energy > 0
   - `similar_set`: entities in `vector_scores` with score > `similarity_threshold`
2. **Bridge opportunities** = `similar_set - activated_set` — semantically close but graph-unreachable
3. **Questionable edges** = `activated_set - similar_set` (with activation > threshold but vector_score < similarity_threshold) — graph says connected, vectors say unrelated
4. Score and sort each list

**Tests:**
- Unit test: all similar entities are also activated → no anomalies
- Unit test: similar but non-activated entities → bridge_opportunities populated
- Unit test: activated but dissimilar entities → questionable_edges populated
- Unit test: empty inputs → empty results

---

### Layer 4: `find_semantic_opportunities()` — Batch Project Scanner

**File:** `src/claude_memory/analysis.py` (new method on `AnalysisMixin`)

**What it does:** Scans an entire project (or all entities) to find the highest-value bridge opportunities across the graph. This is the "show me all the missing connections" view.

```python
async def find_semantic_opportunities(
    self,
    project_id: str | None = None,
    similarity_threshold: float = 0.65,
    limit: int = 20,
    min_graph_distance: int = 3,
) -> dict[str, Any]:
    """Batch scan for high-value bridge opportunities across the graph.

    For each entity, runs a lightweight radar scan and collects the
    top opportunities. Deduplicates bidirectional pairs (A→B = B→A).

    Returns ranked suggestions with aggregate stats.
    """
```

**Algorithm:**
1. Fetch all entity IDs (optionally filtered by `project_id`)
2. For each entity (batched, with concurrency limit):
   - Call `vector_store.find_similar_by_id(entity_id, limit=5, threshold=similarity_threshold)`
   - For each candidate pair, compute graph distance (batch Cypher if possible)
   - Filter: only keep pairs where `graph_distance >= min_graph_distance` or `graph_distance is None`
3. Deduplicate: pair (A,B) = pair (B,A) — keep the higher-scoring one
4. Sort by radar_score descending, take top `limit`
5. Return with aggregate stats (total entities scanned, total pairs evaluated, bridge count by project)

**Performance considerations:**
- Cap entity scan at 200 entities per call (configurable) — full graph scan on 975 entities with 5 candidates each = 4,875 shortest-path queries
- Use batch Cypher for shortest paths where possible
- Consider caching graph distances within a single scan (distance A→B = distance B→A)
- Add a timeout parameter (default 30s)

**Output structure:**
```python
{
    "opportunities": [
        {
            "entity_a": {"id": str, "name": str, "type": str},
            "entity_b": {"id": str, "name": str, "type": str},
            "cosine_similarity": float,
            "graph_distance": int | None,
            "radar_score": float,
            "suggested_relationship": str,
            "reasoning": str,
        }
    ],
    "stats": {
        "entities_scanned": int,
        "pairs_evaluated": int,
        "bridges_found": int,
        "already_connected": int,
        "scan_time_ms": float,
    }
}
```

**Tests:**
- Unit test: empty graph → empty results
- Unit test: fully connected graph → no opportunities (all filtered)
- Unit test: graph with known disconnected clusters → bridges detected between them
- Unit test: deduplication (A,B) = (B,A)
- Unit test: project_id filtering
- Unit test: limit and min_graph_distance params respected
- Integration test: create 2 isolated clusters with semantically similar entities, verify cross-cluster bridges found

---

## MCP Tool Registration

**File:** `src/claude_memory/tools_extra.py`

Register two new tools:

### `semantic_radar` (entity-level)
```python
async def semantic_radar(
    entity_id: str,
    limit: int = 10,
    similarity_threshold: float = 0.6,
    project_id: str | None = None,
) -> dict[str, Any]:
    """Discover potential relationships for an entity.

    Compares vector similarity with graph distance to find bridge
    opportunities — entities that are semantically related but
    poorly connected in the graph. Returns suggestions only,
    does NOT create edges.
    """
```

### `find_semantic_opportunities` (batch/project-level)
```python
async def find_semantic_opportunities(
    project_id: str | None = None,
    similarity_threshold: float = 0.65,
    limit: int = 20,
    min_graph_distance: int = 3,
) -> dict[str, Any]:
    """Batch scan for missing relationships across the graph.

    Scans entities to find pairs that are semantically similar
    but disconnected or distant in the graph. Returns ranked
    bridge opportunities. Does NOT create edges.
    """
```

Both tools must be added to `configure()` function via `mcp.tool()()`.

---

## Schema Updates

**File:** `src/claude_memory/schema.py`

Add a Pydantic model for radar results (optional but clean):

```python
class RadarSuggestion(BaseModel):
    """A single semantic radar suggestion."""
    candidate_id: str
    candidate_name: str
    candidate_type: str
    cosine_similarity: float
    graph_distance: int | None
    radar_score: float
    suggested_relationship: str
    reasoning: str
```

---

## Interface Updates

**File:** `src/claude_memory/interfaces.py`

Add `find_similar_by_id` to the `VectorStore` Protocol:

```python
async def find_similar_by_id(
    self,
    entity_id: str,
    limit: int = 10,
    threshold: float = 0.6,
    exclude_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Find entities similar to a given entity by vector proximity."""
    ...
```

---

## Files Modified (Summary)

| File | Change |
|------|--------|
| `src/claude_memory/vector_store.py` | Add `find_similar_by_id()` |
| `src/claude_memory/interfaces.py` | Add `find_similar_by_id` to Protocol |
| `src/claude_memory/repository_traversal.py` | Add `shortest_path_length()` |
| `src/claude_memory/search_advanced.py` | Add `semantic_radar()` |
| `src/claude_memory/activation.py` | Add `detect_weak_connections()` |
| `src/claude_memory/analysis.py` | Add `find_semantic_opportunities()` |
| `src/claude_memory/tools_extra.py` | Register 2 new MCP tools |
| `src/claude_memory/schema.py` | Add `RadarSuggestion` model |
| `tests/unit/test_radar.py` | New test file — all unit tests |
| `tests/unit/test_radar_integration.py` | Integration tests (Docker required) |

---

## What NOT To Do

- **Do NOT auto-commit relationships.** Radar is advisory only.
- **Do NOT modify existing search behavior.** Radar is additive.
- **Do NOT break existing tests.** All 1,116 tests must still pass.
- **Do NOT add LLM calls** for relationship type inference — use heuristics. LLM dependency makes this slow and non-deterministic.
- **Do NOT scan more than 200 entities** in `find_semantic_opportunities` without explicit override — performance guard.

## Testing Strategy

- All new code must have unit tests with mocked dependencies
- Use the existing test patterns in `tests/unit/` for structure
- Aim for >90% coverage on new code
- Run full test suite after each layer to verify no regressions:
  ```bash
  cd claude-memory-mcp
  python -m pytest tests/ -x -q
  ```

## Build Sequence for AG

1. Layer 1 → run tests → verify green
2. Layer 2 (+ new repo method) → run tests → verify green
3. Layer 3 → run tests → verify green
4. Layer 4 → run tests → verify green
5. MCP tool registration → run tests → verify green
6. Full test suite → all 1,116+ tests pass
