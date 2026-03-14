# SPEC: Distribution Monitors — Statistical Drift Detection

**Spec ID:** DRIFT-002
**Priority:** P2 — second line of defense, catches what golden queries miss
**Depends on:** ADR-007 (Hybrid Search Unification) — shipped
**Location:** `src/claude_memory/stats.py` + MCP tool `search_stats()`

---

## 1. Problem Statement

Golden queries catch behavioral drift for known queries. Distribution monitors catch
**statistical drift** — aggregate shifts in search behavior that no single query reveals.

Examples:
- Temporal routing was 15% of queries last month, now it's 40% (keyword creep)
- Median score dropped from 0.6 to 0.3 (embedding quality degradation)
- `vector_score=None` results went from 2% to 20% (vector sync falling behind graph)
- `recency_score` is 0.0 for 90% of results (timestamp parsing bug)

These are signals, not alarms. They tell you *something changed* — you investigate why.

## 2. What Gets Tracked

### Per-Search Metrics (Rolling Window)

| Metric | Type | Why It Matters |
|--------|------|----------------|
| `retrieval_strategy` distribution | Counter per strategy | Detects router keyword creep |
| `score` percentiles (p10, p50, p90) | Float | Detects embedding quality shifts |
| `vector_score` null rate | Percentage | Detects vector sync lag |
| `recency_score` zero rate | Percentage | Detects timestamp parsing failures |
| `detected_intent` distribution | Counter per intent | Detects classifier drift |
| `temporal_exhausted` rate | Percentage | Detects temporal window inadequacy |
| `result_count` percentiles | Int | Detects graph sparsity or bloat |
| `search_latency_ms` percentiles | Float | Detects performance regression |

### Rolling Window

- Track last 500 searches (configurable via `STATS_WINDOW_SIZE` env var)
- Older entries rotate out
- Stats are in-memory only — no persistence needed (they rebuild from real-time usage)

## 3. Implementation

### 3.1 Stats Accumulator

```python
# src/claude_memory/stats.py

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, UTC

@dataclass
class SearchSnapshot:
    """Single search execution record."""
    timestamp: datetime
    query: str
    detected_intent: str
    retrieval_strategies: list[str]  # from results
    scores: list[float]
    vector_scores: list[float | None]
    recency_scores: list[float]
    result_count: int
    latency_ms: float
    temporal_exhausted: bool | None

class SearchStatsAccumulator:
    """Rolling-window statistics for search behavior monitoring."""

    def __init__(self, window_size: int = 500):
        self._window: deque[SearchSnapshot] = deque(maxlen=window_size)

    def record(self, snapshot: SearchSnapshot) -> None:
        self._window.append(snapshot)

    def report(self) -> dict:
        if not self._window:
            return {"status": "no data", "searches_recorded": 0}

        snapshots = list(self._window)
        n = len(snapshots)

        # Strategy distribution
        strategy_counts: dict[str, int] = {}
        for s in snapshots:
            for strat in s.retrieval_strategies:
                strategy_counts[strat] = strategy_counts.get(strat, 0) + 1

        # Intent distribution
        intent_counts: dict[str, int] = {}
        for s in snapshots:
            intent_counts[s.detected_intent] = intent_counts.get(s.detected_intent, 0) + 1

        # Score percentiles
        all_scores = [sc for s in snapshots for sc in s.scores]
        all_scores.sort()

        # Vector score null rate
        all_vs = [vs for s in snapshots for vs in s.vector_scores]
        vs_null_count = sum(1 for vs in all_vs if vs is None)

        # Recency zero rate
        all_rec = [rs for s in snapshots for rs in s.recency_scores]
        rec_zero_count = sum(1 for rs in all_rec if rs == 0.0)

        # Temporal exhaustion rate
        temporal_searches = [s for s in snapshots if s.temporal_exhausted is not None]
        temporal_exhausted_count = sum(1 for s in temporal_searches if s.temporal_exhausted)

        # Latency percentiles
        latencies = sorted(s.latency_ms for s in snapshots)

        return {
            "searches_recorded": n,
            "window_start": snapshots[0].timestamp.isoformat(),
            "window_end": snapshots[-1].timestamp.isoformat(),
            "strategy_distribution": {
                k: {"count": v, "pct": round(v / sum(strategy_counts.values()) * 100, 1)}
                for k, v in sorted(strategy_counts.items(), key=lambda x: -x[1])
            },
            "intent_distribution": {
                k: {"count": v, "pct": round(v / n * 100, 1)}
                for k, v in sorted(intent_counts.items(), key=lambda x: -x[1])
            },
            "score_percentiles": {
                "p10": _percentile(all_scores, 10),
                "p50": _percentile(all_scores, 50),
                "p90": _percentile(all_scores, 90),
            } if all_scores else {},
            "vector_score_null_rate_pct": round(vs_null_count / len(all_vs) * 100, 1) if all_vs else 0,
            "recency_score_zero_rate_pct": round(rec_zero_count / len(all_rec) * 100, 1) if all_rec else 0,
            "temporal_exhaustion_rate_pct": (
                round(temporal_exhausted_count / len(temporal_searches) * 100, 1)
                if temporal_searches else None
            ),
            "latency_ms_percentiles": {
                "p50": _percentile(latencies, 50),
                "p90": _percentile(latencies, 90),
                "p99": _percentile(latencies, 99),
            },
            "avg_result_count": round(sum(s.result_count for s in snapshots) / n, 1),
        }


def _percentile(sorted_list: list[float], pct: int) -> float:
    if not sorted_list:
        return 0.0
    idx = int(len(sorted_list) * pct / 100)
    idx = min(idx, len(sorted_list) - 1)
    return round(sorted_list[idx], 4)
```

### 3.2 Integration into Search Pipeline

In `search.py`, after the hybrid pipeline returns results:

```python
# At the end of search()
if self._stats:
    self._stats.record(SearchSnapshot(
        timestamp=datetime.now(UTC),
        query=query,
        detected_intent=detected_intent.value,
        retrieval_strategies=[r.retrieval_strategy for r in results],
        scores=[r.score for r in results],
        vector_scores=[r.vector_score for r in results],
        recency_scores=[r.recency_score for r in results],
        result_count=len(results),
        latency_ms=elapsed_ms,
        temporal_exhausted=temporal_exhausted if detected_intent == QueryIntent.TEMPORAL else None,
    ))
```

### 3.3 MCP Tool

```python
# server.py
@mcp.tool()
async def search_stats() -> dict[str, Any]:
    """Return rolling-window search behavior statistics.

    Reports distribution of retrieval strategies, score percentiles,
    vector score null rates, and latency — useful for detecting
    behavioral drift over time.
    """
    if not hasattr(service, "_stats") or service._stats is None:
        return {"status": "stats not enabled", "searches_recorded": 0}
    return service._stats.report()
```

### 3.4 Enabling/Disabling

Stats collection is opt-in via env var:

```
SEARCH_STATS_ENABLED=true   # default: false
STATS_WINDOW_SIZE=500        # default: 500
```

Zero overhead when disabled. When enabled, each search adds ~0.1ms of overhead
(building the snapshot dataclass).

## 4. Reading the Report

### Healthy Baseline (Establish After ADR-007 Settles)

```json
{
    "searches_recorded": 247,
    "strategy_distribution": {
        "semantic": {"count": 180, "pct": 55.0},
        "hybrid": {"count": 120, "pct": 36.6},
        "temporal": {"count": 18, "pct": 5.5},
        "associative": {"count": 8, "pct": 2.4},
        "relational": {"count": 2, "pct": 0.6}
    },
    "score_percentiles": {"p10": 0.28, "p50": 0.52, "p90": 0.81},
    "vector_score_null_rate_pct": 3.2,
    "recency_score_zero_rate_pct": 12.0,
    "temporal_exhaustion_rate_pct": 33.0,
    "latency_ms_percentiles": {"p50": 45, "p90": 120, "p99": 280}
}
```

### Red Flags

| Signal | Possible Cause |
|--------|---------------|
| `semantic` drops below 40% | Router keyword creep — too many queries classified as temporal/relational |
| `score p50` drops below 0.3 | Embedding quality degradation or vector sync lag |
| `vector_score_null_rate` above 15% | Graph growing faster than vector indexing |
| `recency_score_zero_rate` above 30% | Timestamp parsing bug or missing `occurred_at` fields |
| `temporal_exhaustion_rate` above 60% | Default 7-day window too narrow for usage patterns |
| `latency p90` above 500ms | Performance regression in hybrid pipeline |

## 5. Test Plan

| Test | Assertion |
|------|-----------|
| Record snapshot, report has correct counts | Basic accumulation works |
| Window overflow drops oldest entries | Deque maxlen behavior |
| Empty accumulator returns "no data" | No crash on empty |
| Percentile calculation with edge cases | Single element, all same values |
| Stats disabled by default | No accumulator created unless env var set |
| MCP tool returns report | Integration with server.py |

## 6. Success Criteria

- [ ] `search_stats()` MCP tool returns a report after 10+ searches
- [ ] Strategy distribution matches expected routing patterns
- [ ] Score percentiles are non-zero and plausible
- [ ] Zero overhead when disabled
- [ ] Integrated into existing test suite
