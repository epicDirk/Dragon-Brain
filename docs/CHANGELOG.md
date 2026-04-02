# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

> **Note**: Some environment variables use the prefix `EXOCORTEX_` — this was the project's original codename during early development. These names are kept for backwards compatibility.

## [Unreleased]

### Added

- **Semantic Radar** — Relationship discovery layer (Layers 1–5):
  - **Layer 1**: `find_similar_by_id` — Vector primitive on Qdrant using
    `RecommendQuery` with self-exclusion and caller-specified filters.
  - **Layer 2**: `semantic_radar` MCP tool (#32) — Compares vector similarity
    with graph distance to surface bridge opportunities. Uses
    `shortest_path_length` (FalkorDB-safe directed + bidirectional pattern)
    and `radar_score = cosine_sim * log(1 + distance)` formula.
  - **Layer 3**: `detect_weak_connections` — Standalone utility on
    `ActivationEngine`. Partitions activation vs vector results into
    bridge opportunities and questionable edges.
  - **Layer 4**: `find_semantic_opportunities` MCP tool (#33) — Batch scanner
    using `asyncio.Semaphore` for concurrency. Deduplicates bidirectional
    pairs. Env vars: `RADAR_MAX_DISTANCE_FACTOR`, `RADAR_CONCURRENCY`.
  - **Layer 5**: Schema (`RadarSuggestion`), MCP registration, enriched
    `_infer_relationship_type` covering all Dragon Brain node types.
  - 26 new tests across 4 files (3-evil/1-sad/1-happy per layer).
  - Spec: `SPEC-semantic-radar.md`

- **DRIFT Detection Suite** — Behavioural and structural drift detection:
  - **Feature A: Auto-Update Checker** — Fire-and-forget GitHub releases API
    check on server startup. 5s timeout, HTTPS, silent failure. Opt-out via
    `UPDATE_CHECK=false`. New files: `update_check.py`, `VERSION`.
  - **Feature B: Golden Queries** — 22 frozen queries across 4 intents with
    behavioural contracts. 9 framework tests (mocked, CI-fast). Detects
    router drift and score-0 regression (ADR-007).
  - **Feature C: Distribution Monitors** — Rolling-window `SearchStatsAccumulator`
    tracking strategy/intent distributions, score percentiles, vector null rates,
    and latency. New MCP tool: `search_stats` (#31). Opt-out via
    `SEARCH_STATS_ENABLED=false`. New file: `stats.py`.
  - **Feature D: Architectural Invariants** — 15 invariant tests (Hypothesis
    property tests for router + merge) encoding design contracts. Score bounds,
    schema required fields, router determinism, RRF merge properties.
  - Specs: `SPEC-drift-detection-*.md` (3 files in `docs/`)

- **RequestsDependencyWarning fix** — Suppressed via `PYTHONWARNINGS` in
  `tox.ini` + `conftest.py` filter. Upgraded `chardet` 7.0→7.1 and
  `charset-normalizer` 3.4.4→3.4.5.

- **ADR-007: Hybrid Search Unification** — Unified search pipeline replacing the
  previous single-vector approach with a 3-step hybrid strategy:
  1. **Vector search** — Qdrant similarity (unchanged)
  2. **Associative enrichment** — Spreading activation over graph neighbors
  3. **Relational enrichment** — Quoted-entity path traversal

  New capabilities:
  - **`merge.py`** — Reciprocal Rank Fusion (RRF) with configurable `k=60` for
    merging heterogeneous result lists.
  - **`retrieve_by_ids()`** — Batch Qdrant point retrieval with cosine scoring,
    replacing expensive per-entity re-searches.
  - **Enhanced `SearchResult` schema** — 5 new fields: `vector_score`,
    `graph_score`, `recency_score`, `composite_score`, `search_strategy`
    (all backward-compatible with safe defaults).
  - **`temporal_window_days`** parameter on `route()` for time-scoped queries.
  - Spec: [SPEC-hybrid-search-unification.md](SPEC-hybrid-search-unification.md)
  - ADR: [007-hybrid-search-unification.md](adr/007-hybrid-search-unification.md)

- **Gold Stack Tier 4: The Reaper** — `vulture` dead code detection added as a
  tox environment. Scans `src/` with 80% confidence threshold + whitelist.
  Catches unused functions, classes, variables, and dead imports.

- **Coverage remediation** — 110 new tests across 5 files pushing per-module
  coverage to 90%+ minimum (overall: 91% → 98.27%).

- **Dragon Brain Gauntlet** — 20-round automated quality audit. Scored A- (95/100).
  See [spec](DRAGON_BRAIN_GAUNTLET.md) and [results](GAUNTLET_RESULTS.md).
- **Security hardening** — hardcoded password removed, PII scrubbed, git history
  rewritten via `git filter-repo`. `mcp_config.json` → `mcp_config.example.json`.
- **CI pipeline** — GitHub Actions workflow (`.github/workflows/ci.yml`) running
  `tox -e pulse` with FalkorDB + Qdrant service containers.
- **CONTRIBUTING.md** — Testing policy, Gold Stack tiers, code style guide.
- **Dashboard screenshot** in README.
- **MIT License** (`02e6861`) — Project open-sourced under MIT license.
- **`EXOCORTEX_BACKUP_DIR` env var** (`092f37b`) — Backup destination is now
  configurable via environment variable instead of a hardcoded path. Defaults
  to `backups/gdrive` if unset.

- **E-1: Bottle Reader with Content** (`bd2df89`) — `include_content` parameter
  in `BottleQueryParams` and `get_bottles()` to hydrate bottles with observation
  content in a single call.
- **E-2: Deep Search** (`1f3a1e5`) — `deep: bool` parameter on `search()` that
  hydrates results with associated observations and relationships.
- **E-3: Observation Vectorization** (`9f31e12`) — Automatic embedding of
  observation content upon creation; vectors upserted to Qdrant.
- **E-4: Session Reconnect** (`0588888`) — `reconnect()` method returning a
  structured briefing (recent entities, graph health, time window) for returning
  agents.
- **E-5: System Diagnostics** (`028fa3f`) — Unified `system_diagnostics()` method
  with graph health, vector store count, and split-brain detection.
- **E-6: Procedural Memory** (`ec588ab`) — `Procedure` entity type added to the
  ontology with step-based structure.
- **E-7: E2E Script Enhancement** (`907b5dc`) — `argparse` CLI (`--phase`,
  `--skip-cleanup`, `--strict`, `--verbose`), 8 new E2E phases (19–26), and
  per-phase latency threshold warnings (5s).
- `scripts/validate_brain.py` — 9-check live brain health validator (split-brain,
  bottle chain, temporal completeness, obs vectors, maxmemory, ghost graphs,
  orphan vectors, indices, HNSW threshold).
- `scripts/purge_ghost_vectors.py` (`6167ae6`) — Utility to remove orphan Qdrant
  vectors with no matching graph entity.
- **purge_ghost_vectors Pass 2** (`b6f555a`) — Graph cross-reference: scrolls all
  Qdrant IDs, set-diffs against FalkorDB nodes, reports orphan vectors. Batch
  FalkorDB lookup (single Cypher query), dual-category dry-run reporting.
- **`list_orphans` MCP tool** (`2bee963`) — Tool #30. Lists graph nodes with zero
  relationships for triage/cleanup.
- **Observation Embedding Backfill** (`83bef10`) — `embed_observations.py` import
  path fix + batch size reduction (20→5). Backfilled 464/464 observation vectors.
- **Mutation Test Campaign** (`1b633cf`) — 12 new `test_mutant_*.py` files
  targeting mutation survival patterns (schema enums, ontology, router,
  Pydantic defaults, dict values, config defaults, default params, graph
  algorithms, clustering, lock manager, temporal).
- **`flush_background_tasks()`** (`28af2cd`) — Public method on
  `CrudMaintenanceMixin` / `MemoryService` to deterministically await all
  pending background tasks. Useful for graceful shutdown and test assertions.
- **ADR-007: Hybrid Search Unification** — Vector-first pipeline: vector search →
  intent classification → graph enrichment → RRF merge → hydration → recency scoring.
  New `merge.py` module, `HybridSearchResponse` model, `retrieve_by_ids()` batch
  API on `QdrantVectorStore`, real exponential decay recency (`2^(-age/hl)`).
  `search_memory` tool gains `temporal_window_days` and `include_meta` params.
- **10 Gauntlet RRF property tests** — Hypothesis-based property testing for
  `rrf_merge` (limit cap, sort order, determinism, dual-source boost, etc.).

### Fixed

- **25 test failures** (`bf0e669`) — Repaired accumulated API drift across 12
  test files: async/await mismatches, mock shape fixes, removed-code guards,
  QDRANT_HOST config drift, import path fixes.
- **Stale scheduled task** — Removed `ExocortexDailyBackup` (3 AM) from Windows
  Task Scheduler; correct `ExocortexBackup` (11 PM) task retained.
- **P0-0** (`6167ae6`) — Re-embedded all 464 entities after vector store rebuild.
- **P0-1** (`eea3ed8`) — Surface `PRECEDED_BY` errors in `EntityCommitReceipt.warnings`.
- **P0-2** (`26d7870`) — Set FalkorDB `maxmemory 1GB` + `noeviction` via `REDIS_ARGS`.
- **P0-3** (`7b9276f`) — Wrap `search()` and `search_associative()` in error handling.
- **P0-4** (`370919b`) — Add Qdrant `UnexpectedResponse` + `RpcError` to
  `retry_on_transient`.
- **P0-5** (`3976b65`) — Remove phantom dependencies (`graphiti-core`, `neo4j`, `pandas`).
- **P0-6** (`b8c4d34`) — Ghost graph cleanup, temporal backfill, FalkorDB query fix.
- `traverse_path` shortestPath FalkorDB compatibility (`f33ab01`).
- `get_bottles` label query fix (`f33ab01`).
- Custom Louvain replaced with NetworkX + `log2` salience bug fix (`54dcaec`).
- `OntologyManager` CWD-relative path fix (`6c7e616`).
- Bare `except Exception` catches narrowed to specific types across 7 files.
- **`test_retry.py`** patch path bug (`fe0bcca`) — `patch("time.sleep")` was
  intercepted by the autouse `_fast_retries` fixture, causing delay-cap
  assertions to pass vacuously over an empty list. Fixed to
  `patch("claude_memory.retry.time.sleep")` with an `assert call_args_list`
  guard.
- **Async test hang** (`48164a7`) — Fixed `asyncio.run()` event loop conflict
  in test fixtures.
- **Pre-existing test collection errors** — Fixed `test_purge_ghost_vectors.py`
  and `test_backfill_temporal.py` broken imports after scripts moved to
  `scripts/internal/`. 24 tests recovered.

### Changed

- E2E test suite expanded from 18 to 26 phases.
- Unit test suite: **1,120 tests** across 81 files, 0 failures.
- Gold Stack tiers: 4 → 5 (added `reaper`/Vulture dead code tier to tox.ini).
- MCP tools: 31 → 33 (added `semantic_radar`, `find_semantic_opportunities`).
- Pre-commit hooks: ruff, ruff-format, codespell, detect-secrets all passing.
- Removed 3 redundant `with patch("claude_memory.retry.time.sleep")` wrappers
  in `test_mutant_config_defaults.py` — the autouse `_fast_retries` fixture
  covers them (`fe0bcca`).
- **Backup path hardcoded** (`092f37b`) — `scheduled_backup.py` had
  a hardcoded Google Drive path baked in. Now reads `EXOCORTEX_BACKUP_DIR`
  env var. Task Scheduler setup upgraded with better settings.

### Removed

- Internal development docs purged for public release (`02e6861`):
  `REHYDRATION_DOCUMENT.md`, `POST_BUILD_FINDINGS.md`, `ENHANCEMENT_SPEC.md`,
  `structural_analysis.md`, `implementation_plan.md`, `task.md`,
  `walkthrough.md`, and `docs/project_history/`.
