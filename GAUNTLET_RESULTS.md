# Dragon Brain Gauntlet — Results

**Date:** 2026-03-10
**Runner:** Antigravity (local execution with Docker)
**Spec:** `DRAGON_BRAIN_GAUNTLET.md`

---

## ROUND 1: IRON BASELINE ✅ PASS

### 1A. Gold Stack (pulse tier)

| Metric | Value |
|--------|-------|
| Tests collected | 826 |
| Tests passed | 821 |
| Tests skipped | 5 |
| Tests failed | 0 |
| Duration | 140.63s |
| Pre-commit hooks | All passing (ruff, ruff-format, trim-ws, codespell, detect-secrets) |

**Note:** Test count increased from documented 784 to 826 after fixing missing `nest_asyncio` dep that prevented `test_dashboard.py` collection.

### 1C. Test Inventory

| Metric | Value |
|--------|-------|
| Source modules | 28 |
| Test files | 60 |
| Scripts (py) | 32 |
| Scripts (ps1) | 7 |
| MCP tools | 30 (19 decorator + 11 runtime) |

---

## ROUND 2: STRESS TEST ✅ PASS

### 2A. Random Ordering (3 seeds)

| Seed | Passed | Skipped | Failed | Duration |
|------|--------|---------|--------|----------|
| 42 | 821 | 5 | 0 | 140.09s |
| 1337 | 821 | 5 | 0 | 139.47s |
| 31415 | 821 | 5 | 0 | ~140s |

**Flaky test found and fixed:** `test_dashboard_app.py` had 3 order-dependent `StopIteration` failures caused by module-level `MagicMock` state leakage — `reset_mock()` does not reliably clear `side_effect` on nested child mocks (`mock_st.sidebar.button`). Fixed by adding explicit `side_effect = None` cleanup in the autouse fixture. Commit `d8307b6`.

### 2B. Parallel Execution (4 workers)

| Workers | Passed | Skipped | Failed | Duration | Speedup |
|---------|--------|---------|--------|----------|---------|
| 4 | 821 | 5 | 0 | 57.77s | **2.4x** |

Serial result matches parallel — no thread-safety issues in test fixtures.

---

## ROUND 4: MUTATION MASSACRE ⏭️ SKIPPED

Already completed via `mutmut` in a prior session. 12 `test_mutant_*.py` files exist targeting mutation survival patterns.

---

## ROUND 6: STATIC INQUISITION ✅ PASS

### 6A. Type Checking (mypy)

```
Success: no issues found in 28 source files
```

### 6B. Linting (ruff)

```
All checks passed!
```

**Fixed during gauntlet:** 3 errors in `test_purge_ghost_vectors.py` (unused imports `asyncio`, `_report_ids`; unsorted import block). Commit `37ac4a8`.

### 6C. Complexity Analysis (radon CC ≥ C)

| Module | Function | Grade | CC |
|--------|----------|-------|----|
| `search.py` | `SearchMixin._hydrate_search_results` | C | 13 |
| `librarian.py` | `LibrarianAgent.run_cycle` | C | 20 |
| `graph_algorithms.py` | `compute_pagerank` | C | 15 |
| `clustering.py` | `_find_bridge_candidates` | C | 12 |
| `analysis.py` | `AnalysisMixin.detect_structural_gaps` | C | 11 |
| `search_advanced.py` | `SearchAdvancedMixin` (class) | C | 11 |
| `search.py` | `SearchMixin.search` | **A** | **5** |

**Resolved:** `search.py:SearchMixin.search` refactored from grade D (CC=23) → grade A (CC=5) via method extraction. Commit `a7157ee`.

### 6D. Dead Code (vulture)

6 findings, all `__aexit__` / `__exit__` context manager protocol params in `lock_manager.py` — required by Python spec even if unused. **Acceptable.**

### 6E. Exception Census

| Pattern | Count |
|---------|-------|
| `except Exception` | 11 |
| `logger.error` without `exc_info` | 19 |
| bare `except:` | **0** ✅ |

---

## ROUND 7: SECURITY SWEEP ✅ PASS

### 7A. Bandit

| Metric | Value |
|--------|-------|
| Total lines scanned | 3,794 |
| Medium issues | 1 (B104: `0.0.0.0` bind in `embedding_server.py`, already `# noqa: S104`) |
| High issues | **0** |

### 7C. Cypher Injection Audit

| Check | Result |
|-------|--------|
| f-string Cypher queries | **0** ✅ |
| `.format()` Cypher queries | **0** ✅ |
| Parameterized queries (safe) | ✅ |

**All Cypher queries use parameterization.** No injection surface.

### 7D. Credentials Audit

| Check | Result |
|-------|--------|
| Hardcoded passwords/tokens | **0** — all from `os.getenv()` |
| `detect-secrets` baseline | Clean ✅ |

---

## ROUND 10: ARCHITECTURE FORENSICS ✅ PASS

### 10A. Module Sizes (LOC)

| Module | LOC | Status |
|--------|-----|--------|
| `analysis.py` | 351 | ⚠️ Over 300 threshold |
| `server.py` | 295 | OK |
| `repository_queries.py` | 287 | OK |
| `search.py` | 284 | OK |
| `crud.py` | 271 | OK |
| `vector_store.py` | 270 | OK |

### 10B. Import Depth (top 5)

| Module | Imports |
|--------|---------|
| `tools.py` | 16 |
| `analysis.py` | 12 |
| `search.py` | 11 |
| `server.py` | 11 |
| `crud.py` | 10 |

---

## SUMMARY

| Round | Name | Result | Key Findings |
|-------|------|--------|-------------|
| 1 | Iron Baseline | ✅ **PASS** | 826 tests, 0 failures |
| 2 | Stress Test | ✅ **PASS** | Flaky test found & fixed; parallel 2.4x speedup |
| 4 | Mutation Massacre | ⏭️ SKIP | Previously completed via mutmut |
| 6 | Static Inquisition | ✅ **PASS** | mypy 0 errors, ruff 0 errors, 5 complexity hotspots |
| 7 | Security Sweep | ✅ **PASS** | 0 Cypher injection, 0 hardcoded creds |
| 10 | Architecture | ✅ **PASS** | 1 module over 300 LOC threshold |

### Fixes Applied

| Commit | Description |
|--------|-------------|
| `37ac4a8` | 3 ruff violations in `test_purge_ghost_vectors.py` + mypy `no-any-return` in `analysis.py` |
| `d8307b6` | Flaky `test_dashboard_app.py` — `reset_mock()` misses nested `side_effect` on `mock_st.sidebar.button` |
