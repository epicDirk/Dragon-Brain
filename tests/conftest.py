"""Root conftest — early warning suppression + global fixtures.

This file loads before any test collection, so the warnings.filterwarnings
call takes effect before requests/__init__.py fires its version check.
"""

# ── Must be FIRST — before any imports that pull in `requests` ──
import warnings

warnings.filterwarnings(
    "ignore",
    message=r"urllib3.*doesn't match a supported version",
)
