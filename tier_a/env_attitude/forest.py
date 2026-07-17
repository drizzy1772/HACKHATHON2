"""Re-export of the base kit's forest generator. INVARIANT 1 stays single-sourced:
Tier-A never re-implements ``mulberry32``, it only calls the frozen reference.
"""

from __future__ import annotations

from tier_d.env.forest import make_forest

__all__ = ["make_forest"]
