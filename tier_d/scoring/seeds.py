"""Seed discipline. INVARIANT 5: hidden seeds never touch training or tuning.

``PUBLIC_SEEDS`` ship to the teams for self-checking. ``HIDDEN_SEEDS`` are used
exclusively by the final scoring run and live in ``tier_d/admin/scoring/
hidden_seeds.py``, which is NOT part of the student handout. In a student copy
(where ``tier_d/admin/`` has been deleted) ``HIDDEN_SEEDS`` is simply empty and
everything else keeps working.

The two sets are drawn from non-overlapping ranges so a team cannot stumble
onto a hidden seed by scanning upward from 0 (asserted below when the admin
half is present).
"""

from __future__ import annotations

PUBLIC_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7)

try:
    from tier_d.admin.scoring.hidden_seeds import HIDDEN_SEEDS
except ImportError:  # student handout: admin/ deleted
    HIDDEN_SEEDS: tuple[int, ...] = ()

N_TREES: int = 16
TRAIN_BUDGET_EPISODES: int = 3000

if HIDDEN_SEEDS:
    assert not (set(PUBLIC_SEEDS) & set(HIDDEN_SEEDS)), "INVARIANT 5: seed sets overlap"
