"""Seeded forest generation. INVARIANT 1: same (n, seed) -> identical forest.

``mulberry32`` is the canonical 32-bit JS PRNG, ported exactly so a browser
Field Scanner and this module agree bit-for-bit. Reference JS::

    function mulberry32(a) {
      return function() {
        var t = a += 0x6D2B79F5;
        t = Math.imul(t ^ t >>> 15, t | 1);
        t ^= t + Math.imul(t ^ t >>> 7, t | 61);
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
      }
    }

This Python module is the frozen reference: ``tier_d/admin/tests/test_forest.py`` pins a
golden sequence, and ``tier_d/viz/field_scanner.html`` reproduces it in JS.
"""

from __future__ import annotations

import numpy as np

from tier_d.env.constants import (
    AB_CLEARANCE,
    BORDER_MARGIN,
    DOMAIN,
    GOAL_XY,
    MIN_TREE_SEP,
    START_XY,
)

_U32 = 0xFFFFFFFF


def _imul(a: int, b: int) -> int:
    """Math.imul: 32-bit signed multiply, returned as unsigned 32-bit."""
    return ((a & _U32) * (b & _U32)) & _U32


class Mulberry32:
    """Deterministic uniform [0,1) generator, byte-compatible with the JS form."""

    __slots__ = ("_a",)

    def __init__(self, seed: int) -> None:
        self._a = seed & _U32

    def __call__(self) -> float:
        self._a = (self._a + 0x6D2B79F5) & _U32
        t = self._a
        t = _imul(t ^ (t >> 15), t | 1)
        t = (t ^ (t + _imul(t ^ (t >> 7), t | 61))) & _U32
        return ((t ^ (t >> 14)) & _U32) / 4294967296.0


def make_forest(n: int, seed: int) -> np.ndarray:
    """Return an ``(m, 2)`` float64 array of tree centres, ``m <= n``.

    Rejection sampling under three constraints (see ``env.constants``):
    border margin, clearance from A/B, and minimum tree-to-tree separation.
    The attempt budget is fixed, so a dense request degrades deterministically
    to fewer trees rather than looping forever.
    """
    rng = Mulberry32(seed)
    lo, hi = BORDER_MARGIN, DOMAIN - BORDER_MARGIN
    span = hi - lo

    start = np.asarray(START_XY)
    goal = np.asarray(GOAL_XY)
    trees: list[np.ndarray] = []

    attempts = 0
    max_attempts = 200 * max(n, 1)
    while len(trees) < n and attempts < max_attempts:
        attempts += 1
        t = np.array([lo + rng() * span, lo + rng() * span])
        if np.hypot(*(t - start)) < AB_CLEARANCE:
            continue
        if np.hypot(*(t - goal)) < AB_CLEARANCE:
            continue
        if any(np.hypot(*(t - o)) < MIN_TREE_SEP for o in trees):
            continue
        trees.append(t)

    return np.array(trees, dtype=np.float64).reshape(-1, 2)
