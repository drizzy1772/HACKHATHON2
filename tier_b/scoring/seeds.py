"""Seed discipline для tier_b. INVARIANT 5: hidden seeds ніколи не в тренуванні.

``PUBLIC_SEEDS`` — для самоперевірки команд (validate_team_b.py). Це held-out
master-сіди зі зсуву 900000+ (узгоджено з ``eval.master_seed_offset`` у
config.yaml) — тренувальні master-сіди малі (0, 1, 2, …), тож перетину немає
за побудовою.

``HIDDEN_SEEDS`` — виключно для фінального заліку; живуть у
``tier_b/admin/scoring/hidden_seeds.py``, якого немає в учнівській роздачі
(там цей кортеж порожній, і все інше працює).
"""

from __future__ import annotations

PUBLIC_SEEDS: tuple[int, ...] = tuple(range(900_000, 900_008))

try:
    from tier_b.admin.scoring.hidden_seeds import HIDDEN_SEEDS
except ImportError:  # учнівська роздача: admin/ видалено
    HIDDEN_SEEDS: tuple[int, ...] = ()

if HIDDEN_SEEDS:
    assert not (set(PUBLIC_SEEDS) & set(HIDDEN_SEEDS)), "INVARIANT 5: seed sets overlap"
