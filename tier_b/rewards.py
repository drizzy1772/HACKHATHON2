"""Диспетчер reward-ів tier_b.

Середовище (``envs/pirouette_aviary.py``) і тести імпортують ``tier_b.rewards``;
реальна реалізація живе в одному з двох місць:

  • організаторська копія: ``tier_b/admin/rewards_solution.py`` (референс);
  • учнівська роздача (admin/ видалено): ``tier_b/scaffold/student_rewards.py``.

Примусово перемкнутися на учнівську реалізацію (щоб потренувати СВІЙ reward
навіть в організаторській копії):

    TIER_B_STUDENT_REWARDS=1 conda run -n drones python tier_b/train_ppo.py ...
"""

from __future__ import annotations

import os

if os.environ.get("TIER_B_STUDENT_REWARDS") == "1":
    from tier_b.scaffold.student_rewards import *  # noqa: F401,F403
else:
    try:
        from tier_b.admin.rewards_solution import *  # noqa: F401,F403
    except ImportError:  # учнівська роздача: admin/ видалено
        from tier_b.scaffold.student_rewards import *  # noqa: F401,F403
