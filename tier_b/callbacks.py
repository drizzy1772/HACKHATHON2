"""Callback-и: курикулум (success-rate у вікні → рівень вище) + метрики у TB.

Рівень зараховується лише епізодам ПОТОЧНОГО рівня — після переходу вікно
фактично обнуляється саме собою (епізоди старого рівня не рахуються)."""

from __future__ import annotations

import collections

from stable_baselines3.common.callbacks import BaseCallback

N_LEVELS = 9


class CurriculumCallback(BaseCallback):

    def __init__(self, start_level: int = 0, advance_window: int = 200,
                 advance_rate: float = 0.7, n_levels: int = N_LEVELS,
                 verbose: int = 1):
        super().__init__(verbose)
        self.level = start_level
        self.advance_window = advance_window
        self.advance_rate = advance_rate
        self.n_levels = n_levels
        self._episodes = collections.deque(maxlen=4 * advance_window)
        self._level_up_steps: list[tuple[int, int]] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" not in info:
                continue
            self._episodes.append({
                "success": bool(info.get("success", False)),
                "collision": bool(info.get("collision", False)),
                "jam": bool(info.get("jam", False)),
                "out_of_bounds": bool(info.get("out_of_bounds", False)),
                "level": int(info.get("level", -1)),
                "len": int(info["episode"]["l"]),
            })
        self._maybe_advance()
        return True

    def _maybe_advance(self) -> None:
        cur = [e for e in self._episodes if e["level"] == self.level]
        cur = cur[-self.advance_window:]
        if len(cur) < self.advance_window or self.level >= self.n_levels - 1:
            return
        rate = sum(e["success"] for e in cur) / len(cur)
        if rate >= self.advance_rate:
            self.level += 1
            self.training_env.env_method("set_level", self.level)
            self._level_up_steps.append((self.num_timesteps, self.level))
            if self.verbose:
                print(f"[curriculum] крок {self.num_timesteps}: success_rate "
                      f"{rate:.2f} → рівень {self.level}")

    def _on_rollout_end(self) -> None:
        cur = [e for e in self._episodes if e["level"] == self.level]
        cur = cur[-self.advance_window:]
        if cur:
            n = len(cur)
            self.logger.record("rollout/success_rate",
                               sum(e["success"] for e in cur) / n)
            self.logger.record("rollout/collision_rate",
                               sum(e["collision"] for e in cur) / n)
            self.logger.record("rollout/jam_rate",
                               sum(e["jam"] for e in cur) / n)
            self.logger.record("rollout/oob_rate",
                               sum(e["out_of_bounds"] for e in cur) / n)
            self.logger.record("rollout/mean_ep_len",
                               sum(e["len"] for e in cur) / n)
        self.logger.record("curriculum/level", self.level)
