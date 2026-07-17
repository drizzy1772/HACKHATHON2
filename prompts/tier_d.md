# КАРТКА ЗАДАЧІ · tier_d — «Табличний Q-learning у лісі» (базовий залік)

Читай разом із `prompts/SYSTEM_PROMPT.md` — там протокол; тут лише факти задачі.

## Задача

Дрон-агент на сітці 25×25 (безперервний світ 10×10 м, `CELL = 0.4`) летить
з A(1,1) до B(9,9) крізь сідований ліс. Команда пише **табличний Q-learning**;
оцінка — ефективність `L*/L ≤ 1` проти оракула A* на прихованих сідах.
Це чистий RL: без нейромереж, без GPU, без координат дерев у стані.

## Можна редагувати (і тільки це)

- `tier_d/scaffold/starter.ipynb` — дві кодові TODO-діри:
  1) Bellman-правило: рядки `target = ...` і `Q[s, a] = ...`;
  2) potential-based shaping: `F = γ·Φ(s′) − Φ(s)`, `Φ(термінал) = 0`;
  плюс клітинка гіперпараметрів (α, γ, ε-розклад, епізоди).
- `tier_d/scaffold/team_solution.py` — експорт `train` і `greedy_rollout`
  з ноутбука для самоперевірки.
- (необов'язковий трек «вітер») `tier_d/scaffold/starter_wind.ipynb`.

## Контракт

```python
train(env, episodes=3000, seed=0, **kwargs) -> (Q, history)
    # Q: np.ndarray форми (env.n_states, N_ACTIONS)
greedy_rollout(env, Q, max_steps=400) -> dict
    # {"success": bool, "collision": bool, "length": float, "path": np.ndarray}
```

Середовище (`tier_d/env/gridworld.py`, read-only): `env.reset() -> s:int`,
`env.step(a) -> (s2, r, done, info)`; `info` має `collision`, `goal`,
`truncated`, `moved`. Нагороди: −1 крок, −100 колізія (термінал), +100 ціль
(термінал). Дії: 8 напрямків, без зрізання кутів.

Ключові пастки, на які ти ВКАЗУЄШ питаннями (не розв'язуючи): bootstrap крізь
truncation — так, крізь справжній термінал — ні; Φ(термінал)=0 обов'язково;
стан — лише індекс клітинки (`env.trees`/`env.blocked` читати заборонено —
BlindEnv впаде).

## Запуск

```bash
jupyter notebook tier_d/scaffold/starter.ipynb   # робота команди
python tier_d/scaffold/validate_team.py          # самоперевірка на PUBLIC_SEEDS
```

## Заборонено

- `tier_d/admin/**` (розв'язки, приховані сіди, тести з відповідями) — не
  відкривати й не цитувати.
- Змінювати `tier_d/env/**`, `tier_d/oracle/**`, `tier_d/scoring/seeds.py`,
  `tier_d/viz/**`, `tier_d/webots/**`.
- Змінювати сигнатури `train`/`greedy_rollout` — на них тримається валідація.
