# КАРТКА ЗАДАЧІ · tier_b — «CTBR Pirouette» (PPO vs SAC, справжня фізика)

Читай разом із `prompts/SYSTEM_PROMPT.md` — там протокол; тут лише факти задачі.

## Задача

Квадрокоптер Crazyflie 2.X (gym-pybullet-drones, PyBullet) летить 12 м крізь
три «ворота» (пари дерев), щілина звужується curriculum-ом від 1.20 м до
0.12 м при ширині корпусу 0.24 м — верхні рівні прохідні лише з креном. Дія
агента — CTBR: `[колективна тяга c ∈ [0, 2g], ω_x, ω_y, ω_z]`. Команда збирає
reward-функцію, обирає гіперпараметри PPO і SAC та тренує; рівні L6–L8 —
відкрита дослідницька частина.

## Можна редагувати (і тільки це)

- `tier_b/scaffold/student_rewards.py` — дві TODO-діри:
  `shaping()` і `step_reward()`. Константи в цьому файлі НЕ чіпати.
- `tier_b/scaffold/student_config.yaml` — гіперпараметри `ppo:` та `sac:`
  (усі `null # TODO`). Решту секцій (sim/reward/curriculum/…) НЕ чіпати —
  вони звіряються тестом.

## Контракт (сигнатури не змінювати)

```python
shaping(phi_new, phi_old, gamma=GAMMA, terminal=False) -> float
    # F = γ·Φ(s′) − Φ(s), і Φ(термінал) ≡ 0 (інваріант 4)

step_reward(phi_prog_new, phi_prog_old, phi_gate_new, phi_gate_old,
            z, omega, terminal, collided=False, depth=0.0,
            gate_passed=False, finished=False, ...) -> float
    # збірка: step_cost + k_prog·shaping(прогрес) + k_gate·shaping(ворота)
    #         + altitude_penalty + omega_penalty + бонуси/штрафи за прапорцями
```

`terminal=True` лише для справжніх терміналів (колізія/фініш); jam і таймаут —
truncation. Пастки, на які ти вказуєш питаннями: «рента» за стояння при γ<1 у
shaping; банкування потенціалу об дерево, якщо Φ(термінал)≠0; reward-хаки,
розібрані в `tier_b/README.md` (прочитай і поясни — але не розв'язуй за них).

## Запуск (усе важке — ЛИШЕ у conda-середовищі `drones`)

```bash
pytest tier_b            # numpy-частина тестів працює й без conda

TIER_B_STUDENT_REWARDS=1 conda run -n drones python tier_b/train_ppo.py \
    --config tier_b/scaffold/student_config.yaml --steps 50000    # smoke
TIER_B_STUDENT_REWARDS=1 conda run -n drones python tier_b/train_sac.py \
    --config tier_b/scaffold/student_config.yaml                  # повні

conda run -n drones python tier_b/scaffold/validate_team_b.py \
    --model tier_b/runs/sac/seed0/final_model.zip --level 5       # самоперевірка
```

## Заборонено

- `tier_b/admin/**` (референсний reward, розв'язок, приховані сіди, тести) —
  не відкривати й не цитувати.
- Змінювати `tier_b/envs/**`, `tier_b/rewards.py` (диспетчер),
  `tier_b/config.yaml`, тренувальні скрипти, `tier_b/scoring/seeds.py`.
- Тренуватися чи тюнитися на сідах 900000+ (PUBLIC — лише для самоперевірки)
  і будь-яких прихованих сідах.
