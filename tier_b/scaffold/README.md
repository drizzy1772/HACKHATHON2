# Tier-B · Завдання команди

Ваш дрон має пролетіти 12 м крізь три «ворота» (пари дерев), чия щілина за
curriculum звужується від 1.20 м до 0.12 м при ширині корпусу 0.24 м. Верхні
рівні прохідні **лише з креном** — саме цю поведінку агент має відкрити сам.
Фізика справжня (gym-pybullet-drones, Crazyflie 2.X), дія — CTBR:
`[колективна тяга, ω_x, ω_y, ω_z]`. Деталі — у `tier_b/CONCEPT.md` і
`tier_b/README.md`.

## Що заповнює команда (дві діри)

1. **`tier_b/scaffold/student_rewards.py`** — reward-функція:
   - `shaping()` — потенціал-орієнтований shaping (інваріант 4: `F = γ·Φ(s′) − Φ(s)`,
     `Φ(термінал) = 0`);
   - `step_reward()` — збірка повної винагороди кроку з готових членів.

   Константи не чіпайте — вони дзеркалять `config.yaml` і звіряються тестом.

2. **`tier_b/scaffold/student_config.yaml`** — гіперпараметри PPO та SAC
   (усі `null # TODO`). Обґрунтування вибору — частина захисту перед журі.

## Як тренувати

Все важке живе ТІЛЬКИ в conda-середовищі `drones` (див. `tier_b/README.md`
щодо встановлення):

```bash
# smoke-прогін PPO на вашому конфізі та вашому reward
TIER_B_STUDENT_REWARDS=1 conda run -n drones python tier_b/train_ppo.py \
    --config tier_b/scaffold/student_config.yaml --steps 50000

# повний SAC (годинами)
TIER_B_STUDENT_REWARDS=1 conda run -n drones python tier_b/train_sac.py \
    --config tier_b/scaffold/student_config.yaml
```

`TIER_B_STUDENT_REWARDS=1` гарантує, що тренується саме ВАШ
`student_rewards.py` (в учнівській роздачі він і так єдиний).

## Самоперевірка

```bash
conda run -n drones python tier_b/scaffold/validate_team_b.py \
    --model tier_b/runs/sac/seed0/final_model.zip --level 5
```

Прогін по PUBLIC_SEEDS (held-out master-сіди 900000+ — ніколи не бачені у
тренуванні). Фінальний залік — та сама схема на прихованих сідах; здаєте
`final_model.zip`.

## Підказки

- Спершу прочитайте у `tier_b/README.md` розбір трьох виміряних reward-хаків —
  кожен із них «вистрелить», якщо зібрати `step_reward` неправильно.
- Рівні L6–L8 (0.20/0.16/0.12 м) станом на старт хакатону не розв'язані —
  це відкрита дослідницька частина треку (warm-start з чекпоінта L5,
  статистика по сідах, обережний reward-інжиніринг).
