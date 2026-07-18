# КАРТКА ЗАДАЧІ · tier_c — «Політ крізь ліс» (реалістичний FPV-дрон у Blender)

Читай разом із `prompts/SYSTEM_PROMPT.md` — там протокол; тут лише факти задачі.

## Задача

Автономна навігація крізь процедурний ліс на арені 50×50 м з рельєфом: від
`md.start` до вантажівки-чекпоінта `md.checkpoints[0]`, оминаючи дерева, не
вище стелі (середня висота крон — вище = «Під дією РЕБ», дискваліфікація) і не
торкаючись рельєфу. Ліміт 90 с, цикл керування 30 Гц. Статуси: FINISHED /
COLLISION / DISQUALIFIED / TIMEOUT. Підхід класичний: глобальний план (A*) +
реактивне ухилення (APF за лідаром) + кінематичний автопілот.

## Можна редагувати
- tier_c/solution.py
- tier_c/game_env/scene.py
- tier_c/blender_manual.py
- tier_c/sim_headless.py
## Контракт (сигнатури не змінювати)

Вузький контракт: **(lidar_data, current_position, next_checkpoint) → (vx, vy, vz)**,
розгорнутий у чотири функції, які `sim_headless.py` викликає щотіка:

```python
find_path(md, cfg, cell_size=1.0) -> Optional[List[Vec2]]      # раз на старті
compute_desired_direction(pos, lidar, bin_angles, tracker, stuck,
                          t, p, boost_state) -> (dir_xy, boosted, carrot)
class StuckDetector:  update(t, x, y) -> bool                   # щотік
step_autopilot(state, direction_xy, terrain, dt, p,
               min_lidar=None) -> AutopilotState                # щотік
# + налаштовні дата-класи APFParams, AutopilotParams
```

Дано (read-only, можеш пояснювати): `astar2d.py` (occupancy grid,
world↔cell), `apf_controller.py` (`PathTracker`, carrot-chasing),
`kinematic_autopilot.py` (контракт `AutopilotState`: x,y,z,vx,vy,vz,yaw,
pitch,roll), `game_env/` (генератор мап із сідом, `lidar2d` — кутові біни
відстаней навколо дрона, `terrain.height_at(x, y)`).

Пастки, на які ти вказуєш питаннями: що робити, коли видно кілька дерев;
локальні мінімуми APF (притягання + відштовхування = нуль); стеля РЕБ проти
контурного польоту; застрягання без `StuckDetector`.

## Запуск

```bash
python tier_c/sim_headless.py        # headless-самоперевірка → out/autonomous_runs.json
/Applications/Blender.app/Contents/MacOS/Blender -P tier_c/blender_manual.py
#   або подвійний клік: tier_c/«Запустити гру.app»
python tier_c/metrics_report.py      # HTML-звіт останнього прогону
```

