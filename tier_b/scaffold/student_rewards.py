"""ФАЙЛ ВАШОЇ КОМАНДИ (Tier-B): reward-функція.

Дві діри позначені ``TODO(команда)``. Файл мусить *імпортуватися чисто* —
незаповнена діра падає лише при виклику, тож середовище стартує і без
розв'язку. Константи дзеркалять ``tier_b/config.yaml`` (розділ ``reward``) —
їх не чіпайте; ваша робота — зібрати з них функцію.

Що почитати перед заповненням:
  • інваріант 4 кіта: shaping лише потенціал-орієнтований, Φ(термінал) = 0;
  • tier_b/README.md — три виміряні reward-хаки та їх виправлення;
  • тести-якорі: tier_b/admin/tests/test_rewards.py (в організаторській копії).

Перевірка себе (numpy, без conda):
    python -c "from tier_b.scaffold.student_rewards import step_reward;
               print(step_reward(1,0, 1,0, 1.0, (0,0,0), False))"
"""

from __future__ import annotations

# --- дефолти = config.yaml["reward"] — НЕ ЗМІНЮВАТИ --------------------------
GAMMA = 0.99          # γ алгоритму (PPO/SAC)
SHAPING_GAMMA = 1.0   # телескопічна форма Φ′−Φ: стояння на місці дає рівно 0
K_PROG = 5.0
K_GATE = 2.0
COLLISION_BASE = -100.0
COLLISION_DEPTH_SCALE = -100.0
COLLISION_DEPTH_REF = 0.04
Z_MIN = 0.15
K_ALT = 10.0
K_OMEGA = 0.01
STEP_COST = -0.02
GATE_BONUS = 25.0
FINISH_BONUS = 100.0


def shaping(phi_new: float, phi_old: float, gamma: float = GAMMA,
            terminal: bool = False) -> float:
    """F = γ·Φ(s′) − Φ(s), і Φ(термінал) ≡ 0.

    Без обнулення Φ на терміналі гарантія Ng–Harada–Russell не діє на
    терміналі колізії (інваріант 4): агент зможе «банкувати» накопичений
    потенціал, розбиваючись.
    """
    # твоя логіка (як у tier_d): у терміналі майбутній потенціал = 0
    phi_next = 0.0 if terminal else phi_new
    return gamma * phi_next - phi_old


def collision_penalty(depth: float, base: float = COLLISION_BASE,
                      scale: float = COLLISION_DEPTH_SCALE,
                      ref: float = COLLISION_DEPTH_REF) -> float:
    """Градуйований урон: 0 < depth → base + scale·min(1, depth/ref) (обидва від'ємні)."""
    return base + scale * min(1.0, max(0.0, depth) / ref)


def altitude_penalty(z: float, z_min: float = Z_MIN, k_alt: float = K_ALT) -> float:
    return -k_alt * max(0.0, z_min - z)


def omega_penalty(omega, k_omega: float = K_OMEGA) -> float:
    ox, oy, oz = float(omega[0]), float(omega[1]), float(omega[2])
    return -k_omega * (ox * ox + oy * oy + oz * oz)


def step_reward(phi_prog_new: float, phi_prog_old: float,
                phi_gate_new: float, phi_gate_old: float,
                z: float, omega, terminal: bool,
                collided: bool = False, depth: float = 0.0,
                gate_passed: bool = False, finished: bool = False,
                gamma: float = SHAPING_GAMMA, k_prog: float = K_PROG,
                k_gate: float = K_GATE, step_cost: float = STEP_COST,
                gate_bonus: float = GATE_BONUS,
                finish_bonus: float = FINISH_BONUS, **kw) -> float:
    """Повна винагорода кроку. terminal=True лише для СПРАВЖНІХ терміналів
    (колізія/фініш) — jam і таймаут це truncation, там terminal=False
    і bootstrap зберігається (дисципліна terminal-vs-truncated кіта).

    Зберіть винагороду з членів вище:
      step_cost + k_prog·shaping(прогрес) + k_gate·shaping(ворота)
      + altitude_penalty + omega_penalty
      + gate_bonus (якщо gate_passed) + collision_penalty (якщо collided)
      + finish_bonus (якщо finished).
    Зважайте на kw-перевизначення: z_min/k_alt, k_omega,
    collision_base/collision_depth_scale/collision_depth_ref.
    """
    # база + два потенціал-shaping (той самий terminal!) + штрафи + умовні члени
    r = step_cost
    r += k_prog * shaping(phi_prog_new, phi_prog_old, gamma, terminal)
    r += k_gate * shaping(phi_gate_new, phi_gate_old, gamma, terminal)
    r += altitude_penalty(z, kw.get("z_min", Z_MIN), kw.get("k_alt", K_ALT))
    r += omega_penalty(omega, kw.get("k_omega", K_OMEGA))
    if gate_passed:
        r += gate_bonus
    if collided:
        r += collision_penalty(depth,
                               kw.get("collision_base", COLLISION_BASE),
                               kw.get("collision_depth_scale", COLLISION_DEPTH_SCALE),
                               kw.get("collision_depth_ref", COLLISION_DEPTH_REF))
    if finished:
        r += finish_bonus
    return r
