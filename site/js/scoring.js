// scoring.js — формули SkyPoints. Дзеркалять §1 плану та auto_formula кожного тіра.
// Гібрид: SP_auto (метрики harness) + SP_achievements (розблоковані ачівки).
// Глобальний бал = (SP_auto + SP_achievements) × tier_multiplier.

const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));

// ── Авто-бали за треком ────────────────────────────────────────────────
// Кожна функція приймає team.auto і повертає ціле SP (0..~1000).

export const AUTO = {
  // D: 400·success + 200·(1−collision) + 300·(L*/L) + 100·sample_eff
  D(a = {}) {
    const success = a.success ?? 0;
    const collision = a.collision ?? 1;
    const opt = a.optimality ?? 0;      // L*/L ∈ [0,1]
    const eff = a.sample_eff ?? 0;       // 0..1
    return Math.round(400 * success + 200 * (1 - collision) + 300 * opt + 100 * eff);
  },

  // C: база за статусом фінішу − штрафи + бонус (час, гладкість, мало застрягань)
  C(a = {}) {
    const base = { FINISHED: 600, TIMEOUT: 200, COLLISION: 100, DISQUALIFIED: 50 }[a.status] ?? 0;
    const timeBonus = 250 * clamp(a.time_norm ?? 0, 0, 1);      // 1 = найшвидше
    const smoothBonus = 150 * clamp(a.smoothness ?? 0, 0, 1);
    const stuckPenalty = 20 * (a.stuck_count ?? 0);
    return Math.round(clamp(base + timeBonus + smoothBonus - stuckPenalty, 0, 1000));
  },

  // B: досягнутий рівень L0..L8 (×100) + бонус за крен + бонус за чесне порівняння
  B(a = {}) {
    const levelSP = 100 * clamp(a.level ?? 0, 0, 8);            // до 800
    const bankBonus = Math.min(150, (a.bank_deg ?? 0) * 3);     // до 150
    const fairBonus = a.both_algos ? 50 : 0;
    return Math.round(clamp(levelSP + bankBonus + fairBonus, 0, 1000));
  },

  // A: трекінг (нижчий RMSE → більше), успіхи/10, і bounty за «unknown» результат
  A(a = {}) {
    const ref = a.rmse_ref || 0.3;
    const track = clamp(400 * (1 - (a.rmse ?? ref) / ref), 0, 400);
    const succ = 30 * clamp(a.successes ?? 0, 0, 10);           // до 300
    const bounty = a.bounty_sp ?? 0;                            // суддівський bounty
    return Math.round(clamp(track + succ + bounty, 0, 1000));
  },
};

// Людиночитний розклад авто-балів (для деталей у лідерборді)
export function autoBreakdown(track, a = {}) {
  switch (track) {
    case "D": return [
      ["success", Math.round(400 * (a.success ?? 0))],
      ["no-collision", Math.round(200 * (1 - (a.collision ?? 1)))],
      ["L*/L", Math.round(300 * (a.optimality ?? 0))],
      ["sample-eff", Math.round(100 * (a.sample_eff ?? 0))],
    ];
    case "C": return [
      ["status", { FINISHED: 600, TIMEOUT: 200, COLLISION: 100, DISQUALIFIED: 50 }[a.status] ?? 0],
      ["time", Math.round(250 * clamp(a.time_norm ?? 0, 0, 1))],
      ["smoothness", Math.round(150 * clamp(a.smoothness ?? 0, 0, 1))],
      ["stuck", -20 * (a.stuck_count ?? 0)],
    ];
    case "B": return [
      ["level L" + (a.level ?? 0), 100 * clamp(a.level ?? 0, 0, 8)],
      ["bank " + (a.bank_deg ?? 0) + "°", Math.min(150, (a.bank_deg ?? 0) * 3)],
      ["fair compare", a.both_algos ? 50 : 0],
    ];
    case "A": {
      const ref = a.rmse_ref || 0.3;
      return [
        ["tracking", Math.round(clamp(400 * (1 - (a.rmse ?? ref) / ref), 0, 400))],
        ["successes", 30 * clamp(a.successes ?? 0, 0, 10)],
        ["bounty", a.bounty_sp ?? 0],
      ];
    }
    default: return [];
  }
}

// ── Суддівські бали: сума SP розблокованих ачівок ──────────────────────
export function achievementSP(tier, unlocked = []) {
  const byId = new Map(tier.nodes.map((n) => [n.id, n]));
  return (unlocked || []).reduce((s, id) => s + (byId.get(id)?.sp || 0), 0);
}

// ── Повний підрахунок для команди ──────────────────────────────────────
export function scoreTeam(team, data) {
  const tier = data.tiers[team.track];
  const auto = (AUTO[team.track] || (() => 0))(team.auto);
  const ach = achievementSP(tier, team.unlocked);
  const mult = tier.multiplier ?? 1;
  const trackTotal = auto + ach;                 // внутрішньотрековий бал
  const global = Math.round(trackTotal * mult);  // глобальний бал із престиж-множником
  return { auto, ach, trackTotal, mult, global, track: team.track };
}
