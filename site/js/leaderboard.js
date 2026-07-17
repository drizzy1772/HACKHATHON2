// leaderboard.js — лендинг: плитки-треки (вибір напрямку) + лідерборди.
// Глобальний рейтинг (із престиж-множником) + перемикач per-track.

import { scoreTeam, autoBreakdown } from "./scoring.js";

const $ = (s, r = document) => r.querySelector(s);
const el = (tag, cls, txt) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (txt != null) e.textContent = txt;
  return e;
};

let DATA, TEAMS, mode = "global";

init();

async function init() {
  [DATA, TEAMS] = await Promise.all([
    fetch("data/achievements.json").then((r) => r.json()),
    fetch("data/teams.json").then((r) => r.json()),
  ]);
  renderTracks();
  renderModeSwitch();
  renderBoard();
}

function renderTracks() {
  const box = $("#tracks");
  box.innerHTML = "";
  for (const [key, t] of Object.entries(DATA.tiers)) {
    const nTeams = TEAMS.teams.filter((x) => x.track === key).length;
    const nodes = t.nodes.length;
    const a = el("a", "track-card");
    a.href = "index.html";
    a.dataset.level = t.level_label;
    a.append(
      el("div", "track-key", key),
      el("div", "track-name", t.title.replace(/^Tier [A-D] · /, "")),
      el("div", "track-level", t.level_label),
      el("p", "track-summary", t.summary),
    );
    const foot = el("div", "track-foot");
    foot.append(
      el("span", null, `${nodes} ачівок`),
      el("span", null, `×${t.multiplier} престиж`),
      el("span", null, `${nTeams} команд`),
    );
    a.append(foot);
    box.append(a);
  }
}

function renderModeSwitch() {
  const bar = $("#board-switch");
  bar.innerHTML = "";
  const mk = (key, label) => {
    const b = el("button", "switch-chip", label);
    b.dataset.mode = key;
    b.onclick = () => { mode = key; renderBoard(); };
    return b;
  };
  bar.append(mk("global", "Глобальний рейтинг"));
  for (const key of Object.keys(DATA.tiers)) bar.append(mk("track:" + key, `Tier ${key}`));
}

function renderBoard() {
  document.querySelectorAll(".switch-chip").forEach((b) => b.classList.toggle("on", b.dataset.mode === mode));

  let rows = TEAMS.teams.map((team) => ({ team, s: scoreTeam(team, DATA) }));
  let header, sortKey;

  if (mode === "global") {
    header = "Глобальний бал (× престиж-множник)";
    sortKey = (r) => r.s.global;
  } else {
    const tk = mode.split(":")[1];
    rows = rows.filter((r) => r.team.track === tk);
    header = `${DATA.tiers[tk].title} — внутрішньотрековий бал`;
    sortKey = (r) => r.s.trackTotal;
  }
  rows.sort((a, b) => sortKey(b) - sortKey(a));

  $("#board-title").textContent = header;
  const tb = $("#board-body");
  tb.innerHTML = "";
  rows.forEach((r, i) => {
    const tr = el("tr");
    if (i < 3) tr.classList.add("top", "top-" + (i + 1));
    const tier = DATA.tiers[r.team.track];
    const bd = autoBreakdown(r.team.track, r.team.auto)
      .map(([k, v]) => `${k} ${v >= 0 ? "+" : ""}${v}`).join(" · ");
    tr.append(
      cell(el("span", "rank", `${i + 1}`)),
      cell(el("span", "team-name", r.team.name)),
      cell(trackTag(r.team.track, tier.level_label)),
      cell(document.createTextNode(`${r.s.auto}`), "num", bd),
      cell(document.createTextNode(`${r.s.ach}`), "num", `${r.team.unlocked.length} ачівок`),
      cell(el("b", null, mode === "global" ? `${r.s.global}` : `${r.s.trackTotal}`), "num strong",
        mode === "global" ? `(${r.s.trackTotal} × ${r.s.mult})` : ""),
    );
    tb.append(tr);
  });
}

function cell(node, cls, title) {
  const td = el("td", cls);
  td.append(node);
  if (title) td.title = title;
  return td;
}

function trackTag(key, level) {
  const s = el("span", "track-tag", `${key} · ${level}`);
  s.dataset.level = level;
  return s;
}
