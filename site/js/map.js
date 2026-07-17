// map.js — Minecraft-advancements карта всього змагання.
// Ядро в центрі; чотири гілки (A вгору, D вправо, B вниз, C вліво).
// Рамки — піксель-арт спрайти (canvas 14×14, масштаб ×4, як у грі):
// advancement = квадрат зі зрізаними кутами, goal = заокруглений, challenge = зубчастий.
// Жовтий = невиконано, сірий = виконано. Лінії білі з чорним піксельним обведенням.

const NODE = 70;   // базовий розмір рамки
const CORE = 84;   // ядро
const STEP = 100;  // крок між колонками (вглиб гілки) — щільно, як на референсі
const GAP = 88;    // крок між сусідами в колонці
const PAD = 170;   // поле навколо карти

const DIRS = { A: "up", D: "right", B: "down", C: "left" };
const DONE_KEY = "skyrun.mc.done";

const $ = (s, r = document) => r.querySelector(s);
const el = (tag, cls, txt) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (txt != null) e.textContent = txt;
  return e;
};

let DATA, BY_ID = new Map(), POS = new Map(), TIER_OF = new Map(), SPRITES = {};

const loadDone = () => new Set(JSON.parse(localStorage.getItem(DONE_KEY) || "[]"));
const saveDone = (s) => localStorage.setItem(DONE_KEY, JSON.stringify([...s]));

init();

async function init() {
  DATA = await fetch("data/achievements.json").then((r) => r.json());
  for (const [tk, tier] of Object.entries(DATA.tiers))
    for (const n of tier.nodes) { BY_ID.set(n.id, n); TIER_OF.set(n.id, tk); }
  buildSprites();
  paintGrass();
  layout();
  drawBoard();
  updateRating();
  centerView();
}

/* ══════════ рамки: спрайти, вирізані з assets/reference3.png ══════════ */

const FRAME_DIR = "assets/frames";
Object.assign(SPRITES, {
  story: {
    gold: `url(${FRAME_DIR}/advancement_gold.png)`,
    silver: `url(${FRAME_DIR}/advancement_silver.png)`,
  },
  goal: {
    gold: `url(${FRAME_DIR}/goal_gold.png)`,
    silver: `url(${FRAME_DIR}/goal_silver.png)`,
  },
  challenge: {
    gold: `url(${FRAME_DIR}/challenge_gold.png)`,
    silver: `url(${FRAME_DIR}/challenge_silver.png)`,
  },
  core: `url(${FRAME_DIR}/advancement_gold.png)`,
});

// розміри рамок за формою — як у грі: challenge більша за advancement
const SIZE = { story: 70, goal: 74, challenge: 84 };

// іконки-вовна за кольором статусу задачі (файли організатора в assets/)
const WOOL = {
  green: "assets/green wool.webp",
  yellow: "assets/orange wool.webp",
  purple: "assets/violet wool.webp",
  red: "assets/red wool.webp",
  blue: "assets/Blue_Wool.webp",
};

function buildSprites() {
  // прогрів кешу, щоб рамки з'являлись одразу
  for (const shape of ["story", "goal", "challenge"])
    for (const v of ["gold", "silver"]) {
      const img = new Image();
      img.src = SPRITES[shape][v].slice(4, -1);
    }
}

/* ══════════ фон: поле трав'яних блоків (великі блоки + піксельний шум усередині) ══════════ */

const BLOCK_PX = 64; // розмір одного блока на екрані

function paintGrass() {
  // База — текстура організатора assets/grass.png (один блок).
  // Обрізаємо вбудовану рамку блока (щоб не було ліній між блоками),
  // замощуємо тайл 8×8 з ВИПАДКОВИМ поворотом кожного блока і трохи освітлюємо.
  const vp = $("#viewport");
  const apply = (url, size) => {
    vp.style.backgroundImage = `url(${url})`;
    vp.style.backgroundSize = `${size}px ${size}px`;
    vp.style.backgroundRepeat = "repeat";
  };
  const img = new Image();
  img.onload = () => {
    const N = 8;                                   // блоків у тайлі
    const M = Math.round(Math.min(img.width, img.height) / 16); // зріз рамки ≈ 1 текстурний піксель
    const sw = img.width - 2 * M, sh = img.height - 2 * M;
    const cv = document.createElement("canvas");
    cv.width = cv.height = BLOCK_PX * N;
    const ctx = cv.getContext("2d");
    ctx.imageSmoothingEnabled = false;
    let seed = 987654321;
    const rnd = () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
    for (let by = 0; by < N; by++)
      for (let bx = 0; bx < N; bx++) {
        ctx.save();
        ctx.translate(bx * BLOCK_PX + BLOCK_PX / 2, by * BLOCK_PX + BLOCK_PX / 2);
        ctx.rotate(Math.floor(rnd() * 4) * Math.PI / 2);   // 0/90/180/270°
        ctx.drawImage(img, M, M, sw, sh, -BLOCK_PX / 2, -BLOCK_PX / 2, BLOCK_PX, BLOCK_PX);
        ctx.restore();
      }
    // трохи світліша текстура
    ctx.fillStyle = "rgba(255,255,255,.10)";
    ctx.fillRect(0, 0, cv.width, cv.height);
    apply(cv.toDataURL(), BLOCK_PX * N);
  };
  img.onerror = () => apply(grassBlockDataURL(), BLOCK_PX);
  img.src = "assets/grass.png";
}

// Один блок 16×16 (відтворення наданої текстури): зелений шум зі вкрапленнями
// темніших пікселів і ледь темнішим краєм. Кожен блок на екрані — ідентична копія.
function grassBlockDataURL() {
  const S = 16;
  const cv = document.createElement("canvas");
  cv.width = cv.height = S;
  const ctx = cv.getContext("2d");
  const shades = ["#5a9e4a", "#549646", "#4f8f42", "#5fa64e", "#579948", "#4a873d"];
  const dark = "#3f7a34";
  let seed = 42424242;
  const rnd = () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
  for (let y = 0; y < S; y++)
    for (let x = 0; x < S; x++) {
      let c = shades[Math.floor(rnd() * shades.length)];
      if (rnd() < 0.12) c = dark; // вкраплення темних пікселів
      ctx.fillStyle = c;
      ctx.fillRect(x, y, 1, 1);
      if (x === 0 || y === 0 || x === S - 1 || y === S - 1) {
        ctx.fillStyle = "rgba(0,0,0,.16)"; // ледь темніший край блока
        ctx.fillRect(x, y, 1, 1);
      }
    }
  return cv.toDataURL();
}

/* ══════════ розкладка: depth-колонки в 4 напрямки від ядра ══════════ */

function depthMap(tier) {
  const memo = new Map();
  const d = (id) => {
    if (memo.has(id)) return memo.get(id);
    const n = tier.nodes.find((x) => x.id === id);
    const v = n.parents.length ? Math.max(...n.parents.map(d)) + 1 : 0;
    memo.set(id, v);
    return v;
  };
  tier.nodes.forEach((n) => d(n.id));
  return memo;
}

function layout() {
  POS = new Map();
  POS.set("__core__", { x: 0, y: 0 });
  for (const [tk, tier] of Object.entries(DATA.tiers)) {
    const dir = DIRS[tk];
    const depth = depthMap(tier);
    const cols = new Map();
    tier.nodes.forEach((n) => {
      const d = depth.get(n.id);
      if (!cols.has(d)) cols.set(d, []);
      cols.get(d).push(n);
    });
    for (const [d, list] of cols) {
      list.forEach((n, i) => {
        const main = (d + 1) * STEP + CORE / 2;
        const cross = (i - (list.length - 1) / 2) * GAP;
        let x = 0, y = 0;
        if (dir === "right") { x = main; y = cross; }
        if (dir === "left")  { x = -main; y = cross; }
        if (dir === "down")  { y = main; x = cross; }
        if (dir === "up")    { y = -main; x = cross; }
        POS.set(n.id, { x: Math.round(x), y: Math.round(y) });
      });
    }
  }
  let minX = 1e9, minY = 1e9, maxX = -1e9, maxY = -1e9;
  for (const p of POS.values()) {
    minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x);
    minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y);
  }
  for (const p of POS.values()) { p.x += PAD - minX; p.y += PAD - minY; }
  layout.w = maxX - minX + PAD * 2;
  layout.h = maxY - minY + PAD * 2;
}

function tierBBox(tk) {
  let minX = 1e9, minY = 1e9, maxX = -1e9, maxY = -1e9;
  for (const n of DATA.tiers[tk].nodes) {
    const p = POS.get(n.id);
    minX = Math.min(minX, p.x - NODE / 2); maxX = Math.max(maxX, p.x + NODE / 2);
    minY = Math.min(minY, p.y - NODE / 2); maxY = Math.max(maxY, p.y + NODE / 2);
  }
  return { minX, minY, maxX, maxY };
}

/* ══════════ рендер ══════════ */

function drawBoard() {
  const board = $("#board");
  board.style.width = layout.w + "px";
  board.style.height = layout.h + "px";
  board.innerHTML = "";

  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("width", layout.w);
  svg.setAttribute("height", layout.h);
  svg.id = "wires";
  board.append(svg);

  // маршрути ліній: спершу ВСІ чорні підкладки, потім ВСІ білі серцевини —
  // так стики Т-подібних розгалужень зливаються, як у грі
  const ds = [];
  for (const [, tier] of Object.entries(DATA.tiers)) {
    tier.nodes.filter((n) => !n.parents.length)
      .forEach((n) => ds.push(wireD(POS.get("__core__"), POS.get(n.id))));
    tier.nodes.forEach((n) =>
      n.parents.forEach((p) => ds.push(wireD(POS.get(p), POS.get(n.id)))));
  }
  for (const cls of ["wire-out", "wire-in"])
    for (const d of ds) {
      const p = document.createElementNS("http://www.w3.org/2000/svg", "path");
      p.setAttribute("d", d);
      p.setAttribute("class", cls);
      svg.append(p);
    }

  // ядро
  const core = el("button", "node core");
  const cp = POS.get("__core__");
  place(core, cp, CORE);
  core.style.backgroundImage = SPRITES.core;
  const coreTip = () => tip(core, "AI SkyRun", "Старт змагання. Обери гілку свого треку: A вгору, D вправо, B вниз, C вліво.");
  core.addEventListener("mouseenter", coreTip);
  core.addEventListener("mouseleave", hideTip);
  core.addEventListener("click", coreTip);
  board.append(core);

  // ноди
  for (const [, tier] of Object.entries(DATA.tiers))
    for (const n of tier.nodes) {
      const b = el("button", "node");
      b.dataset.id = n.id;
      b.dataset.shape = n.category;
      place(b, POS.get(n.id), SIZE[n.category] || NODE);
      const slot = el("span", "slot");
      slot.style.backgroundImage = `url("${WOOL[n.color]}")`;
      b.append(slot);
      b.addEventListener("mouseenter", () => tip(b, n.title, n.short, n));
      b.addEventListener("mouseleave", hideTip);
      b.addEventListener("click", () => openModal(n));
      board.append(b);
    }

  // підписи гілок — за межами bbox гілки, щоб не накладались на ачівки
  for (const [tk, tier] of Object.entries(DATA.tiers)) {
    const bb = tierBBox(tk);
    const lbl = el("div", "branch-label", `TIER ${tk} · ${tier.level_label} ×${tier.multiplier}`);
    const cx = Math.round((bb.minX + bb.maxX) / 2);
    if (DIRS[tk] === "down") {
      lbl.style.left = cx + "px";
      lbl.style.top = bb.maxY + 18 + "px";
    } else {
      lbl.style.left = cx + "px";
      lbl.style.top = bb.minY - 46 + "px";
    }
    board.append(lbl);
  }

  refreshStates();
}

function place(elm, p, size) {
  elm.style.left = p.x - size / 2 + "px";
  elm.style.top = p.y - size / 2 + "px";
  elm.style.width = size + "px";
  elm.style.height = size + "px";
}

// елбоу-маршрут: виходить із центру батька, входить у центр дитини
function wireD(a, b) {
  const dx = b.x - a.x, dy = b.y - a.y;
  if (Math.abs(dx) >= Math.abs(dy)) {
    const mx = Math.round(a.x + dx / 2);
    return `M ${a.x} ${a.y} L ${mx} ${a.y} L ${mx} ${b.y} L ${b.x} ${b.y}`;
  }
  const my = Math.round(a.y + dy / 2);
  return `M ${a.x} ${a.y} L ${a.x} ${my} L ${b.x} ${my} L ${b.x} ${b.y}`;
}

/* ══════════ стани ══════════ */

function stateOf(n, done) {
  // без «заблокованих»: усе невиконане — золоте, виконане — срібне
  return done.has(n.id) ? "done" : "open";
}

function refreshStates() {
  const done = loadDone();
  document.querySelectorAll(".node[data-id]").forEach((b) => {
    const n = BY_ID.get(b.dataset.id);
    const st = stateOf(n, done);
    b.dataset.state = st;
    b.style.backgroundImage = SPRITES[n.category][st === "done" ? "silver" : "gold"];
  });
}

/* ══════════ тултип ══════════ */

function tip(nodeEl, title, text, n) {
  const t = $("#tip");
  t.innerHTML = "";
  t.append(el("div", "tip-title", title));
  if (n) {
    const c = DATA.colors[n.color];
    const meta = el("div", "tip-meta");
    const sw = el("span", "tip-sw");
    sw.style.background = c.hex;
    meta.append(sw, el("span", null, `${c.label} · ${n.sp} SP · ${n.scoring}`));
    t.append(meta);
  }
  t.append(el("div", "tip-text", text));
  t.style.display = "block";
  const br = nodeEl.getBoundingClientRect();
  const tw = t.offsetWidth, th = t.offsetHeight;
  let x = br.right + 10, y = br.top;
  if (x + tw > innerWidth - 8) x = br.left - tw - 10;
  if (y + th > innerHeight - 8) y = innerHeight - th - 8;
  t.style.left = Math.max(8, Math.round(x)) + "px";
  t.style.top = Math.max(8, Math.round(y)) + "px";
}
const hideTip = () => ($("#tip").style.display = "none");

/* ══════════ модалка ══════════ */

function openModal(n) {
  hideTip();
  const done = loadDone();
  const tk = TIER_OF.get(n.id);
  const tier = DATA.tiers[tk];
  const c = DATA.colors[n.color];
  $("#m-title").textContent = n.title;
  $("#m-tier").textContent = `TIER ${tk} · ${tier.level_label}`;
  const sw = $("#m-color");
  sw.textContent = c.label;
  sw.style.background = c.hex;
  $("#m-sp").textContent = `${n.sp} SP · скоринг: ${n.scoring}`;
  const st = stateOf(n, done);
  $("#m-state").textContent = { done: "✔ Виконано", open: "Доступно" }[st];
  $("#m-state").dataset.state = st;
  const body = $("#m-body");
  body.innerHTML = "";
  n.body.split("\n\n").forEach((p) => body.append(el("p", null, p)));
  $("#m-metric").textContent = n.metric || "—";
  $("#m-parents").textContent = n.parents.length
    ? "Відкривається після: " + n.parents.map((id) => BY_ID.get(id).title).join(", ")
    : "Коренева ачівка гілки";
  const btn = $("#m-toggle");
  btn.textContent = done.has(n.id) ? "Позначити невиконаною" : "Позначити виконаною";
  btn.onclick = () => {
    const d = loadDone();
    d.has(n.id) ? d.delete(n.id) : d.add(n.id);
    saveDone(d);
    refreshStates();
    updateRating();
    openModal(n);
  };
  $("#modal").dataset.open = "1";
}
$("#m-close").addEventListener("click", () => ($("#modal").dataset.open = "0"));
$("#modal").addEventListener("click", (e) => { if (e.target === $("#modal")) $("#modal").dataset.open = "0"; });
addEventListener("keydown", (e) => { if (e.key === "Escape") $("#modal").dataset.open = "0"; });

/* ══════════ рейтинг ══════════ */

function updateRating() {
  const done = loadDone();
  let raw = 0, weighted = 0;
  const parts = [];
  for (const [tk, tier] of Object.entries(DATA.tiers)) {
    const got = tier.nodes.filter((n) => done.has(n.id)).reduce((s, n) => s + n.sp, 0);
    const total = tier.nodes.reduce((s, n) => s + n.sp, 0);
    raw += got;
    weighted += got * tier.multiplier;
    parts.push(`${tk} ${got}/${total}`);
  }
  $("#rating-main").textContent = `Твій рейтинг: ${raw} SP`;
  $("#rating-weighted").textContent = `З престижем: ${Math.round(weighted)} SP`;
  $("#rating-tiers").textContent = parts.join("  ·  ");
}

function centerView() {
  const vp = $("#viewport");
  const c = POS.get("__core__");
  vp.scrollLeft = c.x - vp.clientWidth / 2;
  vp.scrollTop = c.y - vp.clientHeight / 2;
}
