# -*- coding: utf-8 -*-
"""
HTML-звіт якості автономного алгоритму (A*+APF+кінематичний автопілот):
  1. Інтерактивна 2D-мапа ОСТАННЬОГО прогону (дерева/перешкоди/чекпоінт/
     траєкторія) — панорама/зум мишею, без bpy/Blender-в'юпорта.
  2. Графіки того ж прогону в часі (швидкість, висота, дистанція до перешкод).
  3. Таблиця + трендові графіки ПО ВСІХ збережених прогонах
     (sim_headless.load_run_history() — персистентно, з попередніх сесій теж).

Чиста функція рендерингу (без bpy) — те саме розділення, що й sim_headless.py:
дані рахує/зберігає sim_headless, тут лише візуалізація в один self-contained
HTML-файл (inline CSS/JS, без CDN — відкривається офлайн у будь-якому браузері).
"""

from __future__ import annotations

import json

from game_env import STATUS_FINISHED, STATUS_COLLISION, STATUS_DISQUALIFIED, STATUS_TIMEOUT

_STATUS_COLORS = {
    STATUS_FINISHED: "#5fae7d",
    STATUS_COLLISION: "#e2584c",
    STATUS_DISQUALIFIED: "#e2b34c",
    STATUS_TIMEOUT: "#9aa1ac",
}
_STATUS_LABELS_UK = {
    STATUS_FINISHED: "Фініш",
    STATUS_COLLISION: "Зіткнення",
    STATUS_DISQUALIFIED: "Дискваліфіковано",
    STATUS_TIMEOUT: "Тайм-аут",
}


def build_report_html(map_dict=None, frames=None, history=None,
                      cp_radius=None, drone_radius=None, apf_field=None) -> str:
    """map_dict — MapData.to_dict() останнього прогону (або None, якщо ще
    жодного автономного прогону не було в цій сесії); frames — result["frames"]
    того ж прогону; history — список записів sim_headless.load_run_history()
    (легкі підсумкові метрики з КОЖНОГО прогону, персистентно з out/). apf_field —
    {"xs":[...], "ys":[...], "mix":[[...]], "target":[x,y]} (той самий
    потенціал, що й у _draw_apf_map blender_manual.py, тут — перемикач-шар на
    ТІЙ САМІЙ мапі, а не окрема секція) або None, якщо мапи ще не було."""
    payload = {
        "map": map_dict,
        "frames": frames,
        "history": history or [],
        "cpRadius": cp_radius,
        "droneRadius": drone_radius,
        "apfField": apf_field,
        "statusColors": _STATUS_COLORS,
        "statusLabels": _STATUS_LABELS_UK,
    }
    data_json = json.dumps(payload, ensure_ascii=False)
    return _TEMPLATE.replace("__DATA_JSON__", data_json)


_TEMPLATE = r"""<!doctype html>
<html lang="uk">
<head>
<meta charset="utf-8">
<title>Метрики автономного алгоритму</title>
<style>
:root{
  --bg:#0f1310; --panel:#161c15; --card:#1b2219; --line:rgba(230,232,224,.12);
  --ink:#e7e9e2; --ink-soft:rgba(231,233,226,.66); --ink-faint:rgba(231,233,226,.42);
  --accent:#5fae7d; --accent-soft:rgba(95,174,125,.16);
  --warn:#e2b34c; --warn-soft:rgba(226,179,76,.16);
  --bad:#e2584c; --bad-soft:rgba(226,88,76,.16);
  --neutral:#9aa1ac; --neutral-soft:rgba(154,161,172,.16);
  --blue:#6fa8dc;
}
*{box-sizing:border-box;}
html,body{margin:0;padding:0;background:var(--bg);color:var(--ink);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;}
.wrap{max-width:1180px;margin:0 auto;padding:28px 22px 80px;}
h1{font-size:24px;margin:0 0 6px;font-weight:700;}
h2{font-size:17px;margin:0 0 4px;font-weight:700;}
.sub{color:var(--ink-soft);font-size:13.5px;margin:0 0 22px;}
.stats{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:28px;}
.stat{background:var(--card);border:1px solid var(--line);border-radius:9px;
  padding:9px 14px;font-size:13px;color:var(--ink-soft);}
.stat b{display:block;font-size:18px;color:var(--ink);font-variant-numeric:tabular-nums;}
section{margin-bottom:38px;}
.section-head{display:flex;align-items:baseline;gap:10px;margin-bottom:12px;}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px;}
.grid2{display:grid;grid-template-columns:2fr 1fr;gap:16px;}
.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;}
canvas{display:block;width:100%;background:var(--card);border-radius:8px;}
#mapCanvas{height:520px;cursor:grab;touch-action:none;}
#mapCanvas.dragging{cursor:grabbing;}
.chart-box{background:var(--card);border-radius:8px;padding:10px 10px 6px;}
.chart-box canvas{height:130px;}
.chart-label{font-size:12px;color:var(--ink-soft);margin:0 0 4px;}
.map-toolbar{display:flex;justify-content:space-between;align-items:center;margin-top:8px;
  font-size:12.5px;color:var(--ink-faint);}
.map-toolbar button{background:var(--card);color:var(--ink);border:1px solid var(--line);
  border-radius:6px;padding:5px 10px;font-size:12px;cursor:pointer;}
.map-toolbar button:hover{border-color:var(--accent);color:var(--accent);}
.legend{display:flex;flex-wrap:wrap;gap:12px;font-size:12px;color:var(--ink-soft);margin-top:8px;}
.legend span{display:inline-flex;align-items:center;gap:5px;}
.dot{width:9px;height:9px;border-radius:50%;display:inline-block;}
.empty{color:var(--ink-faint);font-size:13.5px;padding:30px 10px;text-align:center;}
table{width:100%;border-collapse:collapse;font-size:12.8px;font-variant-numeric:tabular-nums;}
th,td{padding:7px 10px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap;}
th:first-child,td:first-child,th.l,td.l{text-align:left;}
th{color:var(--ink-faint);font-weight:600;cursor:pointer;user-select:none;position:sticky;top:0;background:var(--panel);}
th:hover{color:var(--accent);}
tbody tr:hover{background:rgba(255,255,255,.03);}
.pill{display:inline-block;padding:2px 9px;border-radius:20px;font-size:11px;font-weight:700;
  letter-spacing:.02em;}
.table-wrap{max-height:420px;overflow:auto;border-radius:8px;}
footer{color:var(--ink-faint);font-size:12px;margin-top:30px;}
</style>
</head>
<body>
<div class="wrap">
  <h1>Метрики автономного алгоритму</h1>
  <p class="sub" id="subtitle">—</p>
  <div class="stats" id="headerStats"></div>

  <section id="mapSection">
    <div class="section-head"><h2>Мапа останнього прогону</h2></div>
    <div class="panel">
      <div id="mapEmpty" class="empty" style="display:none;">
        Немає активного прогону в цій сесії — запустіть автономний політ, щоб побачити мапу.
      </div>
      <div id="mapHost">
        <canvas id="mapCanvas"></canvas>
        <div class="map-toolbar">
          <span id="coordReadout">—</span>
          <span>
            <button id="apfToggleBtn" style="display:none;">Показати APF поле</button>
            <button id="resetViewBtn">Скинути вид</button>
          </span>
        </div>
        <div class="legend">
          <span><span class="dot" style="background:#3f6b4a"></span>дерево</span>
          <span><span class="dot" style="background:#8a6a2f"></span>уламок техніки</span>
          <span><span class="dot" style="background:#6fa8dc"></span>Ка-52</span>
          <span><span class="dot" style="background:#c98a4a"></span>Патрон</span>
          <span><span class="dot" style="background:#e2b34c"></span>чекпоінт</span>
          <span><span class="dot" style="background:#5fae7d"></span>старт</span>
          <span>— траєкторія (колір: час прогону, темніший → пізніше)</span>
          <span id="apfLegend" style="display:none;">— APF поле: <span class="dot" style="background:#3474f2"></span>спокійно →
            <span class="dot" style="background:#f23434"></span>небезпечно/далеко</span>
        </div>
      </div>
    </div>
  </section>

  <section id="chartsSection">
    <div class="section-head"><h2>Графіки останнього прогону</h2></div>
    <div class="panel" id="chartsEmpty" style="display:none;">
      <div class="empty">Немає активного прогону в цій сесії.</div>
    </div>
    <div class="grid3" id="chartsHost">
      <div class="chart-box"><p class="chart-label">Швидкість, м/с</p><canvas id="speedChart"></canvas></div>
      <div class="chart-box"><p class="chart-label">Висота (Z), м</p><canvas id="altChart"></canvas></div>
      <div class="chart-box"><p class="chart-label">Мін. дистанція до перешкод, м
        <span style="color:var(--bad)">(червоні риски — спрацював «буст» виходу із застрягання)</span></p>
        <canvas id="clearChart"></canvas></div>
    </div>
  </section>

  <section id="historySection">
    <div class="section-head"><h2>Історія прогонів</h2></div>
    <div id="historyEmpty" class="empty" style="display:none;">
      Ще немає збережених прогонів — запустіть автономний політ хоча б раз.
    </div>
    <div id="historyHost">
      <div class="stats" id="historyStats"></div>
      <div class="grid2">
        <div class="chart-box">
          <p class="chart-label">Ефективність шляху по прогонах (пряма/пройдена, лише для «Фініш»)</p>
          <canvas id="effTrendChart"></canvas>
        </div>
        <div class="chart-box">
          <p class="chart-label">Мін. дистанція до перешкод по прогонах</p>
          <canvas id="clearTrendChart"></canvas>
        </div>
      </div>
      <div style="height:14px;"></div>
      <div class="panel table-wrap">
        <table id="historyTable">
          <thead><tr>
            <th class="l" data-key="timestamp">Час</th>
            <th data-key="seed">Сід</th>
            <th class="l" data-key="final_status">Статус</th>
            <th data-key="duration_s">Трив., с</th>
            <th data-key="path_length">Шлях, м</th>
            <th data-key="path_efficiency">Ефект.</th>
            <th data-key="avg_speed">Ср.швидк.</th>
            <th data-key="max_speed">Макс.швидк.</th>
            <th data-key="min_clearance">Мін.дист.</th>
            <th data-key="stuck_pct">% застряг.</th>
          </tr></thead>
          <tbody id="historyBody"></tbody>
        </table>
      </div>
    </div>
  </section>

  <footer>Згенеровано локально · sim_headless.py + metrics_report.py · без мережевих залежностей</footer>
</div>

<script>
const DATA = __DATA_JSON__;

function fmt(v, d){ return (v===null||v===undefined||Number.isNaN(v)) ? "—" : Number(v).toFixed(d===undefined?2:d); }

// ── Заголовок / загальна статистика ─────────────────────────────────────────
(function initHeader(){
  const h = DATA.history || [];
  const sub = document.getElementById("subtitle");
  if (DATA.map) {
    sub.textContent = "Останній прогін: сід " + DATA.map.meta.seed +
      " · збережено прогонів: " + h.length;
  } else {
    sub.textContent = "Збережено прогонів: " + h.length;
  }
  const stats = document.getElementById("headerStats");
  const finished = h.filter(r => r.final_status === "FINISHED").length;
  const finRate = h.length ? Math.round(100*finished/h.length) : null;
  const effVals = h.filter(r => r.final_status==="FINISHED" && r.path_efficiency!=null).map(r=>r.path_efficiency);
  const avgEff = effVals.length ? effVals.reduce((a,b)=>a+b,0)/effVals.length : null;
  const clearVals = h.filter(r => r.min_clearance!=null).map(r=>r.min_clearance);
  const avgClear = clearVals.length ? clearVals.reduce((a,b)=>a+b,0)/clearVals.length : null;
  const chips = [
    ["Прогонів збережено", h.length],
    ["Частка «Фініш»", finRate===null? "—" : finRate+"%"],
    ["Серед. ефективність шляху", avgEff===null? "—" : fmt(avgEff,2)],
    ["Серед. мін. дистанція", avgClear===null? "—" : fmt(avgClear,2)+" м"],
  ];
  stats.innerHTML = chips.map(([l,v])=>`<div class="stat">${l}<b>${v}</b></div>`).join("");
})();

// ── Мапа останнього прогону ──────────────────────────────────────────────────
(function initMap(){
  const map = DATA.map, frames = DATA.frames;
  if (!map || !frames || !frames.length) {
    document.getElementById("mapHost").style.display = "none";
    document.getElementById("mapEmpty").style.display = "block";
    document.getElementById("chartsHost").style.display = "none";
    document.getElementById("chartsEmpty").style.display = "block";
    return;
  }
  const canvas = document.getElementById("mapCanvas");
  const ctx = canvas.getContext("2d");
  const bounds = map.meta.bounds;
  let scale, offX, offY;
  let showApf = false;   // перемикач-фільтр APF-поля на цій самій мапі (не окрема секція)

  function resizeCanvas(){
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * devicePixelRatio;
    canvas.height = 520 * devicePixelRatio;
    ctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);
  }
  function fitView(){
    const w = canvas.getBoundingClientRect().width, h = 520;
    scale = Math.min(w, h) / (bounds*2*1.15);
    offX = w/2; offY = h/2;
  }
  function w2c(x,y){ return [offX + x*scale, offY - y*scale]; }
  function c2w(cx,cy){ return [(cx-offX)/scale, -(cy-offY)/scale]; }

  function draw(){
    const w = canvas.getBoundingClientRect().width, h = 520;
    ctx.clearRect(0,0,w,h);
    // сітка
    ctx.strokeStyle = "rgba(255,255,255,.06)"; ctx.lineWidth = 1;
    const step = 10;
    for (let gx=-bounds; gx<=bounds; gx+=step){
      const [x1,y1]=w2c(gx,-bounds),[x2,y2]=w2c(gx,bounds);
      ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x2,y2); ctx.stroke();
    }
    for (let gy=-bounds; gy<=bounds; gy+=step){
      const [x1,y1]=w2c(-bounds,gy),[x2,y2]=w2c(bounds,gy);
      ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x2,y2); ctx.stroke();
    }
    // межі арени
    ctx.strokeStyle = "rgba(255,255,255,.22)"; ctx.lineWidth = 1.5;
    const [bx1,by1]=w2c(-bounds,-bounds),[bx2,by2]=w2c(bounds,bounds);
    ctx.strokeRect(bx1, by2, bx2-bx1, by1-by2);

    // APF поле (перемикач-фільтр, під деревами/траєкторією — той самий
    // потенціал, що й у Blender-панелі: attraction_potential до чекпоінта +
    // repulsion_potential від перешкод, незалежно нормалізовані/змішані на
    // Python-стороні — тут лише інтерполяція кольору «спокійно→небезпечно»).
    if (showApf && DATA.apfField){
      const af = DATA.apfField, xs = af.xs, ys = af.ys, mix = af.mix;
      const n = xs.length;
      const cool = [0.20,0.45,0.95], hot = [0.95,0.20,0.20];
      const half = (xs.length>1 ? (xs[1]-xs[0]) : 1) / 2;
      for (let j=0;j<n;j++){
        for (let i=0;i<n;i++){
          const f = mix[j][i];
          const r = Math.round(255*(cool[0]+(hot[0]-cool[0])*f));
          const g = Math.round(255*(cool[1]+(hot[1]-cool[1])*f));
          const b = Math.round(255*(cool[2]+(hot[2]-cool[2])*f));
          ctx.fillStyle = `rgba(${r},${g},${b},0.6)`;
          const [x0,y0] = w2c(xs[i]-half, ys[j]+half);
          const [x1,y1] = w2c(xs[i]+half, ys[j]-half);
          ctx.fillRect(x0, y0, x1-x0, y1-y0);
        }
      }
    }

    // дерева
    (map.trees||[]).forEach((t,i)=>{
      const [x,y,zb,r,hh] = t;
      const [cx,cy] = w2c(x,y);
      const isWreck = i === map.meta.wreck_index;
      ctx.fillStyle = isWreck ? "#8a6a2f" : "#3f6b4a";
      ctx.beginPath(); ctx.arc(cx,cy, Math.max(2, r*scale), 0, 7); ctx.fill();
    });
    // перешкоди
    (map.obstacles||[]).forEach(o=>{
      const [kind,x,y,z,r,collidable] = o;
      const [cx,cy] = w2c(x,y);
      ctx.fillStyle = kind==="ka52" ? "#6fa8dc" : "#c98a4a";
      ctx.beginPath(); ctx.arc(cx,cy, 5, 0, 7); ctx.fill();
    });
    // чекпоінт
    if (map.checkpoints && map.checkpoints.length){
      const [gx,gy,gz] = map.checkpoints[0];
      const [cx,cy] = w2c(gx,gy);
      if (DATA.cpRadius){
        ctx.fillStyle = "rgba(226,179,76,.16)"; ctx.strokeStyle="#e2b34c"; ctx.lineWidth=1.5;
        ctx.beginPath(); ctx.arc(cx,cy, DATA.cpRadius*scale, 0, 7); ctx.fill(); ctx.stroke();
      }
      ctx.fillStyle = "#e2b34c"; ctx.beginPath(); ctx.arc(cx,cy,5,0,7); ctx.fill();
    }
    // старт
    if (map.start){
      const [sx,sy] = w2c(map.start[0], map.start[1]);
      ctx.fillStyle = "#5fae7d"; ctx.beginPath(); ctx.arc(sx,sy,5,0,7); ctx.fill();
    }
    // траєкторія (колір темнішає з часом)
    const n = frames.length;
    for (let i=1;i<n;i++){
      const f0=frames[i-1], f1=frames[i];
      const [x1,y1]=w2c(f0.x,f0.y),[x2,y2]=w2c(f1.x,f1.y);
      const tfrac = i/n;
      const rr = Math.round(111 - tfrac*70), gg = Math.round(200-tfrac*90), bb = Math.round(255-tfrac*160);
      ctx.strokeStyle = `rgb(${rr},${gg},${bb})`;
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x2,y2); ctx.stroke();
    }
    // кінцева точка
    const last = frames[n-1];
    const statusColor = DATA.statusColors[last.status] || "#ffffff";
    const [lx,ly] = w2c(last.x, last.y);
    ctx.fillStyle = statusColor;
    ctx.beginPath(); ctx.arc(lx,ly,6,0,7); ctx.fill();
    ctx.strokeStyle = "#000"; ctx.lineWidth=1; ctx.stroke();
  }

  resizeCanvas(); fitView(); draw();
  window.addEventListener("resize", ()=>{ resizeCanvas(); draw(); });

  let dragging=false, lastX=0, lastY=0;
  canvas.addEventListener("mousedown", e=>{ dragging=true; canvas.classList.add("dragging"); lastX=e.clientX; lastY=e.clientY; });
  window.addEventListener("mouseup", ()=>{ dragging=false; canvas.classList.remove("dragging"); });
  canvas.addEventListener("mousemove", e=>{
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX-rect.left, my = e.clientY-rect.top;
    const [wx,wy] = c2w(mx,my);
    document.getElementById("coordReadout").textContent = "x=" + wx.toFixed(1) + "  y=" + wy.toFixed(1);
    if (dragging){
      offX += (e.clientX-lastX); offY += (e.clientY-lastY);
      lastX=e.clientX; lastY=e.clientY; draw();
    }
  });
  canvas.addEventListener("wheel", e=>{
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX-rect.left, my = e.clientY-rect.top;
    const [wx,wy] = c2w(mx,my);
    const factor = e.deltaY < 0 ? 1.12 : 1/1.12;
    scale *= factor;
    const [nx,ny] = w2c(wx,wy);
    offX += mx-nx; offY += my-ny;
    draw();
  }, {passive:false});
  document.getElementById("resetViewBtn").addEventListener("click", ()=>{ fitView(); draw(); });

  // ── Перемикач APF-поля (фільтр на цій самій мапі) ────────────────────────
  const apfBtn = document.getElementById("apfToggleBtn");
  if (DATA.apfField){
    apfBtn.style.display = "";
    apfBtn.addEventListener("click", ()=>{
      showApf = !showApf;
      apfBtn.textContent = showApf ? "Сховати APF поле" : "Показати APF поле";
      document.getElementById("apfLegend").style.display = showApf ? "" : "none";
      draw();
    });
  }

  // ── графіки останнього прогону ──────────────────────────────────────────
  function drawLineChart(canvasEl, xs, ys, color, tickMarks){
    const rect = canvasEl.getBoundingClientRect();
    canvasEl.width = rect.width*devicePixelRatio; canvasEl.height = 130*devicePixelRatio;
    const c = canvasEl.getContext("2d");
    c.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);
    const w = rect.width, h = 130, pad=6;
    c.clearRect(0,0,w,h);
    const ymin = Math.min(...ys), ymax = Math.max(...ys);
    const yr = (ymax-ymin) || 1;
    const xmin = xs[0], xmax = xs[xs.length-1] || 1;
    const xr = (xmax-xmin) || 1;
    function px(x){ return pad + (x-xmin)/xr * (w-2*pad); }
    function py(y){ return h-pad - (y-ymin)/yr * (h-2*pad); }
    c.strokeStyle = "rgba(255,255,255,.08)";
    for (let i=0;i<=3;i++){ const yy = pad + i*(h-2*pad)/3; c.beginPath(); c.moveTo(pad,yy); c.lineTo(w-pad,yy); c.stroke(); }
    c.strokeStyle = color; c.lineWidth = 1.6; c.beginPath();
    xs.forEach((x,i)=>{ const X=px(x), Y=py(ys[i]); i===0? c.moveTo(X,Y) : c.lineTo(X,Y); });
    c.stroke();
    c.fillStyle = "rgba(255,255,255,.5)"; c.font = "10px sans-serif";
    c.fillText(ymax.toFixed(1), 2, pad+8);
    c.fillText(ymin.toFixed(1), 2, h-2);
    if (tickMarks){
      c.strokeStyle = "#e2584c"; c.lineWidth = 2;
      xs.forEach((x,i)=>{ if (tickMarks[i]){ const X=px(x); c.beginPath(); c.moveTo(X,h-pad); c.lineTo(X,h-pad-6); c.stroke(); } });
    }
  }
  const ts = frames.map(f=>f.t);
  drawLineChart(document.getElementById("speedChart"), ts, frames.map(f=>f.speed), "#6fa8dc");
  drawLineChart(document.getElementById("altChart"), ts, frames.map(f=>f.z), "#5fae7d");
  drawLineChart(document.getElementById("clearChart"), ts, frames.map(f=> f.lidar&&f.lidar.length? Math.min(...f.lidar): 0),
                "#e2b34c", frames.map(f=>!!f.boosted));

  window._reportDrawLineChart = drawLineChart; // reuse for history trend charts below
})();

// ── Історія прогонів ─────────────────────────────────────────────────────────
(function initHistory(){
  const h = DATA.history || [];
  if (!h.length){
    document.getElementById("historyHost").style.display = "none";
    document.getElementById("historyEmpty").style.display = "block";
    return;
  }
  const finishedIdx = [];
  h.forEach((r,i)=>{ if (r.final_status==="FINISHED") finishedIdx.push(i); });

  function drawTrend(canvasEl, idxList, valsFn, color){
    const rect = canvasEl.getBoundingClientRect();
    canvasEl.width = rect.width*devicePixelRatio; canvasEl.height = 130*devicePixelRatio;
    const c = canvasEl.getContext("2d");
    c.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);
    const w = rect.width, hgt = 130, pad=8;
    c.clearRect(0,0,w,hgt);
    const pts = idxList.map(i=>({x:i, y:valsFn(h[i])})).filter(p=>p.y!=null && !Number.isNaN(p.y));
    if (!pts.length){ c.fillStyle="rgba(255,255,255,.4)"; c.font="12px sans-serif"; c.fillText("немає даних", 10, hgt/2); return; }
    const xmin=0, xmax=Math.max(h.length-1,1);
    const ymin = Math.min(...pts.map(p=>p.y)), ymax = Math.max(...pts.map(p=>p.y));
    const yr = (ymax-ymin)||1;
    function px(x){ return pad + (x-xmin)/(xmax-xmin||1) * (w-2*pad); }
    function py(y){ return hgt-pad - (y-ymin)/yr * (hgt-2*pad); }
    c.strokeStyle="rgba(255,255,255,.08)";
    for (let i=0;i<=3;i++){ const yy=pad+i*(hgt-2*pad)/3; c.beginPath(); c.moveTo(pad,yy); c.lineTo(w-pad,yy); c.stroke(); }
    c.strokeStyle=color; c.fillStyle=color; c.lineWidth=1.6; c.beginPath();
    pts.forEach((p,i)=>{ const X=px(p.x),Y=py(p.y); i===0? c.moveTo(X,Y): c.lineTo(X,Y); });
    c.stroke();
    pts.forEach(p=>{ const X=px(p.x),Y=py(p.y); c.beginPath(); c.arc(X,Y,2.4,0,7); c.fill(); });
    c.fillStyle="rgba(255,255,255,.5)"; c.font="10px sans-serif";
    c.fillText(ymax.toFixed(2), 2, pad+8); c.fillText(ymin.toFixed(2), 2, hgt-2);
  }
  drawTrend(document.getElementById("effTrendChart"), finishedIdx, r=>r.path_efficiency, "#5fae7d");
  drawTrend(document.getElementById("clearTrendChart"), h.map((_,i)=>i), r=>r.min_clearance, "#e2b34c");

  // таблиця
  let sortKey = "timestamp", sortAsc = false;
  function render(){
    const rows = h.slice().sort((a,b)=>{
      let av=a[sortKey], bv=b[sortKey];
      if (typeof av === "string") { av=(av||"").toLowerCase(); bv=(bv||"").toLowerCase(); }
      if (av==null) av = -Infinity; if (bv==null) bv = -Infinity;
      return sortAsc ? (av>bv?1:av<bv?-1:0) : (av<bv?1:av>bv?-1:0);
    });
    document.getElementById("historyBody").innerHTML = rows.map(r=>{
      const color = DATA.statusColors[r.final_status] || "#9aa1ac";
      const label = DATA.statusLabels[r.final_status] || r.final_status;
      return `<tr>
        <td class="l">${r.timestamp||"—"}</td>
        <td>${r.seed}</td>
        <td class="l"><span class="pill" style="background:${color}22;color:${color}">${label}</span></td>
        <td>${fmt(r.duration_s,1)}</td>
        <td>${fmt(r.path_length,1)}</td>
        <td>${fmt(r.path_efficiency,2)}</td>
        <td>${fmt(r.avg_speed,2)}</td>
        <td>${fmt(r.max_speed,2)}</td>
        <td>${fmt(r.min_clearance,2)}</td>
        <td>${fmt(r.stuck_pct,1)}%</td>
      </tr>`;
    }).join("");
  }
  document.querySelectorAll("#historyTable th").forEach(th=>{
    th.addEventListener("click", ()=>{
      const k = th.dataset.key;
      if (sortKey===k) sortAsc=!sortAsc; else { sortKey=k; sortAsc=false; }
      render();
    });
  });
  render();

  const stats = document.getElementById("historyStats");
  const byStatus = {};
  h.forEach(r=>{ const s=r.final_status; byStatus[s]=(byStatus[s]||0)+1; });
  stats.innerHTML = Object.keys(byStatus).map(s=>{
    const color = DATA.statusColors[s] || "#9aa1ac";
    const label = DATA.statusLabels[s] || s;
    return `<div class="stat">${label}<b style="color:${color}">${byStatus[s]}</b></div>`;
  }).join("");
})();
</script>
</body>
</html>
"""
