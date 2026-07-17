#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ПІСОЧНИЦЯ РУШІЯ — вільний 3D-політ для тестування реалістичного рушія (тільки для
розробника/тебе). НЕ доступна ні адміну, ні учаснику: окрема stand-alone програма на
pygame, що запускається СВОЄЮ командою:

    cd drone-hackathon
    python3.13 sandbox/engine_test.py        # (pygame береться з .pylibs)

Мапа — БЕЗ обмежень: без стелі й чекпойнтів, лише перешкоди-дерева й земля ДЛЯ
КОЛІЗІЙ. 3D-вигляд: chase-камера позаду-над дроном (C — перемкнути на FPV).

КЕРУВАННЯ (клавіатура АБО нативний джойстик, без Enjoyable):
  W/S — тангаж   ·  A/D — крен   ·  Space/Shift — газ-важіль (колектив, тримається)
  Q/E — курс     ·  C — камера   ·  R — скид   ·  Esc — вихід
  Джойстик Mode 2: лівий стік — газ(Y)+курс(X), правий — тангаж(Y)+крен(X).

Для джойстика дай Терміналу дозвіл: System Settings → Privacy & Security →
Input Monitoring → додай Terminal (і python), і перезапусти термінал.
"""

import math
import os
import random
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))   # корінь проєкту (файл лежить у корені)
sys.path.insert(0, os.path.join(ROOT, ".pylibs"))    # vендований pygame
sys.path.insert(0, ROOT)                             # realistic_flight.py поруч

try:
    import pygame
except ImportError:
    sys.stderr.write("pygame не встановлено:\n"
                     "    python3.13 -m pip install --target=.pylibs pygame\n")
    sys.exit(1)

from realistic_flight import RealisticQuad, QuadParams

W, H = 1120, 720
DRONE_R = 0.4
SKY_TOP = (120, 165, 215)
SKY_BOT = (175, 200, 225)
GROUND = (86, 120, 70)
GRID = (70, 100, 58)
TRUNK = (96, 68, 46)
CANOPY = (58, 110, 62)
DRONE_C = (250, 205, 70)
NOSE_C = (250, 90, 70)
ROTOR_C = (210, 220, 235)
SHADOW = (30, 40, 30)
WHITE = (240, 240, 240)
GREY = (150, 155, 165)
RED = (235, 70, 60)
GREEN = (95, 215, 125)
PANEL = (22, 25, 31, 210)


# ── Векторна алгебра (чистий Python) ─────────────────────────────────────────────
def sub(a, b): return (a[0] - b[0], a[1] - b[1], a[2] - b[2])
def add(a, b): return (a[0] + b[0], a[1] + b[1], a[2] + b[2])
def mul(a, s): return (a[0] * s, a[1] * s, a[2] * s)
def dot(a, b): return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
def cross(a, b): return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                         a[0] * b[1] - a[1] * b[0])
def vlen(a): return math.sqrt(dot(a, a))
def normalize(a):
    n = vlen(a)
    return (a[0] / n, a[1] / n, a[2] / n) if n > 1e-9 else (0.0, 0.0, 1.0)


def rot_zyx(yaw, pitch, roll):
    """Матриця повороту тіла→світ (ZYX: yaw·pitch·roll) — як у рушії."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    return ((cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp,     cp * sr,                cp * cr))


def matvec(R, v):
    return (R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
            R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
            R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2])


# ── Камера з перспективною проєкцією та near-clip ────────────────────────────────
class Camera:
    def __init__(self, fov_deg=68.0):
        self.f = (H / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
        self.near = 0.15
        self.pos = (0.0, 0.0, 5.0)
        self.right = (1.0, 0.0, 0.0)
        self.up = (0.0, 0.0, 1.0)
        self.fwd = (0.0, 1.0, 0.0)

    def place(self, pos, look):
        self.pos = pos
        self.fwd = normalize(sub(look, pos))
        self.right = normalize(cross(self.fwd, (0.0, 0.0, 1.0)))
        self.up = cross(self.right, self.fwd)

    def to_cam(self, p):
        rel = sub(p, self.pos)
        return (dot(rel, self.right), dot(rel, self.up), dot(rel, self.fwd))

    def project(self, c):
        return (W / 2.0 + self.f * c[0] / c[2], H / 2.0 - self.f * c[1] / c[2])


def _clip_near(poly, near):
    """Sutherland–Hodgman проти площини Zc>=near (список cam-точок)."""
    out = []
    n = len(poly)
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        ain = a[2] >= near
        bin_ = b[2] >= near
        if ain:
            out.append(a)
        if ain != bin_:
            t = (near - a[2]) / (b[2] - a[2])
            out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, near))
    return out


def poly_world(surf, cam, pts, color, width=0):
    cpts = [cam.to_cam(p) for p in pts]
    cpts = _clip_near(cpts, cam.near)
    if len(cpts) >= 3:
        pygame.draw.polygon(surf, color, [cam.project(c) for c in cpts], width)


def line_world(surf, cam, a, b, color, width=1):
    ca, cb = cam.to_cam(a), cam.to_cam(b)
    if ca[2] < cam.near and cb[2] < cam.near:
        return
    if ca[2] < cam.near:
        t = (cam.near - ca[2]) / (cb[2] - ca[2])
        ca = (ca[0] + (cb[0] - ca[0]) * t, ca[1] + (cb[1] - ca[1]) * t, cam.near)
    elif cb[2] < cam.near:
        t = (cam.near - cb[2]) / (ca[2] - cb[2])
        cb = (cb[0] + (ca[0] - cb[0]) * t, cb[1] + (ca[1] - cb[1]) * t, cam.near)
    pygame.draw.line(surf, color, cam.project(ca), cam.project(cb), width)


# ── Світ ─────────────────────────────────────────────────────────────────────────
def gen_trees(seed=1, n=70, extent=48.0):
    rng = random.Random(seed)
    trees = []
    while len(trees) < n:
        x, y = rng.uniform(-extent, extent), rng.uniform(-extent, extent)
        if math.hypot(x, y) < 7.0:            # вільний майданчик навколо старту
            continue
        trees.append((x, y, rng.uniform(0.3, 0.7), rng.uniform(6.0, 14.0)))
    return trees


def draw_sky(surf):
    for i in range(0, H, 4):
        t = i / H
        col = (int(SKY_TOP[0] + (SKY_BOT[0] - SKY_TOP[0]) * t),
               int(SKY_TOP[1] + (SKY_BOT[1] - SKY_TOP[1]) * t),
               int(SKY_TOP[2] + (SKY_BOT[2] - SKY_TOP[2]) * t))
        pygame.draw.rect(surf, col, (0, i, W, 4))


def draw_ground(surf, cam, dx, dy):
    E = 70.0
    poly_world(surf, cam, [(dx - E, dy - E, 0), (dx + E, dy - E, 0),
                           (dx + E, dy + E, 0), (dx - E, dy + E, 0)], GROUND)
    g = 5.0
    x0 = math.floor((dx - 45) / g) * g
    while x0 <= dx + 45:
        line_world(surf, cam, (x0, dy - 45, 0.02), (x0, dy + 45, 0.02), GRID, 1)
        x0 += g
    y0 = math.floor((dy - 45) / g) * g
    while y0 <= dy + 45:
        line_world(surf, cam, (dx - 45, y0, 0.02), (dx + 45, y0, 0.02), GRID, 1)
        y0 += g


def draw_tree(surf, cam, t):
    tx, ty, r, h = t
    # орт, перпендикулярний до напрямку на камеру (щоб «білборд»-стовбур був до нас)
    to_cam = normalize((cam.pos[0] - tx, cam.pos[1] - ty, 0.0))
    perp = (-to_cam[1] * r, to_cam[0] * r, 0.0)
    b1 = (tx + perp[0], ty + perp[1], 0.0)
    b2 = (tx - perp[0], ty - perp[1], 0.0)
    t1 = (tx + perp[0], ty + perp[1], h)
    t2 = (tx - perp[0], ty - perp[1], h)
    poly_world(surf, cam, [b1, b2, t2, t1], TRUNK)
    # крона — коло-білборд на вершині
    c = cam.to_cam((tx, ty, h))
    if c[2] > cam.near:
        sx, sy = cam.project(c)
        rad = max(4, int(r * 3.2 * cam.f / c[2]))
        pygame.draw.circle(surf, CANOPY, (int(sx), int(sy)), rad)


def draw_shadow(surf, cam, dpos):
    c = cam.to_cam((dpos[0], dpos[1], 0.02))
    if c[2] > cam.near:
        sx, sy = cam.project(c)
        rad = max(3, int(0.6 * cam.f / c[2]))
        sh = pygame.Surface((rad * 2, rad * 2), pygame.SRCALPHA)
        pygame.draw.ellipse(sh, (*SHADOW, 110), (0, 0, rad * 2, rad))
        surf.blit(sh, (sx - rad, sy - rad // 2))


def draw_drone(surf, cam, quad):
    R = rot_zyx(quad.yaw, quad.pitch, quad.roll)
    a = 0.55
    arms = {"FR": (a, -a, 0), "FL": (a, a, 0), "BL": (-a, a, 0), "BR": (-a, -a, 0)}
    center = tuple(quad.pos)
    world = {k: add(center, matvec(R, v)) for k, v in arms.items()}
    nose = add(center, matvec(R, (a * 1.5, 0, 0)))
    # промені рами
    for k, wp in world.items():
        line_world(surf, cam, center, wp, DRONE_C, 3)
    # ніс (напрямок)
    line_world(surf, cam, center, nose, NOSE_C, 3)
    # ротори-кільця
    for wp in world.values():
        c = cam.to_cam(wp)
        if c[2] > cam.near:
            sx, sy = cam.project(c)
            rad = max(3, int(0.28 * cam.f / c[2]))
            pygame.draw.circle(surf, ROTOR_C, (int(sx), int(sy)), rad, 2)


# ── HUD ──────────────────────────────────────────────────────────────────────────
def draw_hud(surf, quad, f, fb, joy_on, cam_mode):
    pw, ph = 250, 250
    px, py = 14, 14
    panel = pygame.Surface((pw, ph), pygame.SRCALPHA)
    panel.fill(PANEL)
    surf.blit(panel, (px, py))

    def L(s, x, y, c=WHITE, ff=None):
        surf.blit((ff or f).render(s, True, c), (x, y))

    L("ПІСОЧНИЦЯ РУШІЯ", px + 12, py + 8, DRONE_C, fb)
    vsp = quad.vel[2]
    lines = [
        (f"газ    {int(round(quad.throttle*100)):3d} %", GREEN),
        (f"висота {quad.pos[2]:6.1f} м", WHITE),
        (f"верт   {vsp:+5.1f} м/с", WHITE),
        (f"швидк  {quad.speed:6.1f} м/с", WHITE),
        (f"тангаж {math.degrees(quad.pitch):+5.0f}°", WHITE),
        (f"крен   {math.degrees(quad.roll):+5.0f}°", WHITE),
        (f"курс   {math.degrees(quad.yaw):+5.0f}°", WHITE),
    ]
    for i, (s, c) in enumerate(lines):
        L(s, px + 12, py + 40 + i * 22, c)
    # стовпчик газу
    bx, by, bh = px + 205, py + 44, 150
    pygame.draw.rect(surf, (40, 44, 52), (bx, by, 22, bh))
    fh = int(bh * quad.throttle)
    pygame.draw.rect(surf, GREEN, (bx, by + bh - fh, 22, fh))
    hov = by + bh - int(bh * quad.hover_throttle)
    pygame.draw.line(surf, DRONE_C, (bx - 3, hov), (bx + 25, hov), 1)

    hint = "W/S тангаж · A/D крен · Space/Shift газ · Q/E курс · C камера · R скид · Esc"
    surf.blit(f.render(hint, True, GREY), (14, H - 26))
    surf.blit(f.render(f"камера: {cam_mode}   джойстик: {'є' if joy_on else 'нема'}",
                       True, GREY), (14, H - 48))


def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Пісочниця рушія — 3D вільний політ")
    clock = pygame.time.Clock()
    f = pygame.font.SysFont("menlo,consolas,monospace", 15)
    fb = pygame.font.SysFont("menlo,consolas,monospace", 20, bold=True)
    fbig = pygame.font.SysFont("menlo,consolas,monospace", 46, bold=True)

    pygame.joystick.init()
    joy = None
    if pygame.joystick.get_count() > 0:
        joy = pygame.joystick.Joystick(0); joy.init()
        print(f"Джойстик: {joy.get_name()} — осей {joy.get_numaxes()}")
    else:
        print("Джойстик не знайдено — клавіатура. Дай Терміналу дозвіл Input Monitoring.")

    quad = RealisticQuad(QuadParams())
    START = (0.0, 0.0, 3.0)
    quad.reset(START)
    trees = gen_trees()
    cam = Camera()
    cam_modes = ["chase", "fpv"]
    cam_i = 0
    # згладжена позиція камери (щоб рух було приємно видно)
    cam_pos = list(START)
    flash = 0.0
    running = True
    # Утримувані ФІЗИЧНІ клавіші (SDL-scancode, KSCAN_*) — на відміну від K_w/K_a/…
    # (keycode), scancode прив'язаний до РОЗТАШУВАННЯ клавіші на клавіатурі, а не до
    # символу, який вона друкує в поточній розкладці ОС. Тому WASDQE керують і при
    # кириличній розкладці (де звичайний K_w узагалі не спрацьовував — фізична W
    # видавала символ «Ц», якого немає серед K_*).
    held_scan = set()

    while running:
        dt = min(clock.tick(60) / 1000.0, 0.05)
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                held_scan.add(e.scancode)
                if e.scancode == pygame.KSCAN_ESCAPE:
                    running = False
                elif e.scancode == pygame.KSCAN_R:
                    quad.reset(START); flash = 0.0
                elif e.scancode == pygame.KSCAN_C:
                    cam_i = (cam_i + 1) % len(cam_modes)
            elif e.type == pygame.KEYUP:
                held_scan.discard(e.scancode)

        pitch = (1.0 if pygame.KSCAN_W in held_scan else 0.0) \
            - (1.0 if pygame.KSCAN_S in held_scan else 0.0)
        roll = (1.0 if pygame.KSCAN_D in held_scan else 0.0) \
            - (1.0 if pygame.KSCAN_A in held_scan else 0.0)
        throttle = (1.0 if pygame.KSCAN_SPACE in held_scan else 0.0) \
            - (1.0 if (pygame.KSCAN_LSHIFT in held_scan
                       or pygame.KSCAN_RSHIFT in held_scan) else 0.0)
        yaw = (1.0 if pygame.KSCAN_Q in held_scan else 0.0) \
            - (1.0 if pygame.KSCAN_E in held_scan else 0.0)
        if joy is not None:
            pygame.event.pump()

            def ax(i):
                return joy.get_axis(i) if 0 <= i < joy.get_numaxes() else 0.0

            def dz(v):
                return 0.0 if abs(v) < 0.12 else float(v)
            throttle += -dz(ax(1)); yaw += -dz(ax(0))
            pitch += -dz(ax(3)); roll += dz(ax(2))

        quad.step(throttle, pitch, roll, yaw, dt)

        # Колізії (лише дерева + земля)
        x, y, z = quad.pos
        hit = z < DRONE_R
        for tx, ty, tr, th in trees:
            if z <= th and math.hypot(x - tx, y - ty) < tr + DRONE_R:
                hit = True; break
        if hit and flash <= 0.0:
            flash = 1.0; quad.reset(START)
        if flash > 0.0:
            flash -= dt

        # ── Камера ────────────────────────────────────────────────────────────────
        mode = cam_modes[cam_i]
        yaw_h = quad.yaw
        bx, by = math.cos(yaw_h), math.sin(yaw_h)
        if mode == "chase":
            target = (x - bx * 8.0, y - by * 8.0, z + 3.0)
            a = min(1.0, dt * 5.0)
            cam_pos[0] += (target[0] - cam_pos[0]) * a
            cam_pos[1] += (target[1] - cam_pos[1]) * a
            cam_pos[2] += (target[2] - cam_pos[2]) * a
            cam.place(tuple(cam_pos), (x + bx * 2.0, y + by * 2.0, z))
        else:  # fpv — на корпусі, дивиться вперед-трохи-вниз
            cam.place((x + bx * 0.3, y + by * 0.3, z + 0.15),
                      (x + bx * 6.0, y + by * 6.0, z - 0.6))

        # ── Малювання ─────────────────────────────────────────────────────────────
        draw_sky(screen)
        draw_ground(screen, cam, x, y)
        # дерева — далекі спершу (painter)
        order = sorted(trees, key=lambda t: -((t[0] - cam.pos[0]) ** 2 + (t[1] - cam.pos[1]) ** 2))
        for t in order:
            draw_tree(screen, cam, t)
        draw_shadow(screen, cam, quad.pos)
        if mode != "fpv":
            draw_drone(screen, cam, quad)
        draw_hud(screen, quad, f, fb, joy is not None, mode)
        if flash > 0.0:
            pygame.draw.rect(screen, RED, (0, 0, W, H), 6)
            t = fbig.render("ЗІТКНЕННЯ", True, RED)
            screen.blit(t, (W // 2 - t.get_width() // 2, H // 2 - 60))
            h2 = f.render("R — скид у центр", True, WHITE)
            screen.blit(h2, (W // 2 - h2.get_width() // 2, H // 2 - 8))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
