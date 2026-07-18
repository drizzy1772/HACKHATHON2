#!/usr/bin/env python3
"""Відео прогресу: чекпоінти → детермінований rollout у p.DIRECT →
chase-cam кадри (TinyRenderer) → ffmpeg mp4. Видно шлях
«врізається → маневрує» без уповільнення тренування (усе постфактум).

    conda run -n drones python tier_b/render_video.py \
        --ckpt-dir tier_b/runs/ppo/seed0/ckpt --level 2 --out tier_b/admin/reports/tier_b_progress_ppo.mp4
    conda run -n drones python tier_b/render_video.py \
        --model tier_b/runs/ppo/seed0/final_model.zip --level 2 --out tier_b/admin/reports/tier_b_final_ppo.mp4
"""

from __future__ import annotations

import argparse
import glob
import pathlib
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

FFMPEG = "/opt/homebrew/bin/ffmpeg"
FONT = "/System/Library/Fonts/Helvetica.ttc"
W, H = 1280, 720


def _camera(p, pos):
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=pos, distance=1.8, yaw=-40, pitch=-25, roll=0,
        upAxisIndex=2)
    proj = p.computeProjectionMatrixFOV(fov=60, aspect=W / H, nearVal=0.05,
                                        farVal=40.0)
    return view, proj


def render_rollout(model_path: str, level: int, frames_dir: pathlib.Path,
                   master_seed: int, max_steps: int = 600) -> dict:
    import pybullet as p
    from stable_baselines3 import PPO, SAC

    from tier_b.envs.pirouette_aviary import PirouetteAviary

    cls = SAC if "sac" in model_path.lower() else PPO
    model = cls.load(model_path, device="cpu")
    env = PirouetteAviary(level=level, master_seed=master_seed)
    obs, _ = env.reset()
    frames_dir.mkdir(parents=True, exist_ok=True)

    info = {}
    for i in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _r, term, trunc, info = env.step(
            np.asarray(action, dtype=np.float32).reshape(1, 4))
        view, proj = _camera(p, env.pos[0])
        _w, _h, rgb, _d, _s = p.getCameraImage(
            W, H, view, proj, renderer=p.ER_TINY_RENDERER,
            physicsClientId=env.CLIENT)
        img = np.reshape(rgb, (H, W, 4))[:, :, :3].astype(np.uint8)
        _save_png(img, frames_dir / f"frame_{i:05d}.png")
        if term or trunc:
            break
    env.close()
    return info


def _save_png(img: np.ndarray, path: pathlib.Path) -> None:
    from PIL import Image

    Image.fromarray(img).save(path)


def frames_to_mp4(frames_dir: pathlib.Path, out: pathlib.Path,
                  label: str | None = None) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)

    def _run(vf: str) -> None:
        subprocess.run([FFMPEG, "-y", "-framerate", "48",
                        "-i", str(frames_dir / "frame_%05d.png"),
                        "-vf", vf, "-c:v", "libx264", str(out)],
                       check=True, capture_output=True)

    if label and pathlib.Path(FONT).exists():
        # УВАГА: кома/двокрапка у text ламають парсер фільтрів ffmpeg
        safe = label.replace(",", "").replace(":", " ")
        try:
            _run(f"drawtext=fontfile={FONT}:text='{safe}':x=20:y=20:fontsize=36:"
                 f"fontcolor=white:box=1:boxcolor=black@0.4,format=yuv420p")
            return
        except subprocess.CalledProcessError:
            pass  # шрифт/фільтр недоступний — падаємо на варіант без підпису
    _run("format=yuv420p")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--ckpt-dir", type=str, default=None)
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--master-seed", type=int, default=900001)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--every", type=int, default=1,
                    help="кожен k-тий чекпоінт (для швидкого прев'ю)")
    args = ap.parse_args()

    work = pathlib.Path("tier_b/runs/_frames")
    out = pathlib.Path(args.out)

    if args.model:
        info = render_rollout(args.model, args.level, work / "single",
                              args.master_seed)
        frames_to_mp4(work / "single", out,
                      label=f"L{args.level}  {pathlib.Path(args.model).stem}")
        print(f"епізод: {info}; записано {out}")
        return

    ckpts = sorted(glob.glob(str(pathlib.Path(args.ckpt_dir) / "*_steps.zip")),
                   key=lambda s: int(re.search(r"(\d+)_steps", s).group(1)))
    ckpts = ckpts[::args.every]
    if not ckpts:
        sys.exit("чекпоінтів не знайдено")
    clips = []
    for ck in ckpts:
        steps = re.search(r"(\d+)_steps", ck).group(1)
        fdir = work / f"ck_{steps}"
        info = render_rollout(ck, args.level, fdir, args.master_seed)
        clip = fdir.with_suffix(".mp4")
        frames_to_mp4(fdir, clip, label=f"{int(steps)//1000}k steps L{args.level}")
        clips.append(clip)
        print(f"{steps} steps: success={info.get('success')} "
              f"collision={info.get('collision')} gates={info.get('gates_passed')}")

    concat = work / "concat.txt"
    concat.write_text("".join(f"file '{c.resolve()}'\n" for c in clips))
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", str(concat),
                    "-c", "copy", str(out)], check=True, capture_output=True)
    print(f"записано {out} ({len(clips)} кліпів)")


if __name__ == "__main__":
    main()
