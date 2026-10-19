"""
Shared runner for multi-act trailers (``showcase.py``, ``journey.py``).

An act is ``(name, Animation, Style)``. All acts are planned *together*, so a
view that several acts use is simulated once with all wavelengths any act
needs. The finished acts are concatenated into one video.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import hamster_animator as ha
from hamster_animator.animation import default_cache
from hamster_animator.sim import collect, make_plan


def plan_all(acts, backend):
    """One plan for all acts: views shared between acts are simulated once, with all their wavelengths."""
    anim = acts[0][1]
    collect(anim.cache)
    requests = [r for _, a, _ in acts for r in a._requests(a.frames())]
    return make_plan(requests, anim.sim, anim.cache, getattr(backend, "views_per_job", None))


def concat(videos, out: Path) -> Path:
    listing = out.with_suffix(".txt")
    listing.write_text("".join(f"file '{v.resolve()}'\n" for v in videos))
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy"]
    subprocess.run([*cmd, str(out)], check=True)
    listing.unlink()
    return out


def key_frames(acts):
    """First, middle and last frame of every act as single-frame acts (for ``--check``)."""
    out = []
    for name, anim, style in acts:
        n = len(anim)
        for i in sorted({0, n // 2, n - 1}):
            out.append((f"{name}_{i:04d}", anim._single(i), style))
    return out


def run(build, sim: ha.Sim, name: str, fps: int = 24, argv=None) -> int:
    """
    Command line for a trailer. ``build(sim, cache)`` returns the acts.

    --plan    what would be computed       --draft   10x fewer photons, half resolution
    --check   draft stills of key frames   --slurm   submit to SLURM (run again when done)
    """
    ap = argparse.ArgumentParser(description=f"Render the {name} trailer.")
    ap.add_argument("--plan", action="store_true", help="only show what would be computed")
    ap.add_argument("--draft", action="store_true", help="10x fewer photons, half resolution")
    ap.add_argument("--check", action="store_true", help="draft stills of the first, middle and last frame per act")
    ap.add_argument("--out", default=f"animations/{name}")
    ap.add_argument("--slurm", action="store_true", help="submit to SLURM instead of running locally")
    ap.add_argument("--partition", default=None)
    ap.add_argument("--time", default="12:00:00", help="SLURM time limit per array element")
    ap.add_argument("--cpus", type=int, default=32)
    ap.add_argument("--views-per-job", type=int, default=6)
    ap.add_argument("--setup", action="append", default=[], help="shell line before HAMSTER, e.g. 'module load gcc'")
    ap.add_argument("--resubmit", action="store_true", help="submit even though earlier runs are still pending")
    args = ap.parse_args(argv)

    if not args.plan and not args.check and not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found – it is needed for the videos (brew install ffmpeg / module load ffmpeg).")

    draft = args.draft or args.check
    acts = build(sim.draft() if draft else sim, ha.Cache(default_cache()))
    if args.check:
        acts = key_frames(acts)
    cache = acts[0][1].cache
    if args.slurm:
        backend = ha.Slurm(
            partition=args.partition,
            time=args.time,
            cpus=args.cpus,
            views_per_job=args.views_per_job,
            setup=args.setup,
        )
    else:
        backend = ha.Local()

    plan = plan_all(acts, backend)
    n_frames = sum(len(a) for _, a, _ in acts)
    what = "stills" if args.check else f"frames in {len(acts)} acts ({n_frames / fps:.0f} s at {fps} fps)"
    print(f"\n{n_frames} {what}")
    print(plan)
    if plan.cached == 0 and not args.check:
        separately = sum(a.plan(backend).n_detectors for _, a, _ in acts)
        print(f"→ {plan.n_detectors} HAMSTER detector(s) for {n_frames} frames ({separately} if planned per act)\n")
    if args.plan:
        return 0

    if not plan.empty:
        # Pending runs are not part of the plan, so a second call while SLURM jobs are
        # still running would submit everything again.
        if any((cache.root / "runs").glob("*/manifest.json")) and not args.resubmit:
            print("Earlier runs are still pending (check `squeue -u $USER`). Call again once they are done,")
            print("or pass --resubmit if they failed.")
            return 1
        if not backend.execute(plan, cache):
            print("Jobs submitted. Run the same command again once they are done.")
            return 0

    out = Path(args.out) / ("check" if args.check else "draft" if draft else "final")
    if args.check:
        for act, anim, style in acts:
            anim.still(out / f"{act}.png", style=style)
        print(f"\n✓ stills in {out}")
        return 0
    videos = []
    for act, anim, style in acts:
        videos.append(anim.save(out / f"{act}.mp4", fps=fps, style=style))
        anim.still(out / "stills" / f"{act}.png", frame=len(anim) // 2, style=style)
    final = concat(videos, out / f"{name}.mp4")
    print(f"\n✓ {final}  (stills for slides: {out / 'stills'})")
    return 0
