"""
Showcase: a ~37 s trailer in seven acts, made for presentations.

One structured disk (spiral arms, a planet with gap and wake, a vortex and a
warped inner disk) is shown in seven acts: approach, orbit, a journey through
the spectrum, a J/H/K false-colour composite, polarimetry, a planet moving on
its orbit and a tilt to edge-on.

All acts are planned *together*. Acts 2-5 fly the same orbit, so every orbit
view is simulated once with all six wavelengths, and RGB and polarimetry come
for free. The plan printed at the start shows how few HAMSTER detectors the
876 frames actually need.

    python examples/showcase.py --plan              # what would be computed (no HAMSTER needed)
    python examples/showcase.py --draft             # preview on a laptop (10x fewer photons)
    python examples/showcase.py --slurm --partition long
    python examples/showcase.py --slurm --partition long   # again once the jobs are done → video

Output: ``animations/showcase/showcase.mp4``, one video and one still (for slides) per act.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

import hamster_animator as ha
from hamster_animator.animation import default_cache
from hamster_animator.sim import collect, make_plan

FPS = 24
FOV = 500.0  # au, the whole disk
ORBIT = 144  # frames of one orbit (6 s)

DISK = ha.Disk(
    r_out=250,
    h_ref=8,
    beta=1.2,
    dust_mass=1e-4,
    n_r=100,
    n_phi=96,
    n_theta=91,
    structures=(
        ha.Spiral(arms=2, pitch=14, amplitude=2.5, r_start=20),
        ha.Planet(radius=95, azimuth=120, gap_depth=0.95, wake=1.0),
        ha.Clump(radius=165, azimuth=-60, size=10, amplitude=5, elongation=4, name="vortex"),
        ha.Warp(r_break=15, tilt=12),
    ),
)

SIM = ha.Sim(photons=1e6, heating_photons=1e7, pixels=320)

CINEMATIC = ha.STYLES["cinematic"]
RGB = ha.Style(
    r2=True,
    coronagraph=12.0,
    smooth=0.8,
    temporal=1.0,
    temporal_wrap=True,
    vignette=0.3,
    interpolation="lanczos",
    font_size=17,
    stretch=ha.Stretch("asinh", lo=0, hi=99.5, soft=0.08),
)


def acts(sim: ha.Sim, cache) -> list[tuple[str, ha.Animation, ha.Style]]:
    """(name, animation, style) for every act. All acts share the sim settings and the cache."""
    orbit = ha.Camera.orbit(inclination=50, frames=ORBIT, fov=FOV)
    A = lambda scene, camera, **kw: ha.Animation(scene, camera, sim=sim, cache=cache, **kw)

    approach = ha.Camera.move(72, inclination=(10, 50), azimuth=(-120, 0), fov=(1400, FOV), ease="out")
    spectrum = ha.Steps([0.55e-6, 1.25e-6, 2.2e-6, 10e-6, 870e-6], hold=16, fade=16)
    quantity = ha.Steps(["I", "PI", "Q_PHI"], hold=32, fade=24)
    planet = ha.Steps(DISK.sweep("planet.azimuth", np.arange(0, 360, 30)), hold=2, fade=10, loop=True)
    edge_on = (
        ha.Camera.still(inclination=50, azimuth=0, fov=FOV)
        .then(ha.Camera.move(59, inclination=84, fov=380))
        .then(ha.Camera.still(24))
    )
    return [
        ("1_approach", A(DISK, approach, wavelength=1.6e-6), CINEMATIC.with_(title="Anflug")),
        ("2_orbit", A(DISK, orbit, wavelength=1.6e-6), CINEMATIC.with_(title="Eine Umrundung")),
        (
            "3_spectrum",
            A(DISK, orbit, wavelength=spectrum),
            CINEMATIC.with_(title="Optisch → Millimeter", temporal=0),
        ),
        (
            "4_rgb",
            A(DISK, orbit.reverse(), wavelength=(2.2e-6, 1.6e-6, 1.25e-6)),
            RGB.with_(title="Falschfarben  J · H · K"),
        ),
        (
            "5_polarimetry",
            A(DISK, orbit, wavelength=1.25e-6, quantity=quantity),
            ha.STYLES["polarimetry"].with_(title="Polarimetrie  I → PI → Q_φ", vignette=0.3, font_size=17),
        ),
        (
            "6_planet",
            A(planet, ha.Camera.still(inclination=35, fov=FOV), wavelength=1.6e-6),
            CINEMATIC.with_(title="Ein Planet zieht seine Bahn", temporal=0),
        ),
        (
            "7_edge_on",
            A(DISK, edge_on, wavelength=1.6e-6),
            CINEMATIC.with_(title="Kante voraus"),
        ),
    ]


def plan_all(show, backend):
    """One plan for all acts: views shared between acts are simulated once, with all their wavelengths."""
    anim = show[0][1]
    collect(anim.cache)
    requests = [r for _, a, _ in show for r in a._requests(a.frames())]
    return make_plan(requests, anim.sim, anim.cache, getattr(backend, "views_per_job", None))


def concat(videos, out: Path) -> Path:
    listing = out.with_suffix(".txt")
    listing.write_text("".join(f"file '{v.resolve()}'\n" for v in videos))
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy"]
    subprocess.run([*cmd, str(out)], check=True)
    listing.unlink()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--plan", action="store_true", help="only show what would be computed")
    ap.add_argument("--draft", action="store_true", help="10x fewer photons, half resolution")
    ap.add_argument("--out", default="animations/showcase")
    ap.add_argument("--slurm", action="store_true", help="submit to SLURM instead of running locally")
    ap.add_argument("--partition", default=None)
    ap.add_argument("--views-per-job", type=int, default=6)
    ap.add_argument("--cpus", type=int, default=32)
    ap.add_argument("--resubmit", action="store_true", help="submit even though earlier runs are still pending")
    args = ap.parse_args(argv)

    sim = SIM.draft() if args.draft else SIM
    show = acts(sim, ha.Cache(default_cache()))
    cache = show[0][1].cache
    if args.slurm:
        backend = ha.Slurm(partition=args.partition, cpus=args.cpus, views_per_job=args.views_per_job)
    else:
        backend = ha.Local()

    plan = plan_all(show, backend)
    n_frames = sum(len(a) for _, a, _ in show)
    print(f"\n{n_frames} frames in {len(show)} acts ({n_frames / FPS:.0f} s at {FPS} fps)")
    print(plan)
    if plan.cached == 0:
        separately = sum(a.plan(backend).n_detectors for _, a, _ in show)
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
            print("Jobs submitted. Run this script again once they are done to render the video.")
            return 0

    out = Path(args.out) / ("draft" if args.draft else "final")
    videos = []
    for name, anim, style in show:
        videos.append(anim.save(out / f"{name}.mp4", fps=FPS, style=style))
        anim.still(out / "stills" / f"{name}.png", frame=len(anim) // 2, style=style)
    final = concat(videos, out / "showcase.mp4")
    print(f"\n✓ {final}  (stills for slides: {out / 'stills'})")
    return 0


if __name__ == "__main__":
    if not shutil.which("ffmpeg") and "--plan" not in sys.argv:
        sys.exit("ffmpeg not found – it is needed for the videos (brew install ffmpeg / module load ffmpeg).")
    sys.exit(main())
