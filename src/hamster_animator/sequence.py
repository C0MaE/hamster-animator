"""
:class:`Sequence` plays several animations ("acts") one after another::

    seq = ha.Sequence([
        ("approach", ha.Animation(disk, approach, sim=sim), "cinematic"),
        ("spectrum", ha.Animation(disk, orbit, wavelength=journey, sim=sim), "cinematic"),
        ("polarimetry", ha.Animation(disk, orbit, quantity="PI", sim=sim), "polarimetry"),
    ])
    seq.save("trailer.mp4")     # or: seq.cli("trailer") for a command line

All acts are planned *together*: a view that several acts use is simulated
once, with every wavelength any of them needs. Acts with different styles,
cameras and tracks are fine; they only have to share the :class:`Sim` settings.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from . import render as R
from .animation import Animation
from .sim import Local, Plan, Slurm, collect, make_plan

log = logging.getLogger("hamster_animator")

VIDEO_FORMATS = (".mp4", ".webm", ".mov")  # formats that can be concatenated without re-encoding


@dataclass
class Act:
    name: str
    animation: Animation
    style: object = None  # Style, preset name or None


class Sequence:
    def __init__(self, acts, fps: int = 24):
        self.acts = [a if isinstance(a, Act) else Act(*a) for a in acts]
        if not self.acts:
            raise ValueError("A Sequence needs at least one act.")
        names = [a.name for a in self.acts]
        if len(set(names)) != len(names):
            raise ValueError(f"Act names must be unique: {names}")
        if len({a.animation.sim for a in self.acts}) > 1:
            raise ValueError("All acts must use the same Sim settings, otherwise their views cannot be shared.")
        cache = self.acts[0].animation.cache
        for a in self.acts:
            a.animation.cache = cache
        self.fps = int(fps)

    @property
    def sim(self):
        return self.acts[0].animation.sim

    @property
    def cache(self):
        return self.acts[0].animation.cache

    def __len__(self) -> int:
        return sum(len(a.animation) for a in self.acts)

    def _map(self, fn) -> Sequence:
        return Sequence([Act(a.name, fn(a.animation), a.style) for a in self.acts], self.fps)

    def draft(self, factor: float = 10.0) -> Sequence:
        """Copy with preview quality (fewer photons, half the resolution)."""
        return self._map(lambda anim: anim.draft(factor))

    def key_frames(self) -> Sequence:
        """First, middle and last frame of every act as single-frame acts – a cheap check of the framing."""
        acts = []
        for a in self.acts:
            n = len(a.animation)
            for i in sorted({0, n // 2, n - 1}):
                acts.append(Act(f"{a.name}_{i:04d}", a.animation._single(i), a.style))
        return Sequence(acts, self.fps)

    # ── simulation ──
    def _requests(self):
        for a in self.acts:
            yield from a.animation._requests(a.animation.frames())

    def plan(self, backend=None) -> Plan:
        """What would be computed for all acts together? Nothing is started."""
        collect(self.cache)
        return make_plan(self._requests(), self.sim, self.cache, getattr(backend, "views_per_job", None))

    def pending(self) -> bool:
        """Are there submitted runs whose results have not arrived yet (e.g. running SLURM jobs)?"""
        return any((self.cache.root / "runs").glob("*/manifest.json"))

    def simulate(self, backend=None, resubmit: bool = False) -> bool:
        """
        Compute missing maps for all acts. Returns True when everything is available afterwards.

        Pending runs are not part of the plan, so submitting while earlier jobs are still running
        would compute everything twice. That is refused unless ``resubmit=True`` (e.g. after failed jobs).
        """
        plan = self.plan(backend)
        if plan.empty:
            log.info("all %d (view, wavelength) combinations cached", plan.total)
            return True
        log.info("%s", plan)
        if self.pending() and not resubmit:
            log.warning(
                "earlier runs are still pending (check `squeue -u $USER`) – call again once they are done, "
                "or pass resubmit=True if they failed"
            )
            return False
        return (backend or Local()).execute(plan, self.cache)

    def summary(self, backend=None) -> str:
        """Frames, plan and how much joint planning saves – handy for a slide."""
        plan = self.plan(backend)
        lines = [f"{len(self)} frames in {len(self.acts)} act(s) ({len(self) / self.fps:.0f} s at {self.fps} fps)"]
        lines.append(str(plan))
        if plan.cached == 0 and len(self.acts) > 1:
            separately = sum(a.animation.plan(backend).n_detectors for a in self.acts)
            lines.append(f"→ {plan.n_detectors} HAMSTER detector(s) ({separately} if every act were planned alone)")
        return "\n".join(lines)

    # ── output ──
    def save(self, path, backend=None, workers: int | None = None, quality: int = 18, resubmit: bool = False):
        """
        Simulate what is missing, render every act and join them into one video (.mp4/.webm/.mov).
        The single acts are kept next to it in ``<name>_acts/``. A path without suffix writes one
        PNG folder per act. Returns None while SLURM jobs are running.
        """
        if not self.simulate(backend, resubmit):
            log.info("waiting for SLURM – call save() again when the jobs are done")
            return None
        path = Path(path)
        ext = path.suffix.lower()
        if ext == "":
            for a in self.acts:
                a.animation.save(path / a.name, fps=self.fps, style=a.style, workers=workers)
            return path
        if ext not in VIDEO_FORMATS:
            raise ValueError(f"A Sequence can be saved as {', '.join(VIDEO_FORMATS)} or a PNG folder, not '{ext}'.")
        sizes = {R.as_style(a.style).size for a in self.acts}
        if len(sizes) > 1:
            raise ValueError(f"All acts need the same output size to be joined, got {sorted(sizes)}.")
        parts_dir = path.with_name(f"{path.stem}_acts")
        parts = [
            a.animation.save(
                parts_dir / f"{a.name}{ext}", fps=self.fps, style=a.style, workers=workers, quality=quality
            )
            for a in self.acts
        ]
        out = R.concat_videos(parts, path)
        log.info("%d acts → %s", len(parts), out)
        return out

    def stills(self, folder, frame: str = "middle") -> list[Path]:
        """One PNG per act (``frame`` = "first", "middle" or "last") – e.g. for slides."""
        pick = {"first": lambda n: 0, "middle": lambda n: n // 2, "last": lambda n: n - 1}[frame]
        folder = Path(folder)
        return [
            a.animation.still(folder / f"{a.name}.png", frame=pick(len(a.animation)), style=a.style) for a in self.acts
        ]

    # ── command line ──
    def cli(self, name: str, argv=None) -> int:
        """
        A ready-made command line for a script that builds a Sequence::

            if __name__ == "__main__":
                sys.exit(ha.Sequence(acts(SIM)).cli("trailer"))

        ``--plan`` shows what would be computed, ``--check`` renders draft stills of the key frames,
        ``--draft`` a preview, ``--slurm`` submits to SLURM (run the same command again when done).
        """
        ap = argparse.ArgumentParser(description=f"Render the '{name}' sequence.")
        ap.add_argument("--plan", action="store_true", help="only show what would be computed")
        ap.add_argument("--draft", action="store_true", help="10x fewer photons, half resolution")
        ap.add_argument("--check", action="store_true", help="draft stills of the first, middle and last frame per act")
        ap.add_argument("--out", default=f"animations/{name}", help="output folder (default: %(default)s)")
        ap.add_argument("--slurm", action="store_true", help="submit to SLURM instead of running locally")
        ap.add_argument("--partition", default=None)
        ap.add_argument("--time", default="12:00:00", help="SLURM time limit per array element")
        ap.add_argument("--cpus", type=int, default=32)
        ap.add_argument("--views-per-job", type=int, default=6)
        ap.add_argument(
            "--setup", action="append", default=[], help="shell line before HAMSTER, e.g. 'module load gcc'"
        )
        ap.add_argument("--resubmit", action="store_true", help="submit even though earlier runs are still pending")
        args = ap.parse_args(argv)

        if not (args.plan or args.check) and not shutil.which("ffmpeg"):
            print("ffmpeg not found – it is needed for the videos (brew install ffmpeg / module load ffmpeg).")
            return 1
        seq = self.draft() if args.draft or args.check else self
        if args.check:
            seq = seq.key_frames()
        if args.slurm:
            backend = Slurm(
                partition=args.partition,
                time=args.time,
                cpus=args.cpus,
                views_per_job=args.views_per_job,
                setup=args.setup,
            )
        else:
            backend = Local()

        print(seq.summary(backend), "\n", flush=True)
        if args.plan:
            return 0
        if not seq.simulate(backend, args.resubmit):
            if not seq.pending():
                return 1
            print("Run the same command again once the jobs are done.")
            return 0
        out = Path(args.out) / ("check" if args.check else "draft" if args.draft else "final")
        if args.check:
            seq.stills(out, frame="first")
            print(f"✓ stills in {out}")
            return 0
        video = seq.save(out / f"{name}.mp4")
        seq.stills(out / "stills")
        print(f"✓ {video}  (single acts: {video.with_name(video.stem + '_acts')}, stills: {out / 'stills'})")
        return 0
