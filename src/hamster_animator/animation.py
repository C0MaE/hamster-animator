"""
:class:`Animation` combines four tracks into frames:

==========  ======================================  ==============================
track       value                                   used for
==========  ======================================  ==============================
scene       ``Disk`` or ``Steps([Disk, …])``        parameter evolution
camera      ``View`` or ``Path``                    camera flights
wavelength  ``float``, ``(r, g, b)`` or ``Steps``   wavelength journeys, RGB
quantity    ``"I"``, ``"PI"``, ``"Q_PHI"``, …       polarisation
==========  ======================================  ==============================

Tracks of length 1 are stretched to the full length; all others must have
the same length.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from . import render as R
from .camera import FixedPath, View, as_path
from .scene import Disk
from .sim import Cache, Local, Sim, collect, lam_key, make_plan
from .steps import Mix, as_track

log = logging.getLogger("hamster_animator")


def default_cache() -> Path:
    """``$HAMSTER_ANIMATOR_CACHE``, else ``$XDG_CACHE_HOME/hamster-animator`` (``~/.cache/hamster-animator``)."""
    if os.environ.get("HAMSTER_ANIMATOR_CACHE"):
        return Path(os.environ["HAMSTER_ANIMATOR_CACHE"]).expanduser()
    base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "hamster-animator"


@dataclass
class Frame:
    index: int
    scene: Mix
    view: View
    wavelength: Mix
    quantity: Mix


def _lams(value) -> tuple:
    return tuple(value) if isinstance(value, (tuple, list)) else (value,)


def _fmt_lam(lam: float) -> str:
    um = lam * 1e6
    return f"{um:.0f} µm" if um >= 100 else f"{um:.1f} µm" if um >= 10 else f"{um:.2f} µm"


def _sim_view(disk: Disk, view: View) -> View:
    """The view that is actually simulated: the azimuth does not matter for axisymmetric disks."""
    return replace(view, azimuth=0.0) if disk.axisymmetric and view.azimuth % 360.0 else view


class Animation:
    def __init__(self, scene, camera=None, wavelength=1.6e-6, quantity="I", sim: Sim | None = None, cache=None):
        self.scene = scene
        self.camera = camera
        self.wavelength = wavelength
        self.quantity = quantity
        self.sim = sim or Sim()
        self.cache = cache if isinstance(cache, Cache) else Cache(cache or default_cache())

    def _copy(self, **changes) -> Animation:
        kw = dict(
            scene=self.scene,
            camera=self.camera,
            wavelength=self.wavelength,
            quantity=self.quantity,
            sim=self.sim,
            cache=self.cache,
        )
        kw.update(changes)
        return Animation(**kw)

    def draft(self, factor: float = 10.0) -> Animation:
        """Copy with preview quality (fewer photons, half the resolution)."""
        return self._copy(sim=self.sim.draft(factor))

    # ── frames ──
    def frames(self) -> list[Frame]:
        scenes = as_track(self.scene)
        views = as_path(self.camera).with_defaults(fov=2.0 * scenes[0].a.r_out).views()
        lams = as_track(self.wavelength)
        qs = as_track(self.quantity)
        lengths = {"scene": len(scenes), "camera": len(views), "wavelength": len(lams), "quantity": len(qs)}
        n = max(lengths.values())
        if any(l not in (1, n) for l in lengths.values()):
            raise ValueError(f"Track lengths do not match: {lengths}. Every track needs length 1 or {n}.")
        pick = lambda seq, i: seq[0] if len(seq) == 1 else seq[i]
        return [Frame(i, pick(scenes, i), pick(views, i), pick(lams, i), pick(qs, i)) for i in range(n)]

    def __len__(self) -> int:
        return len(self.frames())

    @staticmethod
    def _requests(frames):
        for f in frames:
            for disk in f.scene.values:
                for lam_val in f.wavelength.values:
                    for lam in _lams(lam_val):
                        yield disk, _sim_view(disk, f.view), lam

    # ── simulation ──
    def plan(self, backend=None):
        """What would be computed? Nothing is started."""
        collect(self.cache)
        return make_plan(self._requests(self.frames()), self.sim, self.cache, getattr(backend, "views_per_job", None))

    def simulate(self, backend=None) -> bool:
        """Compute missing maps. Returns True when everything is available afterwards."""
        plan = self.plan(backend)
        if plan.empty:
            log.info("all %d (view, wavelength) combinations cached", plan.total)
            return True
        log.info("%s", plan)
        return (backend or Local()).execute(plan, self.cache)

    # ── rendering ──
    def render(self, style=None, frames=None) -> list[R.FrameData]:
        """Frames ready for drawing. Requires all simulation data (see :meth:`simulate`)."""
        style = R.as_style(style)
        frames = self.frames() if frames is None else frames
        sim, cache = self.sim, self.cache
        missing = make_plan(self._requests(frames), sim, cache)
        if not missing.empty:
            raise RuntimeError(f"Simulation data missing – call simulate() first.\n{missing}")

        # 1) raw quantities per (scene, view, wavelength, quantity)
        raw: dict = {}

        def get(disk, view, lam, q):
            k = (disk.key, view.key, lam_key(lam), q.upper())
            if k not in raw:
                maps = cache.load(disk, sim, _sim_view(disk, view), lam)
                raw[k] = (R.quantity(maps, q, style.smooth, style.r2), maps["pixel_au"])
            return raw[k]

        def group(lam, q, fi):
            if style.norm == "global":
                return (q.upper(),)
            if style.norm == "frame":
                return (q.upper(), lam_key(lam), fi)
            return (q.upper(), lam_key(lam))

        # 2) normalisations from (sub-sampled) pixels of all frames in a group
        samples: dict = {}
        for fi, f in enumerate(frames):
            for disk in f.scene.values:
                for lam_val in f.wavelength.values:
                    for q in f.quantity.values:
                        for lam in _lams(lam_val):
                            flat = get(disk, f.view, lam, q)[0].ravel()
                            samples.setdefault(group(lam, q, fi), []).append(flat[:: max(1, flat.size // 20000)])
        norms = {g: R.Norm(style.stretch, np.concatenate(s), g[0] in R.SIGNED) for g, s in samples.items()}

        # 3) compose frames (cross-fades are weighted sums in display space)
        changing = _changing_params(as_track(self.scene))
        out = []
        for fi, f in enumerate(frames):
            acc, dominant, pixel_au = None, None, 1.0
            for disk, ws in f.scene.weights():
                for lam_val, wl in f.wavelength.weights():
                    for q, wq in f.quantity.weights():
                        w = ws * wl * wq
                        channels = []
                        for lam in _lams(lam_val):
                            img, pixel_au = get(disk, f.view, lam, q)
                            nrm = norms[group(lam, q, fi)]
                            channels.append(nrm(img))
                            if dominant is None or w > dominant[1]:
                                dominant = (nrm, w, q.upper())
                        disp = channels[0] if len(channels) == 1 else np.stack(channels, -1)
                        acc = disp * w if acc is None else acc + disp * w
            out.append(self._finish(f, acc, dominant, pixel_au, style, changing))
        if style.temporal > 0 and len(out) > 2:
            R.temporal_average(out, style.temporal, style.temporal_wrap)
        return out

    def _finish(self, f: Frame, img, dominant, pixel_au, style: R.Style, changing) -> R.FrameData:
        n = img.shape[0]
        if style.gamma != 1.0:
            img = img ** (1.0 / style.gamma)
        nrm, _, q = dominant
        diverging = q in R.SIGNED
        mask = R.coronagraph_mask(n, pixel_au, style.coronagraph)
        if mask is not None:
            m = mask if img.ndim == 2 else mask[..., None]
            img = 0.5 + (img - 0.5) * m if diverging else img * m
        vig = R.vignette(n, style.vignette)
        if vig is not None:
            img = img * (vig if img.ndim == 2 else vig[..., None])

        vectors = None
        if style.vectors and img.ndim == 2:
            disk = f.scene.dominant
            lam = _lams(f.wavelength.dominant)[0]
            x, y, ang, p = R.polarization_vectors(
                self.cache.load(disk, self.sim, _sim_view(disk, f.view), lam), style.vector_step, style.smooth
            )
            vectors = ((x + 0.5) / n, (y + 0.5) / n, ang, p)

        scaled = style.r2 and q != "P"
        ticks = None
        if img.ndim == 2:
            ticks = (np.array([0.0, 1.0]), ["min", "max"]) if scaled else nrm.ticks()
        unit = "arb. units" if scaled else ("" if q == "P" else "Jy/px")
        name = R.LABELS.get(q, q) + (" · r²" if scaled else "")
        return R.FrameData(
            image=np.asarray(img, dtype=np.float32),
            diverging=diverging,
            ticks=ticks,
            cbar_label=name + (f"  [{unit}]" if unit else ""),
            label=self._label(f, style, changing),
            extent_au=pixel_au * n,
            vectors=vectors,
        )

    def _label(self, f: Frame, style: R.Style, changing) -> str:
        if style.label is None:
            return ""
        v = f.view
        lam_txt = " / ".join(_fmt_lam(l) for l in _lams(f.wavelength.dominant))
        disk = f.scene.dominant
        params = {p: disk.get(p) for p in changing}
        q = str(f.quantity.dominant).upper()
        fields = dict(
            frame=f.index,
            inc=v.inclination,
            az=v.azimuth % 360,
            pa=v.position_angle,
            fov=v.fov,
            lam=lam_txt,
            q=R.LABELS.get(q, q),
            params="   ".join(f"{k} = {val:g}" for k, val in params.items()),
            **params,
        )
        if callable(style.label):
            return style.label(fields)
        if style.label != "auto":
            return style.label.format(**fields)
        parts = ([fields["q"]] if q != "I" else []) + [
            f"λ = {lam_txt}",
            f"i = {v.inclination:.0f}°",
            f"φ = {v.azimuth % 360:.0f}°",
        ]
        return "   ".join(parts + ([fields["params"]] if params else []))

    # ── output ──
    def save(
        self,
        path,
        fps: int = 24,
        style=None,
        backend=None,
        workers: int | None = None,
        quality: int = 18,
        gif_width: int = 640,
    ):
        """Simulate what is missing, render and write .mp4/.webm/.gif/.mov (or a PNG folder)."""
        if not self.simulate(backend):
            log.info("waiting for SLURM – call save() again when the jobs are done")
            return None
        style = R.as_style(style)
        t0 = time.time()
        fds = self.render(style)
        out = R.write_video(R.draw_frames(fds, style, workers), path, style, fps, quality, gif_width)
        log.info("%d frames → %s (%.1f s rendering)", len(fds), out, time.time() - t0)
        return out

    def _single(self, index: int) -> Animation:
        f = self.frames()[index]
        return self._copy(scene=[f.scene], camera=FixedPath([f.view]), wavelength=[f.wavelength], quantity=[f.quantity])

    def still(self, path, frame: int = 0, style=None, backend=None):
        """One frame as PNG – the quickest way to try a style. Only simulates that frame."""
        from PIL import Image

        one = self._single(frame)
        if not one.simulate(backend):
            return None
        style = R.as_style(style).with_(temporal=0)
        W, H = style.size
        data = next(R.draw_frames(one.render(style), style, workers=1))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Image.frombytes("RGB", (W, H), data).save(path)
        log.info("still → %s", path)
        return Path(path)

    def contact_sheet(self, path, n: int = 6, cols: int = 3, style=None, width: int = 1200):
        """Grid of ``n`` evenly spaced frames as PNG (requires the simulation data)."""
        from PIL import Image

        style = R.as_style(style).with_(temporal=0)  # frames are not neighbours
        frames = self.frames()
        idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
        fds = self.render(style, [frames[i] for i in idx])
        W, H = style.size
        tiles = [Image.frombytes("RGB", (W, H), b) for b in R.draw_frames(fds, style, workers=1)]
        rows = -(-n // cols)
        tw = width // cols
        th = round(H * tw / W)
        sheet = Image.new("RGB", (tw * cols, th * rows), style.background)
        for k, t in enumerate(tiles):
            sheet.paste(t.resize((tw, th), Image.LANCZOS), ((k % cols) * tw, (k // cols) * th))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        sheet.save(path)
        return Path(path)


def _changing_params(scene_track) -> list[str]:
    """Names of disk parameters that differ between scene variants (for labels)."""
    disks = []
    for m in scene_track:
        for d in m.values:
            if isinstance(d, Disk) and all(d.key != x.key for x in disks):
                disks.append(d)
    if len(disks) < 2:
        return []
    ref, keys = disks[0].to_dict(), []
    for d in disks[1:]:
        dd = d.to_dict()
        keys += [k for k, v in dd.items() if k != "structures" and v != ref[k] and k not in keys]
        for s0, s1 in zip(ref["structures"], dd["structures"], strict=False):
            keys += [f"{s1['name']}.{k}" for k, v in s1.items() if v != s0.get(k) and f"{s1['name']}.{k}" not in keys]
    return keys
