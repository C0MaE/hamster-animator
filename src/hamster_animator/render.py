"""
Rendering: turn detector maps into good-looking frames and videos.

1. Main process: load the Stokes maps of each frame, compute the displayed
   quantity (I, PI, P, Q_phi, RGB, …), stretch (log/asinh/…) and normalise to
   [0, 1]. Cross-fades happen here in image space. Pure NumPy, fast.
2. Worker processes: Matplotlib draws image + overlays (colour bar, scale
   bar, label, polarisation vectors) into a reused figure and returns RGB bytes.
3. The bytes are piped straight into ffmpeg (MP4/WebM/GIF/MOV) – no
   intermediate files, no FuncAnimation.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import os
import shutil
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

# ── Quantities ────────────────────────────────────────────────────────────────

BASIC = ("I", "Q", "U", "V", "I_DIRECT", "I_SCAT")
SIGNED = {"Q", "U", "V", "Q_PHI", "U_PHI"}
LABELS = {
    "I": "I",
    "Q": "Q",
    "U": "U",
    "V": "V",
    "I_DIRECT": "I (direct)",
    "I_SCAT": "I (scattered)",
    "PI": "PI",
    "P": "pol. degree",
    "Q_PHI": "Q_φ",
    "U_PHI": "U_φ",
}
QUANTITIES = tuple(LABELS)


def _smooth(img: np.ndarray, sigma: float) -> np.ndarray:
    if not sigma:
        return img
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(img, sigma)


def _radius2(n: int) -> np.ndarray:
    c = (n - 1) / 2.0
    y, x = np.mgrid[0:n, 0:n]
    return ((x - c) ** 2 + (y - c) ** 2 + 1.0) / c**2


def quantity(maps: dict, q: str, sigma: float = 0.0, r2: bool = False) -> np.ndarray:
    """Scalar map for one (view, wavelength). ``r2`` multiplies by the squared image-plane radius."""
    q = q.upper()
    g = lambda s: _smooth(maps[s], sigma)
    if q in BASIC:
        img = g(q)
    elif q == "PI":
        img = np.hypot(g("Q"), g("U"))
    elif q == "P":
        I = g("I")
        floor = np.percentile(I[I > 0], 20) if np.any(I > 0) else 0.0
        return np.where(I > floor, np.hypot(g("Q"), g("U")) / np.maximum(I, 1e-300), 0.0)
    elif q in ("Q_PHI", "U_PHI"):
        Q, U = g("Q"), g("U")
        n = Q.shape[0]
        c = (n - 1) / 2.0
        y, x = np.mgrid[0:n, 0:n]
        phi = np.arctan2(y - c, x - c)
        c2, s2 = np.cos(2 * phi), np.sin(2 * phi)
        img = (-Q * c2 - U * s2) if q == "Q_PHI" else (Q * s2 - U * c2)
    else:
        raise ValueError(f"Unknown quantity '{q}'. Available: {', '.join(QUANTITIES)}")
    return img * _radius2(img.shape[0]) if r2 else img


def polarization_vectors(maps: dict, step: int, sigma: float, min_percentile: float = 45.0):
    """Polarisation vectors on a coarse grid: (x, y, angle, degree) in detector pixels."""
    Q, U, I = (_smooth(maps[s], max(sigma, step / 2)) for s in ("Q", "U", "I"))
    idx = np.arange(step // 2, I.shape[0], step)
    yy, xx = np.meshgrid(idx, idx, indexing="ij")
    q, u, i = Q[yy, xx], U[yy, xx], I[yy, xx]
    pos = I[I > 0]
    ok = i > (np.percentile(pos, min_percentile) if pos.size else np.inf)
    p = np.clip(np.hypot(q, u) / np.maximum(i, 1e-300), 0, 1)
    return xx[ok], yy[ok], 0.5 * np.arctan2(u, q)[ok], p[ok]


# ── Stretch / normalisation ───────────────────────────────────────────────────


@dataclass(frozen=True)
class Stretch:
    kind: str = "log"  # log | asinh | sqrt | linear  (signed quantities always use a symmetric asinh)
    lo: float = 1.0  # lower percentile (of positive values)
    hi: float = 99.7  # upper percentile
    decades: float | None = 4.0  # log: maximum dynamic range
    soft: float = 0.02  # asinh: knee relative to the maximum


class Norm:
    """Maps data values to [0, 1] (signed quantities: [-vmax, vmax] → [0, 1])."""

    def __init__(self, stretch: Stretch, samples: np.ndarray, signed: bool):
        self.s, self.signed = stretch, signed
        x = samples[np.isfinite(samples)]
        if signed:
            a = np.abs(x[x != 0])
            self.vmax = float(np.percentile(a, stretch.hi)) if a.size else 1.0
            self.vmin = -self.vmax
            return
        pos = x[x > 0]
        if not pos.size:
            self.vmin, self.vmax = 1e-30, 1.0
            return
        self.vmax = float(np.percentile(pos, stretch.hi))
        self.vmin = float(np.percentile(pos, stretch.lo))
        if stretch.kind == "log" and stretch.decades:
            self.vmin = max(self.vmin, self.vmax * 10**-stretch.decades)
        if stretch.kind in ("linear", "sqrt", "asinh"):
            self.vmin = 0.0
        self.vmin = min(self.vmin, self.vmax * (1 - 1e-6))

    def __call__(self, x):
        if self.signed:
            a = self.s.soft * self.vmax
            return np.clip(0.5 + 0.5 * np.arcsinh(x / a) / math.asinh(self.vmax / a), 0, 1)
        k = self.s.kind
        if k == "log":
            y = (np.log10(np.maximum(x, self.vmin)) - math.log10(self.vmin)) / math.log10(self.vmax / self.vmin)
        elif k == "asinh":
            a = self.s.soft * self.vmax
            y = np.arcsinh(np.maximum(x, 0) / a) / math.asinh(self.vmax / a)
        elif k == "sqrt":
            y = np.sqrt(np.maximum(x, 0) / self.vmax)
        elif k == "linear":
            y = (x - self.vmin) / (self.vmax - self.vmin)
        else:
            raise ValueError(f"Unknown stretch '{k}'")
        return np.clip(y, 0, 1)

    def ticks(self, n: int = 5):
        """(positions in [0, 1], labels) for the colour bar."""
        if self.signed:
            vals = np.array([-self.vmax, -self.vmax * self.s.soft * 5, 0, self.vmax * self.s.soft * 5, self.vmax])
        elif self.s.kind == "log":
            vals = 10.0 ** np.arange(math.ceil(math.log10(self.vmin)), math.floor(math.log10(self.vmax)) + 1)
            if len(vals) > n:
                vals = vals[:: math.ceil(len(vals) / n)]
        else:
            vals = np.linspace(self.vmin, self.vmax, n)
        return self(vals), [_fmt(v) for v in vals]


def _fmt(v: float) -> str:
    if v == 0:
        return "0"
    e = math.floor(math.log10(abs(v)))
    if abs(e) <= 2:
        return f"{v:.{max(0, 1 - e)}f}".rstrip("0").rstrip(".")
    m = v / 10**e
    return f"{m:.0f}e{e}" if abs(m - round(m)) < 0.05 else f"{m:.1f}e{e}"


# ── Style ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Style:
    size: tuple = (1080, 1080)  # output pixels (even numbers for yuv420p)
    background: str = "#000000"
    text: str = "#f2f2f2"
    accent: str = "#ffffff"
    colormap: str = "inferno"
    diverging_colormap: str = "RdBu_r"
    font: str = "DejaVu Sans"
    font_size: float = 15.0
    interpolation: str = "bicubic"
    stretch: Stretch = Stretch()
    norm: str = "group"  # group (per quantity + wavelength) | global (per quantity) | frame
    r2: bool = False  # multiply by r² – reveals the outer disk
    coronagraph: float = 0.0  # mask radius around the star in au
    smooth: float = 0.0  # Gaussian smoothing of the Stokes maps (detector pixels)
    temporal: float = 0.0  # Gaussian averaging over neighbouring frames (sigma in frames)
    temporal_wrap: bool = False  # wrap around (for seamless loops)
    colorbar: bool = True
    scalebar: object = "auto"  # length in au, "auto" or None
    label: object = "auto"  # "auto", a format string, a callable(fields) -> str, or None
    title: str | None = None
    vectors: bool = False  # polarisation vectors
    vector_step: int = 12  # detector pixels between vectors
    vector_color: str = "#7fe3ff"
    vignette: float = 0.0  # 0..1, darkens the corners
    gamma: float = 1.0

    def with_(self, **kw) -> Style:
        return replace(self, **kw)


STYLES = {
    "dark": Style(),
    "cinematic": Style(
        colormap="magma",
        interpolation="lanczos",
        vignette=0.3,
        smooth=0.8,
        temporal=1.0,
        r2=True,
        coronagraph=12.0,
        stretch=Stretch("asinh", lo=0, hi=99.6, soft=0.08),
        font_size=17,
    ),
    "paper": Style(
        background="#ffffff",
        text="#111111",
        accent="#111111",
        colormap="viridis",
        interpolation="nearest",
        font="DejaVu Serif",
    ),
    "polarimetry": Style(
        colormap="afmhot",
        vectors=True,
        r2=True,
        coronagraph=12.0,
        smooth=1.0,
        stretch=Stretch("asinh", lo=0, hi=99.5, soft=0.08),
    ),
    "ice": Style(
        colormap="GnBu_r",
        vignette=0.25,
        smooth=0.5,
        r2=True,
        coronagraph=12.0,
        stretch=Stretch("asinh", lo=0, hi=99.6, soft=0.08),
    ),
}


def as_style(style) -> Style:
    if style is None:
        return STYLES["dark"]
    if isinstance(style, str):
        try:
            return STYLES[style]
        except KeyError:
            raise ValueError(f"Unknown style '{style}'. Available: {', '.join(STYLES)}") from None
    return style


# ── Frame data ────────────────────────────────────────────────────────────────


@dataclass
class FrameData:
    image: np.ndarray  # [ny, nx] or [ny, nx, 3], values in [0, 1]
    diverging: bool = False
    ticks: tuple | None = None
    cbar_label: str = ""
    label: str = ""
    extent_au: float = 1.0
    vectors: tuple | None = None  # (x, y, angle, degree), x/y in [0, 1]


def coronagraph_mask(n: int, pixel_au: float, radius_au: float):
    if radius_au <= 0:
        return None
    c = (n - 1) / 2.0
    y, x = np.mgrid[0:n, 0:n]
    r = np.hypot(x - c, y - c) * pixel_au
    return np.clip((r - radius_au) / max(pixel_au, 1e-9), 0, 1)


def vignette(n: int, strength: float):
    if strength <= 0:
        return None
    c = (n - 1) / 2.0
    y, x = np.mgrid[0:n, 0:n]
    r = np.hypot(x - c, y - c) / c
    return np.clip(1.0 - strength * np.clip(r - 0.55, 0, None) ** 2 / 0.45**2, 0.0, 1.0)


def temporal_average(frames: list[FrameData], sigma: float, wrap: bool) -> None:
    """Gaussian average of neighbouring frames in image space (acts like a light motion blur)."""
    k = int(math.ceil(2.5 * sigma))
    w = np.exp(-0.5 * (np.arange(-k, k + 1) / sigma) ** 2)
    imgs = [f.image for f in frames]
    n = len(imgs)
    for i, f in enumerate(frames):
        acc, norm = 0.0, 0.0
        for d, wd in zip(range(-k, k + 1), w, strict=True):
            j = (i + d) % n if wrap else i + d
            if 0 <= j < n and imgs[j].shape == imgs[i].shape:
                acc = acc + wd * imgs[j]
                norm += wd
        f.image = (acc / norm).astype(np.float32)


# ── Drawing ───────────────────────────────────────────────────────────────────


def _nice_length(x: float) -> float:
    e = 10 ** math.floor(math.log10(x))
    return next(m * e for m in (1, 2, 5, 10) if m * e >= x)


class Canvas:
    """A reusable Matplotlib figure; ``draw()`` returns raw RGB bytes."""

    def __init__(self, style: Style):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection

        self.style = st = style
        W, H = st.size
        plt.rcParams["font.family"] = st.font
        self.fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=st.background)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_axis_off()
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.im = self.ax.imshow(
            np.zeros((2, 2)),
            origin="lower",
            cmap=st.colormap,
            vmin=0,
            vmax=1,
            interpolation=st.interpolation,
            extent=(0, 1, 0, 1),
            aspect="auto",
        )
        fs = st.font_size
        self.label = self.ax.text(0.03, 0.03, "", color=st.text, fontsize=fs, transform=self.ax.transAxes)
        self.ax.text(
            0.03,
            0.97,
            st.title or "",
            color=st.text,
            fontsize=fs * 1.25,
            weight="bold",
            transform=self.ax.transAxes,
            va="top",
        )
        (self.sb_line,) = self.ax.plot([], [], color=st.accent, lw=2.5, solid_capstyle="butt")
        self.sb_text = self.ax.text(0, 0, "", color=st.accent, fontsize=fs * 0.9, ha="center", va="bottom")
        self.vec = LineCollection([], colors=st.vector_color, linewidths=1.6, capstyle="round")
        self.ax.add_collection(self.vec)
        if st.colorbar:
            self.cax = self.fig.add_axes([0.93, 0.25, 0.018, 0.5])
            self.cbar = self.cax.imshow(
                np.linspace(0, 1, 256)[:, None], origin="lower", aspect="auto", cmap=st.colormap, extent=(0, 1, 0, 1)
            )
            self.cax.set_xticks([])
            self.cax.yaxis.tick_left()
            self.cax.tick_params(colors=st.text, labelsize=fs * 0.7, length=3)
            for sp in self.cax.spines.values():
                sp.set_edgecolor(st.text)
                sp.set_linewidth(0.6)
            self.cbar_label = self.fig.text(
                0.965, 0.5, "", color=st.text, fontsize=fs * 0.75, rotation=90, va="center", ha="center"
            )

    def draw(self, fd: FrameData) -> bytes:
        st = self.style
        cmap = st.diverging_colormap if fd.diverging else st.colormap
        self.im.set_data(fd.image)
        self.im.set_cmap(cmap)
        self.label.set_text(fd.label)

        if st.scalebar is not None and fd.extent_au > 0:
            L = _nice_length(fd.extent_au / 5) if st.scalebar == "auto" else float(st.scalebar)
            w = L / fd.extent_au
            self.sb_line.set_data([0.97 - w, 0.97], [0.93, 0.93])
            self.sb_text.set_position((0.97 - w / 2, 0.94))
            self.sb_text.set_text(f"{L:g} au")

        if fd.vectors is not None and len(fd.vectors[0]):
            x, y, ang, p = fd.vectors
            half = 0.02 * p / max(p.max(), 1e-9)
            dx, dy = -np.sin(ang) * half, np.cos(ang) * half
            self.vec.set_segments(np.stack([np.stack([x - dx, y - dy], -1), np.stack([x + dx, y + dy], -1)], 1))
        else:
            self.vec.set_segments([])

        if st.colorbar:
            show = fd.ticks is not None and fd.image.ndim == 2
            self.cax.set_visible(show)
            self.cbar_label.set_visible(show)
            if show:
                self.cbar.set_cmap(cmap)
                pos, labels = fd.ticks
                keep = (np.asarray(pos) >= 0) & (np.asarray(pos) <= 1)
                self.cax.set_yticks(np.asarray(pos)[keep])
                self.cax.set_yticklabels(np.asarray(labels)[keep])
                self.cbar_label.set_text(fd.cbar_label)

        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[..., :3].tobytes()

    def close(self) -> None:
        import matplotlib.pyplot as plt

        plt.close(self.fig)


_WORKER: dict = {}


def _draw_in_worker(i: int) -> bytes:
    if "canvas" not in _WORKER:
        _WORKER["canvas"] = Canvas(_WORKER["style"])
    return _WORKER["canvas"].draw(_WORKER["frames"][i])


def draw_frames(frames: list[FrameData], style: Style, workers: int | None = None):
    """Yield RGB bytes for every frame, in order. Draws in parallel with forked workers."""
    workers = workers or min(len(frames), os.cpu_count() or 1)
    if workers <= 1 or len(frames) < 4 or "fork" not in mp.get_all_start_methods():
        canvas = Canvas(style)
        try:
            for fd in frames:
                yield canvas.draw(fd)
        finally:
            canvas.close()
        return
    _WORKER.clear()
    _WORKER.update(style=style, frames=frames)
    with mp.get_context("fork").Pool(workers) as pool:
        yield from pool.imap(_draw_in_worker, range(len(frames)), chunksize=max(1, len(frames) // (workers * 4)))


# ── Output ────────────────────────────────────────────────────────────────────


def write_video(chunks, path, style: Style, fps: int, quality: int = 18, gif_width: int = 640) -> Path:
    """Write frames to .mp4/.webm/.gif/.mov via ffmpeg, or to a PNG sequence (path without suffix)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    W, H = style.size
    ext = path.suffix.lower()

    if ext in ("", ".png"):
        from PIL import Image

        out = path.with_suffix("") if ext else path
        out.mkdir(parents=True, exist_ok=True)
        for i, b in enumerate(chunks):
            Image.frombytes("RGB", (W, H), b).save(out / f"frame_{i:04d}.png")
        return out

    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found – install it or save as a PNG sequence (path without suffix).")
    encoders = {
        ".mp4": [
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            str(quality),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ],
        ".webm": ["-c:v", "libvpx-vp9", "-crf", str(quality + 12), "-b:v", "0", "-pix_fmt", "yuv420p"],
        ".gif": [
            "-vf",
            f"scale={gif_width}:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];"
            "[b][p]paletteuse=dither=bayer:bayer_scale=4",
        ],
        ".mov": ["-c:v", "prores_ks", "-profile:v", "3"],
    }
    if ext not in encoders:
        raise ValueError(f"Unsupported format '{ext}'. Use {', '.join(encoders)} or a folder for PNGs.")
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{W}x{H}",
        "-r",
        str(fps),
        "-i",
        "-",
        *encoders[ext],
        str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for b in chunks:
            proc.stdin.write(b)
    finally:
        proc.stdin.close()
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed (exit {proc.returncode})")
    return path


def concat_videos(paths, out) -> Path:
    """Join videos with identical encoding settings (e.g. from ``write_video``) without re-encoding."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found – it is needed to join videos.")
    listing = out.with_name(out.name + ".txt")
    listing.write_text("".join(f"file '{Path(p).resolve()}'\n" for p in paths))
    try:
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy"]
        res = subprocess.run([*cmd, str(out)])
    finally:
        listing.unlink(missing_ok=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to join the videos (exit {res.returncode})")
    return out
