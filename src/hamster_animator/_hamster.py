"""
Everything that touches a HAMSTER installation lives here.

HAMSTER itself is not a dependency of this package on PyPI. We only need

* the compiled binary (``bin/HAMSTER``), which must be run from the HAMSTER
  repository root, and
* the ``hamster-tools`` Python package (``import tools``) for writing task
  files, building grids and reading detector FITS files.

All imports of ``tools`` are lazy, so the rest of this package (camera paths,
scenes, planning, rendering) works and can be tested without HAMSTER.
"""

from __future__ import annotations

import contextlib
import io
import os
from functools import cache
from pathlib import Path

AU_TO_M = 1.495978707e11
PC_TO_M = 3.085677581491367e16


class HamsterNotFound(RuntimeError):
    pass


def _tools():
    try:
        import tools
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise HamsterNotFound(
            "The HAMSTER Python tools are not installed. Inside your HAMSTER checkout run\n"
            "    python -m pip install -e .\n"
            "(see the 'HAMSTER-Tools' section of the HAMSTER README)."
        ) from exc
    return tools


def _is_root(path: Path) -> bool:
    return (path / "src" / "main.cpp").is_file()


@cache
def find_root(explicit: str | None = None) -> Path:
    """
    Locate the HAMSTER repository root, in this order:

    1. ``explicit`` argument (e.g. ``Local(hamster="~/HAMSTER")``)
    2. environment variable ``HAMSTER_ROOT``
    3. the checkout the installed ``hamster-tools`` package comes from
    4. the current working directory and its parents
    """
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    if os.environ.get("HAMSTER_ROOT"):
        candidates.append(Path(os.environ["HAMSTER_ROOT"]).expanduser())
    try:
        import tools

        candidates.append(Path(tools.__file__).resolve().parents[1])
    except ImportError:
        pass
    cwd = Path.cwd().resolve()
    candidates += [cwd, *cwd.parents]
    for c in candidates:
        if _is_root(c):
            return c.resolve()
    raise HamsterNotFound(
        "Could not find the HAMSTER repository. Set HAMSTER_ROOT=/path/to/HAMSTER "
        "or pass hamster='/path/to/HAMSTER' to the backend."
    )


def find_binary(root: Path, binary: str | None = None) -> Path:
    b = binary or os.environ.get("HAMSTER_BIN") or "bin/HAMSTER"
    p = Path(b).expanduser()
    p = p if p.is_absolute() else root / p
    if not p.is_file():
        raise HamsterNotFound(f"HAMSTER binary not found at {p}. Compile HAMSTER first (./compile.sh).")
    return p


def resolve_input(path: str, root: Path) -> str:
    """Input files such as dust tables may be given relative to the HAMSTER root."""
    p = Path(path).expanduser()
    return str(p if p.is_absolute() else root / p)


def task_file():
    _tools()
    from tools.task import TaskFile

    return TaskFile()


def write_task_file(tf, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        tf.write(str(path))


def slurm_script():
    _tools()
    from tools.slurm import SlurmBatchScript

    return SlurmBatchScript()


def write_slurm_script(script, path: Path) -> None:
    with contextlib.redirect_stdout(io.StringIO()):
        script.write(str(path))


def spherical_grid(**kwargs):
    _tools()
    from tools.grid import GridSpherical

    return GridSpherical(**kwargs)


def save_grid(grid, path: Path) -> None:
    with contextlib.redirect_stdout(io.StringIO()):
        grid.save(str(path))


def read_detector(path: str):
    """Return (wavelengths [m], {stokes: [n_lam, ny, nx] float32}, pixel size [m])."""
    _tools()
    import numpy as np
    from tools.detector import Detector

    det = Detector(path)
    try:
        wl = np.asarray(det.getWavelengths(), dtype=float)
        stokes = {}
        for s in STOKES:
            stokes[s] = np.stack([np.asarray(det.getMap(i, s), dtype=np.float32) for i in range(len(wl))])
        px, _ = det.getPixelSize()
    finally:
        det.close()
    return wl, stokes, float(px)


STOKES = ("I", "Q", "U", "V", "I_DIRECT", "I_SCAT")
