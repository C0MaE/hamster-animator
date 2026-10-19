"""
hamster-animator – good-looking animations from HAMSTER radiative transfer simulations.

    import hamster_animator as ha

    disk = ha.presets.showcase()
    anim = ha.Animation(disk, ha.Camera.orbit(inclination=55, frames=72), wavelength=1.6e-6)
    anim.draft().save("orbit_draft.mp4")          # quick preview
    anim.save("orbit.mp4", style="cinematic")     # final quality
"""

import logging as _logging

from .animation import Animation, Frame
from .camera import Camera, FixedPath, Path, View
from .render import QUANTITIES, STYLES, Stretch, Style
from .scene import Clump, Disk, Gap, Planet, Spiral, Warp, presets
from .sequence import Act, Sequence
from .sim import Cache, Local, Plan, Sim, Slurm, Star
from .steps import Mix, Steps

__version__ = "0.1.0"

__all__ = [
    "QUANTITIES",
    "STYLES",
    "Act",
    "Animation",
    "Cache",
    "Camera",
    "Clump",
    "Disk",
    "FixedPath",
    "Frame",
    "Gap",
    "Local",
    "Mix",
    "Path",
    "Plan",
    "Planet",
    "Sequence",
    "Sim",
    "Slurm",
    "Spiral",
    "Star",
    "Steps",
    "Stretch",
    "Style",
    "View",
    "Warp",
    "presets",
]

_log = _logging.getLogger("hamster_animator")
if not _log.handlers:
    _h = _logging.StreamHandler()
    _h.setFormatter(_logging.Formatter("[hamster-animator] %(message)s"))
    _log.addHandler(_h)
    _log.setLevel(_logging.INFO)
    _log.propagate = False
