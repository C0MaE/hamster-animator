"""
Discrete tracks with cross-fades.

Wavelengths, scene variants and displayed quantities are discrete – every
scene variant costs a separate simulation. To keep the animation smooth,
:class:`Steps` cross-fades between neighbouring steps in image space, which
costs no simulation time::

    Steps([0.5e-6, 1.6e-6, 10e-6, 870e-6], hold=12, fade=8)
    Steps(disk.sweep("planet.azimuth", range(0, 360, 30)), hold=1, fade=3, loop=True)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .camera import easing


@dataclass(frozen=True)
class Mix:
    """One frame of a track: value ``a``, value ``b`` and blend factor ``t`` (0 → only ``a``)."""

    a: Any
    b: Any = None
    t: float = 0.0

    @property
    def values(self) -> tuple:
        return (self.a,) if self.b is None or self.t <= 0.0 else (self.a, self.b)

    @property
    def dominant(self):
        return self.a if self.b is None or self.t < 0.5 else self.b

    def weights(self) -> list[tuple[Any, float]]:
        if self.b is None or self.t <= 0.0:
            return [(self.a, 1.0)]
        if self.t >= 1.0:
            return [(self.b, 1.0)]
        return [(self.a, 1.0 - self.t), (self.b, self.t)]


class Steps:
    def __init__(self, values, hold: int = 1, fade: int = 0, loop: bool = False, ease="sine"):
        self.values = list(values)
        if not self.values:
            raise ValueError("Steps needs at least one value.")
        self.hold, self.fade, self.loop = int(hold), int(fade), bool(loop)
        self.ease = easing(ease)

    def mixes(self) -> list[Mix]:
        out, n = [], len(self.values)
        for i, v in enumerate(self.values):
            out += [Mix(v)] * self.hold
            if self.fade > 0 and (i + 1 < n or (self.loop and n > 1)):
                nxt = self.values[(i + 1) % n]
                out += [Mix(v, nxt, self.ease(k / (self.fade + 1))) for k in range(1, self.fade + 1)]
        return out

    def __len__(self) -> int:
        return len(self.mixes())

    def __repr__(self) -> str:
        return f"Steps({len(self.values)} values, hold={self.hold}, fade={self.fade}, loop={self.loop})"


def as_track(value) -> list[Mix]:
    if isinstance(value, Steps):
        return value.mixes()
    if isinstance(value, list):
        return [v if isinstance(v, Mix) else Mix(v) for v in value]
    return [Mix(value)]
