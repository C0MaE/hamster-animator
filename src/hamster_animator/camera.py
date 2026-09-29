"""
Camera paths.

A :class:`View` is one detector setting (inclination, azimuth, position angle,
field of view). A :class:`Path` is a sequence of segments chained with
``.then()`` or ``+``. Parameters a segment does not mention are carried over
from the end of the previous segment, so transitions are seamless::

    cam = (Camera.orbit(inclination=45, frames=72)
           .then(Camera.tilt(to=85, frames=30))
           .then(Camera.zoom(to=150, frames=30)))

Every frame becomes one HAMSTER detector; identical views are de-duplicated
when planning. Angles are in degrees, lengths in au.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from dataclasses import dataclass

EASINGS: dict[str, Callable[[float], float]] = {
    "linear": lambda t: t,
    "in": lambda t: t * t,
    "out": lambda t: 1.0 - (1.0 - t) ** 2,
    "in_out": lambda t: t * t * (3.0 - 2.0 * t),
    "sine": lambda t: 0.5 - 0.5 * math.cos(math.pi * t),
    "expo": lambda t: (
        t if t in (0.0, 1.0) else (0.5 * 2 ** (20 * t - 10) if t < 0.5 else 1 - 0.5 * 2 ** (-20 * t + 10))
    ),
}

VIEW_PARAMS = ("inclination", "azimuth", "position_angle", "fov", "center_x", "center_y")


def easing(ease) -> Callable[[float], float]:
    if callable(ease):
        return ease
    try:
        return EASINGS[ease]
    except KeyError:
        raise ValueError(f"Unknown easing '{ease}'. Available: {', '.join(EASINGS)}") from None


@dataclass(frozen=True)
class View:
    inclination: float = 45.0  # 0 = face-on
    azimuth: float = 0.0
    position_angle: float = 90.0
    fov: float | None = None  # image width in au (None = derived from the disk)
    center_x: float = 0.0  # image centre offset in au
    center_y: float = 0.0

    @property
    def key(self) -> tuple:
        r = lambda x: None if x is None else round(float(x), 6)
        return (
            r(self.inclination),
            r(self.azimuth % 360.0),
            r(self.position_angle),
            r(self.fov),
            r(self.center_x),
            r(self.center_y),
        )

    @property
    def key_str(self) -> str:
        return "v_" + "_".join("n" if x is None else f"{x:g}" for x in self.key)


@dataclass
class Segment:
    frames: int
    changes: dict  # param -> (start | None, end) or a constant
    ease: object = "in_out"
    endpoint: bool = True  # whether the last frame reaches the end value

    def resolve(self, state: View) -> list[View]:
        ease = easing(self.ease)
        denom = (self.frames - 1) if self.endpoint else self.frames
        views = []
        for i in range(self.frames):
            t = ease(i / denom) if denom > 0 else 1.0
            kw = {}
            for p, spec in self.changes.items():
                if isinstance(spec, tuple):
                    a, b = spec
                    a = getattr(state, p) if a is None else a
                    kw[p] = b if a is None else (a if b is None else a + (b - a) * t)
                else:
                    kw[p] = spec
            views.append(dataclasses.replace(state, **kw))
        return views

    def end_state(self, state: View) -> View:
        kw = {
            p: (spec[1] if spec[1] is not None else getattr(state, p)) if isinstance(spec, tuple) else spec
            for p, spec in self.changes.items()
        }
        return dataclasses.replace(state, **kw)


class Path:
    """A chain of camera segments. Use the :class:`Camera` factory to create one."""

    def __init__(self, segments=None, start: View | None = None):
        self.segments = list(segments or [])
        self.start = start or View()

    def then(self, other: Path) -> Path:
        if isinstance(other, FixedPath):
            return FixedPath(self.views() + other.views())
        return Path(self.segments + other.segments, self.start)

    __add__ = then

    def __mul__(self, n: int) -> Path:
        return Path(self.segments * int(n), self.start)

    def starting_at(self, view: View) -> Path:
        return Path(self.segments, view)

    def with_defaults(self, **defaults) -> Path:
        """Fill start values that are still ``None`` (e.g. ``fov``)."""
        kw = {k: v for k, v in defaults.items() if getattr(self.start, k) is None}
        return Path(self.segments, dataclasses.replace(self.start, **kw))

    def views(self) -> list[View]:
        state, out = self.start, []
        for seg in self.segments:
            out += seg.resolve(state)
            state = seg.end_state(state)
        return out

    def __len__(self) -> int:
        return sum(s.frames for s in self.segments)

    def reverse(self) -> FixedPath:
        return FixedPath(self.views()[::-1])

    def boomerang(self) -> FixedPath:
        """Play forward and back (seamless ping-pong loop)."""
        v = self.views()
        return FixedPath(v + v[-2:0:-1])


class FixedPath(Path):
    """A path made of already resolved views."""

    def __init__(self, views):
        views = list(views)
        super().__init__([], views[0] if views else View())
        self._views = views

    def views(self) -> list[View]:
        return list(self._views)

    def __len__(self) -> int:
        return len(self._views)

    def then(self, other: Path) -> FixedPath:
        nxt = other if isinstance(other, FixedPath) else Path(other.segments, self._views[-1])
        return FixedPath(self._views + nxt.views())

    __add__ = then

    def with_defaults(self, **defaults) -> FixedPath:
        return FixedPath(
            [
                dataclasses.replace(v, **{k: d for k, d in defaults.items() if getattr(v, k) is None})
                for v in self._views
            ]
        )


def _spec(value):
    if isinstance(value, (tuple, list)):
        return (value[0], value[1])
    return value


class Camera:
    """Factory for camera segments."""

    @staticmethod
    def still(frames: int = 1, **view) -> Path:
        """Hold one view for ``frames`` frames (costs a single detector)."""
        return Path([Segment(frames, dict(view))])

    hold = still

    @staticmethod
    def move(frames: int, ease="in_out", endpoint: bool = True, **params) -> Path:
        """Interpolate any view parameters: ``move(30, inclination=(20, 80), fov=(600, 200))``."""
        changes = {}
        for p, v in params.items():
            if p not in VIEW_PARAMS:
                raise ValueError(f"Unknown view parameter '{p}'. Allowed: {VIEW_PARAMS}")
            changes[p] = tuple(v) if isinstance(v, (list, tuple)) else (None, v)
        return Path([Segment(frames, changes, ease, endpoint)])

    @staticmethod
    def orbit(frames: int = 72, inclination=None, azimuth=(0.0, 360.0), ease="linear", **fixed) -> Path:
        """Fly around the disk. Loops seamlessly (the end point is left out)."""
        a0, a1 = azimuth if isinstance(azimuth, (tuple, list)) else (None, azimuth)
        changes = {"azimuth": (a0, a1)}
        if inclination is not None:
            changes["inclination"] = _spec(inclination)
        changes.update({k: _spec(v) for k, v in fixed.items()})
        return Path([Segment(frames, changes, ease, endpoint=False)])

    @staticmethod
    def tilt(frames: int = 30, to: float = 90.0, frm: float | None = None, ease="in_out") -> Path:
        return Path([Segment(frames, {"inclination": (frm, to)}, ease)])

    @staticmethod
    def zoom(frames: int = 30, to: float = 150.0, frm: float | None = None, ease="in_out", center=None) -> Path:
        changes = {"fov": (frm, to)}
        if center is not None:
            changes["center_x"] = (None, center[0])
            changes["center_y"] = (None, center[1])
        return Path([Segment(frames, changes, ease)])

    @staticmethod
    def roll(frames: int = 30, to: float = 180.0, frm: float | None = None, ease="in_out") -> Path:
        return Path([Segment(frames, {"position_angle": (frm, to)}, ease)])

    @staticmethod
    def flyby(frames: int = 90, inclination=(10.0, 80.0), fov=None, turns: float = 1.0, ease="in_out") -> Path:
        """Orbit while tilting (and optionally zooming)."""
        changes = {"azimuth": (None, 360.0 * turns), "inclination": tuple(inclination)}
        if fov is not None:
            changes["fov"] = tuple(fov)
        return Path([Segment(frames, changes, ease)])

    @staticmethod
    def views(views) -> FixedPath:
        return FixedPath([v if isinstance(v, View) else View(**v) for v in views])


def as_path(camera) -> Path:
    if camera is None:
        return Camera.still()
    if isinstance(camera, Path):
        return camera
    if isinstance(camera, View):
        return Camera.views([camera])
    if isinstance(camera, dict):
        return Camera.views([View(**camera)])
    raise TypeError(f"camera must be a View, dict or Path, not {type(camera).__name__}")
