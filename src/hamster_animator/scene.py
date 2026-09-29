"""
Disk models.

A :class:`Disk` is a Shakura-Sunyaev dust disk plus any number of azimuthal
structures (spirals, gaps, planets, clumps, warps). Densities are evaluated
vectorised with NumPy on all grid cells at once, which is orders of magnitude
faster than calling a Python function per cell.

Disks are immutable and carry a stable content hash (:attr:`Disk.key`) that
the animator uses to cache grids, heating runs and detector maps.

All lengths are in au, all angles in degrees.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ._hamster import AU_TO_M, save_grid, spherical_grid


def _wrap(angle):
    """Map angles to [-pi, pi)."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


# ── Structures ────────────────────────────────────────────────────────────────
# ``modulate`` receives cylindrical coordinates (R [au], phi [rad], z [au]) and
# the current density and returns the new density.


@dataclass(frozen=True)
class Spiral:
    """Logarithmic spiral arms as a multiplicative density enhancement."""

    arms: int = 2
    pitch: float = 15.0  # pitch angle
    amplitude: float = 3.0  # peak relative over-density
    width: float = 20.0  # azimuthal arm width
    r_start: float = 10.0  # radius where the arms start
    phase: float = 0.0  # rotation of the pattern
    name: str = "spiral"

    def modulate(self, R, phi, z, rho, disk):
        R_safe = np.maximum(R, 1e-10)
        phi_arm = np.log(R_safe / self.r_start) / math.tan(math.radians(self.pitch))
        sigma = math.radians(self.width) / 2.0
        boost = np.zeros_like(rho)
        for k in range(self.arms):
            d = _wrap(phi - math.radians(self.phase) - phi_arm - 2.0 * np.pi * k / self.arms)
            boost = np.maximum(boost, np.exp(-0.5 * (d / sigma) ** 2))
        taper = 1.0 / (1.0 + np.exp(-(R - self.r_start) / (0.2 * self.r_start)))
        return rho * (1.0 + self.amplitude * boost * taper)


@dataclass(frozen=True)
class Gap:
    """Axisymmetric gap with a Gaussian radial profile."""

    radius: float = 50.0
    width: float = 8.0
    depth: float = 0.95  # 1.0 = completely empty
    name: str = "gap"

    def modulate(self, R, phi, z, rho, disk):
        return rho * (1.0 - self.depth * np.exp(-0.5 * ((R - self.radius) / self.width) ** 2))


@dataclass(frozen=True)
class Clump:
    """Compact dust clump, e.g. a circumplanetary disk or (with elongation > 1) a vortex."""

    radius: float = 80.0
    azimuth: float = 0.0
    size: float = 5.0  # Gaussian sigma
    amplitude: float = 20.0  # over-density relative to the local midplane density
    elongation: float = 1.0  # > 1 stretches the clump azimuthally
    name: str = "clump"

    def modulate(self, R, phi, z, rho, disk):
        dphi = _wrap(phi - math.radians(self.azimuth))
        ds = R * dphi / self.elongation
        d2 = ((R - self.radius) ** 2 + ds**2 + z**2) / self.size**2
        rho_mid = disk.midplane_density(np.full(1, self.radius))[0]
        return rho + self.amplitude * rho_mid * np.exp(-0.5 * d2)


@dataclass(frozen=True)
class Planet:
    """Gap + clump at the planet position + optional one-armed spiral wake."""

    radius: float = 50.0
    azimuth: float = 0.0
    gap_width: float = 6.0
    gap_depth: float = 0.9
    clump_size: float = 3.0
    clump_amplitude: float = 30.0
    wake: float = 1.0  # wake amplitude (0 = off)
    name: str = "planet"

    def modulate(self, R, phi, z, rho, disk):
        rho = Gap(self.radius, self.gap_width, self.gap_depth).modulate(R, phi, z, rho, disk)
        if self.wake > 0.0:
            R_safe = np.maximum(R, 1e-10)
            h_p = disk.scale_height(self.radius) / self.radius
            phi_wake = -np.sign(R - self.radius) * np.abs(np.log(R_safe / self.radius)) / max(h_p, 1e-3) * 0.5
            d = _wrap(phi - math.radians(self.azimuth) - phi_wake)
            boost = np.exp(-0.5 * (d / 0.15) ** 2) * np.exp(-np.abs(R - self.radius) / (1.5 * self.radius))
            rho = rho * (1.0 + self.wake * boost)
        return Clump(self.radius, self.azimuth, self.clump_size, self.clump_amplitude).modulate(R, phi, z, rho, disk)


@dataclass(frozen=True)
class Warp:
    """Inner disk tilted against the outer disk – casts shadows on the outer disk."""

    r_break: float = 20.0  # transition radius
    tilt: float = 15.0  # tilt of the inner disk
    twist: float = 0.0  # orientation of the tilt axis
    transition: float = 3.0  # width of the transition
    name: str = "warp"


STRUCTURES = {cls.__name__: cls for cls in (Spiral, Gap, Clump, Planet, Warp)}


# ── Disk ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Disk:
    """A protoplanetary disk including its grid resolution and structures."""

    r_in: float = 1.0
    r_out: float = 300.0
    r_ref: float = 100.0
    h_ref: float = 10.0
    alpha: float = 2.0
    beta: float = 1.15
    dust_mass: float = 5e-5  # M_sun
    edge: float = 0.0  # > 0: soft outer edge of this width instead of a hard cut
    structures: tuple = ()

    # grid
    n_r: int = 80
    n_phi: int = 64
    n_theta: int = 81
    f_r: float = 1.08
    f_theta: float = 1.0

    # ── profiles ──
    def scale_height(self, R):
        return self.h_ref * (np.asarray(R) / self.r_ref) ** self.beta

    def midplane_density(self, R):
        R = np.asarray(R, dtype=np.float64)
        return (np.maximum(R, 1e-10) / self.r_ref) ** (-self.alpha)

    def _base(self, x, y, z):
        R = np.hypot(x, y)
        H = self.scale_height(np.maximum(R, 1e-10))
        rho = self.midplane_density(R) * np.exp(-0.5 * (z / H) ** 2)
        if self.edge > 0.0:
            rho = rho * np.where(R > self.r_out, np.exp(-(((R - self.r_out) / self.edge) ** 2)), 1.0)
            return np.where(R < self.r_in, 0.0, rho)
        return np.where((R < self.r_in) | (R > self.r_out), 0.0, rho)

    def _modulated(self, x, y, z):
        rho = self._base(x, y, z)
        R = np.hypot(x, y)
        phi = np.arctan2(y, x)
        for s in self.structures:
            if hasattr(s, "modulate"):
                rho = s.modulate(R, phi, z, rho, self)
        outside = (R < self.r_in) | ((R > self.r_out) if self.edge <= 0.0 else False)
        return np.where(outside, 0.0, rho)

    def density(self, x, y, z):
        """Unnormalised dust density for arrays of positions in au."""
        x, y, z = (np.asarray(v, dtype=np.float64) for v in (x, y, z))
        warps = [s for s in self.structures if isinstance(s, Warp)]
        if not warps:
            return self._modulated(x, y, z)
        w = warps[0]
        t, a = math.radians(w.tilt), math.radians(w.twist)
        ca, sa, ct, st = math.cos(a), math.sin(a), math.cos(t), math.sin(t)
        # rotate into the frame of the tilted inner disk
        xr = ca * x + sa * y
        yr = -sa * x + ca * y
        yt = ct * yr + st * z
        zt = -st * yr + ct * z
        xt = ca * xr - sa * yt
        yt2 = sa * xr + ca * yt
        r_sph = np.sqrt(x**2 + y**2 + z**2)
        inner = 1.0 / (1.0 + np.exp((r_sph - w.r_break) / w.transition))
        return inner * self._modulated(xt, yt2, zt) + (1.0 - inner) * self._modulated(x, y, z)

    # ── variants ──
    def with_(self, **changes) -> Disk:
        """Copy with changes. Structure parameters use ``name.param``: ``with_(**{"planet.radius": 60})``."""
        top = {k: v for k, v in changes.items() if "." not in k}
        nested = {k: v for k, v in changes.items() if "." in k}
        disk = dataclasses.replace(self, **top)
        if nested:
            structs = list(disk.structures)
            for key, value in nested.items():
                sname, param = key.split(".", 1)
                idx = [i for i, s in enumerate(structs) if s.name == sname]
                if not idx:
                    raise KeyError(f"No structure named '{sname}' in this disk.")
                structs[idx[0]] = dataclasses.replace(structs[idx[0]], **{param: value})
            disk = dataclasses.replace(disk, structures=tuple(structs))
        return disk

    def sweep(self, param: str, values) -> list[Disk]:
        """List of variants, e.g. ``disk.sweep("planet.azimuth", range(0, 360, 30))``."""
        return [self.with_(**{param: float(v)}) for v in values]

    def get(self, param: str):
        if "." not in param:
            return getattr(self, param)
        sname, p = param.split(".", 1)
        for s in self.structures:
            if s.name == sname:
                return getattr(s, p)
        raise KeyError(param)

    # ── hashing & grid ──
    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["structures"] = [{"type": type(s).__name__, **dataclasses.asdict(s)} for s in self.structures]
        return d

    @staticmethod
    def from_dict(d: dict) -> Disk:
        d = dict(d)
        structs = []
        for s in d.pop("structures", []):
            s = dict(s)
            structs.append(STRUCTURES[s.pop("type")](**s))
        return Disk(structures=tuple(structs), **d)

    @property
    def key(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, default=float)
        return hashlib.sha1(blob.encode()).hexdigest()[:12]

    @property
    def axisymmetric(self) -> bool:
        return all(isinstance(s, Gap) for s in self.structures)

    def build_grid(self, path) -> Path:
        """Build the HAMSTER grid, normalise it to ``dust_mass`` and save it."""
        n_phi = 1 if self.axisymmetric else self.n_phi
        grid = spherical_grid(
            r_min=self.r_in * AU_TO_M,
            r_max=self.r_out * AU_TO_M,
            N_r=self.n_r,
            N_phi=n_phi,
            N_theta=self.n_theta,
            f_r=self.f_r,
            f_phi=1.0,
            f_theta=self.f_theta,
        )
        xyz = np.asarray(grid.cell_positions_xyz) / AU_TO_M
        rho = np.nan_to_num(self.density(xyz[:, 0], xyz[:, 1], xyz[:, 2]), nan=0.0, posinf=0.0, neginf=0.0)
        grid.dust_density = [rho.astype(np.float64)]
        grid.normalizeDustMass(self.dust_mass, dust_id=0)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_grid(grid, path)
        return path

    def density_map(self, n: int = 400, plane: str = "xy", extent: float | None = None) -> np.ndarray:
        """Quick density slice without building a grid (for checks and plots)."""
        L = extent or self.r_out
        s = np.linspace(-L, L, n)
        a, b = np.meshgrid(s, s)
        zero = np.zeros_like(a)
        return self.density(a, b, zero) if plane == "xy" else self.density(a, zero, b)


# ── Presets ───────────────────────────────────────────────────────────────────


class presets:
    """Ready-made scenes that are worth a camera flight."""

    @staticmethod
    def smooth(**kw) -> Disk:
        return Disk(**kw)

    @staticmethod
    def spiral(**kw) -> Disk:
        return Disk(structures=(Spiral(),), **kw)

    @staticmethod
    def planet(radius: float = 60.0, azimuth: float = 30.0, **kw) -> Disk:
        return Disk(structures=(Planet(radius=radius, azimuth=azimuth),), **kw)

    @staticmethod
    def warped(**kw) -> Disk:
        return Disk(structures=(Warp(), Gap(radius=30.0, width=5.0, depth=0.8)), **kw)

    @staticmethod
    def showcase(**kw) -> Disk:
        """Spiral + planet + vortex – lots of structure for orbits."""
        return Disk(
            structures=(
                Spiral(arms=2, amplitude=2.0, r_start=15.0),
                Planet(radius=90.0, azimuth=120.0, wake=0.0),
                Clump(radius=150.0, azimuth=-60.0, size=10.0, amplitude=5.0, elongation=4.0, name="vortex"),
            ),
            **kw,
        )
