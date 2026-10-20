"""
Journey: a ~48 s flight to a planet, followed by a parameter lab. Sized for
about a day on 32-64 cores at 512 x 512 pixels.

1. Approach from far away, face-on, while the disk rolls into place
2. Tilt to 60° on half an orbit
3. Zoom onto the planet – the image centre follows its projected position
4. At the planet: scattered light (1.6 µm) → 10 µm → dust emission (870 µm)
5. Pull back and tilt to edge-on, where the warped inner disk casts shadows
6. Lab: flaring β, gap depth and warp tilt, each swept over ten values while
   the camera keeps circling

Every parameter value is a scene of its own with its own heating run (29 in
total); the cross-fades between the values cost nothing.

    python examples/journey.py --plan                    # what would be computed (no HAMSTER needed)
    python examples/journey.py --check --slurm ...       # draft stills of the key frames first
    python examples/journey.py --slurm --partition <p> --cpus 64 --setup "module load gcc"
    python examples/journey.py --slurm --partition <p> --cpus 64   # again once the jobs are done → video
"""

import dataclasses
import math
import sys

import numpy as np

import hamster_animator as ha
from hamster_animator.camera import EASINGS

FPS = 24
FOV = 520.0  # au, the whole disk
ZOOM = 200.0  # au, field of view at the planet

DISK = ha.Disk(
    r_out=250,
    h_ref=8,
    beta=1.2,
    dust_mass=1e-4,
    n_r=128,
    n_phi=256,  # ~2.3 au per cell at the planet, fine enough for the zoom
    n_theta=101,
    structures=(
        ha.Spiral(arms=2, pitch=14, amplitude=2.5, r_start=20),
        ha.Planet(radius=95, azimuth=120, gap_width=8, gap_depth=0.95, clump_size=4, wake=1.0),
        ha.Clump(radius=165, azimuth=-60, size=10, amplitude=5, elongation=4, name="vortex"),
        ha.Warp(r_break=15, tilt=12),
    ),
)

SIM = ha.Sim(photons=1e6, heating_photons=1e7, pixels=512)

# Coronagraph and r² scaling are centred on the image, which is the planet during the zoom,
# and the flux per pixel drops as the field of view shrinks: so no r², no mask, and every
# frame is normalised on its own.
FLIGHT = ha.STYLES["cinematic"].with_(
    r2=False,
    coronagraph=0.0,
    norm="frame",
    stretch=ha.Stretch("log", lo=1, hi=99.8, decades=3.5),
)
LAB = ha.STYLES["cinematic"]


def project(view: ha.View, x: float, y: float, z: float = 0.0) -> tuple[float, float]:
    """
    Image-plane position (au) of a point, as HAMSTER's ``Detector`` computes it:
    image x = pos · e_y, image y = pos · e_x, both rotated by -pa - 90° about the line of sight.
    """
    th, ph, pa = (math.radians(a) for a in (view.inclination, view.azimuth, view.position_angle))
    k = np.array([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)])
    e_x = np.array([math.cos(th) * math.cos(ph), math.cos(th) * math.sin(ph), -math.sin(th)])
    e_y = np.array([-math.sin(ph), math.cos(ph), 0.0])
    a = -pa - math.pi / 2
    rot = lambda v: v * math.cos(a) + np.cross(k, v) * math.sin(a)
    pos = np.array([x, y, z])
    return float(pos @ rot(e_y)), float(pos @ rot(e_x))


def follow(views, target, weights):
    """Shift the image centre towards ``target`` (x, y, z in au) by ``weights`` (0 = star, 1 = target)."""
    out = []
    for v, w in zip(views, weights, strict=True):
        cx, cy = project(v, *target)
        out.append(dataclasses.replace(v, center_x=w * cx, center_y=w * cy))
    return out


def flight():
    """The camera path of acts 1-5 as lists of views (one list per act)."""
    segments = [
        ha.Camera.move(96, inclination=(5, 30), azimuth=(-90, 0), position_angle=(0, 90), fov=(1600, 640), ease="out"),
        ha.Camera.move(144, inclination=60, azimuth=180, fov=FOV),
        ha.Camera.move(120, azimuth=215, fov=ZOOM),
        ha.Camera.move(144, azimuth=230, ease="linear"),
        ha.Camera.move(144, inclination=86, azimuth=290, fov=FOV),
    ]
    path = segments[0]
    for s in segments[1:]:
        path = path.then(s)
    views, acts, i = path.views(), [], 0
    for s in segments:
        acts.append(views[i : i + len(s)])
        i += len(s)

    r, az = DISK.get("planet.radius"), math.radians(DISK.get("planet.azimuth"))
    planet = (r * math.cos(az), r * math.sin(az), 0.0)
    ease = EASINGS["in_out"]
    ramp = lambda n: [ease(k / (n - 1)) for k in range(n)]
    acts[2] = follow(acts[2], planet, ramp(len(acts[2])))
    acts[3] = follow(acts[3], planet, [1.0] * len(acts[3]))
    acts[4] = follow(acts[4], planet, [1.0 - w for w in ramp(len(acts[4]))])
    return [ha.Camera.views(a) for a in acts]


def lab(param: str, values, inclination: float, azimuth: float) -> tuple:
    """Sweep one parameter over ten values while the camera circles a quarter turn (168 frames)."""
    scene = ha.Steps(DISK.sweep(param, values), hold=6, fade=12)
    camera = ha.Camera.orbit(frames=len(scene), inclination=inclination, azimuth=(azimuth, azimuth + 90), fov=FOV)
    return scene, camera


def acts(sim: ha.Sim) -> list[tuple[str, ha.Animation, ha.Style]]:
    A = lambda scene, camera, **kw: ha.Animation(scene, camera, sim=sim, **kw)
    approach, orbit, zoom, planet, edge_on = flight()
    to_mm = ha.Steps([1.6e-6, 10e-6, 870e-6], hold=24, fade=36)
    beta = lab("beta", np.round(np.linspace(1.05, 1.32, 10), 2), inclination=55, azimuth=0)
    depths = [0.3, 0.5, 0.65, 0.75, 0.82, 0.88, 0.92, 0.95, 0.97, 0.99]
    gap = lab("planet.gap_depth", depths, inclination=35, azimuth=90)
    warp = lab("warp.tilt", np.arange(0, 20, 2), inclination=70, azimuth=180)
    return [
        ("1_approach", A(DISK, approach, wavelength=1.6e-6), FLIGHT.with_(title="Reise zum Planeten")),
        ("2_orbit", A(DISK, orbit, wavelength=1.6e-6), FLIGHT.with_(title="Reise zum Planeten")),
        ("3_zoom", A(DISK, zoom, wavelength=1.6e-6), FLIGHT.with_(title="Reise zum Planeten")),
        ("4_planet", A(DISK, planet, wavelength=to_mm), FLIGHT.with_(title="Streulicht → Staubemission", temporal=0)),
        ("5_edge_on", A(DISK, edge_on, wavelength=1.6e-6), FLIGHT.with_(title="Kante voraus")),
        ("6_lab_beta", A(*beta, wavelength=1.6e-6), LAB.with_(title="Labor: Aufweitung β", temporal=0)),
        ("7_lab_gap", A(*gap, wavelength=1.6e-6), LAB.with_(title="Labor: Planetenlücke", temporal=0)),
        ("8_lab_warp", A(*warp, wavelength=1.6e-6), LAB.with_(title="Labor: Verkippte Innenscheibe", temporal=0)),
    ]


if __name__ == "__main__":
    sys.exit(ha.Sequence(acts(SIM), fps=FPS).cli("journey"))
