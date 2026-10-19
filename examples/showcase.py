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
    python examples/showcase.py --check             # draft stills of the key frames
    python examples/showcase.py --draft             # preview on a laptop (10x fewer photons)
    python examples/showcase.py --slurm --partition long
    python examples/showcase.py --slurm --partition long   # again once the jobs are done → video

Output: ``animations/showcase/final/showcase.mp4``, the single acts in ``showcase_acts/``
and one still per act (for slides) in ``stills/``.
"""

import sys

import numpy as np

import hamster_animator as ha

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


def acts(sim: ha.Sim) -> list[tuple[str, ha.Animation, ha.Style]]:
    """(name, animation, style) for every act. All acts share the sim settings."""
    orbit = ha.Camera.orbit(inclination=50, frames=ORBIT, fov=FOV)
    A = lambda scene, camera, **kw: ha.Animation(scene, camera, sim=sim, **kw)

    # the field of view shrinks, so the flux per pixel changes: normalise every frame on its own
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
        ("1_approach", A(DISK, approach, wavelength=1.6e-6), CINEMATIC.with_(title="Anflug", norm="frame")),
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


if __name__ == "__main__":
    sys.exit(ha.Sequence(acts(SIM), fps=FPS).cli("showcase"))
