"""Ready-made example animations, also available as ``hamster-animator gallery <name>``."""

from __future__ import annotations

import numpy as np

from .animation import Animation
from .camera import Camera
from .render import STYLES, Stretch, Style
from .scene import presets
from .sim import Sim
from .steps import Steps

SIM = Sim(photons=3e5, heating_photons=5e6, pixels=192)


def orbit(sim: Sim = SIM):
    """Camera flight: orbit at 55°, tilt to almost edge-on, quarter turn."""
    cam = (
        Camera.orbit(inclination=55, frames=48)
        .then(Camera.tilt(to=82, frames=24))
        .then(Camera.orbit(frames=24, azimuth=(0, 90), ease="in_out"))
    )
    return Animation(presets.showcase(), cam, wavelength=1.6e-6, sim=sim), STYLES["cinematic"]


def flyby(sim: Sim = SIM):
    """Tilt and zoom in while orbiting."""
    cam = Camera.flyby(frames=72, inclination=(15, 75), fov=(600, 260))
    return Animation(presets.showcase(), cam, wavelength=1.6e-6, sim=sim), STYLES["cinematic"]


def spectrum(sim: Sim = SIM):
    """Wavelength journey from the optical to the mm regime with cross-fades."""
    wl = Steps([0.5e-6, 1.25e-6, 2.2e-6, 10e-6, 100e-6, 870e-6], hold=10, fade=10)
    cam = Camera.orbit(inclination=40, frames=len(wl), azimuth=(0, 120))
    return Animation(presets.planet(radius=60, azimuth=30), cam, wavelength=wl, sim=sim), STYLES["cinematic"].with_(
        temporal=0
    )


def rgb(sim: Sim = SIM):
    """False-colour J/H/K composite on an orbit."""
    anim = Animation(
        presets.showcase(), Camera.orbit(inclination=60, frames=48), wavelength=(2.2e-6, 1.6e-6, 1.25e-6), sim=sim
    )
    return anim, Style(
        r2=True,
        coronagraph=12.0,
        smooth=0.8,
        temporal=1.0,
        temporal_wrap=True,
        stretch=Stretch("asinh", lo=0, hi=99.5, soft=0.08),
    )


def polarization(sim: Sim = SIM):
    """I → PI → Q_φ → polarisation degree, with vectors, on a slow orbit."""
    q = Steps(["I", "PI", "Q_PHI", "P"], hold=18, fade=8)
    cam = Camera.orbit(inclination=45, frames=len(q), azimuth=(0, 180))
    return Animation(presets.showcase(), cam, wavelength=1.25e-6, quantity=q, sim=sim), STYLES["polarimetry"]


def evolution(sim: Sim = SIM):
    """Parameter evolution: a planet orbiting the star (12 scenes, cross-faded)."""
    variants = presets.planet(radius=60, azimuth=0).sweep("planet.azimuth", np.arange(0, 360, 30))
    anim = Animation(
        Steps(variants, hold=1, fade=3, loop=True), Camera.still(inclination=35), wavelength=1.6e-6, sim=sim
    )
    return anim, STYLES["cinematic"].with_(temporal=0)


EXAMPLES = dict(orbit=orbit, flyby=flyby, spectrum=spectrum, rgb=rgb, polarization=polarization, evolution=evolution)
