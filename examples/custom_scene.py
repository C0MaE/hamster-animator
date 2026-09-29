"""Build your own disk, sweep a parameter and fade through a few wavelengths."""

import numpy as np

import hamster_animator as ha

disk = ha.Disk(
    r_out=250,
    h_ref=8,
    beta=1.2,
    dust_mass=1e-4,
    structures=(
        ha.Planet(radius=45, azimuth=0, gap_depth=0.95),
        ha.Warp(r_break=15, tilt=12),
        ha.Clump(radius=120, azimuth=200, size=8, amplitude=6, elongation=5, name="vortex"),
    ),
)

# the planet moves on its orbit – every step is one simulation, fades are free
scene = ha.Steps(disk.sweep("planet.azimuth", np.arange(0, 360, 30)), hold=2, fade=4, loop=True)
style = ha.STYLES["cinematic"].with_(colormap="inferno", title="Planet in a warped disk")

anim = ha.Animation(scene, ha.View(inclination=40), wavelength=1.25e-6, sim=ha.Sim(photons=3e5, pixels=192))
anim.save("planet_orbit.mp4", fps=24, style=style)

# the same scene through the spectrum (reuses the cached heating runs)
journey = ha.Steps([0.5e-6, 1.25e-6, 2.2e-6, 10e-6, 870e-6], hold=12, fade=12)
ha.Animation(
    disk, ha.Camera.orbit(inclination=40, frames=len(journey), azimuth=(0, 90)), wavelength=journey, sim=anim.sim
).save("spectrum.mp4", style=style.with_(temporal=0))
