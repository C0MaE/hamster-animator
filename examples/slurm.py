"""Render a long, high-quality orbit on a SLURM cluster."""

import hamster_animator as ha

anim = ha.Animation(
    ha.presets.showcase(),
    ha.Camera.orbit(inclination=55, frames=240),
    wavelength=(2.2e-6, 1.6e-6, 1.25e-6),  # RGB composite
    sim=ha.Sim(photons=2e6, pixels=384),
)
backend = ha.Slurm(
    partition="long",
    time="12:00:00",
    cpus=64,
    views_per_job=6,
    setup=["module load gcc"],
)
# First call submits the jobs and returns. Call again when they are done:
# the results are collected from the cache and the video is rendered.
anim.save("orbit_rgb.mp4", fps=30, style="cinematic", backend=backend)
