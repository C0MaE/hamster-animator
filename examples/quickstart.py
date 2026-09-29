"""Minimal example: orbit around a structured disk. Run from anywhere once HAMSTER is set up."""

import hamster_animator as ha

disk = ha.presets.showcase()
camera = ha.Camera.orbit(inclination=55, frames=72).then(ha.Camera.tilt(to=80, frames=24))
anim = ha.Animation(disk, camera, wavelength=1.6e-6, sim=ha.Sim(photons=3e5, pixels=192))

print(anim.plan())  # what will be computed
anim.still("look.png", frame=0, style="cinematic")  # try the look on one frame
anim.draft().save("orbit_draft.mp4", style="cinematic")  # quick preview
anim.save("orbit.mp4", style="cinematic")  # final quality
