# Changelog

## Unreleased

- `Sequence`: several animations planned together (shared views are simulated once with all wavelengths),
  joined into one video without re-encoding, with stills per act and a ready-made command line (`.cli()`)
  including `--check` and protection against submitting SLURM jobs twice
- examples `showcase.py` (seven-act trailer) and `journey.py` (zoom onto a planet, parameter sweeps)

## 0.1.0

First release.

- `Animation` with four tracks: scene, camera, wavelength, quantity
- camera paths: orbit, tilt, zoom, roll, flyby, arbitrary `move`, chaining with `.then()`, easing
- disk models with spirals, gaps, planets, clumps/vortices and warps; vectorised grid construction
- quantities I, Q, U, V, PI, P, Q_φ, U_φ, I_scat, I_direct and RGB composites
- styles with log/asinh stretches, r² scaling, coronagraph, smoothing, temporal denoising, polarisation vectors
- cache per (scene, settings, view, wavelength); only missing combinations are simulated
- local backend and SLURM backend (job arrays, heating → scattering dependency)
- output to MP4, WebM, GIF, ProRes or PNG sequences via ffmpeg
- `hamster-animator` command line tool with a gallery of examples
