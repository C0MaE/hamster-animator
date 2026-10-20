"""
Simulation: planning, caching and running HAMSTER locally or on SLURM.

Cost model
----------
Measured on a structured disk with 16 threads: every photon package interacts
~50 times, and at *every* interaction HAMSTER traces a peel-off ray to *every*
detector. Runtime therefore scales as photons × interactions × detectors;
the photon transport itself is cheap. The planner is built around that:

1. **No unnecessary detectors.** Identical views are de-duplicated; for
   axisymmetric disks the azimuth is irrelevant, so an orbit costs one detector.
2. **All wavelengths and Stokes components come from the same run**, so
   wavelength journeys and polarisation animations are almost free.
3. **Dust heating once per scene**, the temperature grid is cached.
4. **No thermal dust emission where it is irrelevant** (λ < ``dust_emission_min``).
5. **Fine-grained cache per (scene, settings, view, wavelength)** – extending
   a path or adding a wavelength only computes what is missing.
6. **Spread views over jobs.** Cost is linear in detectors, so SLURM array
   jobs (``Slurm(views_per_job=4)``) scale linearly.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import shutil
import subprocess
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from . import _hamster as H
from .camera import View
from .scene import Disk

log = logging.getLogger("hamster_animator")


def lam_key(lam: float) -> str:
    return f"{float(lam):.6e}"


def _hash(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


# ── Settings ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Star:
    temperature: float = 4000.0  # K
    radius: float = 2.0  # R_sun
    extended: bool = False
    photons_direct: int = 1


@dataclass(frozen=True)
class Sim:
    """Physical and numerical settings shared by all frames."""

    photons: float = 2e5  # stellar photon packages per wavelength
    dust_photons: float | None = None  # thermal dust photons (None → same as photons)
    heating_photons: float = 5e6
    pixels: int = 256
    distance_pc: float = 140.0
    dust: str = "input/dust/optool/silicate_wd01_a1-0.optool"  # relative to the HAMSTER root
    star: Star = Star()
    phase_function: str = "MIE"
    precalculations: str = "reduced"
    enforced_first_scattering: bool = True
    dust_emission_min: float = 2e-6  # no thermal dust emission below this wavelength
    threads: int = -1
    seed: int = -1
    max_memory_gb: float = 4.0  # detector memory per HAMSTER task

    def heat_key(self, disk: Disk) -> str:
        return _hash([disk.key, self.dust, dataclasses.asdict(self.star), self.heating_photons])

    def scatter_key(self, disk: Disk) -> str:
        """Everything that affects the image except the view and the wavelength."""
        d = dataclasses.asdict(self)
        for k in ("threads", "seed", "max_memory_gb"):
            d.pop(k)
        return _hash([self.heat_key(disk), d])

    def draft(self, factor: float = 10.0) -> Sim:
        """Fast preview settings: fewer photons, half the resolution."""
        return dataclasses.replace(
            self,
            photons=self.photons / factor,
            dust_photons=None if self.dust_photons is None else self.dust_photons / factor,
            heating_photons=self.heating_photons / factor,
            pixels=max(32, self.pixels // 2),
        )

    def with_(self, **kw) -> Sim:
        return dataclasses.replace(self, **kw)

    def bytes_per_detector(self, n_lam: int) -> int:
        return self.pixels * self.pixels * n_lam * len(H.STOKES) * 8

    @staticmethod
    def from_dict(d: dict) -> Sim:
        d = dict(d)
        return Sim(star=Star(**d.pop("star")), **d)


# ── Cache ─────────────────────────────────────────────────────────────────────


class Cache:
    """
    On-disk cache::

        <root>/scenes/<disk>/disk.json, grid.grid
        <root>/scenes/<disk>/heat_<h>/grid_temp.grid
        <root>/scenes/<disk>/scat_<s>/index.json, maps/*.fits, logs/*.txt
        <root>/runs/<run>/               task files, raw output, manifest.json
    """

    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._index: dict[Path, dict] = {}
        self._maps: dict[str, dict] = {}

    # paths
    def scene_dir(self, disk: Disk) -> Path:
        return self.root / "scenes" / disk.key

    def grid_path(self, disk: Disk) -> Path:
        return self.scene_dir(disk) / "grid.grid"

    def heat_dir(self, disk: Disk, sim: Sim) -> Path:
        return self.scene_dir(disk) / f"heat_{sim.heat_key(disk)}"

    def temp_grid_path(self, disk: Disk, sim: Sim) -> Path:
        return self.heat_dir(disk, sim) / "grid_temp.grid"

    def scat_dir(self, disk: Disk, sim: Sim) -> Path:
        return self.scene_dir(disk) / f"scat_{sim.scatter_key(disk)}"

    def new_run_dir(self) -> Path:
        d = self.root / "runs" / (time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4])
        d.mkdir(parents=True)
        return d

    # grid
    def ensure_grid(self, disk: Disk) -> Path:
        p = self.grid_path(disk)
        if not p.exists():
            disk.build_grid(p)
            (p.parent / "disk.json").write_text(json.dumps(disk.to_dict(), indent=2, default=float))
        return p

    # index
    def index(self, disk: Disk, sim: Sim) -> dict:
        d = self.scat_dir(disk, sim)
        if d not in self._index:
            f = d / "index.json"
            self._index[d] = json.loads(f.read_text()) if f.exists() else {}
        return self._index[d]

    def save_index(self, disk: Disk, sim: Sim) -> None:
        d = self.scat_dir(disk, sim)
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "index.json.tmp"
        tmp.write_text(json.dumps(self._index[d], indent=1))
        tmp.replace(d / "index.json")

    def has(self, disk: Disk, sim: Sim, view: View, lam: float) -> bool:
        return lam_key(lam) in self.index(disk, sim).get(view.key_str, {})

    def add(self, disk: Disk, sim: Sim, view: View, lams, fits: Path) -> None:
        d = self.scat_dir(disk, sim) / "maps"
        d.mkdir(parents=True, exist_ok=True)
        name = f"{view.key_str}__{_hash([lam_key(l) for l in lams])}.fits"
        shutil.move(str(fits), d / name)
        entry = self.index(disk, sim).setdefault(view.key_str, {})
        for l in lams:
            entry[lam_key(l)] = f"maps/{name}"

    # data
    def load(self, disk: Disk, sim: Sim, view: View, lam: float) -> dict:
        """``{stokes: 2D float32 map in Jy/px, "pixel_au": float}``"""
        rel = self.index(disk, sim)[view.key_str][lam_key(lam)]
        path = str(self.scat_dir(disk, sim) / rel)
        if path not in self._maps:
            wl, stokes, px = H.read_detector(path)
            self._maps[path] = {
                lam_key(w): {**{s: stokes[s][i] for s in H.STOKES}, "pixel_au": px / H.AU_TO_M}
                for i, w in enumerate(wl)
            }
        return self._maps[path][lam_key(lam)]

    def clear_memory(self) -> None:
        self._maps.clear()

    def size_bytes(self) -> int:
        return sum(f.stat().st_size for f in self.root.rglob("*") if f.is_file())


# ── Plan ──────────────────────────────────────────────────────────────────────


@dataclass
class ScatterJob:
    disk: Disk
    views: list
    lams: list
    with_dust: bool


@dataclass
class Plan:
    sim: Sim
    n_scenes: int = 0
    heat: list = field(default_factory=list)  # disks that need a heating run
    jobs: list = field(default_factory=list)  # ScatterJobs
    cached: int = 0
    total: int = 0

    @property
    def empty(self) -> bool:
        return not self.jobs

    @property
    def n_detectors(self) -> int:
        return sum(len(j.views) for j in self.jobs)

    def __str__(self) -> str:
        lines = [
            f"{self.n_scenes} scene(s), {self.total} (view, wavelength) combinations, {self.cached} cached",
            f"to compute: {len(self.heat)} heating run(s), {len(self.jobs)} scattering task(s), "
            f"{self.n_detectors} detector(s)",
        ]
        lines += [
            f"  scene {j.disk.key}: {len(j.views):4d} views × {len(j.lams)} λ"
            f"{'  (+ dust emission)' if j.with_dust else ''}"
            for j in self.jobs
        ]
        return "\n".join(lines)


def make_plan(requests, sim: Sim, cache: Cache, views_per_job: int | None = None) -> Plan:
    """Group missing (disk, view, wavelength) requests into as few HAMSTER tasks as possible."""
    needs: dict = {}
    for disk, view, lam in requests:
        needs.setdefault(disk.key, (disk, set()))[1].add((view, float(lam)))

    plan = Plan(sim=sim, n_scenes=len(needs))
    for disk, combos in needs.values():
        plan.total += len(combos)
        missing = defaultdict(set)
        for view, lam in combos:
            if cache.has(disk, sim, view, lam):
                plan.cached += 1
            else:
                missing[view].add(lam)
        if not missing:
            continue
        if not cache.temp_grid_path(disk, sim).exists():
            plan.heat.append(disk)
        by_lams = defaultdict(list)  # views sharing the same missing wavelengths
        for view, lams in missing.items():
            by_lams[tuple(sorted(lams))].append(view)
        for lams, views in by_lams.items():
            views.sort(key=lambda v: v.key)
            for band, with_dust in (
                ([l for l in lams if l < sim.dust_emission_min], False),
                ([l for l in lams if l >= sim.dust_emission_min], True),
            ):
                if not band:
                    continue
                chunk = max(1, int(sim.max_memory_gb * 1e9 // sim.bytes_per_detector(len(band))))
                if views_per_job:
                    chunk = min(chunk, int(views_per_job))
                for i in range(0, len(views), chunk):
                    plan.jobs.append(ScatterJob(disk, views[i : i + chunk], band, with_dust))
    return plan


# ── Task files ────────────────────────────────────────────────────────────────


def _general(tf, sim: Sim, root: Path) -> None:
    # shared settings go to [general]; this works with old and new HAMSTER versions
    tf.setNThreads(sim.threads)
    tf.setSeed(sim.seed)
    tf.addDust(file=H.resolve_input(sim.dust, root), input_source="optool", mass_fraction=1.0, dust_id=0)


def write_heat_tasks(disks, sim: Sim, cache: Cache, path: Path, root: Path) -> None:
    tf = H.task_file()
    _general(tf, sim, root)
    tf.setSimulationMode("DUST_HEATING")
    tf.addSourceStar(
        N_photons=sim.heating_photons,
        position=[0.0, 0.0, 0.0],
        radius=sim.star.radius,
        temperature_eff=sim.star.temperature,
        extended=sim.star.extended,
    )
    for disk in disks:
        out = cache.heat_dir(disk, sim)
        out.mkdir(parents=True, exist_ok=True)
        tid = tf.addTask(name=f"heat_{disk.key}", output=str(out) + "/")
        tf.setGrid(str(cache.grid_path(disk)), task_id=tid)
    H.write_task_file(tf, path)


def write_scatter_tasks(jobs, sim: Sim, cache: Cache, path: Path, out_root: Path, root: Path) -> list[dict]:
    tf = H.task_file()
    _general(tf, sim, root)
    tf.setSimulationMode("MONTE_CARLO_SCATTERING")
    tf.setPhaseFunction(sim.phase_function)
    tf.setPrecalculations(sim.precalculations)
    tf.setEnforcedFirstScattering(sim.enforced_first_scattering)
    tf.setCreateMaps(True)
    tf.setCreateSeds(False)
    manifest = []
    for n, job in enumerate(jobs):
        out = out_root / f"out_{n:04d}"
        tid = tf.addTask(name=f"scat_{job.disk.key}_{n}", output=str(out) + "/")
        tf.setGrid(str(cache.temp_grid_path(job.disk, sim)), task_id=tid)
        tf.setWavelengths(list(job.lams), task_id=tid)
        tf.addSourceStar(
            N_photons=sim.photons,
            position=[0.0, 0.0, 0.0],
            radius=sim.star.radius,
            temperature_eff=sim.star.temperature,
            extended=sim.star.extended,
            N_photons_direct=sim.star.photons_direct,
            task_id=tid,
        )
        if job.with_dust:
            tf.setSourceDust(N_photons=sim.photons if sim.dust_photons is None else sim.dust_photons, task_id=tid)
        for v in job.views:
            size = {} if v.fov is None else dict(length_x=v.fov * H.AU_TO_M, length_y=v.fov * H.AU_TO_M)
            tf.addDetector(
                N_xpixel=sim.pixels,
                N_ypixel=sim.pixels,
                inclination=float(v.inclination),
                azimuth=float(v.azimuth % 360.0),
                position_angle=float(v.position_angle),
                observer_distance=sim.distance_pc * H.PC_TO_M,
                center_x=v.center_x * H.AU_TO_M,
                center_y=v.center_y * H.AU_TO_M,
                task_id=tid,
                **size,
            )
        manifest.append(
            dict(
                out=str(out),
                sim=dataclasses.asdict(sim),
                disk=job.disk.to_dict(),
                lams=list(job.lams),
                views=[dataclasses.asdict(v) for v in job.views],
            )
        )
    H.write_task_file(tf, path)
    return manifest


def ingest(manifest: list[dict], cache: Cache) -> int:
    """Move finished detector maps into the cache. Returns the number of maps taken over."""
    n, touched = 0, {}
    for entry in manifest:
        if entry.get("done"):
            continue
        disk, sim = Disk.from_dict(entry["disk"]), Sim.from_dict(entry["sim"])
        views = [View(**v) for v in entry["views"]]
        out = Path(entry["out"])
        files = [out / f"detector_{k + 1:04d}-map.fits" for k in range(len(views))]
        if not all(f.exists() for f in files):
            continue
        for v, f in zip(views, files, strict=True):
            cache.add(disk, sim, v, entry["lams"], f)
            n += 1
        if (out / "output.txt").exists():
            logs = cache.scat_dir(disk, sim) / "logs"
            logs.mkdir(parents=True, exist_ok=True)
            shutil.move(str(out / "output.txt"), logs / f"{out.parent.name}_{out.name}.txt")
        touched[(disk.key, sim.scatter_key(disk))] = (disk, sim)
        entry["done"] = True
    for disk, sim in touched.values():
        cache.save_index(disk, sim)
    return n


def collect(cache: Cache) -> int:
    """Take over the results of all pending runs (e.g. finished SLURM jobs)."""
    runs = cache.root / "runs"
    if not runs.exists():
        return 0
    n = 0
    for mf in sorted(runs.glob("*/manifest.json")):
        manifest = json.loads(mf.read_text())
        n += ingest(manifest, cache)
        if all(e.get("done") for e in manifest):
            shutil.rmtree(mf.parent, ignore_errors=True)
        else:
            mf.write_text(json.dumps(manifest))
    return n


# ── Backends ──────────────────────────────────────────────────────────────────


def _run(binary: Path, taskfile: Path, root: Path, logfile: Path | None) -> float:
    t0 = time.time()
    cmd = [str(binary), str(taskfile)]
    if logfile is None:
        res = subprocess.run(cmd, cwd=root)
    else:
        with open(logfile, "w") as fh:
            res = subprocess.run(cmd, cwd=root, stdout=fh, stderr=subprocess.STDOUT)
    if res.returncode != 0:
        raise RuntimeError(
            f"HAMSTER exited with code {res.returncode} for {taskfile}" + (f" (log: {logfile})" if logfile else "")
        )
    return time.time() - t0


@dataclass
class Local:
    """
    Run on this machine: one HAMSTER call for all heating tasks and one for
    all scattering tasks (HAMSTER itself uses all cores).
    """

    hamster: str | None = None  # HAMSTER root (default: auto-detect)
    binary: str | None = None  # default: <root>/bin/HAMSTER
    quiet: bool = True  # HAMSTER output only to log files

    def execute(self, plan: Plan, cache: Cache) -> bool:
        root = H.find_root(self.hamster)
        binary = H.find_binary(root, self.binary)
        run_dir = cache.new_run_dir()
        for disk in {j.disk.key: j.disk for j in plan.jobs}.values():
            cache.ensure_grid(disk)
        if plan.heat:
            tf = run_dir / "heat.toml"
            write_heat_tasks(plan.heat, plan.sim, cache, tf, root)
            log.info("heating %d scene(s) …", len(plan.heat))
            dt = _run(binary, tf, root, run_dir / "heat.log" if self.quiet else None)
            log.info("heating done in %.0f s", dt)
        tf = run_dir / "scatter.toml"
        manifest = write_scatter_tasks(plan.jobs, plan.sim, cache, tf, run_dir, root)
        (run_dir / "manifest.json").write_text(json.dumps(manifest))
        log.info("scattering: %d detector(s) in %d task(s) …", plan.n_detectors, len(plan.jobs))
        dt = _run(binary, tf, root, run_dir / "scatter.log" if self.quiet else None)
        n = ingest(manifest, cache)
        log.info("scattering done: %d map(s) in %.0f s (%.1f s per detector)", n, dt, dt / max(n, 1))
        shutil.rmtree(run_dir, ignore_errors=True)
        return True


@dataclass
class Slurm:
    """
    Two SLURM job arrays: heating (one element per scene) and scattering (one
    element per chunk of ``views_per_job`` views), chained with ``afterok``.
    Once the jobs are done, call ``Animation.save()`` again – it collects the
    results automatically.
    """

    hamster: str | None = None
    binary: str | None = None
    views_per_job: int = 8
    partition: str | None = None
    time: str = "04:00:00"
    cpus: int = 32
    memory: str | None = None
    max_parallel: int | None = None
    job_name: str = "hamster-anim"
    extra: dict = field(default_factory=dict)  # additional #SBATCH directives
    setup: list = field(default_factory=list)  # e.g. ["module load gcc"]
    submit: bool = True

    def _script(self, name: str, n: int, run_dir: Path, root: Path, command: str) -> Path:
        s = H.slurm_script()
        s.setJobName(f"{self.job_name}-{name}")
        s.setArray(n, self.max_parallel)
        s.setCpusPerTask(self.cpus)
        s.setTime(self.time)
        s.setOutput(str(run_dir / f"slurm-{name}-%A_%a.out"))
        if self.partition:
            s.setPartition(self.partition)
        if self.memory:
            s.setMemory(self.memory)
        for k, v in self.extra.items():
            s.setDirective(k, v)
        for line in self.setup:
            s.addCommand(line)
        s.addCommand(f"cd {root}")
        s.addCommand(command)
        path = run_dir / f"{name}.sh"
        H.write_slurm_script(s, path)
        return path

    @staticmethod
    def _sbatch(script: Path, dependency: str | None = None) -> str:
        cmd = ["sbatch", "--parsable"] + ([f"--dependency=afterok:{dependency}"] if dependency else [])
        res = subprocess.run(cmd + [str(script)], check=True, capture_output=True, text=True)
        return res.stdout.strip().split(";")[0]

    def execute(self, plan: Plan, cache: Cache) -> bool:
        root = H.find_root(self.hamster)
        binary = H.find_binary(root, self.binary)
        run_dir = cache.new_run_dir()
        for disk in {j.disk.key: j.disk for j in plan.jobs}.values():
            cache.ensure_grid(disk)

        heat_script = None
        if plan.heat:
            for i, disk in enumerate(plan.heat):
                write_heat_tasks([disk], plan.sim, cache, run_dir / f"heat_{i}.toml", root)
            heat_script = self._script(
                "heat", len(plan.heat), run_dir, root, f'{binary} "{run_dir}/heat_$SLURM_ARRAY_TASK_ID.toml"'
            )
        manifest = []
        for i, job in enumerate(plan.jobs):
            manifest += write_scatter_tasks(
                [job], plan.sim, cache, run_dir / f"scat_{i}.toml", run_dir / f"part_{i}", root
            )
        (run_dir / "manifest.json").write_text(json.dumps(manifest))
        scat_script = self._script(
            "scat", len(plan.jobs), run_dir, root, f'{binary} "{run_dir}/scat_$SLURM_ARRAY_TASK_ID.toml"'
        )
        if not self.submit:
            log.info("SLURM scripts written (not submitted): %s", run_dir)
            return False
        dep = None
        try:
            dep = self._sbatch(heat_script) if heat_script else None
            jid = self._sbatch(scat_script, dep)
        except (OSError, subprocess.CalledProcessError):
            if dep is None:
                (run_dir / "manifest.json").rename(run_dir / "manifest.json.stale")
            raise
        log.info(
            "submitted %d heating and %d scattering array element(s) (job %s%s). Run save() again when they are done.",
            len(plan.heat),
            len(plan.jobs),
            jid,
            f", after {dep}" if dep else "",
        )
        return False
