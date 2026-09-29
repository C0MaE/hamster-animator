import pytest

import hamster_animator as ha
from hamster_animator.sim import make_plan


def requests(disk, views, lams):
    return [(disk, v, l) for v in views for l in lams]


def test_plan_groups_views_and_splits_dust_band(tmp_path):
    cache = ha.Cache(tmp_path)
    disk = ha.presets.showcase()
    views = ha.Camera.orbit(inclination=50, frames=6).views()
    plan = make_plan(requests(disk, views, [1.6e-6, 870e-6]), ha.Sim(), cache)
    assert plan.total == 12 and plan.cached == 0
    assert len(plan.heat) == 1
    assert sorted(j.with_dust for j in plan.jobs) == [False, True]
    assert plan.n_detectors == 12


def test_views_per_job_chunks(tmp_path):
    views = ha.Camera.orbit(frames=10).views()
    plan = make_plan(requests(ha.presets.spiral(), views, [1e-6]), ha.Sim(), ha.Cache(tmp_path), views_per_job=4)
    assert [len(j.views) for j in plan.jobs] == [4, 4, 2]


def test_duplicate_and_axisymmetric_views_are_free(tmp_path):
    anim = ha.Animation(ha.presets.smooth(), ha.Camera.orbit(inclination=45, frames=36), cache=tmp_path)
    assert len(anim) == 36
    assert anim.plan().n_detectors == 1
    held = ha.Animation(ha.presets.spiral(), ha.Camera.still(frames=20, inclination=30), cache=tmp_path)
    assert held.plan().n_detectors == 1


def test_track_length_mismatch(tmp_path):
    anim = ha.Animation(
        ha.presets.spiral(), ha.Camera.orbit(frames=4), wavelength=ha.Steps([1e-6, 2e-6]), cache=tmp_path
    )
    with pytest.raises(ValueError, match="Track lengths"):
        anim.frames()


def test_sim_keys():
    disk = ha.presets.spiral()
    assert ha.Sim().scatter_key(disk) == ha.Sim(threads=4, seed=7).scatter_key(disk)
    assert ha.Sim().scatter_key(disk) != ha.Sim(photons=1e6).scatter_key(disk)
    assert ha.Sim.from_dict(__import__("dataclasses").asdict(ha.Sim())) == ha.Sim()
