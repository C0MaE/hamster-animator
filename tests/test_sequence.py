import dataclasses
import json
import shutil

import pytest

import hamster_animator as ha


class Recorder:
    """Backend that records plans instead of running HAMSTER."""

    views_per_job = None

    def __init__(self):
        self.plans = []

    def execute(self, plan, cache):
        self.plans.append(plan)
        return True


def two_acts(cache, sim=None):
    orbit = ha.Camera.orbit(inclination=50, frames=6)
    sim = sim or ha.Sim()
    return [
        ("h_band", ha.Animation(ha.presets.spiral(), orbit, wavelength=1.6e-6, sim=sim, cache=cache)),
        ("j_band", ha.Animation(ha.presets.spiral(), orbit.reverse(), wavelength=1.25e-6, sim=sim, cache=cache)),
    ]


def test_shared_views_are_planned_once(tmp_path):
    seq = ha.Sequence(two_acts(tmp_path))
    assert len(seq) == 12
    assert seq.plan().n_detectors == 6  # both wavelengths of a view come from the same run
    assert sum(a.animation.plan().n_detectors for a in seq.acts) == 12
    assert "6 HAMSTER detector(s) (12 if every act were planned alone)" in seq.summary()


def test_acts_share_sim_and_cache(tmp_path):
    acts = two_acts(tmp_path)
    acts[1][1].cache = ha.Cache(tmp_path / "other")
    seq = ha.Sequence(acts)
    assert seq.acts[1].animation.cache is seq.cache
    with pytest.raises(ValueError, match="same Sim"):
        ha.Sequence([acts[0], ("x", acts[1][1].draft())])
    with pytest.raises(ValueError, match="unique"):
        ha.Sequence([acts[0], acts[0]])
    with pytest.raises(ValueError):
        ha.Sequence([])


def test_draft_and_key_frames(tmp_path):
    seq = ha.Sequence(two_acts(tmp_path))
    assert seq.draft().sim == ha.Sim().draft()
    keys = seq.key_frames()
    assert [a.name for a in keys.acts] == [
        "h_band_0000",
        "h_band_0003",
        "h_band_0005",
        "j_band_0000",
        "j_band_0003",
        "j_band_0005",
    ]
    assert len(keys) == 6


def test_pending_runs_are_not_submitted_twice(tmp_path):
    seq = ha.Sequence(two_acts(tmp_path))
    run = seq.cache.root / "runs" / "earlier"
    run.mkdir(parents=True)
    entry = dict(
        out=str(run / "out_0000"),  # no detector maps there yet → the run stays pending
        sim=dataclasses.asdict(seq.sim),
        disk=ha.presets.spiral().to_dict(),
        lams=[1.6e-6],
        views=[dataclasses.asdict(ha.View())],
    )
    (run / "manifest.json").write_text(json.dumps([entry]))
    backend = Recorder()
    assert seq.pending()
    assert not seq.simulate(backend)
    assert backend.plans == []
    assert seq.simulate(backend, resubmit=True)
    assert backend.plans[0].n_detectors == 6


def test_save_pngs_and_stills(fake_cache, tmp_path):
    style = ha.Style(size=(120, 120))
    seq = ha.Sequence([(name, anim, style) for name, anim in two_acts(fake_cache)], fps=6)
    out = seq.save(tmp_path / "seq", workers=1)
    assert sorted(p.name for p in out.iterdir()) == ["h_band", "j_band"]
    assert len(list((out / "h_band").glob("*.png"))) == 6
    stills = seq.stills(tmp_path / "stills", frame="last")
    assert [p.name for p in stills] == ["h_band.png", "j_band.png"]
    with pytest.raises(ValueError, match="PNG folder"):
        seq.save(tmp_path / "seq.gif")
    if shutil.which("ffmpeg"):
        video = seq.save(tmp_path / "seq.mp4", workers=1)
        assert video.stat().st_size > 0 and (tmp_path / "seq_acts" / "j_band.mp4").exists()


def test_cli_plan(tmp_path, capsys):
    assert ha.Sequence(two_acts(tmp_path)).cli("demo", ["--plan"]) == 0
    assert "12 frames in 2 act(s)" in capsys.readouterr().out
