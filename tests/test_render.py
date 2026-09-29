import shutil

import numpy as np
import pytest
from conftest import fake_maps

import hamster_animator as ha
from hamster_animator import render as R


def test_norm_stretches():
    x = np.logspace(-6, 0, 1000)
    for kind in ("log", "asinh", "sqrt", "linear"):
        y = R.Norm(R.Stretch(kind), x, signed=False)(x)
        assert y.min() >= 0 and y.max() <= 1 and np.all(np.diff(y) >= 0)
    s = R.Norm(R.Stretch(), np.linspace(-1, 1, 101), signed=True)
    assert s(0.0) == pytest.approx(0.5)


def test_quantities():
    m = fake_maps()
    for q in R.QUANTITIES:
        assert R.quantity(m, q).shape == m["I"].shape
    # azimuthal polarisation → Q_phi positive, U_phi ~ 0
    assert R.quantity(m, "Q_PHI").sum() > 0
    assert abs(R.quantity(m, "U_PHI")).sum() < 1e-6 * R.quantity(m, "Q_PHI").sum()
    with pytest.raises(ValueError):
        R.quantity(m, "X")


@pytest.mark.parametrize("style", list(ha.STYLES))
def test_render_all_styles(fake_cache, style):
    anim = ha.Animation(
        ha.presets.showcase(),
        ha.Camera.orbit(frames=4),
        quantity=ha.Steps(["I", "Q_PHI"], hold=1, fade=2),
        cache=fake_cache,
    )
    frames = anim.render(style)
    assert len(frames) == 4
    assert all(f.image.min() >= 0 and f.image.max() <= 1 for f in frames)
    st = ha.STYLES[style].with_(size=(200, 200))
    data = next(R.draw_frames(frames[:1], st, workers=1))
    assert len(data) == 200 * 200 * 3


def test_rgb_and_evolution_labels(fake_cache):
    variants = ha.presets.planet().sweep("planet.azimuth", [0, 90])
    anim = ha.Animation(ha.Steps(variants, fade=1), wavelength=(2.2e-6, 1.6e-6, 1.25e-6), cache=fake_cache)
    frames = anim.render()
    assert frames[0].image.shape[-1] == 3 and frames[0].ticks is None
    assert "planet.azimuth = 0" in frames[0].label


def test_write_outputs(fake_cache, tmp_path):
    anim = ha.Animation(ha.presets.spiral(), ha.Camera.orbit(frames=4), cache=fake_cache)
    style = ha.Style(size=(160, 160))
    frames = anim.render(style)
    out = R.write_video(R.draw_frames(frames, style, workers=1), tmp_path / "seq", style, fps=4)
    assert len(list(out.glob("*.png"))) == 4
    assert anim.contact_sheet(tmp_path / "sheet.png", n=4, cols=2, style=style, width=320).exists()
    if shutil.which("ffmpeg"):
        mp4 = R.write_video(R.draw_frames(frames, style, workers=2), tmp_path / "a.mp4", style, fps=4)
        assert mp4.stat().st_size > 0
