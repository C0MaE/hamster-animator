import pytest

from hamster_animator import Camera, View
from hamster_animator.camera import as_path


def test_orbit_is_seamless_loop():
    views = Camera.orbit(inclination=45, frames=8).views()
    assert len(views) == 8
    assert views[0].azimuth == 0 and views[-1].azimuth == pytest.approx(315)
    assert all(v.inclination == 45 for v in views)


def test_segments_continue_from_previous_end():
    path = Camera.tilt(frames=5, frm=10, to=80).then(Camera.zoom(frames=3, frm=600, to=200))
    views = path.views()
    assert len(path) == len(views) == 8
    assert views[4].inclination == pytest.approx(80)
    assert all(v.inclination == pytest.approx(80) for v in views[5:])
    assert views[-1].fov == pytest.approx(200)


def test_move_rejects_unknown_parameter():
    with pytest.raises(ValueError):
        Camera.move(10, altitude=(0, 1))


def test_defaults_and_fixed_paths():
    path = Camera.orbit(frames=4).with_defaults(fov=500)
    assert all(v.fov == 500 for v in path.views())
    b = Camera.tilt(frames=4, frm=0, to=30, ease="linear").boomerang()
    assert [v.inclination for v in b.views()] == pytest.approx([0, 10, 20, 30, 20, 10])
    assert len(as_path(View(inclination=20))) == 1


def test_view_key_normalises_azimuth():
    assert View(azimuth=360).key == View(azimuth=0).key
    assert View(azimuth=10).key_str != View(azimuth=20).key_str
