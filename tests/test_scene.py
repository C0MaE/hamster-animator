import numpy as np
import pytest

from hamster_animator import Disk, Gap, Planet, Spiral, Warp, presets


def test_density_is_vectorised_and_bounded():
    d = presets.showcase()
    x = np.linspace(-400, 400, 101)
    rho = d.density(x, np.zeros_like(x), np.zeros_like(x))
    assert rho.shape == x.shape
    assert np.all(np.isfinite(rho)) and np.all(rho >= 0)
    assert rho[np.abs(x) > d.r_out].max() == 0


def test_structures_change_density():
    base = Disk()
    assert Disk(structures=(Gap(radius=50, depth=1.0),)).density(50, 0, 0) < 1e-3 * base.density(50, 0, 0)
    assert Disk(structures=(Spiral(),)).density_map(n=64).max() > base.density_map(n=64).max()
    assert Disk(structures=(Warp(),)).density(5, 0, 0.5) != base.density(5, 0, 0.5)


def test_keys_variants_and_roundtrip():
    d = presets.planet(radius=60)
    assert d.key == presets.planet(radius=60).key
    v = d.with_(**{"planet.radius": 70})
    assert v.key != d.key and v.get("planet.radius") == 70
    assert [x.get("planet.azimuth") for x in d.sweep("planet.azimuth", [0, 90])] == [0, 90]
    assert Disk.from_dict(d.to_dict()) == d
    with pytest.raises(KeyError):
        d.with_(**{"spiral.pitch": 10})


def test_axisymmetry():
    assert Disk().axisymmetric
    assert Disk(structures=(Gap(),)).axisymmetric
    assert not Disk(structures=(Planet(),)).axisymmetric
