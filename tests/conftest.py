import numpy as np
import pytest

import hamster_animator as ha


def fake_maps(n=48, pixel_au=12.5):
    """Synthetic Stokes maps: a bright ring with azimuthal polarisation."""
    c = (n - 1) / 2
    y, x = np.mgrid[0:n, 0:n]
    r = np.hypot(x - c, y - c) + 1.0
    phi = np.arctan2(y - c, x - c)
    I = np.exp(-0.5 * ((r - n / 4) / 3) ** 2) + 0.01 / r
    p = 0.3 * I
    return {
        "I": I,
        "Q": -p * np.cos(2 * phi),
        "U": -p * np.sin(2 * phi),
        "V": 0 * I,
        "I_DIRECT": 0 * I,
        "I_SCAT": I,
        "pixel_au": pixel_au,
    }


@pytest.fixture
def fake_cache(tmp_path, monkeypatch):
    """A cache that claims to have every map and returns synthetic data."""
    cache = ha.Cache(tmp_path / "cache")
    monkeypatch.setattr(ha.Cache, "has", lambda self, *a: True)
    monkeypatch.setattr(ha.Cache, "load", lambda self, disk, sim, view, lam: fake_maps())
    return cache
