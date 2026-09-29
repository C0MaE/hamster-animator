import pytest

from hamster_animator import Steps
from hamster_animator.steps import Mix, as_track


def test_hold_and_fade_lengths():
    assert len(Steps([1, 2, 3], hold=2)) == 6
    assert len(Steps([1, 2, 3], hold=2, fade=3)) == 6 + 2 * 3
    assert len(Steps([1, 2, 3], hold=2, fade=3, loop=True)) == 6 + 3 * 3


def test_fade_weights():
    mixes = Steps([1, 2], hold=1, fade=1, ease="linear").mixes()
    assert mixes[1] == Mix(1, 2, 0.5)
    assert mixes[1].weights() == [(1, 0.5), (2, 0.5)]
    assert Mix(1).values == (1,) and Mix(1, 2, 0.7).dominant == 2


def test_as_track():
    assert as_track(3.0) == [Mix(3.0)]
    assert len(as_track([1, 2])) == 2
    with pytest.raises(ValueError):
        Steps([])
