from __future__ import annotations

import pytest

from counterfact.stats import Z_95, newcombe_difference_ci, wilson_ci


def test_wilson_ci__empty_n_returns_zeros() -> None:
    assert wilson_ci(0, 0) == (0.0, 0.0)


def test_wilson_ci__perfect_pass_rate_bounded() -> None:
    lo, hi = wilson_ci(10, 10)
    assert 0.0 <= lo <= hi <= 1.0
    assert hi >= 0.99


def test_wilson_ci__symmetric_around_half_for_balanced_sample() -> None:
    lo, hi = wilson_ci(5, 10)
    assert 0.0 < lo < 0.5 < hi < 1.0
    assert abs((lo + hi) / 2 - 0.5) < 0.05


def test_z_95_is_positive() -> None:
    assert Z_95 > 1.9


def test_newcombe_interval_retains_uncertainty_at_extreme_proportions() -> None:
    assert newcombe_difference_ci(0, 10, 0, 10) == pytest.approx(
        (-0.2775327998628892, 0.2775327998628892)
    )
    low, high = newcombe_difference_ci(10, 10, 0, 10)
    assert 0 < low < 1
    assert high == pytest.approx(1)


def test_newcombe_interval_reverses_with_the_contrast() -> None:
    low, high = newcombe_difference_ci(4, 17, 8, 21)
    reverse_low, reverse_high = newcombe_difference_ci(8, 21, 4, 17)
    assert (reverse_low, reverse_high) == pytest.approx((-high, -low))


@pytest.mark.parametrize("counts", [(0, 0, 1, 2), (3, 2, 1, 2), (1, 2, -1, 2)])
def test_newcombe_interval_rejects_undefined_samples(counts) -> None:
    with pytest.raises(ValueError):
        newcombe_difference_ci(*counts)
