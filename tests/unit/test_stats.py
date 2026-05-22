from __future__ import annotations

from counterfact.stats import Z_95, wilson_ci


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
