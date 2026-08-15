"""The four likelihood-approximation shapes, and the convolution that uses them.

These exercise the same code as the grid/custom/skew-normal fitting tests, but
directly and on small inputs, so the coverage does not depend on the expensive
end-to-end fits (which are marked `slow`).
"""

import numpy as np
import pandas as pd
import pytest

import empiricalcalibration as ec
from empiricalcalibration._rmath import dnorm
from empiricalcalibration.likelihoods import (
    customLlApproximation,
    gridLlApproximation,
    logLikelihoodNullNonNormalLl,
    normalLlApproximaton,
    skewNormalLlApproximation,
)


# --- normal ----------------------------------------------------------------

def test_normal_approximation_is_the_normal_log_density():
    params = {"logRr": 0.3, "seLogRr": 0.2}
    for x in (-0.5, 0.0, 0.3, 1.0):
        assert normalLlApproximaton(x, params) == pytest.approx(
            dnorm(x, 0.3, 0.2, log=True), rel=1e-14)


def test_normal_approximation_peaks_at_the_estimate():
    params = {"logRr": 0.3, "seLogRr": 0.2}
    x = np.linspace(-1, 1.6, 261)
    values = np.array([normalLlApproximaton(xi, params) for xi in x])
    assert x[np.argmax(values)] == pytest.approx(0.3, abs=0.01)


def test_parameters_can_be_a_frame_a_series_or_a_dict():
    frame = pd.DataFrame({"logRr": [0.3], "seLogRr": [0.2]})
    series = frame.iloc[0]
    mapping = {"logRr": 0.3, "seLogRr": 0.2}
    values = [normalLlApproximaton(0.1, p) for p in (frame, series, mapping)]
    assert values[0] == values[1] == values[2]


# --- custom ("gamma") ------------------------------------------------------

def test_custom_approximation_reduces_to_a_scaled_quadratic_at_gamma_zero():
    params = {"mu": 0.2, "sigma": 0.3, "gamma": 0.0}
    for x in (-0.4, 0.2, 0.9):
        assert customLlApproximation(x, params) == pytest.approx(
            -((x - 0.2) ** 2) / (2 * 0.3 ** 2), rel=1e-14)


def test_custom_approximation_peaks_at_mu():
    params = {"mu": 0.2, "sigma": 0.3, "gamma": 0.0}
    x = np.linspace(-1, 1.4, 241)
    assert x[np.argmax(customLlApproximation(x, params))] == pytest.approx(0.2, abs=0.01)


def test_custom_approximation_gamma_tilts_it():
    x = np.linspace(-1, 1.4, 241)
    left = customLlApproximation(x, {"mu": 0.2, "sigma": 0.3, "gamma": -1.0})
    right = customLlApproximation(x, {"mu": 0.2, "sigma": 0.3, "gamma": 1.0})
    assert not np.allclose(left, right)


# --- skew normal -----------------------------------------------------------

def test_skew_normal_at_alpha_zero_is_the_normal_log_density():
    """log(2) + log phi + log Phi(0) = log phi, since Phi(0) = 1/2."""
    params = {"mu": 0.1, "sigma": 0.25, "alpha": 0.0}
    for x in (-0.3, 0.1, 0.8):
        assert skewNormalLlApproximation(x, params) == pytest.approx(
            dnorm(x, 0.1, 0.25, log=True), rel=1e-12)


def test_skew_normal_alpha_shifts_the_mode():
    x = np.linspace(-1.5, 1.5, 601)
    left = x[np.argmax(skewNormalLlApproximation(x, {"mu": 0.0, "sigma": 0.3,
                                                     "alpha": -4.0}))]
    right = x[np.argmax(skewNormalLlApproximation(x, {"mu": 0.0, "sigma": 0.3,
                                                      "alpha": 4.0}))]
    assert left < 0 < right


# --- grid ------------------------------------------------------------------

@pytest.fixture
def profile():
    point = np.array([-1.0, 0.0, 1.0, 2.0])
    value = np.array([-4.0, 0.0, -1.0, -4.0])
    return pd.DataFrame({"point": point, "value": value})


def test_grid_returns_the_node_values_at_the_nodes(profile):
    interior = profile.point.to_numpy()[1:-1]
    got = gridLlApproximation(interior, profile)
    assert got == pytest.approx(profile.value.to_numpy()[1:-1])


def test_grid_interpolates_linearly_between_nodes(profile):
    # halfway between (0, 0) and (1, -1)
    assert gridLlApproximation(0.5, profile) == pytest.approx(-0.5)
    # halfway between (-1, -4) and (0, 0)
    assert gridLlApproximation(-0.5, profile) == pytest.approx(-2.0)


def test_grid_extrapolates_below_the_range_with_a_non_negative_slope(profile):
    """Slope is max(0, ...), so the profile can only fall away to the left."""
    assert gridLlApproximation(-2.0, profile) == pytest.approx(-8.0)


def test_grid_extrapolates_above_the_range_with_a_non_positive_slope(profile):
    """Slope is min(0, ...), so the profile can only fall away to the right."""
    assert gridLlApproximation(3.0, profile) == pytest.approx(-7.0)


def test_grid_extrapolation_slope_is_clamped_flat_when_it_would_rise():
    """A profile increasing at its right edge must not keep increasing outside."""
    rising = pd.DataFrame({"point": [0.0, 1.0, 2.0], "value": [-4.0, -2.0, 0.0]})
    assert gridLlApproximation(5.0, rising) == pytest.approx(0.0)

    falling_at_left = pd.DataFrame({"point": [0.0, 1.0, 2.0],
                                    "value": [0.0, -2.0, -4.0]})
    assert gridLlApproximation(-5.0, falling_at_left) == pytest.approx(0.0)


def test_grid_accepts_a_plain_mapping(profile):
    mapping = {"point": profile.point.to_numpy(), "value": profile.value.to_numpy()}
    assert gridLlApproximation(0.5, mapping) == pytest.approx(-0.5)


def test_grid_always_returns_an_array(profile):
    """A scalar argument still comes back as a length-1 array.

    The function has a ``np.ndim(x) == 0`` branch that would return a bare
    float, but ``x`` is promoted with ``atleast_1d`` beforehand, so it is
    unreachable.  Every caller coerces with ``.item()``, so this is dead code
    rather than a defect -- pinned here so a future change to the return type
    is a deliberate one.
    """
    scalar_result = gridLlApproximation(0.5, profile)
    assert scalar_result.shape == (1,)
    assert float(scalar_result[0]) == pytest.approx(-0.5)
    assert gridLlApproximation(np.array([0.5, 0.25]), profile).shape == (2,)


# --- the convolution -------------------------------------------------------

def _normal_profiles(logRr, seLogRr, n_points=60):
    point = np.linspace(-3, 3, n_points)
    return [pd.DataFrame({"point": point, "value": dnorm(point, m, s, log=True)})
            for m, s in zip(logRr, seLogRr)]


def test_non_normal_likelihood_rejects_a_non_positive_precision():
    profiles = _normal_profiles([0.1, 0.2], [0.3, 0.3])
    assert logLikelihoodNullNonNormalLl([0.0, -1.0], gridLlApproximation,
                                        profiles) == 99999.0


def test_non_normal_likelihood_is_minimised_near_the_truth():
    """With profiles centred on 0.4, the objective should prefer mu = 0.4."""
    profiles = _normal_profiles([0.35, 0.40, 0.45], [0.3, 0.3, 0.3])
    at_truth = logLikelihoodNullNonNormalLl([0.4, 25.0], gridLlApproximation, profiles)
    for wrong_mu in (-0.5, 0.0, 1.2):
        assert logLikelihoodNullNonNormalLl(
            [wrong_mu, 25.0], gridLlApproximation, profiles) > at_truth


def test_non_normal_likelihood_handles_a_degenerate_sd():
    """sd < 1e-6 takes the direct-evaluation branch rather than integrating."""
    profiles = _normal_profiles([0.1, 0.2], [0.3, 0.3])
    value = logLikelihoodNullNonNormalLl([0.15, 1e14], gridLlApproximation, profiles)
    assert np.isfinite(value)


def test_fitNullNonNormalLl_short_circuits_on_normal_columns():
    """A frame with logRr/seLogRr must give exactly fitNull, with no integration."""
    sccs = ec.datasets.sccs()
    negatives = sccs[sccs.groundTruth == 0]
    direct = ec.fitNull(negatives.logRr, negatives.seLogRr)
    viaNonNormal = ec.fitNullNonNormalLl(negatives[["logRr", "seLogRr"]])
    assert float(viaNonNormal[0]) == float(direct[0])
    assert float(viaNonNormal[1]) == float(direct[1])


def test_fitNullNonNormalLl_rejects_non_numeric_grid_columns():
    bad = pd.DataFrame({"a": [1.0], "b": [2.0]})
    with pytest.raises(ValueError, match="not all column names are numeric"):
        ec.fitNullNonNormalLl(bad)


def test_fitNullNonNormalLl_accepts_a_wide_grid_frame():
    """Column names as grid points is the OHDSI likelihood-profile export layout."""
    point = np.linspace(-2, 2, 25)
    rows = [dnorm(point, m, 0.3, log=True) for m in (0.1, 0.2, 0.3)]
    wide = pd.DataFrame(rows, columns=point)
    null = ec.fitNullNonNormalLl(wide)
    assert np.isfinite(float(null[0]))
    assert float(null[0]) == pytest.approx(0.2, abs=0.4)
