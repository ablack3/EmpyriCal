"""MaxSPRT critical values and the LLR/p-value bridge, beyond the R-parity checks."""

import numpy as np
import pandas as pd
import pytest

import empiricalcalibration as ec
from empiricalcalibration.llr_calibration import computeLlrFromP, computePFromLlr

# --- the LLR <-> p-value bridge -------------------------------------------

def test_llr_and_p_are_inverses_in_the_ordinary_range():
    for llr in [0.5, 1.0, 2.0, 5.0, 10.0, 25.0]:
        p = computePFromLlr(llr, mle=1.0)
        assert computeLlrFromP(p) == pytest.approx(llr, rel=1e-6)


def test_p_from_llr_is_monotone_decreasing():
    llr = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
    p = computePFromLlr(llr, mle=np.ones_like(llr))
    assert np.all(np.diff(p) < 0)


def test_zero_llr_gives_a_p_value_of_one_half():
    assert computePFromLlr(0.0, mle=1.0) == pytest.approx(0.5)


def test_p_below_the_null_is_reflected():
    """An mle below the null hypothesis flips the one-sided p-value."""
    above = computePFromLlr(2.0, mle=1.0, nullLogRr=0.0)
    below = computePFromLlr(2.0, mle=-1.0, nullLogRr=0.0)
    assert below == pytest.approx(1 - above)


def test_extreme_llr_uses_log_linear_extrapolation():
    """Above llr = 33 R's pchisq returns exactly 0, so the port extrapolates."""
    p = computePFromLlr(40.0, mle=1.0)
    assert 0 < p < 1e-17
    assert np.isfinite(np.log(p))


def test_extreme_p_uses_log_linear_extrapolation():
    """Below p = 1e-16 R's qchisq returns Inf, so the port extrapolates."""
    llr = computeLlrFromP(1e-30)
    assert np.isfinite(llr)
    assert llr > 60


def test_p_at_or_above_one_half_gives_zero_llr():
    assert computeLlrFromP(0.5) == 0.0
    assert computeLlrFromP(0.9) == 0.0


# --- critical values -------------------------------------------------------

@pytest.fixture(scope="module")
def group_sizes():
    return [1.0] * 10


def test_critical_value_carries_the_selected_alpha(group_sizes):
    ec.set_seed(1)
    cv = ec.computeCvPoisson(group_sizes, alpha=0.05, sampleSize=5_000)
    assert isinstance(cv, float)
    assert float(cv) > 0
    assert 0 < cv.alpha < 0.05
    assert cv.attributes == {"alpha": cv.alpha}


def test_a_stricter_alpha_needs_a_higher_threshold(group_sizes):
    ec.set_seed(2)
    lenient = ec.computeCvPoisson(group_sizes, alpha=0.10, sampleSize=5_000)
    ec.set_seed(2)
    strict = ec.computeCvPoisson(group_sizes, alpha=0.01, sampleSize=5_000)
    assert float(strict) > float(lenient)


def test_more_looks_needs_a_higher_threshold():
    """More looks means more chances to cross, so the boundary must rise."""
    ec.set_seed(3)
    few = ec.computeCvPoisson([5.0] * 2, sampleSize=5_000)
    ec.set_seed(3)
    many = ec.computeCvPoisson([1.0] * 10, sampleSize=5_000)
    assert float(many) > float(few)


def test_an_empirical_null_raises_the_threshold(group_sizes):
    """The whole point of nullMean/nullSd: systematic error costs you alpha."""
    ec.set_seed(4)
    plain = ec.computeCvPoisson(group_sizes, sampleSize=5_000)
    ec.set_seed(4)
    calibrated = ec.computeCvPoisson(group_sizes, sampleSize=5_000,
                                     nullMean=0.5, nullSd=0.3)
    assert float(calibrated) > float(plain)


def test_minimumEvents_lowers_the_threshold(group_sizes):
    """Requiring more events before rejection *truncates* the null distribution
    of the maximum LLR — trials that never accrue the minimum contribute zero —
    so its upper quantile, and hence the critical value, falls."""
    ec.set_seed(5)
    one = ec.computeCvPoisson(group_sizes, minimumEvents=1, sampleSize=5_000)
    ec.set_seed(5)
    five = ec.computeCvPoisson(group_sizes, minimumEvents=5, sampleSize=5_000)
    assert float(five) < float(one)


def test_binomial_critical_value_is_positive_and_finite(group_sizes):
    ec.set_seed(6)
    cv = ec.computeCvBinomial(group_sizes, z=4, sampleSize=5_000)
    assert np.isfinite(float(cv)) and float(cv) > 0
    assert 0 < cv.alpha < 0.05


def test_poisson_regression_critical_value_is_positive_and_finite(group_sizes):
    ec.set_seed(7)
    cv = ec.computeCvPoissonRegression(group_sizes, z=4, sampleSize=5_000)
    assert np.isfinite(float(cv)) and float(cv) > 0
    assert 0 < cv.alpha < 0.05


def test_poisson_regression_validates_its_arguments():
    with pytest.raises(ValueError, match="should be positive"):
        ec.computeCvPoissonRegression([-1, 2], z=1)
    with pytest.raises(ValueError, match="between 0 and 1"):
        ec.computeCvPoissonRegression([1, 2], z=1, alpha=1.5)


def test_scalar_fallback_path_agrees_in_magnitude():
    """Poisson means above 10 leave the vectorised path; the answer must still
    be a sane critical value rather than a numerical artefact."""
    ec.set_seed(8)
    big = ec.computeCvPoisson([25.0] * 4, sampleSize=3_000)
    assert 1.0 < float(big) < 10.0


# --- calibrateLlr ----------------------------------------------------------

@pytest.fixture(scope="module")
def sccs():
    return ec.datasets.sccs()


def test_calibrated_llr_is_never_negative(sccs):
    negatives = sccs[sccs.groundTruth == 0]
    null = ec.fitNull(negatives.logRr, negatives.seLogRr)
    llr = ec.calibrateLlr(null, negatives[["logRr", "seLogRr"]])
    assert np.all(np.asarray(llr) >= 0)


def test_calibration_shrinks_the_llr(sccs):
    """Against a biased null, the calibrated LLR must not exceed the raw one."""
    negatives = sccs[sccs.groundTruth == 0]
    null = ec.fitNull(negatives.logRr, negatives.seLogRr)

    # A moderate estimate, so the raw p-value is comfortably above zero and the
    # comparison is not made trivially true by an infinite raw LLR.
    est = pd.DataFrame({"logRr": [1.1], "seLogRr": [0.35]})
    raw_p = float(np.atleast_1d(
        ec.computeTraditionalP(1.1, 0.35, twoSided=False, upper=True))[0])
    assert 0 < raw_p < 0.05

    raw = computeLlrFromP(raw_p)
    calibrated = float(ec.calibrateLlr(null, est)[0])
    assert np.isfinite(raw)
    assert calibrated < raw


def test_a_degenerate_null_leaves_the_llr_essentially_unchanged():
    """With no systematic error there is nothing to integrate over."""
    null = ec.Null(0.0, 0.0)
    est = pd.DataFrame({"logRr": [0.8], "seLogRr": [0.2]})
    calibrated = float(ec.calibrateLlr(null, est)[0])
    raw = computeLlrFromP(float(np.atleast_1d(
        ec.computeTraditionalP(0.8, 0.2, twoSided=False, upper=True))[0]))
    assert calibrated == pytest.approx(raw, rel=1e-3)


def test_calibrateLlr_handles_missing_estimates():
    null = ec.Null(0.2, 0.2)
    est = pd.DataFrame({"logRr": [0.8, np.nan], "seLogRr": [0.2, 0.2]})
    result = ec.calibrateLlr(null, est)
    assert np.isfinite(result[0])
    assert np.isnan(result[1])


def test_calibrateLlr_with_an_mcmc_null_returns_an_interval(sccs):
    negatives = sccs[sccs.groundTruth == 0]
    positive = sccs[sccs.groundTruth == 1]
    ec.set_seed(30)
    null = ec.fitMcmcNull(negatives.logRr, negatives.seLogRr, iter=300)
    result = ec.calibrateLlr(null, positive)
    assert list(result.columns) == ["llr", "ci95Lb", "ci95Ub"]
    assert (result.ci95Lb <= result.llr).all()
    assert (result.llr <= result.ci95Ub).all()
