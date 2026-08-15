"""Port of ``tests/testthat/test-likelihoods.R``."""

import numpy as np
import pytest

from empiricalcalibration._rmath import dgamma_log, dnorm
from empiricalcalibration.likelihoods import (
    gaussianProduct,
    logLikelihoodNull,
    logLikelihoodNullMcmc,
    minLogLikelihoodErrorModel,
    minLogLikelihoodErrorModelLegacy,
    skewNormalLlApproximation,
)


def test_gaussianProduct_special_values():
    assert np.isnan(gaussianProduct(1, 2, 0.1, np.nan))
    assert gaussianProduct(1, 2, 0.1, np.inf) == 0


def test_gaussianProduct_both_sd_zero():
    assert np.isnan(gaussianProduct(1, 2, 0, 0))


def test_gaussianProduct_result_is_correct():
    assert gaussianProduct(1, 2, 0.1, 0.01) == pytest.approx(1.256095e-21, rel=1e-6)


def test_logLikelihoodNull_negative_precision():
    assert logLikelihoodNull([1, -1], [1, 2], [10.2, 10.2]) == np.inf


def test_logLikelihoodNull_logRr_and_seLogRr_input():
    assert np.isnan(logLikelihoodNull([1, 2], [np.nan, 2], [1, 0.1]))
    assert np.isnan(logLikelihoodNull([1, 2], [1, 2], [1, np.nan]))
    assert logLikelihoodNull([1, 2], [np.inf, 2], [1, 0.1]) == np.inf
    assert logLikelihoodNull([1, 2], [1, 2], [1, np.inf]) == np.inf


def test_logLikelihoodNull_correct_calculation():
    sd = 1e-7
    result = logLikelihoodNull([1, 1 / sd ** 2], [1, 2], [0.01, 0.1])
    expected = -dnorm(1, 1, 0.01, log=True) - dnorm(1, 2, 0.1, log=True)
    assert result == pytest.approx(expected)


def test_logLikelihoodNull_result_is_infinite():
    assert logLikelihoodNull([1, 0.001], [1, 0], [0.1, 2.0 ** 1023]) == np.inf


def test_logLikelihoodNullMcmc_inputs():
    assert np.isnan(logLikelihoodNullMcmc([1, 2], [np.nan, 2], [1, 0.1]))
    assert np.isnan(logLikelihoodNullMcmc([1, 2], [1, 2], [1, np.nan]))


def test_logLikelihoodNullMcmc_result():
    result1 = (-np.log(gaussianProduct(1, 1, 0.01, 1 / np.sqrt(2)))
               - np.log(gaussianProduct(2, 1, 0.1, 1 / np.sqrt(2)))
               - dgamma_log(2, 1e-04, 1e-04))
    result = logLikelihoodNullMcmc([1, 2], [1, 2], [0.01, 0.1])
    assert result == pytest.approx(result1)


def test_minLogLikelihoodErrorModel_cases():
    assert minLogLikelihoodErrorModel([3, 4, -2, 5], [2], [0.2], [0.25]) == 99999

    r = minLogLikelihoodErrorModel([3, 4, 2e-7, 5e-7], [2], [0.2], [0.25])
    assert r == pytest.approx(-dnorm(2, 4, 0.2, log=True))

    r = minLogLikelihoodErrorModel([3, 4, 2e-5, 5e-5], [2], [0.2], [0.25])
    assert r == pytest.approx(-np.log(gaussianProduct(2, 4, 0.2, 3.25e-05)))


def test_minLogLikelihoodErrorModelLegacy_infinite_result():
    r = minLogLikelihoodErrorModelLegacy([3, 4, 2e-5, 5e-5], [2], [2.0 ** 1023], [0.25])
    assert r == 99999


def test_skewNormalLlApproximation_infinite_sigma():
    params = {"mu": 0.0, "sigma": np.inf, "alpha": 1.0}
    assert np.array_equal(skewNormalLlApproximation(np.array([1, 2, 3]), params),
                          np.array([0, 0, 0]))


def test_skewNormalLlApproximation_stays_finite_in_the_tail():
    """R uses pnorm(log.p = TRUE) here; log(pnorm(...)) would give -Inf.

    Reference values from R 4.6.1:
    log(2) + dnorm(x, 0, 0.1, log = TRUE) + pnorm(5 * x, 0, 0.1, log.p = TRUE)
    """
    params = {"mu": 0.0, "sigma": 0.1, "alpha": 5.0}
    got = skewNormalLlApproximation(np.array([-0.5, -1.0, -2.0]), params)
    assert np.all(np.isfinite(got))
    assert got == pytest.approx([-327.0626, -1302.755, -5203.447], rel=1e-6)


def test_pnorm_log_p_matches_r_far_into_the_tail():
    """R: pnorm(c(-45, -100), log.p = TRUE)."""
    from empiricalcalibration._rmath import pnorm
    got = pnorm(np.array([-45.0, -100.0]), log_p=True)
    assert got == pytest.approx([-1017.2260942420, -5005.5242086942], rel=1e-12)
