"""Port of ``test-empiricalCalibrationUsingMcmc.R``, ``test-LlrCalibration.R``,
``test-critical-values.R`` and ``test-simulation.R``."""

import numpy as np
import pandas as pd
import pytest

import empiricalcalibration as ec
from empiricalcalibration import datasets
from empiricalcalibration._rmath import dnorm, pnorm


@pytest.fixture(scope="module")
def sccs():
    return datasets.sccs()


# --- MCMC ------------------------------------------------------------------

def test_fitMcmcNull_requirements():
    with pytest.warns(UserWarning, match="infinite logRr"):
        ec.fitMcmcNull([0, np.inf], [1, 0], iter=100)
    with pytest.warns(UserWarning, match="infinite standard error"):
        ec.fitMcmcNull([0, 0], [1, np.inf], iter=100)
    with pytest.warns(UserWarning, match="NA logRr"):
        ec.fitMcmcNull([0, np.nan], [1, 0], iter=100)
    with pytest.warns(UserWarning, match="NA standard error"):
        ec.fitMcmcNull([0, 0], [1, np.nan], iter=100)


def test_mcmc_calibration_close_to_truth():
    ec.set_seed(124)
    negatives = ec.simulateControls(n=50, mean=0.25, sd=0.16, trueLogRr=0)
    null = ec.fitMcmcNull(negatives.logRr, negatives.seLogRr, iter=10000)
    assert null[0] == pytest.approx(0.25, abs=0.05)

    v = ec.calibrateP(null, 0.2, 0.2, twoSided=False, upper=False)
    assert v["p"].iloc[0] == pytest.approx(pnorm(0.2, 0.25, 0.16 + 0.2 ** 2), abs=0.05)

    v = ec.calibrateP(null, 0.2, 0.2, twoSided=True)
    target = min(pnorm(0.2, 0.25, 0.16 + 0.2 ** 2),
                 pnorm(0.2, 0.25, 0.16 + 0.2 ** 2, lower_tail=False))
    assert v["p"].iloc[0] == pytest.approx(2 * target, abs=0.05)

    for logRr, seLogRr in [(np.nan, 0.2), (0.2, np.nan)]:
        c = ec.calibrateP(null, logRr, seLogRr, twoSided=False, upper=False)
        assert c[["p", "lb95ci", "ub95ci"]].isna().all().all()


def test_fitMcmcNull_all_estimates_na():
    with pytest.warns(UserWarning):
        null = ec.fitMcmcNull(logRr=[np.nan] * 3, seLogRr=[np.nan] * 3)
    repr(null)
    assert np.isnan(ec.calibrateP(null, 1, 1)["p"].iloc[0])
    assert np.isnan(ec.calibrateP(null, 1, 1, pValueOnly=True)[0])
    model = ec.convertNullToErrorModel(null)
    assert np.isnan(model[0])
    ci = ec.calibrateConfidenceInterval(1, 1, model)
    assert np.isnan(ci["logRr"].iloc[0])
    ease = ec.computeExpectedAbsoluteSystematicError(null)
    assert np.isnan(ease["ease"].iloc[0])


def test_mcmc_reproduces_r(sccs):
    """The MCMC chain follows R's RNG stream exactly."""
    negatives = sccs[sccs.groundTruth == 0]
    ec.set_seed(555)
    null = ec.fitMcmcNull(negatives.logRr, negatives.seLogRr, iter=2000)
    assert null[0] == pytest.approx(0.7930473561682035, rel=1e-12)
    assert null[1] == pytest.approx(11.849575025721808, rel=1e-12)


# --- LLR calibration -------------------------------------------------------

def test_calibrateLlr_normal(sccs):
    negatives = sccs[sccs.groundTruth == 0]
    positive = sccs[sccs.groundTruth == 1]
    null = ec.fitNull(negatives.logRr, negatives.seLogRr)
    assert ec.calibrateLlr(null, positive)[0] == 0

    with pytest.raises(ValueError, match="only one-sided upper LLRs are supported"):
        ec.calibrateLlr(null, positive, twoSided=True)
    with pytest.raises(ValueError, match="only one-sided upper LLRs are supported"):
        ec.calibrateLlr(null, positive, upper=False)


def test_calibrateLlr_grid():
    ec.set_seed(123)
    data = ec.simulateControls(n=50, mean=0.2, sd=0.2, trueLogRr=0, seLogRr=0.1)
    point = np.linspace(np.log(0.1), np.log(10), 1000)
    grids = [pd.DataFrame({"point": point,
                           "value": dnorm(point, data.logRr.iloc[i], data.seLogRr.iloc[i])})
             for i in range(len(data))]
    null = ec.fitNullNonNormalLl(grids)
    result = ec.calibrateLlr(null, grids[:5])
    assert len(result) == 5
    assert np.all(np.asarray(result) >= 0)


def test_calibrateLlr_non_normal(sccs):
    negatives = sccs[sccs.groundTruth == 0]
    positive = sccs[sccs.groundTruth == 1]
    null = ec.fitNull(negatives.logRr, negatives.seLogRr)

    custom = positive.copy()
    custom.columns = ["drugName", "mu", "gamma", "sigma"]
    assert ec.calibrateLlr(null, custom)[0] == pytest.approx(0.247, abs=0.1)

    skew = positive.copy()
    skew.columns = ["drugName", "mu", "alpha", "sigma"]
    assert ec.calibrateLlr(null, skew)[0] == pytest.approx(0.344, abs=0.1)


# --- MaxSPRT critical values ----------------------------------------------

def test_computeCvPoisson_boundary_conditions():
    with pytest.raises(ValueError, match="should be positive"):
        ec.computeCvPoisson([-1, 2])
    for kwargs, msg in [(dict(minimumEvents=1.3), "positive integer"),
                        (dict(minimumEvents=-1), "positive integer"),
                        (dict(alpha=-0.4), "between 0 and 1"),
                        (dict(alpha=1.3), "between 0 and 1"),
                        (dict(sampleSize=1.3), "positive integer"),
                        (dict(sampleSize=-1), "positive integer")]:
        with pytest.raises(ValueError, match=msg):
            ec.computeCvPoisson([0, 1, 2], **kwargs)


def test_computeCvBinomial_boundary_conditions():
    with pytest.raises(ValueError, match="should be positive"):
        ec.computeCvBinomial([-1, 2], z=1)
    with pytest.raises(ValueError, match="positive"):
        ec.computeCvBinomial([0, 1, 2], z=-1.2)
    for kwargs, msg in [(dict(minimumEvents=1.3), "positive integer"),
                        (dict(minimumEvents=-1), "positive integer"),
                        (dict(alpha=-0.4), "between 0 and 1"),
                        (dict(alpha=1.3), "between 0 and 1"),
                        (dict(sampleSize=1.3), "positive integer"),
                        (dict(sampleSize=-1), "positive integer")]:
        with pytest.raises(ValueError, match=msg):
            ec.computeCvBinomial([0, 1, 2], z=1, **kwargs)


@pytest.mark.parametrize("fn,kwargs,expected", [
    (ec.computeCvPoisson, dict(), 2.5451774444795623),
    (ec.computeCvBinomial, dict(z=4), 3.0649537425959439),
])
def test_critical_values_match_sequential(fn, kwargs, expected):
    """The R package validates these against the Sequential package; matching R
    at one million Monte-Carlo samples therefore matches Sequential too."""
    ec.set_seed(1 if fn is ec.computeCvPoisson else 2)
    cv = fn([1.0] * 10, **kwargs)
    assert float(cv) == pytest.approx(expected, rel=1e-6)
    assert 0 < cv.alpha < 0.05


def test_critical_values_reproduce_r_at_small_sample():
    ec.set_seed(1)
    cv = ec.computeCvPoisson([1.0] * 10, sampleSize=10000)
    assert float(cv) == 2.5451774444795623
    assert cv.alpha == 0.0408


def test_computeCvPoisson_with_group_size_zero():
    ec.set_seed(5)
    cv = ec.computeCvPoisson([0.0, 1, 2, 3], sampleSize=10000)
    assert np.isfinite(float(cv))


# --- simulation ------------------------------------------------------------

def test_simulateControls_degenerate():
    n, mean, trueLogRr = 100, 2, 1
    data = ec.simulateControls(n=n, mean=mean, sd=0, seLogRr=0, trueLogRr=trueLogRr)
    assert np.allclose(data.logRr, np.repeat(mean + trueLogRr, n))


def test_simulateControls_reproduces_r():
    """Values from R 4.6.1 with set.seed(101).

    Not asserted bit-exactly: R's compiler contracts ``mu + sigma * z`` into a
    fused multiply-add on this platform, so the last bit can differ even though
    the underlying uniform stream is identical (see test below).
    """
    ec.set_seed(101)
    d = ec.simulateControls(n=150, mean=0.25, sd=0.25, trueLogRr=np.log([1, 2, 4]))
    assert d.logRr.iloc[0] == pytest.approx(0.14372009925922047, rel=1e-13)
    assert d.seLogRr.iloc[0] == pytest.approx(0.06092271543573588, rel=1e-13)


def test_raw_rng_stream_is_bit_identical_to_r():
    """R's set.seed(101); runif(5) -- exact, not approximate."""
    from empiricalcalibration._rrng import RRandom
    got = RRandom(101).runif(5)
    assert list(got) == [0.37219837633892894, 0.043824815424159169,
                         0.70968401827849448, 0.65769039653241634,
                         0.24985572323203087]
    # set.seed(123); rpois(6, 3.7) and rbinom(6, 20, 0.2)
    assert list(RRandom(123).rpois_n(6, 3.7)) == [3, 5, 3, 6, 7, 1]
    assert list(RRandom(5).rbinom_n(6, 20, 0.2)) == [2, 5, 7, 3, 2, 5]
    # set.seed(3); sample.int(45, 8, replace = TRUE)
    assert list(RRandom(3).sample_int(45, 8)) == [5, 12, 39, 36, 40, 43, 31, 8]


def test_simulateMaxSprtData_shape():
    ec.set_seed(909)
    data = ec.simulateMaxSprtData(n=200, numberOfNegativeControls=2,
                                  numberOfPositiveControls=1)
    assert set(data.columns) == {"time", "outcome", "exposure", "lookTime", "outcomeId"}
    assert sorted(data.outcomeId.unique()) == [1, 2, 3]
    assert len(data.lookTime.unique()) == 10
    assert (data.time > 0).all()
