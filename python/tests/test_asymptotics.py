"""Port of ``tests/testthat/test-EmpiricalCalibrationUsingAsymptotics.R`` and
``test-asymptoticNull.R``."""

import numpy as np
import pytest

import empiricalcalibration as ec
from empiricalcalibration import datasets


@pytest.fixture(scope="module")
def sccs():
    return datasets.sccs()


def test_calibrateP_argument_requirements():
    logRr = [0.1, 0.3, 0.2]
    seLogRr = [0.05, 0.1]
    null = ec.fitNull([1, 2], [0, 0])
    with pytest.raises(ValueError, match="arguments must be of equal length"):
        ec.calibrateP(null, logRr, seLogRr)


def test_calibrateOneP_argument_requirements():
    logRr = [0.1, 0.3, 0.2]
    seLogRr = [0.05, 0.1, np.nan]
    null = ec.fitNull([0, 0], [0, 0])
    assert np.allclose(ec.calibrateP(null, logRr, seLogRr)[:2],
                       [0.045500266, 0.002699796], atol=1e-6)
    assert np.isnan(ec.calibrateP(null, logRr, seLogRr)[2])
    assert np.allclose(ec.calibrateP(null, logRr, seLogRr, twoSided=False)[:2],
                       [0.022750133, 0.001349898], atol=1e-6)
    lower = np.round(ec.calibrateP(null, logRr, seLogRr, twoSided=False, upper=False), 2)
    assert np.allclose(lower[:2], [0.98, 1.00])


def test_computeTraditionalP_argument_options(sccs):
    positive = sccs[sccs.groundTruth == 1]
    assert ec.computeTraditionalP(positive.logRr, positive.seLogRr)[0] == 0
    assert ec.computeTraditionalP(positive.logRr, positive.seLogRr, twoSided=False)[0] == 0
    assert ec.computeTraditionalP(positive.logRr, positive.seLogRr,
                                  twoSided=False, upper=False)[0] == 1


def test_fitNullNonNormalLl_matches_fitNull(sccs):
    negatives = sccs[sccs.groundTruth == 0]
    null = ec.fitNull(negatives.logRr, negatives.seLogRr)
    nn = ec.fitNullNonNormalLl(negatives)
    assert np.allclose([nn[0], nn[1]], [null[0], null[1]])


def test_fitNullNonNormalLl_grid_approximation():
    """Fitting the null from grid profiles recovers the asymptotic answer."""
    from empiricalcalibration._rmath import dnorm
    import pandas as pd

    ec.set_seed(123)
    mu, sigma = 0.2, 0.2
    data = ec.simulateControls(n=50, mean=mu, sd=sigma, trueLogRr=0, seLogRr=0.1)
    goldStandardNull = ec.fitNull(logRr=data.logRr, seLogRr=data.seLogRr)

    point = np.linspace(np.log(0.1), np.log(10), 1000)
    grids = [pd.DataFrame({"point": point,
                           "value": dnorm(point, data.logRr.iloc[i], data.seLogRr.iloc[i])})
             for i in range(len(data))]
    null = ec.fitNullNonNormalLl(grids)
    assert abs(null[0] - goldStandardNull[0]) < 0.1
    assert abs(null[1] - goldStandardNull[1]) < 0.1


def test_fitNullNonNormalLl_errors_and_warnings(sccs):
    negatives = sccs[sccs.groundTruth == 0].copy()

    bad = negatives.copy()
    bad.columns = ["drugName", "mu", "grid", "sigma"]
    with pytest.raises(ValueError, match="but not all column names are numeric"):
        ec.fitNullNonNormalLl(bad)

    custom = negatives.copy()
    custom.columns = ["drugName", "mu", "gamma", "sigma"]
    custom.iloc[0, custom.columns.get_loc("mu")] = np.nan
    with pytest.warns(UserWarning, match="Approximations with NA parameters detected"):
        ec.fitNullNonNormalLl(custom)

    skew = negatives.copy()
    skew.columns = ["drugName", "mu", "alpha", "sigma"]
    skew.iloc[0, skew.columns.get_loc("mu")] = np.nan
    with pytest.warns(UserWarning, match="Approximations with NA parameters detected"):
        ec.fitNullNonNormalLl(skew)


def test_calibrateP_matches_traditional_when_null_is_degenerate():
    null = ec.Null(0, 0)
    logRr, seLogRr = 0.2, 0.2
    assert np.allclose(ec.calibrateP(null, logRr, seLogRr),
                       ec.computeTraditionalP(logRr, seLogRr))


def test_fitNull_all_estimates_na():
    with pytest.warns(UserWarning):
        null = ec.fitNull(logRr=[np.nan] * 3, seLogRr=[np.nan] * 3)
    repr(null)  # must not raise
    assert np.isnan(ec.calibrateP(null, 1, 1)[0])

    model = ec.convertNullToErrorModel(null)
    assert np.isnan(model[0])

    ci = ec.calibrateConfidenceInterval(1, 1, model)
    assert np.isnan(ci["logRr"].iloc[0])


# --- test-asymptoticNull.R -------------------------------------------------

def test_fitNull_argument_requirements(sccs):
    negatives = sccs[sccs.groundTruth == 0]
    with pytest.warns(UserWarning, match="infinite logRr"):
        ec.fitNull([np.inf], negatives.seLogRr.iloc[:1])
    with pytest.warns(UserWarning, match="infinite standard error"):
        ec.fitNull(negatives.logRr.iloc[:1], [np.inf])
    with pytest.warns(UserWarning, match="NA logRr"):
        ec.fitNull([np.nan], negatives.seLogRr.iloc[:1])
    with pytest.warns(UserWarning, match="NA standard error"):
        ec.fitNull(negatives.logRr.iloc[:1], [np.nan])
    with pytest.warns(UserWarning, match="extreme logRr"):
        ec.fitNull([101.0], negatives.seLogRr.iloc[:1])


def test_fitNull_output():
    null = ec.fitNull([1, 2], [0, 0])
    assert abs(null["mean"] - 1.5) < 1e-3
    assert repr(null).splitlines()[0] == "Estimated null distribution"


def test_fitNull_reproduces_r(sccs):
    """Bit-exact against R 4.6.1 EmpiricalCalibration 3.1.4."""
    negatives = sccs[sccs.groundTruth == 0]
    null = ec.fitNull(negatives.logRr, negatives.seLogRr)
    assert null[0] == 0.79215805192387778
    assert null[1] == 0.28343634583577032
