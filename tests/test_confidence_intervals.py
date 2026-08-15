"""Port of ``tests/testthat/test-confidenceIntervalCalibration.R``,
``test-expected-systematic-error.R`` and ``test-evaluation.R``."""

import numpy as np
import pytest

import empiricalcalibration as ec
from empiricalcalibration import datasets
from empiricalcalibration.expected_systematic_error import closedFormIntegeralAbsolute


@pytest.fixture(scope="module")
def sccs():
    return datasets.sccs()


def test_fitSystematicErrorModel_requirements():
    trueLogRr = [0, 0]
    with pytest.warns(UserWarning, match="infinite standard error"):
        ec.fitSystematicErrorModel([0, 0], [1, np.inf], trueLogRr)
    with pytest.warns(UserWarning, match="infinite logRr"):
        ec.fitSystematicErrorModel([0, np.inf], [1, 0], trueLogRr)
    with pytest.warns(UserWarning, match="NA standard error"):
        ec.fitSystematicErrorModel([0, 0], [1, np.nan], trueLogRr)
    with pytest.warns(UserWarning, match="NA logRr"):
        ec.fitSystematicErrorModel([0, np.nan], [1, 0], trueLogRr)


def test_fitSystematicErrorModel_legacy_with_covariance():
    ec.set_seed(1)
    controls = ec.simulateControls(n=50 * 3, mean=0.25, sd=0.25,
                                   trueLogRr=np.log([1, 2, 4]))
    model = ec.fitSystematicErrorModel(controls.logRr, controls.seLogRr,
                                       controls.trueLogRr, legacy=True,
                                       estimateCovarianceMatrix=True)
    assert model[0] == pytest.approx(0.254, abs=0.1)
    assert model.names[2] == "logSdIntercept"
    assert "CovarianceMatrix" in model.attributes
    assert model.attributes["LB95CI"][0] < model[0] < model.attributes["UB95CI"][0]


def test_convertNullToErrorModel_requirements():
    with pytest.raises(ValueError, match="type 'null'"):
        ec.convertNullToErrorModel(null=0)


def test_convertNullToErrorModel(sccs):
    negatives = sccs[sccs.groundTruth == 0]
    null = ec.fitNull(negatives.logRr, negatives.seLogRr)
    model = ec.convertNullToErrorModel(null)
    positive = sccs[sccs.groundTruth == 1]
    result = ec.calibrateConfidenceInterval(positive.logRr, positive.seLogRr, model)
    assert result["logRr"].iloc[0] == pytest.approx(-0.0593, abs=0.1)


def test_calibrateConfidenceInterval_reproduces_r(sccs):
    """Bit-exact against R 4.6.1 EmpiricalCalibration 3.1.4."""
    negatives = sccs[sccs.groundTruth == 0]
    positive = sccs[sccs.groundTruth == 1]
    model = ec.convertNullToErrorModel(ec.fitNull(negatives.logRr, negatives.seLogRr))
    ci = ec.calibrateConfidenceInterval(positive.logRr, positive.seLogRr, model)
    assert ci["logRr"].iloc[0] == -0.0595345149238824
    assert ci["logLb95Rr"].iloc[0] == -0.6335409149390898
    assert ci["logUb95Rr"].iloc[0] == 0.514471885091325
    assert ci["seLogRr"].iloc[0] == 0.2928657896486348


def test_computeTraditionalCi(sccs):
    positive = sccs[sccs.groundTruth == 1]
    ci = ec.computeTraditionalCi(positive.logRr, positive.seLogRr)
    assert list(ci.columns) == ["rr", "lb", "ub"]
    assert (ci["lb"] < ci["rr"]).all() and (ci["rr"] < ci["ub"]).all()


# --- expected systematic error --------------------------------------------

def test_ease_returns_mean_value(sccs):
    negatives = sccs[sccs.groundTruth == 0]
    null = ec.fitNull(negatives.logRr, negatives.seLogRr)
    error = ec.computeExpectedAbsoluteSystematicError(null)
    assert error == pytest.approx(null["mean"], abs=1e-3)


def test_ease_returns_zero_value():
    assert ec.computeExpectedAbsoluteSystematicError(ec.Null(0, 0)) == 0


def test_ease_mcmc_returns_quantiles(sccs):
    alpha = 0.05
    negatives = sccs[sccs.groundTruth == 0]
    ec.set_seed(42)
    null = ec.fitMcmcNull(negatives.logRr, negatives.seLogRr, iter=2000)
    chain = null.mcmc["chain"]
    dist = np.array([closedFormIntegeralAbsolute(r[0], 1 / np.sqrt(r[1])) for r in chain])
    expected = np.quantile(dist, [0.5, alpha / 2, 1 - alpha / 2])
    error = ec.computeExpectedAbsoluteSystematicError(null)
    assert error["ease"].iloc[0] == expected[0]
    assert error["ciLb"].iloc[0] == expected[1]
    assert error["ciUb"].iloc[0] == expected[2]


def test_compareEase_length_check():
    with pytest.raises(ValueError, match="equal length"):
        ec.compareEase([1, 2], [1, 2], [1, 2, 3], [1, 2, 3])


def test_compareEase_detects_added_bias():
    ec.set_seed(777)
    ncs1 = ec.simulateControls(n=30)
    rng = ec.get_generator()
    logRr2 = ncs1.logRr.to_numpy() + rng.rnorm(len(ncs1), mean=0.1, sd=0.1)
    delta = ec.compareEase(ncs1.logRr, ncs1.seLogRr, logRr2, ncs1.seLogRr,
                           sampleSize=100)
    assert list(delta.columns) == ["delta", "ciLb", "ciUb", "p"]
    assert "ease1" in delta.attrs and "ease2" in delta.attrs
    # the second method is more biased, so its EASE should be larger
    assert delta["delta"].iloc[0] < 0


# --- evaluation ------------------------------------------------------------

def test_evaluateCiCalibration_requirements(sccs):
    negatives = sccs[sccs.groundTruth == 0].iloc[:5]

    with pytest.raises(ValueError, match="factor"):
        ec.evaluateCiCalibration([0, 0], [1, 0], [0, 0], strata=["strat1", "strat2"])

    logRr = list(negatives.logRr) + [np.inf]
    seLogRr = list(negatives.seLogRr) + [0]
    trueLogRr = list(negatives.groundTruth) + [0]
    with pytest.warns(UserWarning, match="infinite logRr"):
        ec.evaluateCiCalibration(logRr, seLogRr, trueLogRr)

    logRr = list(negatives.logRr) + [0]
    seLogRr = list(negatives.seLogRr) + [np.inf]
    with pytest.warns(UserWarning, match="infinite standard error"):
        ec.evaluateCiCalibration(logRr, seLogRr, trueLogRr)

    logRr = list(negatives.logRr) + [np.nan]
    seLogRr = list(negatives.seLogRr) + [0]
    with pytest.warns(UserWarning, match="NA logRr"):
        ec.evaluateCiCalibration(logRr, seLogRr, trueLogRr)

    logRr = list(negatives.logRr) + [0]
    seLogRr = list(negatives.seLogRr) + [np.nan]
    with pytest.warns(UserWarning, match="NA standard error"):
        ec.evaluateCiCalibration(logRr, seLogRr, trueLogRr)


def test_evaluateCiCalibration_shape():
    ec.set_seed(303)
    d = ec.simulateControls(n=30, mean=0.25, sd=0.25, trueLogRr=np.log([1, 2, 4]))
    ev = ec.evaluateCiCalibration(d.logRr, d.seLogRr, d.trueLogRr)
    assert set(ev.columns) == {"label", "Confidence interval calculation",
                               "ciWidth", "coverage", "trueRr"}
    # 3 strata x 2 calculations x 3 labels x 99 widths
    assert len(ev) == 3 * 2 * 3 * 99
    assert ((ev["coverage"] >= 0) & (ev["coverage"] <= 1)).all()
