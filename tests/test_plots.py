"""Port of ``tests/testthat/test-plots.R``.

R checks that each function returns a ggplot object; here the equivalent check is
that it returns a matplotlib Figure.
"""

import os

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
from matplotlib.figure import Figure  # noqa: E402

import empiricalcalibration as ec  # noqa: E402
from empiricalcalibration import datasets  # noqa: E402


@pytest.fixture(scope="module")
def data():
    sccs = datasets.sccs()
    negatives = sccs[sccs.groundTruth == 0]
    positives = sccs[sccs.groundTruth == 1]
    null = ec.fitNull(negatives.logRr, negatives.seLogRr)
    ec.set_seed(1)
    mcmcNull = ec.fitMcmcNull(negatives.logRr, negatives.seLogRr, iter=1000)
    return negatives, positives, null, mcmcNull


def test_plotMcmcTrace(data):
    _, _, _, mcmcNull = data
    assert isinstance(ec.plotMcmcTrace(mcmcNull=mcmcNull), Figure)


def test_plotForest(data):
    negatives, _, _, _ = data
    assert isinstance(ec.plotForest(negatives.logRr, negatives.seLogRr,
                                    negatives.drugName), Figure)


def test_plotCalibration_warnings(data):
    negatives, _, _, _ = data
    lr, se = list(negatives.logRr), list(negatives.seLogRr)
    with pytest.warns(UserWarning, match="infinite logRr"):
        ec.plotCalibration(lr + [np.inf], se + [0])
    with pytest.warns(UserWarning, match="infinite standard error"):
        ec.plotCalibration(lr + [0], se + [np.inf])
    with pytest.warns(UserWarning, match="NA logRr"):
        ec.plotCalibration(lr + [np.nan], se + [0])
    with pytest.warns(UserWarning, match="NA standard error"):
        ec.plotCalibration(lr + [0], se + [np.nan])
    assert isinstance(ec.plotCalibration(negatives.logRr, negatives.seLogRr), Figure)


def test_plotCalibrationEffect(data):
    negatives, positives, null, _ = data
    with pytest.raises(ValueError, match="fitMcmcNull"):
        ec.plotCalibrationEffect(negatives.logRr, negatives.seLogRr,
                                 positives.logRr, positives.seLogRr,
                                 showCis=True, null=null)

    for kwargs in [
        dict(logRrPositives=positives.logRr, seLogRrPositives=positives.seLogRr),
        dict(logRrPositives=positives.logRr, seLogRrPositives=positives.seLogRr,
             showExpectedAbsoluteSystematicError=True),
        dict(logRrPositives=positives.logRr, seLogRrPositives=positives.seLogRr,
             null=null),
        dict(null=null),
    ]:
        assert isinstance(ec.plotCalibrationEffect(
            negatives.logRr, negatives.seLogRr, **kwargs), Figure)


def test_plotCalibrationEffect_limit_warnings(data):
    negatives, _, _, _ = data
    with pytest.warns(UserWarning, match="xLimits"):
        ec.plotCalibrationEffect(negatives.logRr, negatives.seLogRr,
                                 logRrPositives=[-3, -2, 11],
                                 seLogRrPositives=[0.1, 1.2, 2.5])
    with pytest.warns(UserWarning, match="yLimits"):
        ec.plotCalibrationEffect(negatives.logRr, negatives.seLogRr,
                                 logRrPositives=[-3, -2, 2],
                                 seLogRrPositives=[0.1, 1.2, 2.5])


def test_plotCiCalibration(data):
    negatives, _, _, _ = data
    assert isinstance(ec.plotCiCalibration(negatives.logRr, negatives.seLogRr,
                                           negatives.groundTruth), Figure)


def test_plotCiCalibrationEffect(data):
    negatives, _, _, _ = data
    assert isinstance(ec.plotCiCalibrationEffect(negatives.logRr, negatives.seLogRr,
                                                 negatives.groundTruth), Figure)


def test_plotCiCoverage(data):
    negatives, _, _, _ = data
    assert isinstance(ec.plotCiCoverage(negatives.logRr, negatives.seLogRr,
                                        negatives.groundTruth), Figure)


def test_plotErrorModel(data):
    negatives, _, _, _ = data
    assert isinstance(ec.plotErrorModel(negatives.logRr, negatives.seLogRr,
                                        negatives.groundTruth), Figure)


def test_plotExpectedType1Error(data):
    negatives, positives, null, _ = data
    with pytest.raises(ValueError, match="fitMcmcNull"):
        ec.plotExpectedType1Error(negatives.logRr, negatives.seLogRr,
                                  positives.seLogRr, showCis=True, null=null)
    assert isinstance(ec.plotExpectedType1Error(
        negatives.logRr, negatives.seLogRr, positives.seLogRr), Figure)
    assert isinstance(ec.plotExpectedType1Error(
        negatives.logRr, negatives.seLogRr, positives.seLogRr,
        showEffectSizes=True), Figure)


def test_plotTrueAndObserved_saves_file(tmp_path):
    ec.set_seed(123)
    data = ec.simulateControls(n=50 * 3, mean=0.25, sd=0.25,
                               trueLogRr=np.log([1, 2, 4]))
    fileName = tmp_path / "output.png"
    plot = ec.plotTrueAndObserved(data.logRr, data.seLogRr, data.trueLogRr,
                                  fileName=str(fileName))
    assert os.path.exists(fileName)
    assert plot is not None
