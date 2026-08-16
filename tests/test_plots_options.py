"""Plot options not exercised by the port of the R testthat suite:
saving, titles, legend placement, and the MCMC credible-interval paths."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
from matplotlib.figure import Figure  # noqa: E402

import empiricalcalibration as ec  # noqa: E402


@pytest.fixture(scope="module")
def fitted():
    sccs = ec.datasets.sccs()
    negatives = sccs[sccs.groundTruth == 0]
    positives = sccs[sccs.groundTruth == 1]
    null = ec.fitNull(negatives.logRr, negatives.seLogRr)
    ec.set_seed(1)
    mcmcNull = ec.fitMcmcNull(negatives.logRr, negatives.seLogRr, iter=1000)
    return negatives, positives, null, mcmcNull


# --- saving ----------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "plotForest", "plotCalibration", "plotCalibrationEffect",
    "plotExpectedType1Error",
])
def test_fileName_writes_a_png(fitted, tmp_path, name):
    negatives, positives, _, _ = fitted
    out = tmp_path / f"{name}.png"
    fn = getattr(ec, name)

    if name == "plotForest":
        fig = fn(negatives.logRr, negatives.seLogRr, negatives.drugName,
                 fileName=str(out))
    elif name == "plotExpectedType1Error":
        fig = fn(negatives.logRr, negatives.seLogRr, positives.seLogRr,
                 fileName=str(out))
    elif name == "plotCalibrationEffect":
        fig = fn(negatives.logRr, negatives.seLogRr,
                 positives.logRr, positives.seLogRr, fileName=str(out))
    else:
        fig = fn(negatives.logRr, negatives.seLogRr, fileName=str(out))

    assert isinstance(fig, Figure)
    assert out.exists()
    assert out.stat().st_size > 1000


def test_no_fileName_writes_nothing(fitted, tmp_path, monkeypatch):
    """Omitting fileName must not write a file anywhere.

    The check only means something if the process CWD is the directory being
    inspected -- a relative default filename would land there.
    """
    negatives, _, _, _ = fitted
    monkeypatch.chdir(tmp_path)
    ec.plotCalibration(negatives.logRr, negatives.seLogRr)
    assert list(tmp_path.iterdir()) == []


# --- titles ----------------------------------------------------------------

@pytest.mark.parametrize("name", ["plotForest", "plotCalibration",
                                  "plotCalibrationEffect"])
def test_title_is_accepted(fitted, name):
    negatives, _, _, _ = fitted
    fn = getattr(ec, name)
    if name == "plotForest":
        fig = fn(negatives.logRr, negatives.seLogRr, negatives.drugName,
                 title="a title")
    else:
        fig = fn(negatives.logRr, negatives.seLogRr, title="a title")
    assert isinstance(fig, Figure)


# --- legend placement ------------------------------------------------------

@pytest.mark.parametrize("position", ["right", "left", "top", "none"])
def test_legendPosition_options(fitted, position):
    negatives, _, _, _ = fitted
    fig = ec.plotCalibration(negatives.logRr, negatives.seLogRr,
                             legendPosition=position)
    assert isinstance(fig, Figure)


# --- MCMC paths ------------------------------------------------------------

def test_plotCalibrationEffect_with_credible_intervals(fitted):
    negatives, positives, _, mcmcNull = fitted
    fig = ec.plotCalibrationEffect(negatives.logRr, negatives.seLogRr,
                                   positives.logRr, positives.seLogRr,
                                   null=mcmcNull, showCis=True)
    assert isinstance(fig, Figure)


def test_plotExpectedType1Error_with_credible_intervals(fitted):
    negatives, positives, _, mcmcNull = fitted
    fig = ec.plotExpectedType1Error(negatives.logRr, negatives.seLogRr,
                                    positives.seLogRr, null=mcmcNull,
                                    showCis=True)
    assert isinstance(fig, Figure)


@pytest.mark.slow
def test_plotCalibration_useMcmc(fitted):
    """useMcmc runs a full 100,000-iteration MCMC fit *per* leave-one-out
    control, so the cost is quadratic-ish in the control count.  Six controls
    keeps it to about a minute; the full 45 would take twenty."""
    negatives, _, _, _ = fitted
    subset = negatives.head(6)
    ec.set_seed(2)
    fig = ec.plotCalibration(subset.logRr, subset.seLogRr, useMcmc=True)
    assert isinstance(fig, Figure)


def test_showCis_requires_an_mcmc_null(fitted):
    negatives, positives, null, _ = fitted
    with pytest.raises(ValueError, match="fitMcmcNull"):
        ec.plotCalibrationEffect(negatives.logRr, negatives.seLogRr,
                                 positives.logRr, positives.seLogRr,
                                 null=null, showCis=True)


# --- axis limits -----------------------------------------------------------

def test_explicit_limits_that_contain_the_data_do_not_warn(fitted):
    negatives, _, _, _ = fitted
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fig = ec.plotCalibrationEffect(negatives.logRr, negatives.seLogRr,
                                       xLimits=(0.1, 10), yLimits=(0, 1.5))
    assert isinstance(fig, Figure)


# --- forest plot input handling -------------------------------------------

def test_plotForest_accepts_plain_lists():
    fig = ec.plotForest([0.1, 0.3, -0.2], [0.1, 0.2, 0.15],
                        ["drug A", "drug B", "drug C"])
    assert isinstance(fig, Figure)


def test_plotTrueAndObserved_runs(fitted):
    ec.set_seed(3)
    d = ec.simulateControls(n=60, mean=0.2, sd=0.2, trueLogRr=np.log([1, 2, 4]))
    fig = ec.plotTrueAndObserved(d.logRr, d.seLogRr, d.trueLogRr)
    assert isinstance(fig, Figure)
