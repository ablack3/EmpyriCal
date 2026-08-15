"""Scalar arguments must work everywhere a single estimate makes sense.

R is vectorised by default and a length-1 vector is indistinguishable from a
scalar, so R code translated function-by-function routinely passes a bare float
where the examples pass a column.  These are regression tests for that path.
"""

import numpy as np
import pandas as pd
import pytest

import empiricalcalibration as ec


@pytest.fixture(scope="module")
def null():
    sccs = ec.datasets.sccs()
    negatives = sccs[sccs.groundTruth == 0]
    return ec.fitNull(negatives.logRr, negatives.seLogRr)


def test_computeTraditionalCi_accepts_a_scalar():
    """Regression: this raised 'If using all scalar values, you must pass an index'."""
    ci = ec.computeTraditionalCi(0.2, 0.1)
    assert isinstance(ci, pd.DataFrame)
    assert len(ci) == 1
    assert ci.rr.iloc[0] == pytest.approx(np.exp(0.2))
    assert ci.lb.iloc[0] < ci.rr.iloc[0] < ci.ub.iloc[0]


def test_computeTraditionalCi_scalar_matches_length_one_sequence():
    scalar = ec.computeTraditionalCi(0.2, 0.1)
    sequence = ec.computeTraditionalCi([0.2], [0.1])
    pd.testing.assert_frame_equal(scalar, sequence)


def test_computeTraditionalCi_honours_ciWidth():
    narrow = ec.computeTraditionalCi(0.2, 0.1, ciWidth=0.80)
    wide = ec.computeTraditionalCi(0.2, 0.1, ciWidth=0.99)
    assert wide.lb.iloc[0] < narrow.lb.iloc[0]
    assert wide.ub.iloc[0] > narrow.ub.iloc[0]


def test_computeTraditionalP_accepts_a_scalar():
    p = ec.computeTraditionalP(0.2, 0.1)
    assert np.isscalar(p) or np.ndim(p) == 0
    assert 0 < float(p) < 1


def test_calibrateP_accepts_scalars(null):
    p = ec.calibrateP(null, 0.2, 0.1)
    assert p.shape == (1,)
    assert 0 <= p[0] <= 1


def test_calibrateConfidenceInterval_accepts_scalars(null):
    model = ec.convertNullToErrorModel(null)
    ci = ec.calibrateConfidenceInterval(0.2, 0.1, model)
    assert len(ci) == 1
    assert set(ci.columns) == {"logRr", "logLb95Rr", "logUb95Rr", "seLogRr"}
    assert ci.logLb95Rr.iloc[0] < ci.logRr.iloc[0] < ci.logUb95Rr.iloc[0]


def test_a_full_single_estimate_report_runs(null):
    """The pattern the documentation recommends, on one estimate."""
    model = ec.convertNullToErrorModel(null)
    logRr, seLogRr = 0.2, 0.1

    trad_ci = ec.computeTraditionalCi(logRr, seLogRr)
    cal_ci = ec.calibrateConfidenceInterval(logRr, seLogRr, model)
    trad_p = ec.computeTraditionalP(logRr, seLogRr)
    cal_p = ec.calibrateP(null, logRr, seLogRr)
    ease = ec.computeExpectedAbsoluteSystematicError(null)

    assert np.isfinite([float(trad_p), float(cal_p[0]), float(ease)]).all()
    # This design is biased upward, so calibration must widen the interval.
    trad_width = np.log(trad_ci.ub.iloc[0]) - np.log(trad_ci.lb.iloc[0])
    cal_width = cal_ci.logUb95Rr.iloc[0] - cal_ci.logLb95Rr.iloc[0]
    assert cal_width > trad_width
    assert float(cal_p[0]) > float(trad_p)


def test_calibrateP_scalar_and_vector_agree(null):
    single = ec.calibrateP(null, 0.2, 0.1)[0]
    batch = ec.calibrateP(null, [0.2, 0.5], [0.1, 0.1])[0]
    assert single == pytest.approx(batch, rel=1e-15)
