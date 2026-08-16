"""Evaluation options and the two simulators."""

import numpy as np
import pandas as pd
import pytest

import empiricalcalibration as ec

# --- evaluateCiCalibration options ----------------------------------------

@pytest.fixture(scope="module")
def controls():
    """A stratified subsample of the Southworth controls.

    `evaluateCiCalibration` refits the error model once per control, so the cost
    is quadratic in the control count; 12 per true effect size keeps these tests
    quick while retaining all three strata.
    """
    df = ec.datasets.southworthReplication()
    df = df[df.trueLogRr.notna()]
    df = df[df.trueLogRr.isin(sorted(df.trueLogRr.unique())[:3])]
    return (df.groupby("trueLogRr", group_keys=False)
              .head(12)
              .reset_index(drop=True))


def test_evaluation_covers_both_calculations_and_all_widths(controls):
    ec.set_seed(1)
    ev = ec.evaluateCiCalibration(controls.logRr, controls.seLogRr, controls.trueLogRr)

    assert set(ev["Confidence interval calculation"]) == {"Uncalibrated", "Calibrated"}
    assert set(ev["label"]) == {"Below confidence interval",
                                "Within confidence interval",
                                "Above confidence interval"}
    widths = np.sort(ev.ciWidth.unique())
    assert len(widths) == 99
    assert widths[0] == pytest.approx(0.01)
    assert widths[-1] == pytest.approx(0.99)


def test_calibration_improves_coverage_at_95(controls):
    """The central claim: calibrated intervals are closer to nominal coverage."""
    ec.set_seed(2)
    ev = ec.evaluateCiCalibration(controls.logRr, controls.seLogRr, controls.trueLogRr)
    within = ev[(ev.label == "Within confidence interval")
                & np.isclose(ev.ciWidth, 0.95)]

    uncal = within[within["Confidence interval calculation"] == "Uncalibrated"]
    cal = within[within["Confidence interval calculation"] == "Calibrated"]

    assert abs(cal.coverage.mean() - 0.95) < abs(uncal.coverage.mean() - 0.95)


def test_explicit_strata_are_honoured(controls):
    ec.set_seed(3)
    single = np.zeros(len(controls))
    ev = ec.evaluateCiCalibration(controls.logRr, controls.seLogRr, controls.trueLogRr,
                                  strata=single)
    assert ev.trueRr.nunique() == 1


def test_crossValidationGroup_changes_the_result(controls):
    """Grouping controls together must actually alter the leave-one-out fits."""
    ec.set_seed(4)
    default = ec.evaluateCiCalibration(controls.logRr, controls.seLogRr,
                                       controls.trueLogRr)
    # Put every control at the same true effect size into one CV group.
    ec.set_seed(4)
    grouped = ec.evaluateCiCalibration(controls.logRr, controls.seLogRr,
                                       controls.trueLogRr,
                                       crossValidationGroup=controls.trueLogRr)
    assert not np.allclose(default.coverage.to_numpy(), grouped.coverage.to_numpy())


def test_legacy_model_runs_end_to_end(controls):
    ec.set_seed(5)
    ev = ec.evaluateCiCalibration(controls.logRr, controls.seLogRr, controls.trueLogRr,
                                  legacy=True)
    assert ((ev.coverage >= 0) & (ev.coverage <= 1)).all()


# --- simulateControls ------------------------------------------------------

def test_simulateControls_shape_and_columns():
    ec.set_seed(10)
    d = ec.simulateControls(n=40, mean=0.2, sd=0.15, trueLogRr=0)
    assert len(d) == 40
    assert {"logRr", "seLogRr", "trueLogRr"} <= set(d.columns)
    assert (d.seLogRr > 0).all()
    assert (d.trueLogRr == 0).all()


def test_simulateControls_cycles_multiple_true_effect_sizes():
    ec.set_seed(11)
    trueLogRr = np.log([1, 2, 4])
    d = ec.simulateControls(n=90, mean=0.1, sd=0.1, trueLogRr=trueLogRr)
    assert len(d) == 90
    assert sorted(d.trueLogRr.unique()) == pytest.approx(sorted(trueLogRr))
    # each true effect used equally often
    assert d.trueLogRr.value_counts().nunique() == 1


def test_simulateControls_recovers_the_null_it_was_given():
    """fitNull must recover the *between-control* SD, net of each control's own
    standard error.

    The tolerance has to be tight enough to exclude the marginal SD
    sqrt(sd**2 + seLogRr**2) -- what you would get if seLogRr were never netted
    out.  With sd=0.2 and seLogRr=0.35 the two differ by 0.20, so a conflation
    of the two can no longer pass.
    """
    ec.set_seed(12)
    sd, se = 0.2, 0.35
    d = ec.simulateControls(n=4000, mean=0.3, sd=sd, trueLogRr=0, seLogRr=se)
    null = ec.fitNull(d.logRr, d.seLogRr)

    marginal = np.sqrt(sd ** 2 + se ** 2)
    assert null[0] == pytest.approx(0.3, abs=0.03)
    assert null[1] == pytest.approx(sd, abs=0.03)
    assert abs(null[1] - marginal) > 0.1, "tolerance must exclude the marginal SD"


def test_simulateControls_fixed_seLogRr_is_used_exactly():
    ec.set_seed(13)
    d = ec.simulateControls(n=25, mean=0.0, sd=0.1, trueLogRr=0, seLogRr=0.123)
    assert np.allclose(d.seLogRr, 0.123)


def test_simulateControls_is_reproducible_under_set_seed():
    ec.set_seed(14)
    a = ec.simulateControls(n=20, mean=0.2, sd=0.2, trueLogRr=0)
    ec.set_seed(14)
    b = ec.simulateControls(n=20, mean=0.2, sd=0.2, trueLogRr=0)
    pd.testing.assert_frame_equal(a, b)


# --- simulateMaxSprtData ---------------------------------------------------

def test_simulateMaxSprtData_look_and_outcome_structure():
    ec.set_seed(20)
    data = ec.simulateMaxSprtData(n=500, numberOfNegativeControls=4,
                                  numberOfPositiveControls=2, looks=5)
    assert set(data.columns) == {"time", "outcome", "exposure", "lookTime", "outcomeId"}
    assert data.lookTime.nunique() == 5
    assert sorted(data.outcomeId.unique()) == [1, 2, 3, 4, 5, 6]
    assert (data.time > 0).all()
    assert set(data.exposure.unique()) <= {0, 1}
    assert set(data.outcome.unique()) <= {0, 1}


def test_simulateMaxSprtData_accrues_over_looks():
    """Each look must see at least as much data as the one before."""
    ec.set_seed(21)
    data = ec.simulateMaxSprtData(n=2000, numberOfNegativeControls=2,
                                  numberOfPositiveControls=1, looks=5)
    counts = data.groupby("lookTime").size()
    assert list(counts) == sorted(counts)


def test_simulateMaxSprtData_positive_control_has_the_larger_effect():
    ec.set_seed(22)
    data = ec.simulateMaxSprtData(n=20000, numberOfNegativeControls=5,
                                  numberOfPositiveControls=1,
                                  positiveControlEffectSize=4)
    last = data[data.lookTime == data.lookTime.max()]

    def rate_ratio(g):
        exposed, unexposed = g[g.exposure == 1], g[g.exposure == 0]
        a, b = exposed.outcome.sum(), unexposed.outcome.sum()
        if a == 0 or b == 0:
            return np.nan
        return (a / exposed.time.sum()) / (b / unexposed.time.sum())

    # Build the per-outcome ratios with an explicit loop rather than
    # groupby().apply(include_groups=...): that keyword only exists from pandas
    # 2.2, while this package declares pandas>=1.4.
    rrs = pd.Series({oid: rate_ratio(g) for oid, g in last.groupby("outcomeId")})
    positive_id = last.outcomeId.max()          # positives get the higher IDs
    negatives = rrs.drop(index=positive_id).dropna()
    assert rrs[positive_id] > negatives.median()


def test_simulateMaxSprtData_is_reproducible_under_set_seed():
    ec.set_seed(23)
    a = ec.simulateMaxSprtData(n=300, numberOfNegativeControls=2,
                               numberOfPositiveControls=1, looks=3)
    ec.set_seed(23)
    b = ec.simulateMaxSprtData(n=300, numberOfNegativeControls=2,
                               numberOfPositiveControls=1, looks=3)
    pd.testing.assert_frame_equal(a, b)
