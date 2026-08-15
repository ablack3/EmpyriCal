"""Port of ``R/Evaluation.R``: leave-one-out evaluation of confidence interval
calibration."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ._rmath import qnorm
from .confidence_intervals import calibrateConfidenceInterval, fitSystematicErrorModel

__all__ = ["evaluateCiCalibration"]


def evaluateCiCalibration(logRr, seLogRr, trueLogRr, strata=None,
                          crossValidationGroup=None, legacy=False) -> pd.DataFrame:
    """Evaluate confidence interval calibration by leave-one-out cross-validation.

    The confidence interval of an effect is computed by fitting a systematic
    error model on all *other* controls.

    Parameters
    ----------
    logRr, seLogRr : array_like
        Effect estimates on the log scale and their standard errors.
    trueLogRr : array_like
        The true log relative risk.
    strata : array_like, optional
        Variable used to stratify the plot.  Defaults to ``trueLogRr``; pass an
        explicit value to override, or an array of a single constant for no
        stratification.
    crossValidationGroup : array_like, optional
        Unit of cross-validation.  By default each control is its own unit, but
        a different grouping can be provided, for example linking a negative
        control to synthetic positive controls derived from it.
    legacy : bool
        Fit the legacy error model (standard deviation linear on the log scale).

    Returns
    -------
    pandas.DataFrame
        Coverage per stratum for a wide range of confidence interval widths,
        including the fractions below and above the interval.  Columns:
        ``label``, ``Confidence interval calculation``, ``ciWidth``,
        ``coverage`` and ``trueRr``.
    """
    logRr = np.atleast_1d(np.asarray(logRr, dtype=float))
    seLogRr = np.atleast_1d(np.asarray(seLogRr, dtype=float))
    trueLogRr = np.atleast_1d(np.asarray(trueLogRr, dtype=float))

    if strata is None:
        strata = trueLogRr.copy()
    else:
        strata = np.atleast_1d(np.asarray(strata))
        if strata.dtype.kind not in "biufc":
            raise ValueError("Strata argument should be a factor (or null)")
    if crossValidationGroup is None:
        crossValidationGroup = np.arange(1, logRr.size + 1)
    crossValidationGroup = np.atleast_1d(np.asarray(crossValidationGroup))

    data = pd.DataFrame({
        "logRr": logRr,
        "seLogRr": seLogRr,
        "trueLogRr": trueLogRr,
        "strata": strata,
        "crossValidationGroup": crossValidationGroup,
    })

    if np.any(np.isinf(data["seLogRr"])):
        warnings.warn("Estimate(s) with infinite standard error detected. "
                      "Removing before fitting error model")
        data = data[~np.isinf(data["seLogRr"])]
    if np.any(np.isinf(data["logRr"])):
        warnings.warn("Estimate(s) with infinite logRr detected. "
                      "Removing before fitting error model")
        data = data[~np.isinf(data["logRr"])]
    if data["seLogRr"].isna().any():
        warnings.warn("Estimate(s) with NA standard error detected. "
                      "Removing before fitting error model")
        data = data[~data["seLogRr"].isna()]
    if data["logRr"].isna().any():
        warnings.warn("Estimate(s) with NA logRr detected. "
                      "Removing before fitting error model")
        data = data[~data["logRr"].isna()]

    ciWidths = np.round(np.arange(0.01, 0.995, 0.01), 10)

    def computeCoverage(subset, ciWidth, model):
        if len(subset) == 0:
            return np.array([0.0, 0.0, 0.0])
        ci = calibrateConfidenceInterval(
            logRr=subset["logRr"].to_numpy(),
            seLogRr=subset["seLogRr"].to_numpy(),
            ciWidth=ciWidth, model=model)
        lb = ci["logLb95Rr"].fillna(0).to_numpy()
        ub = ci["logUb95Rr"].fillna(999).to_numpy()
        true = subset["trueLogRr"].to_numpy()
        return np.array([float(np.sum(true < lb)),
                         float(np.sum((lb <= true) & (ub >= true))),
                         float(np.sum(true > ub))])

    def computeTheoreticalCoverage(subset, ciWidth):
        true = subset["trueLogRr"].to_numpy()
        lb = subset["logRr"].to_numpy() + qnorm((1 - ciWidth) / 2) * subset["seLogRr"].to_numpy()
        ub = subset["logRr"].to_numpy() - qnorm((1 - ciWidth) / 2) * subset["seLogRr"].to_numpy()
        return np.array([float(np.sum(true < lb)),
                         float(np.sum((true >= lb) & (true <= ub))),
                         float(np.sum(true > ub))])

    print("Fitting error models within leave-one-out cross-validation")
    rows = []
    for leaveOutGroup in pd.unique(data["crossValidationGroup"]):
        dataLeaveOneOut = data[data["crossValidationGroup"] != leaveOutGroup]
        dataLeftOut = data[data["crossValidationGroup"] == leaveOutGroup]
        if len(dataLeaveOneOut) == 0 or len(dataLeftOut) == 0:
            continue
        model = fitSystematicErrorModel(
            logRr=dataLeaveOneOut["logRr"].to_numpy(),
            seLogRr=dataLeaveOneOut["seLogRr"].to_numpy(),
            trueLogRr=dataLeaveOneOut["trueLogRr"].to_numpy(),
            estimateCovarianceMatrix=False, legacy=legacy)
        for stratum in pd.unique(dataLeftOut["strata"]):
            subset = dataLeftOut[dataLeftOut["strata"] == stratum]
            for ciWidth in ciWidths:
                below, within, above = computeCoverage(subset, ciWidth, model)
                tb, tw, ta = computeTheoreticalCoverage(subset, ciWidth)
                rows.append((stratum, ciWidth, below, within, above, tb, tw, ta))

    coverage = pd.DataFrame(rows, columns=[
        "strata", "ciWidth", "below", "within", "above",
        "theoreticalBelow", "theoreticalWithin", "theoreticalAbove"])

    counts = data.groupby("strata", sort=True).size().rename("count").reset_index()

    pieces = []
    spec = [("below", "Below confidence interval", "Calibrated"),
            ("within", "Within confidence interval", "Calibrated"),
            ("above", "Above confidence interval", "Calibrated"),
            ("theoreticalBelow", "Below confidence interval", "Uncalibrated"),
            ("theoreticalWithin", "Within confidence interval", "Uncalibrated"),
            ("theoreticalAbove", "Above confidence interval", "Uncalibrated")]
    for col, label, type_ in spec:
        agg = coverage.groupby(["strata", "ciWidth"], as_index=False)[col].sum()
        agg = agg.merge(counts, on="strata")
        agg["coverage"] = agg[col] / agg["count"]
        agg["label"] = label
        agg["type"] = type_
        pieces.append(agg[["strata", "label", "type", "ciWidth", "coverage"]])

    vizData = pd.concat(pieces, ignore_index=True)
    vizData = vizData.rename(columns={"type": "Confidence interval calculation"})
    vizData["trueRr"] = np.exp(vizData["strata"].astype(float))
    vizData = vizData.drop(columns=["strata"])
    return vizData
