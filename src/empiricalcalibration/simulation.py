"""Port of ``R/Simulation.R``."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._rrng import get_generator

__all__ = ["simulateControls", "simulateMaxSprtData"]


def simulateControls(n=50, mean=0.0, sd=0.1, seLogRr=None, trueLogRr=0.0) -> pd.DataFrame:
    """Simulate (negative) controls.

    Generates point estimates given known true effect sizes and standard errors.

    Parameters
    ----------
    n : int
        Number of controls to simulate.
    mean : float
        Mean of the error distribution (on the log RR scale).
    sd : float
        Standard deviation of the error distribution (on the log RR scale).
    seLogRr : array_like, optional
        Standard error of the log relative risk, recycled across the controls.
        The default samples these from ``Uniform(0.01, 0.2)``.
    trueLogRr : array_like
        The true relative risk (log scale) used to generate the controls,
        recycled across the controls.

    Returns
    -------
    pandas.DataFrame
        Columns ``logRr``, ``seLogRr`` and ``trueLogRr``.
    """
    rng = get_generator()
    trueLogRr_full = np.resize(np.atleast_1d(np.asarray(trueLogRr, dtype=float)), n)

    # R's `seLogRr` default is a promise: `runif(n, 0.01, 0.2)` is not evaluated
    # until seLogRr is first used, which is when `rnorm` below matches its `sd`
    # argument -- i.e. *after* theta has been drawn.  Preserving that order is
    # what makes the stream match R's.
    theta = rng.rnorm(n, mean=mean, sd=sd)
    if seLogRr is None:
        seLogRr = rng.runif(n, 0.01, 0.2)
    seLogRr = np.resize(np.atleast_1d(np.asarray(seLogRr, dtype=float)), n)
    logRr = rng.rnorm(n, mean=trueLogRr_full + theta, sd=seLogRr)
    return pd.DataFrame({"logRr": logRr, "seLogRr": seLogRr,
                         "trueLogRr": trueLogRr_full})


def simulateMaxSprtData(n=10000, pExposure=0.5, backgroundHazard=0.001, tar=10,
                        nullMu=0.2, nullSigma=0.2, maxT=100, looks=10,
                        numberOfNegativeControls=50, numberOfPositiveControls=1,
                        positiveControlEffectSize=4) -> pd.DataFrame:
    """Simulate survival data for MaxSPRT computation.

    Simulates data for negative and positive controls, providing multiple looks
    at data accruing over time, each look having more data than the one before.
    Systematic error for each outcome is drawn from the prespecified null
    distribution.

    Outcome IDs are assigned sequentially starting at 1, with the lower IDs used
    for the negative controls and the higher IDs for the positive controls.

    Parameters
    ----------
    n : int
        Number of subjects.
    pExposure : float
        Probability of being in the target cohort.
    backgroundHazard : float
        Background hazard (risk of the outcome per day).
    tar : float
        Time at risk for each exposure.
    nullMu, nullSigma : float
        Null distribution mean and SD (on the log HR scale).
    maxT : float
        Maximum time to simulate.
    looks : int
        Number of (evenly spaced) looks at the data.
    numberOfNegativeControls, numberOfPositiveControls : int
        How many of each to simulate.
    positiveControlEffectSize : float
        True effect size of the positive controls.

    Returns
    -------
    pandas.DataFrame
        Columns ``time`` (time from index date to the event or end of
        observation, whichever came first), ``outcome`` (1/0), ``exposure``
        (True/False), ``lookTime`` (when the look occurred) and ``outcomeId``.
    """
    rng = get_generator()

    def simulateOutcome(trueEffectSize):
        tIndex = rng.runif(n, 0, maxT)
        exposure = rng.runif(n) < pExposure
        systematicError = float(rng.rnorm(1, mean=nullMu, sd=nullSigma)[0])
        hazard = np.where(exposure,
                          backgroundHazard * trueEffectSize * np.exp(systematicError),
                          backgroundHazard)
        tOutcome = rng.rexp(n, hazard)
        outcome = tOutcome < tar
        time = np.full(n, float(tar))
        time[outcome] = tOutcome[outcome]

        t_looks = np.linspace(0, maxT, looks + 1)[1:]
        frames = []
        for t in t_looks:
            truncatedTime = time.copy()
            idxTruncated = tIndex + time > t
            truncatedTime[idxTruncated] = t - tIndex[idxTruncated]
            truncatedOutcome = outcome.astype(int).copy()
            truncatedOutcome[idxTruncated] = 0
            data = pd.DataFrame({"time": truncatedTime,
                                 "outcome": truncatedOutcome,
                                 "exposure": exposure})
            data = data[data["time"] > 0].copy()
            data["lookTime"] = t
            frames.append(data)
        return pd.concat(frames, ignore_index=True)

    dataSets = []
    for i in range(1, numberOfNegativeControls + 1):
        ds = simulateOutcome(1)
        ds["outcomeId"] = i
        dataSets.append(ds)
    for i in range(numberOfNegativeControls + 1,
                   numberOfNegativeControls + numberOfPositiveControls + 1):
        ds = simulateOutcome(positiveControlEffectSize)
        ds["outcomeId"] = i
        dataSets.append(ds)

    return pd.concat(dataSets, ignore_index=True)
