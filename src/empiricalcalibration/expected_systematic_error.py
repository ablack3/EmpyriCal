"""Port of ``R/ExpectedSystematicError.R``.

The expected absolute systematic error (EASE) summarises a null distribution in
a single number that reflects both its mean and its spread.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._rmath import dnorm, pnorm
from ._rrng import get_generator
from ._types import McmcNull

__all__ = ["computeExpectedAbsoluteSystematicError", "compareEase",
           "closedFormIntegral", "closedFormIntegeralAbsolute"]


def closedFormIntegral(x, mu, sigma):
    """R's ``closedFormIntegral``."""
    return mu * pnorm(x, mu, sigma) - 1 - sigma ** 2 * dnorm(x, mu, sigma)


def closedFormIntegeralAbsolute(mu, sigma):
    """R's ``closedFormIntegeralAbsolute`` (name keeps R's typo)."""
    return (closedFormIntegral(np.inf, mu, sigma)
            - 2 * closedFormIntegral(0, mu, sigma)
            + closedFormIntegral(-np.inf, mu, sigma))


def computeExpectedAbsoluteSystematicError(null, alpha=0.05):
    """Compute the expected absolute systematic error of a null distribution.

    For a random study estimate, this is the expected value of the absolute
    systematic error.  The expected (signed) systematic error of a null
    distribution equals its mean and is insensitive to its spread; taking the
    absolute value expresses both mean and spread in one number.

    Parameters
    ----------
    null : Null or McmcNull
    alpha : float
        Expected type I error used for the credible interval (``McmcNull`` only).

    Returns
    -------
    float or pandas.DataFrame
        A float for a :class:`Null`; for an :class:`McmcNull` a one-row data
        frame with ``ease``, ``ciLb`` and ``ciUb``.

    See Also
    --------
    compareEase : compare the EASE of two sets of estimates for the same controls.
    """
    if isinstance(null, McmcNull):
        return _ease_mcmcNull(null, alpha)
    return _ease_null(null, alpha)


def _ease_null(null, alpha=0.05):
    null = np.asarray(null, dtype=float)
    if null[0] == 0 and null[1] == 0:
        return 0.0
    return float(closedFormIntegeralAbsolute(null[0], null[1]))


def _ease_mcmcNull(null, alpha=0.05) -> pd.DataFrame:
    if np.isnan(float(np.asarray(null)[0])):
        return pd.DataFrame({"ease": [np.nan], "ciLb": [np.nan], "ciUb": [np.nan]})
    chain = null.mcmc["chain"]
    with np.errstate(divide="ignore", invalid="ignore"):
        dist = np.array([closedFormIntegeralAbsolute(row[0], 1 / np.sqrt(row[1]))
                         for row in chain])
    q = np.quantile(dist, [0.5, alpha / 2, 1 - (alpha / 2)])
    return pd.DataFrame({"ease": [q[0]], "ciLb": [q[1]], "ciUb": [q[2]]})


def _computeEaseBootstrap(logRr, seLogRr, alpha, sampleSize) -> pd.DataFrame:
    """R's ``computeEaseBoostrap`` (spelling as in the original)."""
    from .asymptotics import fitNull

    rng = get_generator()
    logRr = np.asarray(logRr, dtype=float)
    seLogRr = np.asarray(seLogRr, dtype=float)
    n = logRr.size
    sample = np.empty(sampleSize, dtype=float)
    for i in range(sampleSize):
        idx = rng.sample_int(n, n) - 1
        null = fitNull(logRr[idx], seLogRr[idx])
        sample[i] = computeExpectedAbsoluteSystematicError(null)
    q = np.quantile(sample, [0.5, 0.025, 0.975])
    return pd.DataFrame({"ease": [q[0]], "ciLb": [q[1]], "ciUb": [q[2]]})


def compareEase(logRr1, seLogRr1, logRr2, seLogRr2,
                alpha=0.05, sampleSize=1000) -> pd.DataFrame:
    """Compare the EASE of two correlated sets of estimates.

    Compares the expected absolute systematic error of two sets of estimates for
    the *same* set of negative controls.

    Important: the two sets (``logRr1``/``seLogRr1`` and ``logRr2``/``seLogRr2``)
    must be in identical order, so that the i-th entry of each corresponds to
    the same negative control.

    Parameters
    ----------
    logRr1, seLogRr1 : array_like
        Estimates generated with the first method.
    logRr2, seLogRr2 : array_like
        Estimates generated with the second method.
    alpha : float
        Expected type I error for confidence intervals and p-values.
    sampleSize : int
        Number of bootstrap samples.

    Returns
    -------
    pandas.DataFrame
        One row with ``delta``, ``ciLb``, ``ciUb`` and ``p`` -- the difference in
        EASE between the two sets, its confidence interval, and a p-value against
        the null hypothesis that the EASE is the same for both.

        ``.attrs["ease1"]`` and ``.attrs["ease2"]`` hold the bootstrap EASE
        estimates for the two sets.  These may differ somewhat from
        :func:`computeExpectedAbsoluteSystematicError` because the confidence
        interval is computed differently -- here in a way that aligns with the
        computation of the difference.
    """
    from .asymptotics import fitNull

    lengths = {len(np.atleast_1d(x)) for x in (logRr1, seLogRr1, logRr2, seLogRr2)}
    if len(lengths) != 1:
        raise ValueError("Arguments logRr1, seLogRr1, logRr2, and seLogRr2 "
                         "should be of equal length.")

    logRr1 = np.asarray(logRr1, dtype=float)
    seLogRr1 = np.asarray(seLogRr1, dtype=float)
    logRr2 = np.asarray(logRr2, dtype=float)
    seLogRr2 = np.asarray(seLogRr2, dtype=float)

    ease1 = _computeEaseBootstrap(logRr1, seLogRr1, alpha, sampleSize)
    ease2 = _computeEaseBootstrap(logRr2, seLogRr2, alpha, sampleSize)
    delta = float(ease1["ease"].iloc[0] - ease2["ease"].iloc[0])

    rng = get_generator()
    n = logRr1.size
    sample = np.empty(sampleSize, dtype=float)
    for i in range(sampleSize):
        idx = rng.sample_int(n, n) - 1
        null1 = fitNull(logRr1[idx], seLogRr1[idx])
        e1 = computeExpectedAbsoluteSystematicError(null1)
        null2 = fitNull(logRr2[idx], seLogRr2[idx])
        e2 = computeExpectedAbsoluteSystematicError(null2)
        sample[i] = e1 - e2

    deltaCi = np.quantile(sample, [0.025, 0.975])
    epsilon = 0.0001
    if delta > 0:
        deltaP = float(np.mean(sample < epsilon))
    else:
        deltaP = float(np.mean(sample > -epsilon))

    out = pd.DataFrame({"delta": [delta], "ciLb": [deltaCi[0]],
                        "ciUb": [deltaCi[1]], "p": [deltaP]})
    out.attrs["ease1"] = ease1
    out.attrs["ease2"] = ease2
    return out
