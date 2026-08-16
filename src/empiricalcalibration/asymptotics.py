"""Port of ``R/EmpiricalCalibrationUsingAsymptotics.R``.

Implements the method of Schuemie et al., *Interpreting observational studies:
why empirical calibration is needed to correct p-values*, Statistics in Medicine
33(2):209-18, 2014.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ._rmath import pnorm
from ._roptim import optim
from ._types import Null
from .likelihoods import (
    customLlApproximation,
    gridLlApproximation,
    logLikelihoodNull,
    logLikelihoodNullNonNormalLl,
    skewNormalLlApproximation,
)

__all__ = ["fitNull", "calibrateP", "computeTraditionalP", "fitNullNonNormalLl"]


def _clean_estimates(logRr, seLogRr, what="null distribution", dropExtreme=True):
    """The validity filters R applies before fitting, with the same warnings.

    ``dropExtreme`` selects between R's two variants.  ``fitNull`` and
    ``fitMcmcNull`` apply five filters, the fifth dropping ``abs(logRr) >
    log(100)``.  R's ``plotCalibration`` inlines only the first four, so it
    plots extreme estimates rather than silently discarding them; pass
    ``dropExtreme=False`` to match it.
    """
    logRr = np.atleast_1d(np.asarray(logRr, dtype=float)).copy()
    seLogRr = np.atleast_1d(np.asarray(seLogRr, dtype=float)).copy()
    if logRr.size != seLogRr.size:
        logRr, seLogRr = np.broadcast_arrays(logRr, seLogRr)
        logRr, seLogRr = logRr.copy(), seLogRr.copy()

    if np.any(np.isinf(seLogRr)):
        warnings.warn("Estimate(s) with infinite standard error detected. "
                      f"Removing before fitting {what}", stacklevel=3)
        keep = ~np.isinf(seLogRr)
        logRr, seLogRr = logRr[keep], seLogRr[keep]
    if np.any(np.isinf(logRr)):
        warnings.warn("Estimate(s) with infinite logRr detected. "
                      f"Removing before fitting {what}", stacklevel=3)
        keep = ~np.isinf(logRr)
        logRr, seLogRr = logRr[keep], seLogRr[keep]
    if np.any(np.isnan(seLogRr)):
        warnings.warn("Estimate(s) with NA standard error detected. "
                      f"Removing before fitting {what}", stacklevel=3)
        keep = ~np.isnan(seLogRr)
        logRr, seLogRr = logRr[keep], seLogRr[keep]
    if np.any(np.isnan(logRr)):
        warnings.warn("Estimate(s) with NA logRr detected. "
                      f"Removing before fitting {what}", stacklevel=3)
        keep = ~np.isnan(logRr)
        logRr, seLogRr = logRr[keep], seLogRr[keep]
    if dropExtreme and np.any(np.abs(logRr) > np.log(100)):
        warnings.warn("Estimate(s) with extreme logRr detected: abs(logRr) > log(100). "
                      f"Removing before fitting {what}", stacklevel=3)
        keep = np.abs(logRr) <= np.log(100)
        logRr, seLogRr = logRr[keep], seLogRr[keep]
    return logRr, seLogRr


def fitNull(logRr, seLogRr) -> Null:
    """Fit the null distribution to a set of negative controls.

    Fits a Gaussian to the negative control estimates as described in Schuemie
    et al. (2014).

    Parameters
    ----------
    logRr : array_like
        Effect estimates on the log scale.
    seLogRr : array_like
        Standard error of the log of the effect estimates.  Hint: often the
        standard error = ``(log(lower bound 95% CI) - log(effect estimate)) /
        qnorm(0.025)``.

    Returns
    -------
    Null
        The parameters (``mean``, ``sd``) of the null distribution.

    Examples
    --------
    >>> from empiricalcalibration import datasets, fitNull
    >>> sccs = datasets.sccs()
    >>> negatives = sccs[sccs.groundTruth == 0]
    >>> null = fitNull(negatives.logRr, negatives.seLogRr)
    """
    logRr, seLogRr = _clean_estimates(logRr, seLogRr)
    if logRr.size == 0:
        warnings.warn("No estimates remaining", stacklevel=2)
        return Null(np.nan, np.nan)

    theta = np.array([0.0, 1.0])
    fit = optim(theta, lambda t: logLikelihoodNull(t, logRr, seLogRr))
    par = fit["par"]
    return Null(par[0], 1.0 / np.sqrt(par[1]))


def computeTraditionalP(logRr, seLogRr, twoSided=True, upper=True):
    """Compute the traditional p-value from the log RR and its standard error.

    Parameters
    ----------
    logRr, seLogRr : array_like
        Effect estimates on the log scale and their standard errors.
    twoSided : bool
        Compute a two-sided (``True``) or one-sided (``False``) p-value.
    upper : bool
        If one-sided, compute the p-value for the upper (``True``) or lower
        (``False``) bound.
    """
    logRr = np.asarray(logRr, dtype=float)
    seLogRr = np.asarray(seLogRr, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = logRr / seLogRr
    pUpperBound = 1 - np.asarray(pnorm(z))
    pLowerBound = np.asarray(pnorm(z))
    if twoSided:
        out = 2 * np.minimum(pUpperBound, pLowerBound)
    elif upper:
        out = pUpperBound
    else:
        out = pLowerBound
    return out.item() if np.ndim(out) == 0 else out


def calibrateP(null, logRr, seLogRr, twoSided=True, upper=True, pValueOnly=None, **kwargs):
    """Compute calibrated p-values using a fitted null distribution.

    Dispatches on the type of ``null``, mirroring R's S3 methods
    ``calibrateP.null`` and ``calibrateP.mcmcNull``.

    Parameters
    ----------
    null : Null or McmcNull
        As produced by :func:`fitNull` or :func:`fitMcmcNull`.
    logRr, seLogRr : array_like
        One or more effect estimates on the log scale and their standard errors.
    twoSided : bool
        Two-sided (``True``) or one-sided (``False``) p-value.
    upper : bool
        If one-sided: the upper (``True``) or lower (``False``) bound.
    pValueOnly : bool, optional
        ``mcmcNull`` only -- return just the p-value rather than the data frame
        with the credible interval.

    Returns
    -------
    numpy.ndarray or pandas.DataFrame
        For a ``Null``: the calibrated p-values.  For an ``McmcNull``: a data
        frame with columns ``p``, ``lb95ci`` and ``ub95ci`` (or, with
        ``pValueOnly=True``, just the p-values).
    """
    from ._types import McmcNull  # local import to avoid a cycle at import time
    from .mcmc import _calibrateP_mcmcNull

    if isinstance(null, McmcNull):
        return _calibrateP_mcmcNull(null, logRr, seLogRr, twoSided, upper, pValueOnly)
    return _calibrateP_null(null, logRr, seLogRr, twoSided, upper)


def _calibrateP_null(null, logRr, seLogRr, twoSided=True, upper=True):
    logRr = np.atleast_1d(np.asarray(logRr, dtype=float))
    seLogRr = np.atleast_1d(np.asarray(seLogRr, dtype=float))
    if logRr.size != seLogRr.size:
        raise ValueError("The logRr and seLogRr arguments must be of equal length")

    null = np.asarray(null, dtype=float)
    out = np.empty(logRr.size, dtype=float)
    for i in range(logRr.size):
        if (np.isnan(logRr[i]) or np.isinf(logRr[i])
                or np.isnan(seLogRr[i]) or np.isinf(seLogRr[i])):
            out[i] = np.nan
            continue
        denom = np.sqrt(null[1] ** 2 + seLogRr[i] ** 2)
        pUpperBound = float(pnorm((null[0] - logRr[i]) / denom))
        pLowerBound = float(pnorm((logRr[i] - null[0]) / denom))
        if twoSided:
            out[i] = 2 * min(pUpperBound, pLowerBound)
        elif upper:
            out[i] = pUpperBound
        else:
            out[i] = pLowerBound
    return out


def fitNullNonNormalLl(likelihoodApproximations) -> Null:
    """Fit the null distribution using non-normal log-likelihood approximations.

    Parameters
    ----------
    likelihoodApproximations : pandas.DataFrame or sequence
        Either a data frame of normal (``logRr``/``seLogRr``), custom
        (``mu``/``sigma``/``gamma``) or skew-normal (``mu``/``sigma``/``alpha``)
        parametric approximations, a data frame whose column names are the grid
        points, or a list of grid likelihood profiles (data frames with
        ``point`` and ``value`` columns).

    Returns
    -------
    Null
    """
    llApproximationFunction = gridLlApproximation

    if isinstance(likelihoodApproximations, pd.DataFrame):
        cols = list(likelihoodApproximations.columns)
        if "logRr" in cols:
            print("Detected data following normal distribution")
            return fitNull(logRr=likelihoodApproximations["logRr"],
                           seLogRr=likelihoodApproximations["seLogRr"])
        elif "gamma" in cols:
            print("Detected data following custom parameric distribution")
            llApproximationFunction = customLlApproximation
            bad = (likelihoodApproximations["mu"].isna()
                   | likelihoodApproximations["sigma"].isna()
                   | likelihoodApproximations["gamma"].isna())
            if bad.any():
                warnings.warn("Approximations with NA parameters detected. "
                              "Removing before fitting null distribution", stacklevel=2)
                likelihoodApproximations = likelihoodApproximations[~bad]
            approximations = [likelihoodApproximations.iloc[[i]]
                              for i in range(len(likelihoodApproximations))]
        elif "alpha" in cols:
            print("Detected data following skew normal distribution")
            llApproximationFunction = skewNormalLlApproximation
            bad = (likelihoodApproximations["mu"].isna()
                   | likelihoodApproximations["sigma"].isna()
                   | likelihoodApproximations["alpha"].isna())
            if bad.any():
                warnings.warn("Approximations with NA parameters detected. "
                              "Removing before fitting null distribution", stacklevel=2)
                likelihoodApproximations = likelihoodApproximations[~bad]
            approximations = [likelihoodApproximations.iloc[[i]]
                              for i in range(len(likelihoodApproximations))]
        else:
            print("Detected data following grid distribution")
            try:
                point = np.asarray([float(c) for c in cols], dtype=float)
            except (TypeError, ValueError) as err:
                raise ValueError(
                    "Expecting grid data, but not all column names are numeric") from err
            if np.any(np.isnan(point)):
                raise ValueError("Expecting grid data, but not all column names are numeric")
            approximations = [
                pd.DataFrame({"value": likelihoodApproximations.iloc[i].to_numpy(dtype=float),
                              "point": point})
                for i in range(len(likelihoodApproximations))
            ]
    else:
        print("Detected data following grid distribution")
        approximations = []
        for approx in likelihoodApproximations:
            approx = pd.DataFrame(approx).copy()
            approx["value"] = approx["value"] - approx["value"].max()
            approximations.append(approx)

    theta = np.array([0.0, 1.0])
    fit = optim(
        theta,
        lambda t: logLikelihoodNullNonNormalLl(t, llApproximationFunction, approximations),
    )
    par = fit["par"]
    return Null(par[0], 1.0 / np.sqrt(par[1]))
