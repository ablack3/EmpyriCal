"""Port of ``R/ConfidenceIntervalCalibration.R``.

Implements the method of Schuemie et al., *Empirical confidence interval
calibration for population-level effect estimation studies in observational
healthcare data*, PNAS 115(11):2571-2577, 2018.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ._rmath import qnorm
from ._roptim import optim, uniroot
from ._types import McmcNull, Null, SystematicErrorModel
from .likelihoods import minLogLikelihoodErrorModel, minLogLikelihoodErrorModelLegacy

__all__ = ["fitSystematicErrorModel", "calibrateConfidenceInterval",
           "computeTraditionalCi", "convertNullToErrorModel"]


def fitSystematicErrorModel(logRr, seLogRr, trueLogRr,
                            estimateCovarianceMatrix=False,
                            legacy=False) -> SystematicErrorModel:
    """Fit a model of the systematic error as a function of true effect size.

    The mean and standard deviation of the error distribution are assumed to be
    linear in the true effect size, so each is represented by an intercept and a
    slope.

    Parameters
    ----------
    logRr, seLogRr : array_like
        Effect estimates on the log scale and their standard errors.
    trueLogRr : array_like
        The true effect sizes.
    estimateCovarianceMatrix : bool
        Compute a covariance matrix, making confidence intervals for the model
        parameters available as the ``CovarianceMatrix``, ``LB95CI`` and
        ``UB95CI`` entries of ``.attributes``.
    legacy : bool
        Fit the legacy model, in which the standard deviation is linear on the
        log scale rather than simply linear.

    Returns
    -------
    SystematicErrorModel
    """
    logRr = np.atleast_1d(np.asarray(logRr, dtype=float)).copy()
    seLogRr = np.atleast_1d(np.asarray(seLogRr, dtype=float)).copy()
    trueLogRr = np.atleast_1d(np.asarray(trueLogRr, dtype=float)).copy()
    logRr, seLogRr, trueLogRr = (np.broadcast_arrays(logRr, seLogRr, trueLogRr))
    logRr, seLogRr, trueLogRr = logRr.copy(), seLogRr.copy(), trueLogRr.copy()

    def drop(keep):
        return trueLogRr[keep], logRr[keep], seLogRr[keep]

    if np.any(np.isinf(seLogRr)):
        warnings.warn("Estimate(s) with infinite standard error detected. "
                      "Removing before fitting error model", stacklevel=2)
        trueLogRr, logRr, seLogRr = drop(~np.isinf(seLogRr))
    if np.any(np.isinf(logRr)):
        warnings.warn("Estimate(s) with infinite logRr detected. "
                      "Removing before fitting error model", stacklevel=2)
        trueLogRr, logRr, seLogRr = drop(~np.isinf(logRr))
    if np.any(np.isnan(seLogRr)):
        warnings.warn("Estimate(s) with NA standard error detected. "
                      "Removing before fitting error model", stacklevel=2)
        trueLogRr, logRr, seLogRr = drop(~np.isnan(seLogRr))
    if np.any(np.isnan(logRr)):
        warnings.warn("Estimate(s) with NA logRr detected. "
                      "Removing before fitting error model", stacklevel=2)
        trueLogRr, logRr, seLogRr = drop(~np.isnan(logRr))

    if legacy:
        theta = np.array([0.0, 1.0, -2.0, 0.0])
        logLikelihood = minLogLikelihoodErrorModelLegacy
        parscale = np.array([1.0, 1.0, 10.0, 10.0])
        names = ["meanIntercept", "meanSlope", "logSdIntercept", "logSdSlope"]
    else:
        theta = np.array([0.0, 1.0, 0.1, 0.0])
        logLikelihood = minLogLikelihoodErrorModel
        parscale = np.array([1.0, 1.0, 1.0, 1.0])
        names = ["meanIntercept", "meanSlope", "sdIntercept", "sdSlope"]

    fit = optim(
        theta,
        lambda t: logLikelihood(t, logRr, seLogRr, trueLogRr),
        method="BFGS",
        hessian=True,
        control={"parscale": parscale},
    )
    model = SystematicErrorModel(fit["par"], names)
    if estimateCovarianceMatrix:
        fisher_info = np.linalg.inv(fit["hessian"])
        prop_sigma = np.sqrt(np.diag(fisher_info))
        model.attributes["CovarianceMatrix"] = fisher_info
        model.attributes["LB95CI"] = fit["par"] + qnorm(0.025) * prop_sigma
        model.attributes["UB95CI"] = fit["par"] + qnorm(0.975) * prop_sigma
    return model


def _opt(x, z, logRr, se, interceptMean, slopeMean, interceptSd, slopeSd, legacy):
    """R's inner ``opt``: the function whose root is a calibrated bound."""
    mean = interceptMean + slopeMean * x
    if legacy:
        sd = np.exp(interceptSd + slopeSd * x)
    else:
        sd = interceptSd + slopeSd * abs(x)
    numerator = mean - logRr
    denominator = np.sqrt(sd ** 2 + se ** 2)
    if np.isinf(denominator):
        return z + 1e-10 if numerator > 0 else z - 1e-10
    return z + numerator / denominator


def _logBound(ciWidth, lb, logRr, se, interceptMean, slopeMean,
              interceptSd, slopeSd, legacy):
    """R's inner ``logBound``."""
    z = qnorm((1 - ciWidth) / 2)
    if lb:
        z = -z
    args = (z, logRr, se, interceptMean, slopeMean, interceptSd, slopeSd, legacy)

    if legacy:
        # Simple grid search for a bracketing interval
        if slopeSd > 0:
            lower = -100.0
            upper = lower
            while _opt(upper, *args) < 0 and upper < 100:
                upper += 1
            if upper == lower or upper == 100:
                return np.nan
        else:
            upper = 100.0
            lower = upper
            while _opt(lower, *args) > 0 and lower > -100:
                lower -= 1
            if upper == lower or lower == -100:
                return np.nan
    else:
        lower, upper = -100.0, 100.0
        lowerValue = _opt(lower, *args)
        upperValue = _opt(upper, *args)
        if (lowerValue < 0 and upperValue < 0) or (lowerValue > 0 and upperValue > 0):
            return np.nan

    return uniroot(_opt, (lower, upper), args=args)["root"]


def calibrateConfidenceInterval(logRr, seLogRr, model, ciWidth=0.95) -> pd.DataFrame:
    """Compute calibrated confidence intervals from a systematic error model.

    Parameters
    ----------
    logRr, seLogRr : array_like
        Effect estimates on the log scale and their standard errors.
    model : SystematicErrorModel
        As created by :func:`fitSystematicErrorModel` or
        :func:`convertNullToErrorModel`.
    ciWidth : float
        Width of the confidence interval, e.g. ``0.95``.

    Returns
    -------
    pandas.DataFrame
        Columns ``logRr``, ``logLb95Rr``, ``logUb95Rr`` and ``seLogRr``.
    """
    logRr = np.atleast_1d(np.asarray(logRr, dtype=float))
    seLogRr = np.atleast_1d(np.asarray(seLogRr, dtype=float))
    logRr, seLogRr = np.broadcast_arrays(logRr, seLogRr)

    names = getattr(model, "names", None)
    legacy = names is not None and names[2] == "logSdIntercept"
    m = np.asarray(model, dtype=float)

    n = logRr.size
    result = pd.DataFrame({
        "logRr": np.full(n, np.nan),
        "logLb95Rr": np.full(n, np.nan),
        "logUb95Rr": np.full(n, np.nan),
    })

    if not np.isnan(m[0]):
        for i in range(n):
            if (np.isinf(logRr[i]) or np.isnan(logRr[i])
                    or np.isinf(seLogRr[i]) or np.isnan(seLogRr[i])):
                continue
            result.loc[i, "logRr"] = _logBound(
                0, True, logRr[i], seLogRr[i], m[0], m[1], m[2], m[3], legacy)
            result.loc[i, "logLb95Rr"] = _logBound(
                ciWidth, True, logRr[i], seLogRr[i], m[0], m[1], m[2], m[3], legacy)
            result.loc[i, "logUb95Rr"] = _logBound(
                ciWidth, False, logRr[i], seLogRr[i], m[0], m[1], m[2], m[3], legacy)

    result["seLogRr"] = ((result["logLb95Rr"] - result["logUb95Rr"])
                         / (2 * qnorm((1 - ciWidth) / 2)))
    return result


def computeTraditionalCi(logRr, seLogRr, ciWidth=0.95) -> pd.DataFrame:
    """Compute the traditional confidence interval.

    Returns
    -------
    pandas.DataFrame
        Columns ``rr``, ``lb`` and ``ub`` (on the ratio scale).
    """
    # atleast_1d so a scalar estimate yields a one-row frame, as in R, rather
    # than failing pandas' "all scalar values" construction check; then
    # broadcast so that mixing a scalar with a vector recycles the way R does
    # (atleast_1d alone turns a scalar into length 1, which pandas rejects
    # against a longer column).
    logRr = np.atleast_1d(np.asarray(logRr, dtype=float))
    seLogRr = np.atleast_1d(np.asarray(seLogRr, dtype=float))
    logRr, seLogRr = np.broadcast_arrays(logRr, seLogRr)
    z = qnorm((1 - ciWidth) / 2)
    return pd.DataFrame({
        "rr": np.exp(logRr),
        "lb": np.exp(logRr + z * seLogRr),
        "ub": np.exp(logRr - z * seLogRr),
    })


def convertNullToErrorModel(null, meanSlope=1.0, sdSlope=0.0) -> SystematicErrorModel:
    """Convert an empirical null distribution into a systematic error model.

    Where :func:`fitSystematicErrorModel` uses positive controls to learn how the
    error distribution changes with true effect size, this function requires the
    user to assume it.  The default (``meanSlope = 1``, ``sdSlope = 0``) encodes
    the belief that the error distribution is the same for all true effect
    sizes.  If the estimation method is biased towards the null this assumption
    is violated and the calibrated intervals will have below-nominal coverage.

    Parameters
    ----------
    null : Null or McmcNull
        Fitted with :func:`fitNull` or :func:`fitMcmcNull`.
    meanSlope : float
        Slope for the mean of the error distribution.
    sdSlope : float
        Slope for the standard deviation of the error distribution.

    Returns
    -------
    SystematicErrorModel
    """
    if isinstance(null, Null):
        values = [null[0], meanSlope, null[1], sdSlope]
    elif isinstance(null, McmcNull):
        values = [null[0], meanSlope, 1 / np.sqrt(null[1]), sdSlope]
    else:
        raise ValueError("Null argument should be of type 'null' or 'mcmcNull'")
    return SystematicErrorModel(
        values, ["meanIntercept", "meanSlope", "sdIntercept", "sdSlope"])
