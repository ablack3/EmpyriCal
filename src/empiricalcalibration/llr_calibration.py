"""Port of ``R/LlrCalibration.R``: calibration of the log likelihood ratio used
in MaxSPRT surveillance."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._rintegrate import integrate
from ._rmath import dnorm, pchisq, qchisq
from ._roptim import optim
from ._types import McmcNull
from .likelihoods import (
    customLlApproximation,
    gridLlApproximation,
    normalLlApproximaton,
    skewNormalLlApproximation,
)

__all__ = ["calibrateLlr", "computePFromLlr", "computeLlrFromP"]


def computePFromLlr(llr, mle, nullLogRr=0.0):
    """One-sided p-value corresponding to a log likelihood ratio.

    For very large ``llr`` R's ``pchisq`` returns exactly 0; since the p-value
    behaves roughly log-linearly in that region, linear extrapolation is used
    above ``llr = 33`` (the coefficients are R's, fitted for smooth joining).
    """
    llr = np.asarray(llr, dtype=float)
    mle = np.asarray(mle, dtype=float)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        p = np.where(
            llr > 33,
            np.exp(-2.4039960592000753081 - 1.0193835554520327413 * llr),
            (1 - pchisq(2 * llr, 1)) / 2,
        )
    out = np.where(mle < nullLogRr, 1 - p, p)
    return out.item() if np.ndim(out) == 0 else out


def computeLlrFromP(p):
    """Log likelihood ratio corresponding to a one-sided p-value.

    For ``p < 1e-16`` R's ``qchisq`` returns ``Inf``, so linear extrapolation of
    the log-linear region is used instead (R's fitted coefficients)."""
    p = float(p)
    if p >= 0.5:
        return 0.0
    if p < 1e-16:
        return -2.0604312 - 0.9677458 * np.log(p)
    return float(qchisq(1 - 2 * p, 1) / 2)


def _computePAtNull(x, logLikelihood, parameters, mle, ml, null):
    """R's ``computePAtNull``: the p-value at true effect ``x``, weighted by the null."""
    llr = ml - float(np.asarray(logLikelihood(x=x, parameters=parameters)).item())
    p = computePFromLlr(llr, mle=mle, nullLogRr=x)
    return p * float(dnorm(x, null[0], null[1]))


def _calibrateOneLlrOneNull(nullMu, nullSigma, logLikelihood, parameters, mle, ml):
    """R's ``calibrateOneLlrOneNull``."""
    if nullSigma < 0.001:
        if mle < nullMu:
            return 0.0
        return ml - float(np.asarray(
            logLikelihood(x=nullMu, parameters=parameters)).item())
    calibratedP = integrate(
        _computePAtNull,
        nullMu - 10 * nullSigma, nullMu + 10 * nullSigma,
        args=(logLikelihood, parameters, mle, ml, (nullMu, nullSigma)),
        stop_on_error=False,
    )["value"]
    return computeLlrFromP(calibratedP)


def calibrateLlr(null, likelihoodApproximation, twoSided=False, upper=True):
    """Compute a calibrated log likelihood ratio using a fitted null distribution.

    Parameters
    ----------
    null : Null or McmcNull
        As created by :func:`fitNull` or :func:`fitMcmcNull`.
    likelihoodApproximation : pandas.DataFrame or sequence
        Either a data frame of normal (``logRr``/``seLogRr``), custom
        (``mu``/``sigma``/``gamma``) or skew-normal (``mu``/``sigma``/``alpha``)
        likelihood approximations, or one or more grid likelihood profiles.
    twoSided : bool
        Must be ``False`` -- only one-sided upper LLRs are supported.
    upper : bool
        Must be ``True`` -- only one-sided upper LLRs are supported.

    Returns
    -------
    numpy.ndarray or pandas.DataFrame
        The calibrated LLR(s).  With an :class:`McmcNull`, a data frame with
        ``llr``, ``ci95Lb`` and ``ci95Ub``.
    """
    if twoSided or not upper:
        raise ValueError("Currently only one-sided upper LLRs are supported")

    approximations = None
    if isinstance(likelihoodApproximation, pd.Series):
        # R's is.numeric() branch: a single named numeric vector is one grid
        # profile whose names are the grid points.  A Series indexed by grid
        # point is the direct analogue, so transpose it to a one-row frame.
        likelihoodApproximation = likelihoodApproximation.to_frame().T
    if isinstance(likelihoodApproximation, pd.DataFrame):
        cols = list(likelihoodApproximation.columns)
        if "logRr" in cols:
            print("Detected data following normal distribution")
            type_ = "normal"
            logLikelihood = normalLlApproximaton
            approximations = [likelihoodApproximation.iloc[[i]]
                              for i in range(len(likelihoodApproximation))]
        elif "gamma" in cols:
            print("Detected data following custom parameric distribution")
            type_ = "custom"
            logLikelihood = customLlApproximation
            approximations = [likelihoodApproximation.iloc[[i]]
                              for i in range(len(likelihoodApproximation))]
        elif "alpha" in cols:
            print("Detected data following skew normal distribution")
            type_ = "skewNormal"
            logLikelihood = skewNormalLlApproximation
            approximations = [likelihoodApproximation.iloc[[i]]
                              for i in range(len(likelihoodApproximation))]
        elif "point" in cols:
            print("Detected data following grid distribution")
            type_ = "grid"
            logLikelihood = gridLlApproximation
            approximations = [likelihoodApproximation]
        else:
            print("Detected data following grid distribution")
            type_ = "grid"
            logLikelihood = gridLlApproximation
            try:
                point = np.asarray([float(c) for c in cols], dtype=float)
            except (TypeError, ValueError) as err:
                raise ValueError(
                    "Expecting grid data, but not all column names are numeric") from err
            approximations = []
            for i in range(len(likelihoodApproximation)):
                value = likelihoodApproximation.iloc[i].to_numpy(dtype=float)
                approximations.append(pd.DataFrame({"value": value - value.max(),
                                                    "point": point}))
    else:
        print("Detected data following grid distribution")
        type_ = "grid"
        logLikelihood = gridLlApproximation
        approximations = list(likelihoodApproximation)

    useMcmc = isinstance(null, McmcNull)
    results = []

    for parameters in approximations:
        if type_ == "grid":
            value = np.asarray(parameters["value"], dtype=float)
            point = np.asarray(parameters["point"], dtype=float)
            idx = int(np.flatnonzero(value == value.max())[0])
            mle = float(point[idx])
            ml = float(value[idx])
        elif type_ == "normal":
            lr = float(np.asarray(parameters["logRr"]).ravel()[0])
            se = float(np.asarray(parameters["seLogRr"]).ravel()[0])
            if np.isnan(lr) or np.isnan(se) or np.isinf(lr) or np.isinf(se):
                results.append(np.nan)
                continue
            mle = lr
            ml = float(np.asarray(logLikelihood(mle, parameters=parameters)).item())
        else:
            fit = optim([0.0], lambda x, p=parameters: -float(np.asarray(
                logLikelihood(x=x[0], parameters=p)).item()))
            mle = float(fit["par"][0])
            ml = -float(fit["value"])

        if useMcmc:
            chain = null.mcmc["chain"]
            with np.errstate(divide="ignore", invalid="ignore"):
                sigmas = 1 / np.sqrt(chain[:, 1])
            vals = np.array([
                _calibrateOneLlrOneNull(mu, sigma, logLikelihood, parameters, mle, ml)
                for mu, sigma in zip(chain[:, 0], sigmas)
            ])
            results.append(np.quantile(vals, [0.5, 0.025, 0.975]))
        else:
            results.append(_calibrateOneLlrOneNull(
                float(np.asarray(null)[0]), float(np.asarray(null)[1]),
                logLikelihood, parameters, mle, ml))

    if useMcmc:
        return pd.DataFrame(np.vstack(results), columns=["llr", "ci95Lb", "ci95Ub"])
    return np.asarray(results, dtype=float)
