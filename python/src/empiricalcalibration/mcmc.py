"""Port of ``R/EmpiricalCalibrationUsingMcmc.R``.

Fits the null distribution by Metropolis MCMC, which additionally yields a
credible interval for the calibrated p-value.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ._rmath import pnorm, qchisq
from ._roptim import optim
from ._rrng import get_generator
from ._types import McmcNull
from .asymptotics import _clean_estimates
from .likelihoods import logLikelihoodNullMcmc

__all__ = ["fitMcmcNull"]


def _proposalFunction(param, scale, rng):
    """R's ``proposalFunction``: a normal random walk with a reflected precision."""
    param = np.asarray(param, dtype=float)
    draw = rng.rnorm(param.size, mean=param, sd=scale)
    draw = np.asarray(draw, dtype=float).copy()
    # Precision cannot be negative
    draw[1] = abs(draw[1])
    return draw


def _runMetropolisMcmc(startValue, iterations, scale, logRr, seLogRr, rng):
    """R's ``runMetropolisMcmc``."""
    startValue = np.asarray(startValue, dtype=float)
    dim = startValue.size
    chain = np.empty((iterations + 1, dim), dtype=float)
    logLik = np.empty((iterations + 1, 1), dtype=float)
    acc = np.empty((iterations + 1, 1), dtype=float)

    logLik[0, 0] = -logLikelihoodNullMcmc(startValue, logRr, seLogRr)
    chain[0, :] = startValue
    acc[0, 0] = 1

    for i in range(iterations):
        proposal = _proposalFunction(chain[i, :], scale, rng)
        try:
            newLogLik = -logLikelihoodNullMcmc(proposal, logRr, seLogRr)
        except Exception:
            newLogLik = -1e10
        with np.errstate(over="ignore"):
            prob = np.exp(newLogLik - logLik[i, 0])
        if rng.unif_rand() < prob:
            chain[i + 1, :] = proposal
            logLik[i + 1, 0] = newLogLik
            acc[i + 1, 0] = 1
        else:
            chain[i + 1, :] = chain[i, :]
            logLik[i + 1, 0] = logLik[i, 0]
            acc[i + 1, 0] = 0
    return {"logLik": logLik, "chain": chain, "acc": acc}


def _binarySearchMu(modeMu, modeSigma, logRrNegatives, seLogRrNegatives,
                    alpha=0.1, precision=1e-07):
    """R's ``binarySearchMu``: profile-likelihood scale for the mean."""
    q = qchisq(1 - alpha, 1) / 2
    L = modeMu
    H = 10.0
    llMode = -logLikelihoodNullMcmc([modeMu, modeSigma], logRrNegatives, seLogRrNegatives)
    while H >= L:
        M = L + (H - L) / 2
        llM = -logLikelihoodNullMcmc([M, modeSigma], logRrNegatives, seLogRrNegatives)
        metric = llMode - llM - q
        if metric > precision:
            H = M
        elif -metric > precision:
            L = M
        else:
            return abs(M - modeMu)
        if M == modeMu or M == 10:
            return 0.0
    return None


def _binarySearchSigma(modeMu, modeSigma, logRrNegatives, seLogRrNegatives,
                       alpha=0.1, precision=1e-07):
    """R's ``binarySearchSigma``: profile-likelihood scale for the precision."""
    q = qchisq(1 - alpha, 1) / 2
    llMode = -logLikelihoodNullMcmc([modeMu, modeSigma], logRrNegatives, seLogRrNegatives)
    L = modeSigma
    H = modeSigma
    i = 0
    for i in range(1, 11):
        H = modeSigma + np.exp(i)
        llM = -logLikelihoodNullMcmc([modeMu, H], logRrNegatives, seLogRrNegatives)
        metric = llMode - llM - q
        if metric > 0:
            break
    if i == 10:
        return 0.0
    while H >= L:
        M = L + (H - L) / 2
        llM = -logLikelihoodNullMcmc([modeMu, M], logRrNegatives, seLogRrNegatives)
        metric = llMode - llM - q
        if metric > precision:
            H = M
        elif -metric > precision:
            L = M
        else:
            return abs(M - modeSigma)
        if M == modeSigma:
            return 0.0
    return None


def fitMcmcNull(logRr, seLogRr, iter=100000) -> McmcNull:
    """Fit the null distribution using Markov Chain Monte Carlo.

    Experimental function for computing the 95% credible interval of a
    calibrated p-value.

    Parameters
    ----------
    logRr, seLogRr : array_like
        Effect estimates on the log scale and their standard errors.
    iter : int
        Number of MCMC iterations.

    Returns
    -------
    McmcNull
        Posterior median of the mean and the **precision**, with the MCMC trace
        available as ``.mcmc``.
    """
    logRr, seLogRr = _clean_estimates(logRr, seLogRr)
    rng = get_generator()

    if logRr.size == 0:
        warnings.warn("No valid estimates left. Returning undefined null distribution")
        mcmc = {"chain": np.array([[np.nan, np.nan]]),
                "logLik": np.array([[np.nan]]),
                "acc": np.array([[np.nan]])}
    else:
        fit = optim(np.array([0.0, 1.0]),
                    lambda t: logLikelihoodNullMcmc(t, logRr, seLogRr))
        par = fit["par"]
        # Profile likelihood for a roughly correct proposal scale
        scale = [_binarySearchMu(par[0], par[1], logRr, seLogRr),
                 _binarySearchSigma(par[0], par[1], logRr, seLogRr)]
        scale = np.array([0.0 if s is None else s for s in scale], dtype=float)
        mcmc = _runMetropolisMcmc(par, iter, scale, logRr, seLogRr, rng)

    result = McmcNull(np.median(mcmc["chain"][:, 0]),
                      np.median(mcmc["chain"][:, 1]),
                      mcmc=mcmc)
    return result


def _calibrateP_mcmcNull(null, logRr, seLogRr, twoSided=True, upper=True, pValueOnly=None):
    """R's ``calibrateP.mcmcNull``."""
    mcmc = null.mcmc
    logRr = np.atleast_1d(np.asarray(logRr, dtype=float))
    seLogRr = np.atleast_1d(np.asarray(seLogRr, dtype=float))
    n = logRr.size
    adjustedP = pd.DataFrame({
        "p": np.full(n, np.nan),
        "lb95ci": np.full(n, np.nan),
        "ub95ci": np.full(n, np.nan),
    })

    if not np.isnan(float(np.asarray(null)[0])):
        chain = mcmc["chain"]
        for i in range(n):
            if (np.isnan(logRr[i]) or np.isinf(logRr[i])
                    or np.isnan(seLogRr[i]) or np.isinf(seLogRr[i])):
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                sd = np.sqrt((1 / np.sqrt(chain[:, 1])) ** 2 + seLogRr[i] ** 2)
                pUpperBound = np.asarray(pnorm((chain[:, 0] - logRr[i]) / sd))
                pLowerBound = np.asarray(pnorm((logRr[i] - chain[:, 0]) / sd))
            if twoSided:
                p = pUpperBound.copy()
                mask = pLowerBound < p
                p[mask] = pLowerBound[mask]
                p = p * 2
            elif upper:
                p = pUpperBound
            else:
                p = pLowerBound
            adjustedP.loc[i, "p"] = np.quantile(p, 0.5)
            adjustedP.loc[i, "lb95ci"] = np.quantile(p, 0.025)
            adjustedP.loc[i, "ub95ci"] = np.quantile(p, 0.975)

    if pValueOnly is None or pValueOnly is False:
        adjustedP.attrs["mcmc"] = mcmc
        return adjustedP
    return adjustedP["p"].to_numpy()
