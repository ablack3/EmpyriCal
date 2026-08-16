"""Port of ``R/Likelihoods.R`` plus the two likelihood helpers that live in
``src/RcppWrapper.cpp`` (``logLikelihoodNull`` and ``gridLlApproximation``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._rintegrate import integrate
from ._rmath import dgamma_log, dnorm, pnorm

__all__ = [
    "gaussianProduct", "logLikelihoodNull", "logLikelihoodNullMcmc",
    "minLogLikelihoodErrorModel", "minLogLikelihoodErrorModelLegacy",
    "logLikelihoodNullNonNormalLl", "normalLlApproximaton",
    "customLlApproximation", "skewNormalLlApproximation", "gridLlApproximation",
]


def gaussianProduct(mu1, mu2, sd1, sd2):
    """R's ``gaussianProduct`` from ``Likelihoods.R``."""
    mu1 = np.asarray(mu1, dtype=float)
    mu2 = np.asarray(mu2, dtype=float)
    sd1 = np.asarray(sd1, dtype=float)
    sd2 = np.asarray(sd2, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        var = sd1 ** 2 + sd2 ** 2
        out = ((2 * np.pi) ** (-1 / 2) * var ** (-1 / 2)
               * np.exp(-(mu1 - mu2) ** 2 / (2 * var)))
    return out.item() if out.ndim == 0 else out


def _gaussian_product_c(mu1, mu2, sd1, sd2):
    """The C++ variant used inside ``logLikelihoodNull`` (same value, R's ordering)."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        v = sd1 ** 2 + sd2 ** 2
        return (1.0 / (np.sqrt(2 * np.pi) * np.sqrt(v))) * np.exp(-(mu1 - mu2) ** 2 / (2 * v))


def logLikelihoodNull(theta, logRr, seLogRr):
    """Port of the C++ ``logLikelihoodNull``.

    ``theta`` is ``(mean, precision)``.  Returns ``+Inf`` for a non-positive
    precision and (as in the C++) whenever the accumulated result is exactly 0.
    """
    theta = np.asarray(theta, dtype=float)
    logRr = np.atleast_1d(np.asarray(logRr, dtype=float))
    seLogRr = np.atleast_1d(np.asarray(seLogRr, dtype=float))
    if theta[1] <= 0:
        return np.inf
    sd = 1.0 / np.sqrt(theta[1])
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        if sd < 1e-6:
            terms = dnorm(theta[0], logRr, seLogRr, log=True)
            result = -np.sum(np.atleast_1d(terms))
        else:
            gp = _gaussian_product_c(logRr, theta[0], seLogRr, sd)
            result = -np.sum(np.log(np.atleast_1d(gp)))
    result = float(result)
    if result == 0:
        result = np.inf
    return result


def logLikelihoodNullMcmc(theta, logRr, seLogRr):
    """R's ``logLikelihoodNullMcmc`` -- the null likelihood plus a weak gamma prior."""
    result = logLikelihoodNull(theta, logRr, seLogRr)
    theta = np.asarray(theta, dtype=float)
    return result - float(dgamma_log(theta[1], shape=1e-04, rate=1e-04))


def minLogLikelihoodErrorModel(theta, logRr, seLogRr, trueLogRr):
    """R's ``minLogLikelihoodErrorModel`` (standard deviation linear in |true|)."""
    theta = np.asarray(theta, dtype=float)
    logRr = np.atleast_1d(np.asarray(logRr, dtype=float))
    seLogRr = np.atleast_1d(np.asarray(seLogRr, dtype=float))
    trueLogRr = np.atleast_1d(np.asarray(trueLogRr, dtype=float))

    total = 0.0
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        for i in range(logRr.size):
            mean = theta[0] + theta[1] * trueLogRr[i]
            sd = theta[2] + theta[3] * abs(trueLogRr[i])
            if sd < 0:
                total += np.inf
            elif sd < 1e-6:
                total += -float(dnorm(logRr[i], mean, seLogRr[i], log=True))
            else:
                # R-level gaussianProduct: this is an R function, not the C++ one
                total += -float(np.log(gaussianProduct(logRr[i], mean, seLogRr[i], sd)))
    if not np.isfinite(total):
        total = 99999.0
    return float(total)


def minLogLikelihoodErrorModelLegacy(theta, logRr, seLogRr, trueLogRr):
    """R's ``minLogLikelihoodErrorModelLegacy`` (log standard deviation linear)."""
    theta = np.asarray(theta, dtype=float)
    logRr = np.atleast_1d(np.asarray(logRr, dtype=float))
    seLogRr = np.atleast_1d(np.asarray(seLogRr, dtype=float))
    trueLogRr = np.atleast_1d(np.asarray(trueLogRr, dtype=float))

    result = 0.0
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        for i in range(logRr.size):
            mean = theta[0] + theta[1] * trueLogRr[i]
            sd = np.exp(theta[2] + theta[3] * trueLogRr[i])
            result = result - float(np.log(gaussianProduct(logRr[i], mean, seLogRr[i], sd)))
    if not np.isfinite(result):
        result = 99999.0
    return float(result)


def _profileProduct(x, mu, sigma, llApproximationFunction, likelihoodApproximation):
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        return np.exp(dnorm(x, mu, sigma, log=True)
                      + llApproximationFunction(x=x, parameters=likelihoodApproximation))


def _cheap_params(parameters):
    """Hoist a one-row parameter frame into a plain dict of floats.

    ``_get`` reads its constants out of ``parameters`` on *every* integrand
    evaluation, and a pandas scalar lookup is roughly two orders of magnitude
    dearer than a dict lookup -- it dominated the runtime of the non-normal
    fits.  Converting once per call is arithmetically identical, since ``_get``
    would have applied the same ``float()`` to the same values.

    Grid profiles are left alone: their ``point``/``value`` columns are arrays
    that :func:`gridLlApproximation` indexes as a whole.
    """
    if isinstance(parameters, pd.DataFrame):
        if len(parameters) != 1 or "point" in parameters.columns:
            return parameters
        row = parameters.iloc[0]
        return {name: float(row[name]) for name in parameters.columns
                if isinstance(row[name], (int, float, np.integer, np.floating))}
    return parameters


def logLikelihoodNullNonNormalLl(theta, llApproximationFunction, likelihoodApproximations):
    """R's ``logLikelihoodNullNonNormalLl``."""
    theta = np.asarray(theta, dtype=float)
    if theta[1] <= 0:
        return 99999.0
    result = 0.0
    sd = 1.0 / np.sqrt(theta[1])
    approximations = [_cheap_params(a) for a in likelihoodApproximations]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        if sd < 1e-6:
            for approx in approximations:
                result = result - float(
                    np.asarray(llApproximationFunction(x=theta[0], parameters=approx)).item())
        else:
            for approx in approximations:
                val = integrate(
                    lambda x, a=approx: float(np.asarray(
                        _profileProduct(x, theta[0], sd, llApproximationFunction, a)).item()),
                    theta[0] - 10 * sd, theta[0] + 10 * sd,
                    stop_on_error=False,
                )["value"]
                result = result - float(np.log(val))
    if not np.isfinite(result):
        result = 99999.0
    return float(result)


# --------------------------------------------------------------------------
# The four likelihood-approximation shapes
# --------------------------------------------------------------------------

def _get(parameters, name):
    """Read a field from a one-row DataFrame, a Series, a dict or an object."""
    if isinstance(parameters, pd.DataFrame):
        return float(parameters[name].iloc[0])
    if isinstance(parameters, pd.Series):
        return float(parameters[name])
    if isinstance(parameters, dict):
        return float(parameters[name])
    return float(getattr(parameters, name))


def normalLlApproximaton(x, parameters):
    """Normal log-likelihood approximation (name matches R's typo)."""
    return dnorm(x, mean=_get(parameters, "logRr"), sd=_get(parameters, "seLogRr"), log=True)


def customLlApproximation(x, parameters):
    """R's ``customLlApproximation`` (the 'gamma' parameterisation)."""
    x = np.asarray(x, dtype=float)
    mu = _get(parameters, "mu")
    sigma = _get(parameters, "sigma")
    gamma = _get(parameters, "gamma")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        return (np.exp(gamma * (x - mu))) * ((-(x - mu) ** 2) / (2 * sigma ** 2))


def skewNormalLlApproximation(x, parameters):
    """R's ``skewNormalLlApproximation``."""
    x = np.asarray(x, dtype=float)
    mu = _get(parameters, "mu")
    sigma = _get(parameters, "sigma")
    alpha = _get(parameters, "alpha")
    if np.isinf(sigma):
        return np.zeros(x.shape) if x.ndim else 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        # R uses pnorm(..., log.p = TRUE); taking log() of the probability
        # instead would underflow to -Inf once alpha*(x-mu)/sigma drops below
        # about -38, which happens routinely while integrating over +/-10 sd.
        return (np.log(2)
                + dnorm(x, mu, sigma, log=True)
                + pnorm(alpha * (x - mu), 0, sigma, log_p=True))


def gridLlApproximation(x, parameters):
    """Port of the C++ ``gridLlApproximation``: linear interpolation of a
    likelihood profile, with flat-or-decreasing linear extrapolation outside
    the grid."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    if isinstance(parameters, pd.DataFrame):
        point = parameters["point"].to_numpy(dtype=float)
        value = parameters["value"].to_numpy(dtype=float)
    else:
        point = np.asarray(parameters["point"], dtype=float)
        value = np.asarray(parameters["value"], dtype=float)
    n = point.size
    result = np.empty(x.shape, dtype=float)

    for i in range(x.size):
        xi = x.flat[i]
        if xi < point[0]:
            slope = max(0.0, (value[1] - value[0]) / (point[1] - point[0]))
            result.flat[i] = (xi - point[0]) * slope + value[0]
        elif xi >= point[n - 1]:
            slope = min(0.0, (value[n - 1] - value[n - 2]) / (point[n - 1] - point[n - 2]))
            result.flat[i] = (xi - point[n - 1]) * slope + value[n - 1]
        else:
            for j in range(n):
                if point[j] >= xi:
                    slope = (value[j] - value[j - 1]) / (point[j] - point[j - 1])
                    result.flat[i] = (xi - point[j - 1]) * slope + value[j - 1]
                    break
    if np.ndim(x) == 0:
        return result.item()
    return result
