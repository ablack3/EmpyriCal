"""Port of ``R/MaxSprtCriticalValues.R`` and the three Monte-Carlo samplers in
``src/RcppWrapper.cpp``.

These re-implement ``Sequential::CV.Poisson`` and ``Sequential::CV.Binomial``
by simulation, and additionally support a Poisson regression design and an
empirical null distribution.

Because the samplers draw from R's RNG in the same order as the C++ code, a
given seed produces the same critical value as the R package.  The default
``sampleSize`` of one million makes these the slowest functions in the package;
a vectorised fast path is used whenever every Poisson mean is below 10 and every
binomial ``n*p`` is below 30 (the regimes in which R's samplers consume exactly
one uniform per variate), with an exact scalar fallback otherwise.
"""

from __future__ import annotations

import warnings

import numpy as np

from ._rmath import dbinom, dpois
from ._rrng import get_generator

__all__ = ["computeCvPoisson", "computeCvBinomial", "computeCvPoissonRegression",
           "samplePoissonMaxLrr", "sampleBinomialMaxLrr",
           "samplePoissonRegressionMaxLrr"]


# --------------------------------------------------------------------------
# Vectorised inverse-CDF sampling that reproduces R's stream exactly
# --------------------------------------------------------------------------

def _pois_cdf_table(mu):
    """R's ``rpois`` Case B cumulative table, accumulated in R's order."""
    cum = np.empty(36, dtype=float)
    p = q = np.exp(-mu)
    cum[0] = q
    for k in range(1, 36):
        p *= mu / k
        q += p
        cum[k] = q
    return cum


def _binom_pmf_table(n, p, limit=111):
    """R's ``rbinom`` inverse-cdf ladder (``np < 30`` branch), in R's order."""
    pp = min(p, 1.0 - p)
    q = 1.0 - pp
    r = pp / q
    g = r * (n + 1)
    f = q ** n
    cum = np.empty(limit, dtype=float)
    total = 0.0
    for ix in range(limit):
        total += f
        cum[ix] = total
        f *= (g / (ix + 1) - r)
    return cum


class _StateGuard:
    """Save/restore the generator so an invalid fast path costs nothing."""

    def __init__(self, rng):
        self.rng = rng

    def __enter__(self):
        r = self.rng
        self._saved = (r._mt.copy(), r._mti, r._buf.copy(), r._bpos)
        return self

    def restore(self):
        r = self.rng
        r._mt, r._mti, r._buf, r._bpos = (
            self._saved[0].copy(), self._saved[1], self._saved[2].copy(), self._saved[3])

    def __exit__(self, *exc):
        return False


# --------------------------------------------------------------------------
# The three samplers
# --------------------------------------------------------------------------

def samplePoissonMaxLrr(groupSizes, minimumEvents, sampleSize, nullMean, nullSd):
    """Port of the C++ ``samplePoissonMaxLrr``."""
    rng = get_generator()
    groupSizes = np.atleast_1d(np.asarray(groupSizes, dtype=float))
    sampleSize = int(sampleSize)

    if nullSd == 0:
        # systematicError is exp(nullMean) for every sample and consumes no draws,
        # so every Poisson mean is known up front and R's inverse-cdf branch
        # (mu < 10) consumes exactly one uniform per variate.
        lam = groupSizes * np.exp(nullMean)
        if np.all(lam < 10):
            return _poisson_fast(rng, groupSizes, lam, minimumEvents, sampleSize,
                                 nullMean, nullSd)

    return _poisson_scalar(rng, groupSizes, minimumEvents, sampleSize, nullMean, nullSd)


def _poisson_fast(rng, groupSizes, lam, minimumEvents, sampleSize, nullMean, nullSd):
    draw_cols = np.flatnonzero(lam > 0)  # rpois(0) consumes nothing
    G = groupSizes.size
    tables = {j: _pois_cdf_table(lam[j]) for j in draw_cols}

    with _StateGuard(rng) as guard:
        u = rng.unif_rand_n(sampleSize * draw_cols.size)
        u = u.reshape(sampleSize, draw_cols.size)
        counts = np.zeros((sampleSize, G), dtype=float)
        for c, j in enumerate(draw_cols):
            idx = np.searchsorted(tables[j], u[:, c], side="left")
            if np.any(idx >= 36):
                # a draw fell past the 36-entry table, where R re-draws; redo the
                # whole thing scalar-wise from the original generator state
                guard.restore()
                return _poisson_scalar(rng, groupSizes, minimumEvents, sampleSize,
                                       nullMean, nullSd)
            counts[:, j] = idx

    observed = np.cumsum(counts, axis=1)
    expected = np.cumsum(groupSizes)[None, :]
    return _max_llr_poisson(observed, expected, minimumEvents)


def _max_llr_poisson(observed, expected, minimumEvents):
    maxLlr = np.zeros(observed.shape[0], dtype=float)
    for j in range(observed.shape[1]):
        obs = observed[:, j]
        exp_ = expected[0, j]
        active = (obs >= minimumEvents) & (obs >= exp_)
        if not np.any(active):
            continue
        o = obs[active]
        llr = _dpois_vec(o, o) - _dpois_vec(o, exp_)
        maxLlr[active] = np.maximum(maxLlr[active], llr)
    return maxLlr


def _dpois_vec(x, lam):
    """log dpois for arrays, matching R's dpois (used only for integral x).

    Monte-Carlo output contains only a few dozen distinct event counts, so the
    scalar routine is evaluated once per distinct value and broadcast back.
    """
    from ._rmath import _dpois_raw
    x = np.atleast_1d(np.asarray(x, dtype=float))
    lam = np.broadcast_to(np.asarray(lam, dtype=float), x.shape)
    pairs = np.stack([x.ravel(), lam.ravel()], axis=1)
    uniq, inv = np.unique(pairs, axis=0, return_inverse=True)
    vals = np.array([_dpois_raw(float(a), float(b), True) for a, b in uniq])
    return vals[inv].reshape(x.shape)


def _poisson_scalar(rng, groupSizes, minimumEvents, sampleSize, nullMean, nullSd):
    G = groupSizes.size
    values = np.zeros(sampleSize, dtype=float)
    for i in range(sampleSize):
        maxLlr = 0.0
        observed = 0.0
        expected = 0.0
        systematicError = np.exp(_rnorm1(rng, nullMean, nullSd))
        for j in range(G):
            expected += groupSizes[j]
            observed += rng.rpois(groupSizes[j] * systematicError)
            if observed >= minimumEvents and observed >= expected:
                llr = (dpois(observed, observed, log=True)
                       - dpois(observed, expected, log=True))
                if llr > maxLlr:
                    maxLlr = llr
        values[i] = maxLlr
    return values


def _rnorm1(rng, mu, sigma):
    """R's scalar ``rnorm``: consumes no draw when ``sigma == 0``."""
    if sigma == 0 or not np.isfinite(mu):
        return mu
    return mu + sigma * rng.norm_rand()


def sampleBinomialMaxLrr(groupSizes, p, minimumEvents, sampleSize, nullMean, nullSd):
    """Port of the C++ ``sampleBinomialMaxLrr``."""
    rng = get_generator()
    groupSizes = np.atleast_1d(np.asarray(groupSizes, dtype=float))
    sampleSize = int(sampleSize)

    if nullSd == 0:
        se = np.exp(nullMean)
        prob = p * se / (1 + p * (se - 1))
        n_round = np.floor(groupSizes + 0.5)
        if 0 < prob < 1 and np.all(n_round * min(prob, 1 - prob) < 30):
            return _binomial_fast(rng, groupSizes, n_round, prob, p,
                                  minimumEvents, sampleSize, nullMean, nullSd)

    return _binomial_scalar(rng, groupSizes, p, minimumEvents, sampleSize,
                            nullMean, nullSd)


def _binomial_fast(rng, groupSizes, n_round, prob, p, minimumEvents, sampleSize,
                   nullMean, nullSd):
    G = groupSizes.size
    draw_cols = np.flatnonzero(n_round > 0)
    tables = {j: _binom_pmf_table(n_round[j], prob) for j in draw_cols}
    flip = prob > 0.5

    with _StateGuard(rng) as guard:
        u = rng.unif_rand_n(sampleSize * draw_cols.size).reshape(sampleSize, draw_cols.size)
        counts = np.zeros((sampleSize, G), dtype=float)
        for c, j in enumerate(draw_cols):
            idx = np.searchsorted(tables[j], u[:, c], side="right")
            if np.any(idx > 110):
                guard.restore()
                return _binomial_scalar(rng, groupSizes, p, minimumEvents,
                                        sampleSize, nullMean, nullSd)
            counts[:, j] = (n_round[j] - idx) if flip else idx

    cumulativeObserved = np.cumsum(counts, axis=1)
    cumulativeCount = np.cumsum(groupSizes)[None, :]
    return _max_llr_binomial(cumulativeObserved, cumulativeCount, p, minimumEvents)


def _max_llr_binomial(cumObs, cumCount, p, minimumEvents):
    maxLlr = np.zeros(cumObs.shape[0], dtype=float)
    for j in range(cumObs.shape[1]):
        obs = cumObs[:, j]
        cnt = cumCount[0, j]
        if cnt == 0:
            continue
        active = (obs >= minimumEvents) & (obs >= cnt * p)
        if not np.any(active):
            continue
        o = obs[active]
        llr = (_dbinom_vec(o, cnt, o / cnt) - _dbinom_vec(o, cnt, p))
        maxLlr[active] = np.maximum(maxLlr[active], llr)
    return maxLlr


def _dbinom_vec(x, n, prob):
    """log dbinom for arrays, evaluated once per distinct (x, prob) pair."""
    from ._rmath import _dbinom_raw
    x = np.atleast_1d(np.asarray(x, dtype=float))
    prob = np.broadcast_to(np.asarray(prob, dtype=float), x.shape)
    pairs = np.stack([x.ravel(), prob.ravel()], axis=1)
    uniq, inv = np.unique(pairs, axis=0, return_inverse=True)
    vals = np.array([_dbinom_raw(float(a), float(n), float(b), 1.0 - float(b), True)
                     for a, b in uniq])
    return vals[inv].reshape(x.shape)


def _binomial_scalar(rng, groupSizes, p, minimumEvents, sampleSize, nullMean, nullSd):
    G = groupSizes.size
    values = np.zeros(sampleSize, dtype=float)
    for i in range(sampleSize):
        maxLlr = 0.0
        cumulativeObserved = 0.0
        cumulativeCount = 0.0
        systematicError = np.exp(_rnorm1(rng, nullMean, nullSd))
        prob = p * systematicError / (1 + p * (systematicError - 1))
        for j in range(G):
            cumulativeCount += groupSizes[j]
            cumulativeObserved += rng.rbinom(groupSizes[j], prob)
            if (cumulativeObserved >= minimumEvents
                    and cumulativeObserved >= cumulativeCount * p):
                llr = (dbinom(cumulativeObserved, cumulativeCount,
                              cumulativeObserved / cumulativeCount, log=True)
                       - dbinom(cumulativeObserved, cumulativeCount, p, log=True))
                if llr > maxLlr:
                    maxLlr = llr
        values[i] = maxLlr
    return values


def samplePoissonRegressionMaxLrr(groupSizes, z, minimumEvents, sampleSize):
    """Port of the C++ ``samplePoissonRegressionMaxLrr``."""
    rng = get_generator()
    groupSizes = np.atleast_1d(np.asarray(groupSizes, dtype=float))
    sampleSize = int(sampleSize)

    lambda1 = groupSizes / (z + 1)
    lambda2 = lambda1 * z
    if np.all(lambda1 < 10) and np.all(lambda2 < 10):
        return _pois_reg_fast(rng, groupSizes, lambda1, lambda2, z,
                              minimumEvents, sampleSize)
    return _pois_reg_scalar(rng, groupSizes, lambda1, lambda2, z,
                            minimumEvents, sampleSize)


def _pois_reg_fast(rng, groupSizes, lambda1, lambda2, z, minimumEvents, sampleSize):
    G = groupSizes.size
    # per iteration the C++ draws rpois(lambda1_j) then rpois(lambda2_j) for each j
    cols = []
    for j in range(G):
        if lambda1[j] > 0:
            cols.append((j, 1, _pois_cdf_table(lambda1[j])))
        if lambda2[j] > 0:
            cols.append((j, 2, _pois_cdf_table(lambda2[j])))

    with _StateGuard(rng) as guard:
        u = rng.unif_rand_n(sampleSize * len(cols)).reshape(sampleSize, len(cols))
        c1 = np.zeros((sampleSize, G), dtype=float)
        c2 = np.zeros((sampleSize, G), dtype=float)
        for c, (j, which, table) in enumerate(cols):
            idx = np.searchsorted(table, u[:, c], side="left")
            if np.any(idx >= 36):
                guard.restore()
                return _pois_reg_scalar(rng, groupSizes, lambda1, lambda2, z,
                                        minimumEvents, sampleSize)
            (c1 if which == 1 else c2)[:, j] = idx

    observed1 = np.cumsum(c1, axis=1)
    observed2 = np.cumsum(c2, axis=1)
    maxLlr = np.zeros(sampleSize, dtype=float)
    for j in range(G):
        o1, o2 = observed1[:, j], observed2[:, j]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(o1 > 0, o2 / np.where(o1 > 0, o1, 1), np.inf)
        active = (o1 >= minimumEvents) & (ratio < z)
        if not np.any(active):
            continue
        a1, a2 = o1[active], o2[active]
        expected1 = (a1 + a2) / (z + 1)
        llr = ((_dpois_vec(a1, a1) + _dpois_vec(a2, a2))
               - (_dpois_vec(a1, expected1) + _dpois_vec(a2, expected1 * z)))
        maxLlr[active] = np.maximum(maxLlr[active], llr)
    return maxLlr


def _pois_reg_scalar(rng, groupSizes, lambda1, lambda2, z, minimumEvents, sampleSize):
    G = groupSizes.size
    values = np.zeros(sampleSize, dtype=float)
    for i in range(sampleSize):
        maxLlr = 0.0
        observed1 = 0.0
        observed2 = 0.0
        for j in range(G):
            observed1 += rng.rpois(lambda1[j])
            observed2 += rng.rpois(lambda2[j])
            if observed1 >= minimumEvents and observed2 / observed1 < z:
                expected1 = (observed1 + observed2) / (z + 1)
                llr = ((dpois(observed1, observed1, log=True)
                        + dpois(observed2, observed2, log=True))
                       - (dpois(observed1, expected1, log=True)
                          + dpois(observed2, expected1 * z, log=True)))
                if llr > maxLlr:
                    maxLlr = llr
        values[i] = maxLlr
    return values


# --------------------------------------------------------------------------
# Turning sampled LLRs into a critical value
# --------------------------------------------------------------------------

def _pick_critical_value(values, alpha, sampleSize):
    """R's shared tail: sort descending, drop duplicates, take the least
    conservative value whose alpha is still below the requested one."""
    order = np.argsort(-values, kind="stable")
    values = values[order]
    alphas = np.arange(1, sampleSize + 1, dtype=float) / sampleSize
    # R's !duplicated(): keep the first occurrence of each distinct value
    keep = np.empty(values.size, dtype=bool)
    keep[0] = True
    keep[1:] = values[1:] != values[:-1]
    alphas = alphas[keep]
    values = values[keep]
    below = np.flatnonzero(alphas < alpha)
    if below.size == 0:
        raise ValueError("No critical value with an alpha below the one specified")
    pick = below.max()
    print(f"Selected alpha: {alphas[pick]:0.3f} "
          f"(least conservative value below {alpha})")
    result = float(values[pick])
    return result, float(alphas[pick])


class CriticalValue(float):
    """A critical value carrying the selected alpha, mirroring R's ``alpha`` attribute."""

    def __new__(cls, value, alpha):
        obj = super().__new__(cls, value)
        obj.alpha = alpha
        return obj

    @property
    def attributes(self):
        return {"alpha": self.alpha}


def _validate(groupSizes, minimumEvents, alpha, sampleSize):
    groupSizes = np.atleast_1d(np.asarray(groupSizes, dtype=float))
    if np.any(groupSizes < 0):
        raise ValueError("Group sizes should be positive")
    if minimumEvents < 0 or minimumEvents != round(minimumEvents):
        raise ValueError("Minimum events should be a positive integer")
    if alpha <= 0 or alpha >= 1:
        raise ValueError("Alpha should be a number between 0 and 1")
    if sampleSize < 0 or sampleSize != round(sampleSize):
        raise ValueError("sampleSize should be a positive integer")
    return groupSizes


def computeCvPoisson(groupSizes, minimumEvents=1, alpha=0.05, sampleSize=int(1e6),
                     nullMean=0.0, nullSd=0.0) -> CriticalValue:
    """Compute critical values for Poisson data.

    Obtains critical values for the group and continuous sequential MaxSPRT test
    with Poisson data, using a Wald type upper boundary (flat with respect to the
    likelihood ratio function) and a pre-specified upper limit on the sample
    size.

    It is often not possible to select a critical value corresponding to the
    exact alpha specified; this function selects the least conservative critical
    value whose alpha is below the one specified, so the sequential analysis is
    conservative.

    This is a Monte-Carlo re-implementation of ``Sequential::CV.Poisson``.

    Parameters
    ----------
    groupSizes : array_like
        Expected number of events under H0 for each test.
    minimumEvents : int
        Minimum number of events needed before the null can be rejected.
    alpha : float
        Significance level (type 1 error probability).
    sampleSize : int
        Sample size for the Monte-Carlo simulation.
    nullMean, nullSd : float
        Mean and standard deviation of the empirical null distribution.

    Returns
    -------
    CriticalValue
        The critical value; ``.alpha`` reports the alpha actually selected.
    """
    groupSizes = _validate(groupSizes, minimumEvents, alpha, sampleSize)
    values = samplePoissonMaxLrr(groupSizes, minimumEvents, sampleSize, nullMean, nullSd)
    if values.max() == 0:
        warnings.warn("Unable to find a critical value that could lead to "
                      "rejection of the null", stacklevel=2)
        return CriticalValue(np.inf, 0)
    value, sel = _pick_critical_value(values, alpha, int(sampleSize))
    return CriticalValue(value, sel)


def computeCvBinomial(groupSizes, z, minimumEvents=1, alpha=0.05, sampleSize=int(1e6),
                      nullMean=0.0, nullSd=0.0) -> CriticalValue:
    """Compute critical values for Binomial data.

    Monte-Carlo re-implementation of ``Sequential::CV.Binomial``.

    Parameters
    ----------
    groupSizes : array_like
        Expected number of events under H0 for each test.
    z : float
        For a matched case-control analysis, the number of controls matched to
        each case under the null hypothesis.  For a self-controlled analysis,
        the control time divided by the time at risk.
    minimumEvents : int
        Minimum number of events needed before the null can be rejected.
    alpha : float
        Significance level.
    sampleSize : int
        Monte-Carlo sample size.
    nullMean, nullSd : float
        Parameters of the empirical null distribution.

    Returns
    -------
    CriticalValue
    """
    groupSizes = _validate(groupSizes, minimumEvents, alpha, sampleSize)
    if z < 0:
        raise ValueError("Z should be positive")
    p = 1 / (1 + z)
    values = sampleBinomialMaxLrr(groupSizes, p, minimumEvents, sampleSize,
                                  nullMean, nullSd)
    if values.max() == 0:
        warnings.warn("Unable to find a critical value that could lead to "
                      "rejection of the null", stacklevel=2)
        return CriticalValue(np.inf, 0)
    value, sel = _pick_critical_value(values, alpha, int(sampleSize))
    return CriticalValue(value, sel)


def computeCvPoissonRegression(groupSizes, z, minimumEvents=1, alpha=0.05,
                               sampleSize=int(1e6)) -> CriticalValue:
    """Compute critical values for Poisson regression data.

    Parameters
    ----------
    groupSizes : array_like
        Expected number of events under H0 for each test.
    z : float
        Control time divided by the time at risk.
    minimumEvents : int
        Minimum number of events needed before the null can be rejected.
    alpha : float
        Significance level.
    sampleSize : int
        Monte-Carlo sample size.

    Returns
    -------
    CriticalValue
    """
    groupSizes = _validate(groupSizes, minimumEvents, alpha, sampleSize)
    if z < 0:
        raise ValueError("Z should be positive")
    values = samplePoissonRegressionMaxLrr(groupSizes, z, minimumEvents, sampleSize)
    value, sel = _pick_critical_value(values, alpha, int(sampleSize))
    return CriticalValue(value, sel)
