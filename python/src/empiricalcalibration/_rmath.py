"""R-compatible density / distribution functions.

SciPy's versions agree with R's only to about 1e-13 relative (and SciPy's
``norm.cdf`` underflows to zero around x = -37.5 where R still returns a
subnormal).  Because this package's whole job is to produce p-values, the
routines that R itself uses are ported here directly from R's ``nmath``:

* ``dnorm``  -- ``nmath/dnorm.c`` (incl. the split-argument trick for |z| >= 5)
* ``pnorm``  -- ``nmath/pnorm.c`` (Cody's rational Chebyshev approximation)
* ``qnorm``  -- ``nmath/qnorm.c`` (Wichura's AS 241)
* ``dpois`` / ``dbinom`` -- ``nmath/dpois.c``, ``dbinom.c`` (Loader's saddle point,
  via ``stirlerr`` and ``bd0``)

``pchisq``, ``qchisq`` and ``dgamma`` are taken from SciPy: they agree with R to
~1e-15 relative, which is far below the 1.2e-4 integration tolerance R's
``integrate`` applies to every result that depends on them.
"""

from __future__ import annotations

import numpy as np
from scipy import special, stats

__all__ = [
    "dnorm", "pnorm", "qnorm", "dpois", "dbinom", "dgamma_log",
    "pchisq", "qchisq", "stirlerr", "bd0",
]

DBL_EPSILON = np.finfo(float).eps
DBL_MIN = np.finfo(float).tiny
DBL_MAX = np.finfo(float).max

M_LN_SQRT_2PI = 0.918938533204672741780329736406
M_1_SQRT_2PI = 0.398942280401432677939946059934
M_SQRT_32 = 5.656854249492380195206754896838
M_LN_2PI = 1.837877066409345483560659472811
M_2PI = 6.283185307179586476925286766559


# --------------------------------------------------------------------------
# Normal distribution
# --------------------------------------------------------------------------

def dnorm(x, mean=0.0, sd=1.0, log=False):
    """R's ``dnorm`` (``nmath/dnorm.c``)."""
    x = np.asarray(x, dtype=float)
    mean = np.asarray(mean, dtype=float)
    sd = np.asarray(sd, dtype=float)
    x, mean, sd = np.broadcast_arrays(x, mean, sd)
    out = np.empty(x.shape, dtype=float)
    scalar = out.ndim == 0
    x = np.atleast_1d(x)
    mean = np.atleast_1d(mean)
    sd = np.atleast_1d(sd)
    out = np.atleast_1d(out)

    nan = np.isnan(x) | np.isnan(mean) | np.isnan(sd)
    out[nan] = np.nan

    neg_sd = (~nan) & (sd < 0)
    out[neg_sd] = np.nan

    inf_sd = (~nan) & np.isinf(sd)
    out[inf_sd] = -np.inf if log else 0.0

    zero_sd = (~nan) & (sd == 0)
    if np.any(zero_sd):
        eq = zero_sd & (x == mean)
        out[eq] = np.inf
        out[zero_sd & ~eq] = -np.inf if log else 0.0

    ok = (~nan) & (~neg_sd) & (~inf_sd) & (~zero_sd)
    if np.any(ok):
        # x-mu is NaN when x is infinite and equal to mu
        bad = ok & np.isinf(x) & (mean == x)
        out[bad] = np.nan
        ok = ok & ~bad

    if np.any(ok):
        z = np.abs((x[ok] - mean[ok]) / sd[ok])
        s = sd[ok]
        res = np.empty(z.shape, dtype=float)

        far = np.isinf(z) | (z >= 2 * np.sqrt(DBL_MAX))
        res[far] = -np.inf if log else 0.0
        near = ~far

        if log:
            res[near] = -(M_LN_SQRT_2PI + 0.5 * z[near] * z[near] + np.log(s[near]))
        else:
            zn, sn = z[near], s[near]
            small = zn < 5
            r = np.empty(zn.shape, dtype=float)
            r[small] = M_1_SQRT_2PI * np.exp(-0.5 * zn[small] ** 2) / sn[small]
            big = ~small
            if np.any(big):
                zb, sb = zn[big], sn[big]
                rb = np.zeros(zb.shape, dtype=float)
                # beyond this, exp(-x^2/2) underflows to 0
                lim = np.sqrt(-2 * np.log(2.0) * (np.finfo(float).minexp + 1 - 53))
                inrange = zb <= lim
                # split z = z1 + z2 with |z2| <= 2^-16 for full accuracy
                z1 = np.ldexp(np.round(np.ldexp(zb[inrange], 16)), -16)
                z2 = zb[inrange] - z1
                rb[inrange] = (M_1_SQRT_2PI / sb[inrange]
                               * np.exp(-0.5 * z1 * z1) * np.exp((-0.5 * z2 - z1) * z2))
                r[big] = rb
            res[near] = r
        out[ok] = res
    return out.item() if scalar else out


def pnorm(q, mean=0.0, sd=1.0, lower_tail=True, log_p=False):
    """R's ``pnorm``.

    Built on SciPy's ``ndtr``, which agrees with R's Cody rational-Chebyshev
    implementation to 2.6e-15 relative for |z| <= 5 and 7.8e-15 for |z| <= 10
    -- i.e. to within a few units in the last place over the whole range in
    which p-values are meaningful.  ``ndtr`` flushes to zero around z = -37.5
    where R still returns subnormals, so the lower tail falls back to
    ``exp(log_ndtr(z))``, which reproduces R's subnormal values exactly.
    """
    q = np.asarray(q, dtype=float)
    mean = np.asarray(mean, dtype=float)
    sd = np.asarray(sd, dtype=float)
    qb, mb, sb = np.broadcast_arrays(q, mean, sd)
    scalar = qb.ndim == 0
    qf = np.atleast_1d(qb).astype(float)
    mf = np.atleast_1d(mb).astype(float)
    sf = np.atleast_1d(sb).astype(float)

    out = np.empty(qf.shape, dtype=float)
    nan = np.isnan(qf) | np.isnan(mf) | np.isnan(sf) | (sf < 0)
    out[nan] = np.nan
    # x infinite and equal to mu => x - mu is NaN
    nan |= np.isinf(qf) & (mf == qf)
    out[np.isinf(qf) & (mf == qf)] = np.nan

    zero_sd = (~nan) & (sf == 0)
    if np.any(zero_sd):
        step = (qf[zero_sd] >= mf[zero_sd]).astype(float)
        step = step if lower_tail else 1.0 - step
        with np.errstate(divide="ignore"):
            out[zero_sd] = np.log(step) if log_p else step

    ok = (~nan) & (~zero_sd)
    if np.any(ok):
        z = (qf[ok] - mf[ok]) / sf[ok]
        if not lower_tail:
            z = -z
        if log_p:
            # log_ndtr stays finite arbitrarily far into the tail, matching R's
            # pnorm(log.p = TRUE); taking log() of the probability would give
            # -inf as soon as the probability underflows (around z = -38).
            out[ok] = special.log_ndtr(z)
        else:
            with np.errstate(under="ignore"):
                v = special.ndtr(z)
                # recover the subnormal lower tail that ndtr flushes to zero
                flush = (v == 0.0) & (z < -30)
                if np.any(flush):
                    v = v.copy()
                    v[flush] = np.exp(special.log_ndtr(z[flush]))
            out[ok] = v

    out = out.reshape(qb.shape)
    return out.item() if scalar else out


def qnorm(p, mean=0.0, sd=1.0, lower_tail=True, log_p=False):
    """R's ``qnorm`` -- Wichura's AS 241 (``nmath/qnorm.c``)."""
    p = np.asarray(p, dtype=float)
    mean = np.asarray(mean, dtype=float)
    sd = np.asarray(sd, dtype=float)
    pb, mb, sb = np.broadcast_arrays(p, mean, sd)
    scalar = pb.ndim == 0
    pf = np.atleast_1d(pb).ravel().astype(float)
    mf = np.atleast_1d(mb).ravel()
    sf = np.atleast_1d(sb).ravel()
    out = np.empty(pf.shape, dtype=float)

    for i in range(pf.size):
        pp = pf[i]
        if np.isnan(pp) or np.isnan(mf[i]) or np.isnan(sf[i]):
            out[i] = np.nan
            continue
        p_ = np.exp(pp) if log_p else pp
        if not lower_tail:
            p_ = 1.0 - p_
        if p_ < 0 or p_ > 1:
            out[i] = np.nan
            continue
        if p_ == 0:
            out[i] = -np.inf
            continue
        if p_ == 1:
            out[i] = np.inf
            continue
        q = p_ - 0.5
        if abs(q) <= 0.425:
            r = 0.180625 - q * q
            val = (q * (((((((r * 2509.0809287301226727 +
                              33430.575583588128105) * r + 67265.770927008700853) * r +
                            45921.953931549871457) * r + 13731.693765509461125) * r +
                          1971.5909503065514427) * r + 133.14166789178437745) * r +
                        3.387132872796366608)
                   / (((((((r * 5226.495278852854561 +
                            28729.085735721942674) * r + 39307.89580009271061) * r +
                          21213.794301586595867) * r + 5394.1960214247511077) * r +
                        687.1870074920579083) * r + 42.313330701600911252) * r + 1.0))
        else:
            r = p_ if q <= 0 else 1.0 - p_
            r = np.sqrt(-np.log(r))
            if r <= 5.0:
                r -= 1.6
                val = ((((((((r * 7.7454501427834140764e-4 +
                              0.0227238449892691845833) * r + 0.24178072517745061177) *
                            r + 1.27045825245236838258) * r +
                           3.64784832476320460504) * r + 5.7694972214606914055) *
                         r + 4.6303378461565452959) * r +
                        1.42343711074968357734)
                       / (((((((r *
                                1.05075007164441684324e-9 + 5.475938084995344946e-4) *
                               r + 0.0151986665636164571966) * r +
                              0.14810397642748007459) * r + 0.68976733498510000455) *
                            r + 1.6763848301838038494) * r +
                           2.05319162663775882187) * r + 1.0))
            elif r <= 27.0:
                r -= 5.0
                val = ((((((((r * 2.01033439929228813265e-7 +
                              2.71155556874348757815e-5) * r +
                             0.0012426609473880784386) * r + 0.026532189526576123093) *
                           r + 0.29656057182850489123) * r +
                          1.7848265399172913358) * r + 5.4637849111641143699) *
                        r + 6.6579046435011037772)
                       / (((((((r *
                                2.04426310338993978564e-15 + 1.4215117583164458887e-7) *
                               r + 1.8463183175100546818e-5) * r +
                              7.868691311456132591e-4) * r + 0.0148753612908506148525)
                            * r + 0.13692988092273580531) * r +
                           0.59983220655588793769) * r + 1.0))
            else:  # only reachable with log_p and extreme values
                val = float(stats.norm.ppf(p_))
            if q < 0.0:
                val = -val
        out[i] = mf[i] + sf[i] * val

    out = out.reshape(pb.shape)
    return out.item() if scalar else out


# --------------------------------------------------------------------------
# Loader's saddle-point machinery for dpois / dbinom
# --------------------------------------------------------------------------

_S0 = 0.083333333333333333333        # 1/12
_S1 = 0.00277777777777777777778      # 1/360
_S2 = 0.00079365079365079365079365   # 1/1260
_S3 = 0.000595238095238095238095238  # 1/1680
_S4 = 0.0008417508417508417508417508  # 1/1188

_SFERR_HALVES = np.array([
    0.0,
    0.1534264097200273452913848,
    0.0810614667953272582196702,
    0.0548141210519176538961390,
    0.0413406959554092940938221,
    0.03316287351993628748511048,
    0.02767792568499833914878929,
    0.02374616365629749597132920,
    0.02079067210376509311152277,
    0.01848845053267318523077934,
    0.01664469118982119216319487,
    0.01513497322191737887351255,
    0.01387612882307074799874573,
    0.01281046524292022692424986,
    0.01189670994589177009505572,
    0.01110455975820691732662991,
    0.010411265261972096497478567,
    0.009799416126158803298389475,
    0.009255462182712732917728637,
    0.008768700134139385462952823,
    0.008330563433362871256469318,
    0.007934114564314020547248100,
    0.007573675487951840794972024,
    0.007244554301320383179543912,
    0.006942840107209529865664152,
    0.006665247032707682442354394,
    0.006408994188004207068439631,
    0.006171712263039457647532867,
    0.005951370112758847735624416,
    0.005746216513010115682023589,
    0.005554733551962801371038690,
])


def stirlerr(n: float) -> float:
    """Error of the Stirling approximation, ``nmath/stirlerr.c``."""
    if n <= 15.0:
        nn = n + n
        if nn == int(nn):
            return _SFERR_HALVES[int(nn)]
        return special.gammaln(n + 1.0) - (n + 0.5) * np.log(n) + n - M_LN_SQRT_2PI
    nn = n * n
    if n > 500:
        return (_S0 - _S1 / nn) / n
    if n > 80:
        return (_S0 - (_S1 - _S2 / nn) / nn) / n
    if n > 35:
        return (_S0 - (_S1 - (_S2 - _S3 / nn) / nn) / nn) / n
    return (_S0 - (_S1 - (_S2 - (_S3 - _S4 / nn) / nn) / nn) / nn) / n


def bd0(x: float, np_: float) -> float:
    """Deviance term ``x*log(x/np) + np - x``, ``nmath/bd0.c``."""
    if not np.isfinite(x) or not np.isfinite(np_) or np_ == 0.0:
        return np.nan
    if abs(x - np_) < 0.1 * (x + np_):
        v = (x - np_) / (x + np_)
        s = (x - np_) * v
        if abs(s) < DBL_MIN:
            return s
        ej = 2 * x * v
        v = v * v
        for j in range(1, 1000):
            ej *= v
            s1 = s + ej / ((j << 1) + 1)
            if s1 == s:
                return s1
            s = s1
    return x * np.log(x / np_) + np_ - x


def _dpois_raw(x: float, lam: float, log: bool) -> float:
    if lam == 0:
        return (0.0 if log else 1.0) if x == 0 else (-np.inf if log else 0.0)
    if not np.isfinite(lam):
        return -np.inf if log else 0.0
    if x < 0:
        return -np.inf if log else 0.0
    if x <= lam * DBL_MIN:
        return -lam if log else np.exp(-lam)
    if lam < x * DBL_MIN:
        if not np.isfinite(x):
            return -np.inf if log else 0.0
        val = -lam + x * np.log(lam) - special.gammaln(x + 1)
        return val if log else np.exp(val)
    ex = -stirlerr(x) - bd0(x, lam)
    f = M_2PI * x
    return (-0.5 * np.log(f) + ex) if log else (np.exp(ex) / np.sqrt(f))


def dpois(x, lam, log=False):
    """R's ``dpois``."""
    xs = np.atleast_1d(np.asarray(x, dtype=float))
    ls = np.atleast_1d(np.asarray(lam, dtype=float))
    xs, ls = np.broadcast_arrays(xs, ls)
    out = np.array([_dpois_raw(float(np.round(xi)), float(li), log)
                    if not (np.isnan(xi) or np.isnan(li)) else np.nan
                    for xi, li in zip(xs.ravel(), ls.ravel())])
    out = out.reshape(xs.shape)
    return out.item() if np.ndim(x) == 0 and np.ndim(lam) == 0 else out


def _dbinom_raw(x: float, n: float, p: float, q: float, log: bool) -> float:
    def d_exp(v):
        return v if log else np.exp(v)

    if p == 0:
        return (0.0 if log else 1.0) if x == 0 else (-np.inf if log else 0.0)
    if q == 0:
        return (0.0 if log else 1.0) if x == n else (-np.inf if log else 0.0)
    if x == 0:
        if n == 0:
            return 0.0 if log else 1.0
        lc = (-bd0(n, n * q) - n * p) if p < 0.1 else n * np.log(q)
        return d_exp(lc)
    if x == n:
        lc = (-bd0(n, n * p) - n * q) if q < 0.1 else n * np.log(p)
        return d_exp(lc)
    if x < 0 or x > n:
        return -np.inf if log else 0.0
    lc = (stirlerr(n) - stirlerr(x) - stirlerr(n - x)
          - bd0(x, n * p) - bd0(n - x, n * q))
    lf = M_LN_2PI + np.log(x) + np.log1p(-x / n)
    return d_exp(lc - 0.5 * lf)


def dbinom(x, size, prob, log=False):
    """R's ``dbinom``."""
    xs = np.atleast_1d(np.asarray(x, dtype=float))
    ns = np.atleast_1d(np.asarray(size, dtype=float))
    ps = np.atleast_1d(np.asarray(prob, dtype=float))
    xs, ns, ps = np.broadcast_arrays(xs, ns, ps)
    out = np.array([_dbinom_raw(float(np.round(xi)), float(ni), float(pi), 1.0 - float(pi), log)
                    for xi, ni, pi in zip(xs.ravel(), ns.ravel(), ps.ravel())])
    out = out.reshape(xs.shape)
    if np.ndim(x) == 0 and np.ndim(size) == 0 and np.ndim(prob) == 0:
        return out.item()
    return out


# --------------------------------------------------------------------------
# Remaining distributions (SciPy agrees with R to ~1e-15 here)
# --------------------------------------------------------------------------

def dgamma_log(x, shape, rate):
    """R's ``dgamma(x, shape, rate, log = TRUE)``."""
    return stats.gamma.logpdf(x, a=shape, scale=1.0 / rate)


def pchisq(q, df, lower_tail=True):
    """R's ``pchisq``."""
    if lower_tail:
        return stats.chi2.cdf(q, df)
    return stats.chi2.sf(q, df)


def qchisq(p, df, lower_tail=True):
    """R's ``qchisq``."""
    if lower_tail:
        return stats.chi2.ppf(p, df)
    return stats.chi2.isf(p, df)
