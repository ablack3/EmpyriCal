"""Faithful ports of the R optimisers used by EmpiricalCalibration.

``fitNull``, ``fitNullNonNormalLl``, ``fitMcmcNull`` and ``calibrateLlr`` all call
``stats::optim`` with its default Nelder-Mead method; ``fitSystematicErrorModel``
calls it with ``method = "BFGS"`` and ``hessian = TRUE``.  R's optimisers are not
the same algorithms as SciPy's, and they stop on different criteria, so calling
``scipy.optimize.minimize`` would give answers that differ in the 3rd or 4th
decimal.  The routines below are line-by-line ports of ``nmmin``, ``vmmin`` and
``optimhess`` from R's ``src/appl/optim.c`` / ``src/main/optim.c`` so that the
Python package returns the same numbers as the R package.

The same reasoning applies to ``uniroot`` (R's ``zeroin``), ported here as well.
"""

from __future__ import annotations

import numpy as np

DBL_EPSILON = np.finfo(float).eps
BIG = 1.0e35  # `big` in nmmin


def _as_array(x) -> np.ndarray:
    return np.asarray(x, dtype=float).ravel()


def nmmin(par, fn, *, abstol=-np.inf, intol=None, alpha=1.0, beta=0.5, gamma=2.0,
          maxit=500, fnscale=1.0, parscale=None, trace=False):
    """R's Nelder-Mead (``src/appl/optim.c:nmmin``).

    Returns ``(par, value, fncount, convergence)`` where ``par`` is on the
    user's scale.  ``fn`` is evaluated on the user's scale; internally R works
    with ``par / parscale`` and objective ``fn / fnscale``.
    """
    if intol is None:
        intol = np.sqrt(DBL_EPSILON)  # `reltol` default in optim()
    Bvec = _as_array(par)
    n = Bvec.size
    parscale = np.ones(n) if parscale is None else _as_array(parscale)
    if parscale.size == 1:
        parscale = np.repeat(parscale, n)

    def fminfn(scaled):
        return fn(scaled * parscale) / fnscale

    # R works in the scaled space
    Bvec = Bvec / parscale

    fail = 0
    if maxit <= 0:
        return Bvec * parscale, fminfn(Bvec) * fnscale, 0, 0

    f = fminfn(Bvec)
    if not np.isfinite(f):
        raise ValueError("function cannot be evaluated at initial parameters")
    funcount = 1
    convtol = intol * (abs(f) + intol)
    n1 = n + 1
    C = n + 2
    # P is n1 x (n+2) in R (`matrix(n, n+1)` allocates (n+1) x (n+2))
    P = np.zeros((n1, n1 + 1))
    P[n1 - 1, 0] = f
    P[:n, 0] = Bvec

    L = 1
    size = 0.0

    step = 0.0
    for i in range(n):
        if 0.1 * abs(Bvec[i]) > step:
            step = 0.1 * abs(Bvec[i])
    if step == 0.0:
        step = 0.1

    # BUILD the initial simplex (done once)
    for j in range(2, n1 + 1):
        P[:n, j - 1] = Bvec
        trystep = step
        while P[j - 2, j - 1] == Bvec[j - 2]:
            P[j - 2, j - 1] = Bvec[j - 2] + trystep
            trystep *= 10
        size += trystep
    oldsize = size
    calcvert = True

    while True:  # do { ... } while (funcount <= maxit)
        if calcvert:
            for j in range(n1):
                if j + 1 != L:
                    Bvec = P[:n, j].copy()
                    f = fminfn(Bvec)
                    if not np.isfinite(f):
                        f = BIG
                    funcount += 1
                    P[n1 - 1, j] = f
            calcvert = False

        VL = P[n1 - 1, L - 1]
        VH = VL
        H = L
        for j in range(1, n1 + 1):
            if j != L:
                f = P[n1 - 1, j - 1]
                if f < VL:
                    L = j
                    VL = f
                if f > VH:
                    H = j
                    VH = f
        if VH <= VL + convtol or VL <= abstol:
            break

        # centroid of all vertices except the high one
        for i in range(n):
            temp = -P[i, H - 1]
            for j in range(n1):
                temp += P[i, j]
            P[i, C - 1] = temp / n
        # REFLECTION
        for i in range(n):
            Bvec[i] = (1.0 + alpha) * P[i, C - 1] - alpha * P[i, H - 1]
        f = fminfn(Bvec)
        if not np.isfinite(f):
            f = BIG
        funcount += 1
        VR = f
        if VR < VL:
            # EXTENSION
            P[n1 - 1, C - 1] = f
            for i in range(n):
                f = gamma * Bvec[i] + (1 - gamma) * P[i, C - 1]
                P[i, C - 1] = Bvec[i]
                Bvec[i] = f
            f = fminfn(Bvec)
            if not np.isfinite(f):
                f = BIG
            funcount += 1
            if f < VR:
                P[:n, H - 1] = Bvec
                P[n1 - 1, H - 1] = f
            else:
                P[:n, H - 1] = P[:n, C - 1]
                P[n1 - 1, H - 1] = VR
        else:
            # HI-REDUCTION / LO-REDUCTION
            if VR < VH:
                P[:n, H - 1] = Bvec
                P[n1 - 1, H - 1] = VR
            for i in range(n):
                Bvec[i] = (1 - beta) * P[i, H - 1] + beta * P[i, C - 1]
            f = fminfn(Bvec)
            if not np.isfinite(f):
                f = BIG
            funcount += 1
            if f < P[n1 - 1, H - 1]:
                P[:n, H - 1] = Bvec
                P[n1 - 1, H - 1] = f
            elif VR >= VH:
                # SHRINK toward the low vertex
                calcvert = True
                size = 0.0
                for j in range(n1):
                    if j + 1 != L:
                        for i in range(n):
                            P[i, j] = beta * (P[i, j] - P[i, L - 1]) + P[i, L - 1]
                            size += abs(P[i, j] - P[i, L - 1])
                if size < oldsize:
                    oldsize = size
                else:
                    if trace:
                        print("Polytope size measure not decreasing in shrink")
                    fail = 10
                    break
        if funcount > maxit:
            break

    if trace:
        print("Exiting from Nelder Mead minimizer")
        print(f"    {funcount} function evaluations used")
    Fmin = P[n1 - 1, L - 1]
    Bvec = P[:n, L - 1].copy()
    convergence = 1 if funcount > maxit else fail
    return Bvec * parscale, Fmin * fnscale, funcount, convergence


def _fmingr(fn, scaled, parscale, fnscale, ndeps):
    """R's numerical gradient (``optim.c:fmingr``) for the no-``gr`` case."""
    n = scaled.size
    df = np.empty(n)
    x = scaled.copy()
    for i in range(n):
        eps = ndeps[i]
        x[i] = scaled[i] + eps
        s1 = fn(x * parscale) / fnscale
        x[i] = scaled[i] - eps
        s2 = fn(x * parscale) / fnscale
        df[i] = (s1 - s2) / (2 * eps)
        if not np.isfinite(df[i]):
            raise ValueError(
                f"non-finite finite-difference value [{i + 1}]")
        x[i] = scaled[i]
    return df


def vmmin(par, fn, *, abstol=-np.inf, reltol=None, maxit=100, fnscale=1.0,
          parscale=None, ndeps=1e-3, report=10, trace=False):
    """R's BFGS (``src/appl/optim.c:vmmin``) with R's finite-difference gradient."""
    if reltol is None:
        reltol = np.sqrt(DBL_EPSILON)
    b = _as_array(par)
    n = b.size
    parscale = np.ones(n) if parscale is None else _as_array(parscale)
    if parscale.size == 1:
        parscale = np.repeat(parscale, n)
    ndeps = np.repeat(float(ndeps), n) if np.ndim(ndeps) == 0 else _as_array(ndeps)

    def fminfn(scaled):
        val = fn(scaled * parscale) / fnscale
        return val

    def fmingr(scaled):
        return _fmingr(fn, scaled, parscale, fnscale, ndeps)

    b = b / parscale
    stepredn = 0.2
    acctol = 0.0001
    reltest = 10.0

    if maxit <= 0:
        return b * parscale, fminfn(b) * fnscale, 0, 0, 0

    B = np.zeros((n, n))
    f = fminfn(b)
    if not np.isfinite(f):
        raise ValueError("initial value in 'vmmin' is not finite")
    Fmin = f
    funcount = gradcount = 1
    g = fmingr(b)
    iter_ = 1
    ilast = gradcount
    fail = 0
    count = 0

    X = np.zeros(n)
    c = np.zeros(n)
    t = np.zeros(n)

    while True:
        if ilast == gradcount:
            B[:] = 0.0
            np.fill_diagonal(B, 1.0)
        X[:] = b
        c[:] = g
        gradproj = 0.0
        for i in range(n):
            s = -np.dot(B[i, : i + 1], g[: i + 1])
            if i < n - 1:
                s -= np.dot(B[i + 1:, i], g[i + 1:])
            t[i] = s
            gradproj += s * g[i]

        if gradproj < 0.0:
            steplength = 1.0
            accpoint = False
            while True:
                count = 0
                for i in range(n):
                    b[i] = X[i] + steplength * t[i]
                    if reltest + X[i] == reltest + b[i]:
                        count += 1
                if count < n:
                    f = fminfn(b)
                    funcount += 1
                    accpoint = np.isfinite(f) and (
                        f <= Fmin + gradproj * steplength * acctol)
                    if not accpoint:
                        steplength *= stepredn
                if not (count != n and not accpoint):
                    break
            enough = (f > abstol) and abs(f - Fmin) > reltol * (abs(Fmin) + reltol)
            if not enough:
                count = n
                Fmin = f
            if count < n:
                Fmin = f
                g = fmingr(b)
                gradcount += 1
                iter_ += 1
                D1 = 0.0
                for i in range(n):
                    t[i] = steplength * t[i]
                    c[i] = g[i] - c[i]
                    D1 += t[i] * c[i]
                if D1 > 0:
                    D2 = 0.0
                    for i in range(n):
                        s = np.dot(B[i, : i + 1], c[: i + 1])
                        if i < n - 1:
                            s += np.dot(B[i + 1:, i], c[i + 1:])
                        X[i] = s
                        D2 += s * c[i]
                    D2 = 1.0 + D2 / D1
                    for i in range(n):
                        for j in range(i + 1):
                            B[i, j] += (D2 * t[i] * t[j] - X[i] * t[j] - t[i] * X[j]) / D1
                else:
                    ilast = gradcount
            else:
                if ilast < gradcount:
                    count = 0
                    ilast = gradcount
        else:
            count = 0
            if ilast == gradcount:
                count = n
            else:
                ilast = gradcount
        if iter_ >= maxit:
            break
        if gradcount - ilast > 2 * n:
            ilast = gradcount
        if not (count != n or ilast != gradcount):
            break

    fail = 0 if iter_ < maxit else 1
    return b * parscale, Fmin * fnscale, funcount, gradcount, fail


def optimhess(par, fn, *, fnscale=1.0, parscale=None, ndeps=1e-3):
    """R's ``optimhess`` for the case where no analytic gradient is supplied.

    R differences its *numerical* gradient, then symmetrises:
    ``H <- 0.5 * (H + t(H))``, and rescales by ``fnscale / (parscale_i * parscale_j)``.
    """
    x = _as_array(par)
    n = x.size
    parscale = np.ones(n) if parscale is None else _as_array(parscale)
    if parscale.size == 1:
        parscale = np.repeat(parscale, n)
    ndeps = np.repeat(float(ndeps), n) if np.ndim(ndeps) == 0 else _as_array(ndeps)

    scaled = x / parscale

    def fmingr(s):
        return _fmingr(fn, s, parscale, fnscale, ndeps)

    H = np.empty((n, n))
    p = scaled.copy()
    for i in range(n):
        eps = ndeps[i]
        p[i] = scaled[i] + eps
        df1 = fmingr(p)
        p[i] = scaled[i] - eps
        df2 = fmingr(p)
        for j in range(n):
            H[i, j] = fnscale * (df1[j] - df2[j]) / (2 * eps * parscale[i] * parscale[j])
        p[i] = scaled[i]
    # symmetrise exactly as R does
    return 0.5 * (H + H.T)


def optim(par, fn, method="Nelder-Mead", hessian=False, control=None):
    """A thin stand-in for ``stats::optim`` covering the calls this package makes.

    Returns a dict with keys ``par``, ``value``, ``counts``, ``convergence`` and
    (when requested) ``hessian`` -- mirroring R's returned list.
    """
    control = dict(control or {})
    fnscale = control.get("fnscale", 1.0)
    parscale = control.get("parscale", None)
    ndeps = control.get("ndeps", 1e-3)
    par = _as_array(par)

    if method == "Nelder-Mead":
        maxit = control.get("maxit", 500)
        p, value, fncount, conv = nmmin(
            par, fn,
            abstol=control.get("abstol", -np.inf),
            intol=control.get("reltol", None),
            alpha=control.get("alpha", 1.0),
            beta=control.get("beta", 0.5),
            gamma=control.get("gamma", 2.0),
            maxit=maxit, fnscale=fnscale, parscale=parscale,
            trace=control.get("trace", False),
        )
        counts = (fncount, None)
    elif method == "BFGS":
        maxit = control.get("maxit", 100)
        p, value, fncount, grcount, conv = vmmin(
            par, fn,
            abstol=control.get("abstol", -np.inf),
            reltol=control.get("reltol", None),
            maxit=maxit, fnscale=fnscale, parscale=parscale, ndeps=ndeps,
            trace=control.get("trace", False),
        )
        counts = (fncount, grcount)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported optim method: {method}")

    out = {"par": p, "value": value, "counts": counts, "convergence": conv}
    if hessian:
        out["hessian"] = optimhess(p, fn, fnscale=fnscale, parscale=parscale, ndeps=ndeps)
    return out


def uniroot(f, interval, *, tol=None, maxiter=1000, extendInt="no", args=()):
    """R's ``uniroot`` -- Brent's ``zeroin`` from ``src/library/stats/src/zeroin.c``.

    Default ``tol`` is R's ``.Machine$double.eps^0.25``.
    """
    if tol is None:
        tol = DBL_EPSILON ** 0.25
    ax, bx = float(interval[0]), float(interval[1])
    fa = f(ax, *args)
    fb = f(bx, *args)
    if fa == 0.0:
        return {"root": ax, "f.root": 0.0, "iter": 0, "estim.prec": 0.0}
    if fb == 0.0:
        return {"root": bx, "f.root": 0.0, "iter": 0, "estim.prec": 0.0}
    if fa * fb > 0:
        raise ValueError("f() values at end points not of opposite sign")

    a, b, c = ax, bx, ax
    fc = fa
    maxit = maxiter + 1
    it = 0
    while maxit > 0:
        prev_step = b - a
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tol_act = 2 * DBL_EPSILON * abs(b) + tol / 2
        new_step = (c - b) / 2
        if abs(new_step) <= tol_act or fb == 0.0:
            return {"root": b, "f.root": fb, "iter": maxiter - maxit + 1,
                    "estim.prec": abs(c - b)}
        if abs(prev_step) >= tol_act and abs(fa) > abs(fb):
            cb = c - b
            if a == c:
                t1 = fb / fa
                p = cb * t1
                q = 1.0 - t1
            else:
                q = fa / fc
                t1 = fb / fc
                t2 = fb / fa
                p = t2 * (cb * q * (q - t1) - (b - a) * (t1 - 1.0))
                q = (q - 1.0) * (t1 - 1.0) * (t2 - 1.0)
            if p > 0.0:
                q = -q
            else:
                p = -p
            if (p < (0.75 * cb * q - abs(tol_act * q) / 2)
                    and p < abs(prev_step * q / 2)):
                new_step = p / q
        if abs(new_step) < tol_act:
            new_step = tol_act if new_step > 0 else -tol_act
        a, fa = b, fb
        b += new_step
        fb = f(b, *args)
        if (fb > 0) == (fc > 0):
            c, fc = a, fa
        maxit -= 1
        it += 1
    return {"root": b, "f.root": fb, "iter": -1, "estim.prec": abs(c - b)}
