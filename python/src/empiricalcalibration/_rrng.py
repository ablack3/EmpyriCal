"""R's random number generator, ported so that ``set_seed(n)`` reproduces R's
``set.seed(n)`` stream exactly.

Several functions in this package are stochastic -- ``simulateControls``,
``simulateMaxSprtData``, ``fitMcmcNull``, ``compareEase`` and the three
``computeCv*`` Monte-Carlo routines.  Using NumPy's generator would make their
results merely *distributionally* the same as R's; using R's own generator makes
them *identical* given the same seed, which is what makes the port testable.

Ported from R's ``src/main/RNG.c`` (Mersenne-Twister + ``set.seed`` scrambling,
``R_unif_index`` rejection sampling) and ``src/nmath/`` (``snorm.c`` inversion,
``sexp.c``, ``rpois.c``, ``rbinom.c``).
"""

from __future__ import annotations

import os as _os

import numpy as np

from ._rmath import qnorm

__all__ = ["RRandom", "set_seed", "get_generator"]

_N = 624
_M = 397
_MATRIX_A = 0x9908B0DF
_UPPER_MASK = 0x80000000
_LOWER_MASK = 0x7FFFFFFF
_TEMPERING_MASK_B = 0x9D2C5680
_TEMPERING_MASK_C = 0xEFC60000

_I2_32M1 = 2.328306437080797e-10  # 1 / (2^32 - 1)
_BIG = 134217728.0  # 2^27, used by the inversion normal generator

_UINT32 = 0xFFFFFFFF

# q[k] = sum_{j=1..k+1} log(2)^j / j!   (R's sexp.c table)
_EXP_Q = np.array([
    0.6931471805599453, 0.9333736875190459, 0.9888777961838675,
    0.9984959252914960040, 0.9998292811061389, 0.9999833164100727,
    0.99999855082196313338, 0.99999988631034542, 0.99999999193381246,
    0.99999999947158118, 0.99999999996714100, 0.99999999999808990,
    0.99999999999989410, 0.99999999999999450, 0.99999999999999970,
    0.99999999999999999,
])

_FACT = np.array([1., 1., 2., 6., 24., 120., 720., 5040., 40320., 362880.])

# rpois polynomial coefficients (Ahrens & Dieter 1982)
_A0, _A1, _A2, _A3 = -0.5, 0.3333333, -0.2500068, 0.2000118
_A4, _A5, _A6, _A7 = -0.1661269, 0.1421878, -0.1384794, 0.1250060
_ONE_7 = 0.1428571428571428571
_ONE_12 = 0.0833333333333333333
_ONE_24 = 0.0416666666666666667


class RRandom:
    """R's Mersenne-Twister stream."""

    def __init__(self, seed: int | None = None):
        self._mt = np.zeros(_N, dtype=np.uint32)
        self._mti = _N + 1
        # buffer of pre-generated uniforms, refilled 624 at a time
        self._buf = np.empty(0, dtype=np.float64)
        self._bpos = 0
        # persistent state of rpois / rbinom, exactly as R keeps it static
        self._pois_muprev = -1.0
        self._pois_muprev2 = -1.0
        self._pois_pp = np.zeros(36)
        self._pois_l = 0
        self._pois_m = 0
        self._pois_p = self._pois_q = self._pois_p0 = 0.0
        self._pois_s = self._pois_d = self._pois_big_l = 0.0
        self._pois_omega = 0.0
        self._pois_b1 = self._pois_b2 = 0.0
        self._pois_c = self._pois_c0 = self._pois_c1 = 0.0
        self._pois_c2 = self._pois_c3 = 0.0
        self._binom_nsave = -1
        self._binom_psave = -1.0
        if seed is None:
            # R seeds from the clock when .Random.seed does not yet exist; do the
            # same rather than leave an all-zero state, which would degenerate.
            seed = int.from_bytes(_os.urandom(4), "little")
        self.set_seed(seed)

    # ---------------------------------------------------------------- seeding
    def set_seed(self, seed: int) -> None:
        """Equivalent to R's ``set.seed(seed)`` with the default RNG kind."""
        s = int(seed) & _UINT32
        # R applies 50 rounds of initial scrambling before filling i_seed
        for _ in range(50):
            s = (69069 * s + 1) & _UINT32
        i_seed = np.empty(_N + 1, dtype=np.uint32)
        for j in range(_N + 1):
            s = (69069 * s + 1) & _UINT32
            i_seed[j] = s
        # FixupSeeds: i_seed[0] is mti and is forced to N on initialisation
        self._mti = _N
        self._mt = i_seed[1:].copy()
        self._buf = np.empty(0, dtype=np.float64)
        self._bpos = 0
        self._pois_muprev = -1.0
        self._pois_muprev2 = -1.0
        self._binom_nsave = -1
        self._binom_psave = -1.0

    # ------------------------------------------------------------ core stream
    def _next_block(self) -> np.ndarray:
        """Generate the next 624 tempered words (vectorised MT recurrence)."""
        mt = self._mt.astype(np.uint32)
        A = np.uint32(_MATRIX_A)
        one = np.uint32(1)

        def twist(lo, hi):
            """Apply the recurrence to mt[lo:hi]; the slice must only read values
            that are already final."""
            k = np.arange(lo, hi, dtype=np.intp)
            y = (mt[k] & np.uint32(_UPPER_MASK)) | (mt[k + 1] & np.uint32(_LOWER_MASK))
            src = k + _M if hi <= _N - _M else k + (_M - _N)
            mt[k] = mt[src] ^ (y >> one) ^ np.where((y & one).astype(bool), A, np.uint32(0))

        # kk = 0 .. N-M-1 reads only old values
        twist(0, _N - _M)
        # kk = N-M .. N-2 reads mt[kk + M - N], which the previous stage wrote.
        # Split so each stage only reads values finalised by an earlier stage.
        twist(_N - _M, 2 * (_N - _M))          # reads indices 0 .. N-M-1
        twist(2 * (_N - _M), _N - 1)           # reads indices N-M .. 2(N-M)-1
        y = (mt[_N - 1] & np.uint32(_UPPER_MASK)) | (mt[0] & np.uint32(_LOWER_MASK))
        mt[_N - 1] = mt[_M - 1] ^ (y >> one) ^ (A if (y & 1) else np.uint32(0))
        self._mt = mt

        # tempering
        z = mt.copy()
        z ^= z >> np.uint32(11)
        z ^= (z << np.uint32(7)) & np.uint32(_TEMPERING_MASK_B)
        z ^= (z << np.uint32(15)) & np.uint32(_TEMPERING_MASK_C)
        z ^= z >> np.uint32(18)
        return z.astype(np.float64) * 2.3283064365386963e-10

    def _refill(self) -> None:
        vals = self._next_block()
        # R's fixup(): 0 and 1 are never returned
        np.clip(vals, 0.5 * _I2_32M1, 1.0 - 0.5 * _I2_32M1, out=vals)
        self._buf = vals
        self._bpos = 0

    def unif_rand(self) -> float:
        """One draw from R's ``unif_rand``."""
        if self._bpos >= self._buf.size:
            self._refill()
        v = self._buf[self._bpos]
        self._bpos += 1
        return float(v)

    def unif_rand_n(self, n: int) -> np.ndarray:
        """``n`` consecutive draws (same stream, just faster)."""
        out = np.empty(n, dtype=np.float64)
        filled = 0
        while filled < n:
            if self._bpos >= self._buf.size:
                self._refill()
            take = min(n - filled, self._buf.size - self._bpos)
            out[filled:filled + take] = self._buf[self._bpos:self._bpos + take]
            self._bpos += take
            filled += take
        return out

    # ------------------------------------------------------------- normal
    def norm_rand(self) -> float:
        """R's ``norm_rand`` with the default INVERSION method."""
        u1 = self.unif_rand()
        u1 = int(_BIG * u1) + self.unif_rand()
        return float(qnorm(u1 / _BIG, 0.0, 1.0))

    def norm_rand_n(self, n: int) -> np.ndarray:
        if n == 0:
            return np.empty(0)
        u = self.unif_rand_n(2 * n)
        u1 = np.trunc(_BIG * u[0::2]) + u[1::2]
        return np.asarray(qnorm(u1 / _BIG, 0.0, 1.0), dtype=float).reshape(n)

    # -------------------------------------------------------- exponential
    def exp_rand(self) -> float:
        """R's ``exp_rand`` (``nmath/sexp.c``)."""
        a = 0.0
        u = self.unif_rand()
        while u <= 0.0 or u >= 1.0:
            u = self.unif_rand()
        while True:
            u += u
            if u > 1.0:
                break
            a += _EXP_Q[0]
        u -= 1.0
        if u <= _EXP_Q[0]:
            return a + u
        i = 0
        ustar = self.unif_rand()
        umin = ustar
        while True:
            ustar = self.unif_rand()
            umin = min(umin, ustar)
            i += 1
            if u <= _EXP_Q[i]:
                break
        return a + umin * _EXP_Q[0]

    # ------------------------------------------------------------- Poisson
    def rpois(self, mu: float) -> float:
        """R's ``rpois`` (Ahrens & Dieter 1982, ``nmath/rpois.c``)."""
        if not np.isfinite(mu) or mu < 0:
            return np.nan
        if mu <= 0.0:
            return 0.0

        big_mu = mu >= 10.0
        new_big_mu = False
        pois = -1.0

        if not (big_mu and mu == self._pois_muprev):
            if big_mu:
                new_big_mu = True
                self._pois_muprev = mu
                self._pois_s = np.sqrt(mu)
                self._pois_d = 6.0 * mu * mu
                self._pois_big_l = np.floor(mu - 1.1484)
            else:
                # Case B: mu < 10, table-driven inversion
                if mu != self._pois_muprev:
                    self._pois_muprev = mu
                    self._pois_m = max(1, int(mu))
                    self._pois_l = 0
                    self._pois_q = self._pois_p0 = self._pois_p = np.exp(-mu)
                while True:
                    u = self.unif_rand()
                    if u <= self._pois_p0:
                        return 0.0
                    if self._pois_l != 0:
                        start = 1 if u <= 0.458 else min(self._pois_l, self._pois_m)
                        for kk in range(start, self._pois_l + 1):
                            if u <= self._pois_pp[kk]:
                                return float(kk)
                        if self._pois_l == 35:
                            continue
                    self._pois_l += 1
                    hit = None
                    for kk in range(self._pois_l, 36):
                        self._pois_p *= mu / kk
                        self._pois_q += self._pois_p
                        self._pois_pp[kk] = self._pois_q
                        if u <= self._pois_q:
                            self._pois_l = kk
                            hit = kk
                            break
                    if hit is not None:
                        return float(hit)
                    self._pois_l = 35

        # ---- Case A: mu >= 10
        s, big_l = self._pois_s, self._pois_big_l
        g = mu + s * self.norm_rand()
        difmuk = fk = u = 0.0
        if g >= 0.0:
            pois = np.floor(g)
            if pois >= big_l:
                return pois
            fk = pois
            difmuk = mu - fk
            u = self.unif_rand()
            if self._pois_d * u >= difmuk * difmuk * difmuk:
                return pois

        if new_big_mu or mu > self._pois_muprev2:
            self._pois_muprev2 = mu
            omega = 0.3989422804014327 / s
            b1 = _ONE_24 / mu
            b2 = 0.3 * b1 * b1
            c3 = _ONE_7 * b1 * b2
            c2 = b2 - 15.0 * c3
            c1 = b1 - 6.0 * b2 + 45.0 * c3
            c0 = 1.0 - b1 + 3.0 * b2 - 15.0 * c3
            c = 0.1069 / mu
            self._pois_omega, self._pois_b1, self._pois_b2 = omega, b1, b2
            self._pois_c0, self._pois_c1 = c0, c1
            self._pois_c2, self._pois_c3, self._pois_c = c2, c3, c
        omega = self._pois_omega
        b1, b2 = self._pois_b1, self._pois_b2
        c0, c1, c2, c3, c = (self._pois_c0, self._pois_c1, self._pois_c2,
                             self._pois_c3, self._pois_c)

        px = py = fx = fy = 0.0
        if g >= 0.0:
            px, py, fx, fy = self._pois_f(pois, mu, s, omega, b1, b2, c0, c1, c2, c3)
            if fy - u * fy <= py * np.exp(px - fx):
                return pois

        while True:
            E = self.exp_rand()
            u = 2.0 * self.unif_rand() - 1.0
            t = 1.8 + np.copysign(E, u)
            if t <= -0.6744:
                continue
            pois = np.floor(mu + s * t)
            fk = pois
            difmuk = mu - fk
            px, py, fx, fy = self._pois_f(pois, mu, s, omega, b1, b2, c0, c1, c2, c3)
            if c * abs(u) <= py * np.exp(px + E) - fy * np.exp(fx + E):
                return pois

    @staticmethod
    def _pois_f(pois, mu, s, omega, b1, b2, c0, c1, c2, c3):
        """Step F of ``rpois``: compute px, py, fx, fy."""
        if pois < 10:
            px = -mu
            py = mu ** pois / _FACT[int(pois)]
        else:
            delta = _ONE_12 / pois
            delta = delta * (1.0 - 4.8 * delta * delta)
            v = (mu - pois) / pois
            if abs(v) <= 0.25:
                px = pois * v * v * (((((((_A7 * v + _A6) * v + _A5) * v + _A4)
                                        * v + _A3) * v + _A2) * v + _A1) * v + _A0) - delta
            else:
                px = pois * np.log(1.0 + v) - (mu - pois) - delta
            py = 0.3989422804014327 / np.sqrt(pois)
        x = (0.5 - (mu - pois)) / s
        xx = x * x
        fx = -0.5 * xx
        fy = omega * (((c3 * xx + c2) * xx + c1) * xx + c0)
        return px, py, fx, fy

    # ------------------------------------------------------------- Binomial
    def rbinom(self, nin: float, pp: float) -> float:
        """R's ``rbinom`` (Kachitvichyanukul & Schmeiser 1988, ``nmath/rbinom.c``)."""
        n = float(np.floor(nin + 0.5))
        if n == 0.0 or pp == 0.0:
            return 0.0
        if pp == 1.0:
            return n
        if not np.isfinite(n) or not np.isfinite(pp) or n < 0 or pp < 0 or pp > 1:
            return np.nan

        p = min(pp, 1.0 - pp)
        q = 1.0 - p
        np_ = n * p
        r = p / q
        g = r * (n + 1)

        if np_ < 30.0:
            # inverse cdf logic for mean less than 30
            qn = q ** n
            while True:
                ix = 0
                f = qn
                u = self.unif_rand()
                done = False
                while True:
                    if u < f:
                        done = True
                        break
                    if ix > 110:
                        break
                    u -= f
                    ix += 1
                    f *= (g / ix - r)
                if done:
                    break
            return float(n - ix) if pp > 0.5 else float(ix)

        # BTPE
        ffm = np_ + p
        m = int(ffm)
        fm = m
        npq = np_ * q
        p1 = int(2.195 * np.sqrt(npq) - 4.6 * q) + 0.5
        xm = fm + 0.5
        xl = xm - p1
        xr = xm + p1
        cc = 0.134 + 20.5 / (15.3 + fm)
        al = (ffm - xl) / (ffm - xl * p)
        xll = al * (1.0 + 0.5 * al)
        al = (xr - ffm) / (xr * q)
        xlr = al * (1.0 + 0.5 * al)
        p2 = p1 * (1.0 + cc + cc)
        p3 = p2 + cc / xll
        p4 = p3 + cc / xlr

        while True:
            u = self.unif_rand() * p4
            v = self.unif_rand()
            if u <= p1:
                ix = int(xm - p1 * v + u)
                return float(n - ix) if pp > 0.5 else float(ix)
            if u <= p2:  # parallelogram region
                x = xl + (u - p1) / cc
                v = v * cc + 1.0 - abs(xm - x) / p1
                if v > 1.0 or v <= 0.0:
                    continue
                ix = int(x)
            elif u > p3:  # right tail
                ix = int(xr - np.log(v) / xlr)
                if ix > n:
                    continue
                v = v * (u - p3) * xlr
            else:  # left tail
                ix = int(xl + np.log(v) / xll)
                if ix < 0:
                    continue
                v = v * (u - p2) * xll

            k = abs(ix - m)
            if k <= 20 or k >= npq / 2 - 1:
                # explicit evaluation
                f = 1.0
                if m < ix:
                    for i in range(m + 1, ix + 1):
                        f *= (g / i - r)
                elif m > ix:
                    for i in range(ix + 1, m + 1):
                        f /= (g / i - r)
                if v <= f:
                    return float(n - ix) if pp > 0.5 else float(ix)
            else:
                # squeezing using upper and lower bounds on log(f(x))
                amaxp = (k / npq) * ((k * (k / 3. + 0.625) + 0.1666666666666) / npq + 0.5)
                ynorm = -k * k / (2.0 * npq)
                alv = np.log(v)
                if alv < ynorm - amaxp:
                    return float(n - ix) if pp > 0.5 else float(ix)
                if alv <= ynorm + amaxp:
                    # Stirling's formula to machine accuracy
                    x1 = float(ix + 1)
                    f1 = fm + 1.0
                    z = n + 1 - fm
                    w = n - ix + 1.0
                    z2 = z * z
                    x2 = x1 * x1
                    f2 = f1 * f1
                    w2 = w * w
                    t = (xm * np.log(f1 / x1) + (n - m + 0.5) * np.log(z / w)
                         + (ix - m) * np.log(w * p / (x1 * q))
                         + (13860.0 - (462.0 - (132.0 - (99.0 - 140.0 / f2) / f2) / f2) / f2) / f1 / 166320.0
                         + (13860.0 - (462.0 - (132.0 - (99.0 - 140.0 / z2) / z2) / z2) / z2) / z / 166320.0
                         + (13860.0 - (462.0 - (132.0 - (99.0 - 140.0 / x2) / x2) / x2) / x2) / x1 / 166320.0
                         + (13860.0 - (462.0 - (132.0 - (99.0 - 140.0 / w2) / w2) / w2) / w2) / w / 166320.0)
                    if alv <= t:
                        return float(n - ix) if pp > 0.5 else float(ix)

    # ------------------------------------------------------- discrete index
    def _rbits(self, bits: int) -> float:
        v = 0
        n = 0
        while n <= bits:
            v1 = int(np.floor(self.unif_rand() * 65536))
            v = 65536 * v + v1
            n += 16
        return float(v & ((1 << bits) - 1))

    def unif_index(self, dn: float) -> float:
        """R's ``R_unif_index`` with the (post 3.6.0) default rejection sampling."""
        if dn <= 0:
            return 0.0
        bits = int(np.ceil(np.log2(dn)))
        while True:
            dv = self._rbits(bits)
            if dn > dv:
                return dv

    def sample_int(self, n: int, size: int, replace: bool = True) -> np.ndarray:
        """R's ``sample.int(n, size, replace = TRUE)`` -- 1-based, as in R."""
        if not replace:
            raise NotImplementedError("only sampling with replacement is used here")
        return np.array([int(self.unif_index(float(n))) + 1 for _ in range(size)],
                        dtype=np.int64)

    # ------------------------------------------------------ vector wrappers
    def runif(self, n: int, min=0.0, max=1.0) -> np.ndarray:
        """R's ``runif``.  Note R returns ``a`` without consuming a draw when
        ``a == b``, so degenerate arguments must not disturb the stream."""
        a = np.broadcast_to(np.asarray(min, dtype=float), (n,))
        b = np.broadcast_to(np.asarray(max, dtype=float), (n,))
        if np.all(a != b) and np.all(np.isfinite(a)) and np.all(np.isfinite(b)):
            return a + (b - a) * self.unif_rand_n(n)  # fast path
        out = np.empty(n, dtype=float)
        for i in range(n):
            if not np.isfinite(a[i]) or not np.isfinite(b[i]) or b[i] < a[i]:
                out[i] = np.nan
            elif a[i] == b[i]:
                out[i] = a[i]
            else:
                out[i] = a[i] + (b[i] - a[i]) * self.unif_rand()
        return out

    def rnorm(self, n: int, mean=0.0, sd=1.0) -> np.ndarray:
        """R's ``rnorm``.  R returns ``mu`` without consuming a draw when
        ``sigma == 0`` or ``mu`` is infinite."""
        mu = np.broadcast_to(np.asarray(mean, dtype=float), (n,))
        sigma = np.broadcast_to(np.asarray(sd, dtype=float), (n,))
        if np.all(sigma > 0) and np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma)):
            return mu + sigma * self.norm_rand_n(n)  # fast path
        out = np.empty(n, dtype=float)
        for i in range(n):
            if np.isnan(mu[i]) or not np.isfinite(sigma[i]) or sigma[i] < 0:
                out[i] = np.nan
            elif sigma[i] == 0 or not np.isfinite(mu[i]):
                out[i] = mu[i]
            else:
                out[i] = mu[i] + sigma[i] * self.norm_rand()
        return out

    def rexp(self, n: int, rate=1.0) -> np.ndarray:
        """R's ``rexp``; R passes ``1 / rate`` to the C routine as the scale."""
        scale = 1.0 / np.broadcast_to(np.asarray(rate, dtype=float), (n,))
        out = np.empty(n, dtype=float)
        for i in range(n):
            s = scale[i]
            if not np.isfinite(s) or s <= 0.0:
                out[i] = 0.0 if s == 0.0 else np.nan
            else:
                out[i] = s * self.exp_rand()
        return out

    def rpois_n(self, n: int, lam) -> np.ndarray:
        lam = np.broadcast_to(np.asarray(lam, dtype=float), (n,))
        return np.array([self.rpois(float(l)) for l in lam])

    def rbinom_n(self, n: int, size, prob) -> np.ndarray:
        size = np.broadcast_to(np.asarray(size, dtype=float), (n,))
        prob = np.broadcast_to(np.asarray(prob, dtype=float), (n,))
        return np.array([self.rbinom(float(s), float(p)) for s, p in zip(size, prob)])


# A module-level stream mirroring R's single global .Random.seed
_GLOBAL = RRandom()


def set_seed(seed: int) -> None:
    """Equivalent to R's ``set.seed(seed)``; affects the package's global stream."""
    _GLOBAL.set_seed(seed)


def get_generator() -> RRandom:
    """Return the package-global generator (R's ``.Random.seed`` equivalent)."""
    return _GLOBAL
