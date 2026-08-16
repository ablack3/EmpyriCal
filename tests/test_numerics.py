"""The ported R numerics: nmath densities, optim, uniroot and integrate.

These are the pieces that were ported rather than delegated to SciPy, so they
are tested both for internal consistency and against SciPy, which they should
agree with to near machine precision without being identical to it.
"""

import numpy as np
import pytest
from scipy import integrate as sp_integrate
from scipy import stats

from empiricalcalibration._rintegrate import integrate
from empiricalcalibration._rmath import (
    dbinom,
    dgamma_log,
    dnorm,
    dpois,
    pchisq,
    pnorm,
    qchisq,
    qnorm,
)
from empiricalcalibration._roptim import optim, optimhess, uniroot

# --- densities and distribution functions ---------------------------------

def test_dnorm_at_zero_is_the_analytic_constant():
    assert dnorm(0.0) == pytest.approx(1 / np.sqrt(2 * np.pi), rel=1e-15)


def test_qnorm_matches_rs_printed_quantiles():
    # R: qnorm(0.025) and qnorm(0.975)
    assert qnorm(0.025) == pytest.approx(-1.959963984540054, rel=1e-14)
    assert qnorm(0.975) == pytest.approx(1.959963984540054, rel=1e-14)


def test_qnorm_and_pnorm_are_inverses():
    p = np.array([1e-8, 0.001, 0.025, 0.5, 0.975, 0.999])
    assert np.allclose(pnorm(qnorm(p)), p, rtol=1e-12)


@pytest.mark.parametrize("mean,sd", [(0.0, 1.0), (0.3, 0.25), (-1.5, 2.0)])
def test_dnorm_and_pnorm_agree_with_scipy(mean, sd):
    x = np.linspace(-6, 6, 101)
    assert np.allclose(dnorm(x, mean, sd), stats.norm.pdf(x, mean, sd), rtol=1e-12)
    assert np.allclose(pnorm(x, mean, sd), stats.norm.cdf(x, mean, sd), rtol=1e-12)


def test_dnorm_log_avoids_underflow():
    """log = True must stay finite where the density itself underflows to 0."""
    x = 60.0
    assert dnorm(x, 0.0, 1.0) == 0.0
    assert np.isfinite(dnorm(x, 0.0, 1.0, log=True))
    assert dnorm(x, 0.0, 1.0, log=True) < -1000


def test_pnorm_log_p_avoids_underflow():
    """The skew-normal approximation relies on this over +/- 10 sd."""
    assert pnorm(-50.0, log_p=True) < -1000
    assert np.isfinite(pnorm(-50.0, log_p=True))
    assert pnorm(-50.0) == 0.0


def test_pnorm_lower_tail_false():
    assert pnorm(1.5, lower_tail=False) == pytest.approx(1 - pnorm(1.5), rel=1e-14)


def test_dpois_matches_scipy():
    k = np.arange(0, 30)
    for lam in (0.5, 3.7, 12.0, 250.0):
        assert np.allclose(dpois(k, lam), stats.poisson.pmf(k, lam), rtol=1e-12)


def test_dbinom_matches_scipy():
    k = np.arange(0, 21)
    for n, p in [(20, 0.2), (20, 0.5), (200, 0.01)]:
        assert np.allclose(dbinom(k, n, p), stats.binom.pmf(k, n, p), rtol=1e-12)


def test_dpois_and_dbinom_sum_to_one():
    assert np.sum(dpois(np.arange(0, 200), 12.0)) == pytest.approx(1.0, rel=1e-12)
    assert np.sum(dbinom(np.arange(0, 21), 20, 0.3)) == pytest.approx(1.0, rel=1e-12)


def test_dgamma_log_matches_scipy():
    """Used as the MCMC prior on the precision."""
    for shape, rate in [(1e-4, 1e-4), (2.0, 3.0)]:
        for x in (0.1, 1.0, 10.0):
            assert dgamma_log(x, shape=shape, rate=rate) == pytest.approx(
                stats.gamma.logpdf(x, a=shape, scale=1 / rate), rel=1e-12)


def test_pchisq_and_qchisq_are_inverses():
    p = np.array([0.01, 0.1, 0.5, 0.9, 0.99])
    assert np.allclose(pchisq(qchisq(p, 1), 1), p, rtol=1e-10)


def test_qchisq_matches_scipy():
    for p in (0.5, 0.9, 0.95, 0.99):
        assert qchisq(p, 1) == pytest.approx(stats.chi2.ppf(p, 1), rel=1e-10)


# --- optimisation ----------------------------------------------------------

def test_nelder_mead_finds_a_quadratic_minimum():
    fit = optim(np.array([5.0, -3.0]),
                lambda t: (t[0] - 1.5) ** 2 + (t[1] + 0.5) ** 2 + 2.0)
    assert fit["par"][0] == pytest.approx(1.5, abs=1e-3)
    assert fit["par"][1] == pytest.approx(-0.5, abs=1e-3)
    assert fit["value"] == pytest.approx(2.0, abs=1e-6)


def test_bfgs_finds_the_rosenbrock_minimum():
    def rosenbrock(t):
        return (1 - t[0]) ** 2 + 100 * (t[1] - t[0] ** 2) ** 2

    fit = optim(np.array([-1.2, 1.0]), rosenbrock, method="BFGS",
                control={"maxit": 2000})
    assert fit["par"][0] == pytest.approx(1.0, abs=1e-2)
    assert fit["par"][1] == pytest.approx(1.0, abs=2e-2)


def test_bfgs_returns_a_hessian_when_asked():
    fit = optim(np.array([0.0]), lambda t: 3.0 * t[0] ** 2,
                method="BFGS", hessian=True)
    assert fit["hessian"].shape == (1, 1)
    # d2/dx2 of 3x^2 is 6
    assert fit["hessian"][0, 0] == pytest.approx(6.0, rel=1e-3)


def test_optimhess_on_a_known_quadratic():
    h = optimhess(np.array([0.0, 0.0]),
                  lambda t: 2.0 * t[0] ** 2 + 5.0 * t[1] ** 2)
    assert h[0, 0] == pytest.approx(4.0, rel=1e-3)
    assert h[1, 1] == pytest.approx(10.0, rel=1e-3)
    assert abs(h[0, 1]) < 1e-6


def test_optim_rejects_an_unknown_method():
    with pytest.raises(ValueError):
        optim(np.array([0.0]), lambda t: t[0] ** 2, method="NoSuchMethod")


# --- root finding ----------------------------------------------------------

def test_uniroot_finds_a_simple_root():
    result = uniroot(lambda x: x ** 3 - 2 * x - 5, (2.0, 3.0))
    assert result["root"] == pytest.approx(2.0945514815423265, abs=1e-4)


def test_uniroot_passes_extra_args():
    result = uniroot(lambda x, target: x ** 2 - target, (0.0, 5.0), args=(9.0,))
    assert result["root"] == pytest.approx(3.0, abs=1e-4)


def test_uniroot_uses_rs_loose_default_tolerance():
    """R's default is eps^0.25, far looser than SciPy's; the port must match."""
    result = uniroot(lambda x: x - 1 / 3, (0.0, 1.0))
    assert result["root"] == pytest.approx(1 / 3, abs=1e-4)


# --- integration -----------------------------------------------------------

def test_integrate_a_normal_density_over_the_real_line():
    result = integrate(lambda x: float(dnorm(x, 0.0, 1.0)), -np.inf, np.inf)
    assert result["value"] == pytest.approx(1.0, rel=1e-6)


def test_integrate_matches_scipy_on_a_finite_interval():
    def f(x):
        return np.exp(-(x ** 2)) * np.cos(x)

    ours = integrate(lambda x: float(f(x)), -3.0, 3.0)["value"]
    theirs, _ = sp_integrate.quad(f, -3.0, 3.0)
    assert ours == pytest.approx(theirs, rel=1e-8)


def test_integrate_passes_extra_args():
    result = integrate(lambda x, m: float(dnorm(x, m, 1.0)), -20.0, 20.0, args=(2.0,))
    assert result["value"] == pytest.approx(1.0, rel=1e-6)


def test_integrate_can_suppress_errors():
    """stop_on_error=False is how the LLR calibration tolerates rough integrands."""
    result = integrate(lambda x: float(np.sin(1 / x)) if x != 0 else 0.0,
                       0.0, 1.0, stop_on_error=False)
    assert np.isfinite(result["value"])
