"""R's ``stats::integrate`` defaults on top of the same QUADPACK routines.

R uses ``rel.tol = abs.tol = .Machine$double.eps^0.25`` (1.22e-4) and
``subdivisions = 100``; SciPy defaults to 1.49e-8 and ``limit = 50``.  Since both
call QAGS/QAGI, matching R only requires matching the control parameters.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import integrate as _si

DEFAULT_TOL = np.finfo(float).eps ** 0.25  # 1.220703125e-4


def integrate(f, lower, upper, args=(), subdivisions=100,
              rel_tol=DEFAULT_TOL, abs_tol=DEFAULT_TOL, stop_on_error=True):
    """Mimics ``stats::integrate``; returns a dict with ``value`` and ``abs.error``."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            value, abserr = _si.quad(
                f, lower, upper, args=args, epsabs=abs_tol, epsrel=rel_tol,
                limit=subdivisions,
            )
        except Exception:
            if stop_on_error:
                raise
            return {"value": np.nan, "abs.error": np.nan}
    return {"value": value, "abs.error": abserr}
