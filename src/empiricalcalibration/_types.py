"""The three S3 classes the R package returns: ``null``, ``mcmcNull`` and
``systematicErrorModel``.

In R these are named numeric vectors carrying a class attribute, so users write
``null[1]`` for the mean and ``null[2]`` for the standard deviation.  Here they
are NumPy array subclasses, so the same positional access works (0-based, as
Python users expect) and name-based access is available too::

    null[0], null["mean"], null.mean   # all the null distribution's mean
"""

from __future__ import annotations

import numpy as np


class RNamedVector(np.ndarray):
    """A NumPy vector that also supports R-style name lookup."""

    def __new__(cls, values, names=None):
        obj = np.asarray(values, dtype=float).view(cls)
        obj.names = list(names) if names is not None else None
        obj.attributes = {}
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.names = getattr(obj, "names", None)
        self.attributes = getattr(obj, "attributes", {})

    def __getitem__(self, key):
        if isinstance(key, str):
            if self.names is None or key not in self.names:
                raise KeyError(key)
            return float(np.asarray(self)[self.names.index(key)])
        return super().__getitem__(key)

    def to_dict(self):
        if self.names is None:
            return {i: float(v) for i, v in enumerate(np.asarray(self))}
        return dict(zip(self.names, (float(v) for v in np.asarray(self))))


def _fmt(value, digits=6):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "NA"
    return f"{value:.{digits}g}"


class Null(RNamedVector):
    """Result of :func:`fitNull` / :func:`fitNullNonNormalLl` (R class ``null``)."""

    def __new__(cls, mean, sd):
        return super().__new__(cls, [mean, sd], names=["mean", "sd"])

    @property
    def mean(self):
        return float(np.asarray(self)[0])

    @property
    def sd(self):
        return float(np.asarray(self)[1])

    def __repr__(self):
        # Mirrors R's print.null()
        lines = ["Estimated null distribution", ""]
        lines.append(f"{'':>4} {'Estimate':>10}")
        lines.append(f"{'Mean':>4} {_fmt(self.mean):>10}")
        lines.append(f"{'SD':>4} {_fmt(self.sd):>10}")
        return "\n".join(lines)

    __str__ = __repr__


class McmcNull(RNamedVector):
    """Result of :func:`fitMcmcNull` (R class ``mcmcNull``).

    Element 0 is the posterior median of the mean, element 1 the posterior
    median of the **precision** (not the standard deviation) -- exactly as in R.
    The MCMC output is available as ``.mcmc``, a dict with ``chain``, ``logLik``
    and ``acc``.
    """

    def __new__(cls, mean, precision, mcmc=None):
        obj = super().__new__(cls, [mean, precision], names=["mean", "precision"])
        obj.mcmc = mcmc
        return obj

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        if obj is not None:
            self.mcmc = getattr(obj, "mcmc", None)

    @property
    def mean(self):
        return float(np.asarray(self)[0])

    @property
    def precision(self):
        return float(np.asarray(self)[1])

    def __repr__(self):
        # Mirrors R's print.mcmcNull()
        lines = ["Estimated null distribution (using MCMC)", ""]
        if np.isnan(self.mean):
            lines.append("Undefined")
            return "\n".join(lines)
        chain = self.mcmc["chain"]
        lb_mean, ub_mean = np.quantile(chain[:, 0], [0.025, 0.975])
        lb_prec, ub_prec = np.quantile(chain[:, 1], [0.025, 0.975])
        lines.append(f"{'':>9} {'Estimate':>10} {'lower .95':>10} {'upper .95':>10}")
        lines.append(f"{'Mean':>9} {_fmt(self.mean):>10} {_fmt(lb_mean):>10} {_fmt(ub_mean):>10}")
        lines.append(f"{'Precision':>9} {_fmt(self.precision):>10} {_fmt(lb_prec):>10} {_fmt(ub_prec):>10}")
        lines.append("")
        lines.append(f"Acceptance rate: {float(np.mean(self.mcmc['acc']))}")
        return "\n".join(lines)

    __str__ = __repr__


class SystematicErrorModel(RNamedVector):
    """Result of :func:`fitSystematicErrorModel` / :func:`convertNullToErrorModel`.

    Names are ``meanIntercept, meanSlope, sdIntercept, sdSlope`` or, when
    ``legacy = True``, ``meanIntercept, meanSlope, logSdIntercept, logSdSlope``.
    """

    def __new__(cls, values, names):
        return super().__new__(cls, values, names=names)

    @property
    def legacy(self) -> bool:
        return self.names is not None and self.names[2] == "logSdIntercept"

    def __repr__(self):
        width = max(len(n) for n in self.names)
        head = " ".join(f"{n:>{max(width, 12)}}" for n in self.names)
        vals = " ".join(f"{_fmt(v):>{max(width, 12)}}" for v in np.asarray(self))
        return f"{head}\n{vals}"

    __str__ = __repr__
