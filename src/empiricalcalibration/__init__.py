"""EmpiricalCalibration -- routines for performing empirical calibration of
observational study estimates.

A Python port of the OHDSI R package of the same name (version 3.1.4).  Function
names, argument names, argument order and defaults follow the R package, so R
code translates across directly::

    # R                                        # Python
    null <- fitNull(neg$logRr, neg$seLogRr)    null = fitNull(neg.logRr, neg.seLogRr)
    calibrateP(null, logRr, seLogRr)           calibrateP(null, logRr, seLogRr)

By using a set of negative control hypotheses we can estimate the empirical null
distribution of a particular observational study setup.  This empirical null
distribution can be used to compute a calibrated p-value, which reflects the
probability of observing an estimated effect size when the null hypothesis is
true, taking both random and systematic error into account.  A similar approach
calibrates confidence intervals, using both negative and positive controls.

References
----------
Schuemie MJ, Ryan PB, DuMouchel W, Suchard MA, Madigan D. Interpreting
observational studies: why empirical calibration is needed to correct p-values.
Statistics in Medicine 33(2):209-18, 2014.  doi:10.1002/sim.5925

Schuemie MJ, Hripcsak G, Ryan PB, Madigan D, Suchard MA. Empirical confidence
interval calibration for population-level effect estimation studies in
observational healthcare data. PNAS 115(11):2571-2577, 2018.
doi:10.1073/pnas.1708282114

Notes
-----
Two deliberate differences from the R package:

* Plotting uses matplotlib and returns a :class:`matplotlib.figure.Figure`
  rather than a ggplot object.  Signatures, defaults and the ``fileName``
  argument are unchanged.
* Random draws come from a port of R's own Mersenne-Twister, so
  :func:`set_seed` reproduces R's ``set.seed`` streams exactly.
"""

from __future__ import annotations

#: Version of this Python package.  It has its own release lineage and does
#: not track the R package's version.
__version__ = "0.1.1"

#: Version of the OHDSI R package this port reproduces.  The R-parity gold
#: standard in ``extras/r-parity/`` was generated against exactly this version.
__r_package_version__ = "3.1.4"

from . import datasets
from ._rrng import RRandom, get_generator, set_seed
from ._types import McmcNull, Null, SystematicErrorModel
from .asymptotics import (
    calibrateP,
    computeTraditionalP,
    fitNull,
    fitNullNonNormalLl,
)
from .confidence_intervals import (
    calibrateConfidenceInterval,
    computeTraditionalCi,
    convertNullToErrorModel,
    fitSystematicErrorModel,
)
from .evaluation import evaluateCiCalibration
from .expected_systematic_error import (
    compareEase,
    computeExpectedAbsoluteSystematicError,
)
from .llr_calibration import calibrateLlr
from .maxsprt import (
    computeCvBinomial,
    computeCvPoisson,
    computeCvPoissonRegression,
)
from .mcmc import fitMcmcNull
from .plots import (
    plotCalibration,
    plotCalibrationEffect,
    plotCiCalibration,
    plotCiCalibrationEffect,
    plotCiCoverage,
    plotErrorModel,
    plotExpectedType1Error,
    plotForest,
    plotMcmcTrace,
    plotTrueAndObserved,
)
from .simulation import simulateControls, simulateMaxSprtData

__all__ = [
    # null distribution and p-value calibration
    "fitNull", "fitNullNonNormalLl", "fitMcmcNull", "calibrateP",
    "computeTraditionalP",
    # confidence interval calibration
    "fitSystematicErrorModel", "calibrateConfidenceInterval",
    "computeTraditionalCi", "convertNullToErrorModel",
    # systematic error summaries
    "computeExpectedAbsoluteSystematicError", "compareEase",
    # log likelihood ratio calibration and MaxSPRT
    "calibrateLlr", "computeCvPoisson", "computeCvBinomial",
    "computeCvPoissonRegression",
    # evaluation and simulation
    "evaluateCiCalibration", "simulateControls", "simulateMaxSprtData",
    # plots
    "plotForest", "plotCalibration", "plotCalibrationEffect",
    "plotCiCalibration", "plotCiCalibrationEffect", "plotCiCoverage",
    "plotErrorModel", "plotExpectedType1Error", "plotMcmcTrace",
    "plotTrueAndObserved",
    # classes, data and RNG
    "Null", "McmcNull", "SystematicErrorModel", "datasets",
    "set_seed", "get_generator", "RRandom",
    # versions
    "__version__", "__r_package_version__",
]
