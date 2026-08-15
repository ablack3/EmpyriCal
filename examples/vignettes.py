"""Python equivalents of the three R vignettes.

Run with::

    python examples/vignettes.py

Writes the figures next to this file.
"""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import empiricalcalibration as ec  # noqa: E402

OUT = Path(__file__).parent


def empirical_p_calibration():
    """EmpiricalPCalibrationVignette.Rmd"""
    print("=" * 70)
    print("Empirical p-value calibration")
    print("=" * 70)

    data = ec.datasets.sccs()
    negatives = data[data.groundTruth == 0]
    positive = data[data.groundTruth == 1]

    ec.plotForest(negatives.logRr, negatives.seLogRr, negatives.drugName,
                  fileName=str(OUT / "p_forest.png"))

    null = ec.fitNull(negatives.logRr, negatives.seLogRr)
    print(null)

    print("\nTraditional p-value for sertraline: "
          f"{ec.computeTraditionalP(positive.logRr, positive.seLogRr)[0]:.4f}")
    print("Calibrated p-value for sertraline:  "
          f"{ec.calibrateP(null, positive.logRr, positive.seLogRr)[0]:.4f}")

    ease = ec.computeExpectedAbsoluteSystematicError(null)
    print(f"Expected absolute systematic error: {ease:.4f}")

    ec.plotCalibrationEffect(negatives.logRr, negatives.seLogRr,
                             positive.logRr, positive.seLogRr,
                             fileName=str(OUT / "p_calibration_effect.png"))
    ec.plotCalibration(negatives.logRr, negatives.seLogRr,
                       fileName=str(OUT / "p_calibration.png"))
    ec.plotExpectedType1Error(negatives.logRr, negatives.seLogRr,
                              positive.seLogRr,
                              fileName=str(OUT / "p_type1error.png"))


def empirical_ci_calibration():
    """EmpiricalCiCalibrationVignette.Rmd"""
    print()
    print("=" * 70)
    print("Empirical confidence interval calibration")
    print("=" * 70)

    data = ec.datasets.southworthReplication()
    outcome = data[data.trueLogRr.isna()]        # the outcome of interest: GiBleed
    controls = data[~data.trueLogRr.isna()]      # negative and positive controls

    print("Uncalibrated estimate for GI bleed:")
    print(ec.computeTraditionalCi(outcome.logRr, outcome.seLogRr).to_string(index=False))

    model = ec.fitSystematicErrorModel(controls.logRr, controls.seLogRr,
                                       controls.trueLogRr)
    print("\nSystematic error model:")
    print(model)

    ec.plotErrorModel(controls.logRr, controls.seLogRr, controls.trueLogRr,
                      fileName=str(OUT / "ci_error_model.png"))

    calibrated = ec.calibrateConfidenceInterval(outcome.logRr, outcome.seLogRr, model)
    print("\nCalibrated estimate for GI bleed (ratio scale):")
    print(f"  rr = {np.exp(calibrated.logRr.iloc[0]):.3f} "
          f"({np.exp(calibrated.logLb95Rr.iloc[0]):.3f} - "
          f"{np.exp(calibrated.logUb95Rr.iloc[0]):.3f})")

    ec.plotCiCalibrationEffect(controls.logRr, controls.seLogRr, controls.trueLogRr,
                               fileName=str(OUT / "ci_calibration_effect.png"))
    ec.plotCiCoverage(controls.logRr, controls.seLogRr, controls.trueLogRr,
                      fileName=str(OUT / "ci_coverage.png"))


def maxsprt_calibration():
    """EmpiricalMaxSprtCalibrationVignette.Rmd (abbreviated)"""
    print()
    print("=" * 70)
    print("Empirical MaxSPRT calibration")
    print("=" * 70)

    groupSizes = [1.0] * 10
    # smaller sampleSize than the default to keep the example quick
    ec.set_seed(1)
    cv = ec.computeCvPoisson(groupSizes, sampleSize=100_000)
    print(f"Uncalibrated Poisson critical value: {float(cv):.4f} "
          f"(alpha = {cv.alpha:.4f})")

    ec.set_seed(1)
    cvCal = ec.computeCvPoisson(groupSizes, sampleSize=100_000,
                                nullMean=0.2, nullSd=0.2)
    print(f"Critical value under an empirical null (mu=0.2, sd=0.2): "
          f"{float(cvCal):.4f} (alpha = {cvCal.alpha:.4f})")

    # Calibrating a log likelihood ratio against an empirical null
    sccs = ec.datasets.sccs()
    negatives = sccs[sccs.groundTruth == 0]
    positive = sccs[sccs.groundTruth == 1]
    null = ec.fitNull(negatives.logRr, negatives.seLogRr)
    print(f"Calibrated LLR for sertraline: "
          f"{ec.calibrateLlr(null, positive)[0]:.4f}")


if __name__ == "__main__":
    empirical_p_calibration()
    empirical_ci_calibration()
    maxsprt_calibration()
    print(f"\nFigures written to {OUT}")
