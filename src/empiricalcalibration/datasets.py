"""The five example datasets shipped with the R package.

Each is exposed as a function returning a fresh :class:`pandas.DataFrame`, so
callers cannot mutate the package's copy::

    from empiricalcalibration import datasets
    sccs = datasets.sccs()

The CSVs were written from the ``.rda`` files with 17 significant digits and are
read back with round-trip float parsing, so the values are bit-identical to R's.

One deliberate difference: in the R package ``cohortMethod$groundTruth`` is a
*factor* with levels ``"0"``/``"1"`` while ``sccs`` and ``caseControl`` store it
as a number.  Here it is an integer in all three, matching the documented format
("Whether the drug is a positive (1) or negative (0) control") and making
``df.groundTruth == 1`` behave the same way everywhere.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_DATA_DIR = Path(__file__).parent / "data"

__all__ = ["sccs", "caseControl", "cohortMethod", "southworthReplication",
           "grahamReplication", "load"]


def load(name: str) -> pd.DataFrame:
    """Load one of the packaged datasets by name."""
    path = _DATA_DIR / f"{name}.csv"
    if not path.exists():
        raise ValueError(f"Unknown dataset: {name!r}")
    df = pd.read_csv(path, float_precision="round_trip")
    if "groundTruth" in df.columns:
        df["groundTruth"] = df["groundTruth"].astype("int64")
    return df


def sccs() -> pd.DataFrame:
    """Incidence rate ratios from a Self-Controlled Case Series design.

    Upper GI bleeding; the drug of interest (``groundTruth == 1``) is
    sertraline, alongside 45 negative control drugs.  46 rows, 4 columns:
    ``drugName``, ``groundTruth``, ``logRr``, ``seLogRr``.
    """
    return load("sccs")


def caseControl() -> pd.DataFrame:
    """Odds ratios from a case-control design (47 rows).

    Upper GI bleeding; the drug of interest (``groundTruth == 1``) is
    sertraline, alongside 46 negative control drugs.
    """
    return load("caseControl")


def cohortMethod() -> pd.DataFrame:
    """Relative risks from a new-user cohort design (31 rows).

    Acute liver injury; the drug of interest (``groundTruth == 1``) is
    isoniazid, alongside 30 negative control drugs.
    """
    return load("cohortMethod")


def southworthReplication() -> pd.DataFrame:
    """Relative risks from an unadjusted new-user cohort design (173 rows).

    Dabigatran vs warfarin for GI hemorrhage, with negative and positive
    controls.  Columns: ``outcomeName``, ``trueLogRr``, ``logRr``, ``seLogRr``.
    ``trueLogRr`` is missing for the outcome of interest.
    """
    return load("southworthReplication")


def grahamReplication() -> pd.DataFrame:
    """Relative risks from an adjusted new-user cohort design (125 rows).

    Dabigatran vs warfarin for GI hemorrhage, with negative and positive
    controls.  Columns: ``outcomeName``, ``trueLogRr``, ``logRr``, ``seLogRr``.
    """
    return load("grahamReplication")
