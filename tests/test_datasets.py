"""The five packaged datasets, and the loader's contract."""

import numpy as np
import pandas as pd
import pytest

from empiricalcalibration import datasets

ALL = ["sccs", "caseControl", "cohortMethod",
       "southworthReplication", "grahamReplication"]


@pytest.mark.parametrize("name", ALL)
def test_every_dataset_loads_and_has_the_core_columns(name):
    df = getattr(datasets, name)()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert {"logRr", "seLogRr"} <= set(df.columns)
    assert df.logRr.dtype.kind == "f"
    assert df.seLogRr.dtype.kind == "f"


@pytest.mark.parametrize("name", ALL)
def test_load_by_name_matches_the_accessor(name):
    pd.testing.assert_frame_equal(datasets.load(name), getattr(datasets, name)())


def test_load_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="Unknown dataset"):
        datasets.load("noSuchDataset")


@pytest.mark.parametrize("name", ALL)
def test_each_call_returns_a_fresh_frame(name):
    """Mutating a returned frame must not affect the package's copy."""
    first = getattr(datasets, name)()
    first.loc[first.index[0], "logRr"] = 999.0
    second = getattr(datasets, name)()
    assert second.logRr.iloc[0] != 999.0


@pytest.mark.parametrize("name", ["sccs", "caseControl", "cohortMethod"])
def test_groundTruth_is_an_integer_everywhere(name):
    """Deliberate difference from R, where cohortMethod$groundTruth is a factor."""
    df = getattr(datasets, name)()
    assert df.groundTruth.dtype.kind == "i"
    assert set(df.groundTruth.unique()) <= {0, 1}
    assert (df.groundTruth == 1).any()
    assert (df.groundTruth == 0).any()


@pytest.mark.parametrize("name", ["southworthReplication", "grahamReplication"])
def test_replication_datasets_carry_true_effect_sizes(name):
    df = getattr(datasets, name)()
    assert "trueLogRr" in df.columns
    controls = df[df.trueLogRr.notna()]
    assert len(controls) > 0
    # Negative controls at zero, plus at least one non-zero positive control.
    assert (controls.trueLogRr == 0).any()
    assert (controls.trueLogRr > 0).any()


def test_standard_errors_are_positive_and_finite():
    for name in ALL:
        df = getattr(datasets, name)()
        se = df.seLogRr.to_numpy()
        assert np.all(np.isfinite(se)), name
        assert np.all(se > 0), name


def test_sccs_values_are_bit_identical_to_r():
    """The CSVs round-trip at full precision, so values match R's .rda exactly."""
    sccs = datasets.sccs()
    assert sccs.logRr.iloc[0] == 0.433902074
    assert sccs.seLogRr.iloc[0] == 0.761753760137732
    assert len(sccs) == 46
