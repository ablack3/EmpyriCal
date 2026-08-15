"""The three R-style result classes: positional access, name access, repr."""

import numpy as np
import pytest

import empiricalcalibration as ec
from empiricalcalibration._types import RNamedVector


def test_null_positional_name_and_attribute_access_agree():
    null = ec.Null(0.25, 0.4)
    assert null[0] == null["mean"] == null.mean == 0.25
    assert null[1] == null["sd"] == null.sd == 0.4


def test_null_is_a_numpy_array():
    null = ec.Null(0.25, 0.4)
    assert isinstance(null, np.ndarray)
    assert np.asarray(null).tolist() == [0.25, 0.4]
    # arithmetic still works, which the calibration code relies on
    assert float(np.sum(null)) == pytest.approx(0.65)


def test_null_repr_matches_r_print_null():
    lines = repr(ec.Null(0.25, 0.4)).splitlines()
    assert lines[0] == "Estimated null distribution"
    assert lines[3].split() == ["Mean", "0.25"]
    assert lines[4].split() == ["SD", "0.4"]


def test_null_repr_handles_nan():
    assert "NA" in repr(ec.Null(np.nan, np.nan))


def test_unknown_name_raises_keyerror():
    with pytest.raises(KeyError):
        ec.Null(0, 1)["nosuchname"]


def test_to_dict_uses_names():
    assert ec.Null(0.25, 0.4).to_dict() == {"mean": 0.25, "sd": 0.4}


def test_to_dict_falls_back_to_positions_without_names():
    assert RNamedVector([1.0, 2.0]).to_dict() == {0: 1.0, 1: 2.0}


def test_systematic_error_model_names_and_legacy_flag():
    current = ec.SystematicErrorModel(
        [0.1, 0.9, 0.2, 0.05],
        ["meanIntercept", "meanSlope", "sdIntercept", "sdSlope"])
    assert current.legacy is False
    assert current["meanSlope"] == 0.9

    legacy = ec.SystematicErrorModel(
        [0.1, 0.9, -2.0, 0.0],
        ["meanIntercept", "meanSlope", "logSdIntercept", "logSdSlope"])
    assert legacy.legacy is True
    assert "logSdIntercept" in repr(legacy)


def test_mcmc_null_stores_precision_not_sd():
    """R's mcmcNull carries the precision in slot 2; sd is 1/sqrt(precision)."""
    sccs = ec.datasets.sccs()
    negatives = sccs[sccs.groundTruth == 0]
    ec.set_seed(11)
    null = ec.fitMcmcNull(negatives.logRr, negatives.seLogRr, iter=500)

    assert null[1] == null["precision"] == null.precision
    assert null.precision > 1          # a precision, not a standard deviation
    sd = 1 / np.sqrt(null.precision)
    assert 0 < sd < 1

    text = repr(null)
    assert text.splitlines()[0] == "Estimated null distribution (using MCMC)"
    assert "Precision" in text
    assert "Acceptance rate" in text


def test_mcmc_null_repr_when_undefined():
    with pytest.warns(UserWarning):
        null = ec.fitMcmcNull([np.nan] * 3, [np.nan] * 3)
    assert "Undefined" in repr(null)


def test_mcmc_chain_is_exposed():
    sccs = ec.datasets.sccs()
    negatives = sccs[sccs.groundTruth == 0]
    ec.set_seed(12)
    null = ec.fitMcmcNull(negatives.logRr, negatives.seLogRr, iter=500)

    chain = null.mcmc["chain"]
    assert chain.shape == (501, 2)                 # iter + 1 rows
    assert null.mcmc["logLik"].shape == (501, 1)
    assert null.mcmc["acc"].shape == (501, 1)
    assert np.median(chain[:, 0]) == null[0]
    assert np.median(chain[:, 1]) == null[1]
    assert np.all(chain[:, 1] >= 0)                # precision is reflected
