"""Shared pytest configuration."""

import matplotlib
import pytest

# Select a non-interactive backend before any test imports pyplot, so the suite
# runs headless in CI.
matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def _close_figures():
    """Close every figure a test leaves open.

    The plotting functions build figures with ``plt.subplots`` and return them
    without closing, which is the right behaviour for a user who wants to keep
    adjusting them.  Across a whole test session it accumulates, and matplotlib
    starts warning past twenty open figures.
    """
    yield
    import matplotlib.pyplot as plt

    plt.close("all")
