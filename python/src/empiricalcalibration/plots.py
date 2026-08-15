"""Port of ``R/Plots.R``.

The R package builds these with ggplot2; here they are built with matplotlib and
each function returns a :class:`matplotlib.figure.Figure`.  Function names,
argument names, argument order, defaults, colours, axis breaks, labels and the
saved figure dimensions all follow the R original, so the figures are the same
plots -- they are simply not the same object type, since there is no ggplot
object in Python.

Every function accepts ``fileName``; when given, the figure is written there
with the same width/height/dpi ``ggsave`` used in R.
"""

from __future__ import annotations

import warnings

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter

from ._rmath import pnorm, qnorm
from ._types import McmcNull, Null
from .asymptotics import fitNull
from .confidence_intervals import fitSystematicErrorModel
from .evaluation import evaluateCiCalibration
from .expected_systematic_error import computeExpectedAbsoluteSystematicError
from .mcmc import fitMcmcNull

__all__ = ["plotForest", "plotCalibrationEffect", "plotCalibration",
           "plotCiCalibration", "plotTrueAndObserved", "plotMcmcTrace",
           "plotCiCoverage", "plotErrorModel", "plotExpectedType1Error",
           "plotCiCalibrationEffect"]

_PANEL_BG = "#FAFAFA"
_GRID = "#AAAAAA"
_MAJOR = "#EEEEEE"

# ggplot2's default discrete colours are not used; these are R's explicit rgb() calls
_COL = [(0, 0, 0.8), (0.8, 0.4, 0)]
_COL_FILL = [(0, 0, 1, 0.5), (1, 0.4, 0, 0.5)]


def _style_panel(ax, bg=_PANEL_BG, major=False):
    ax.set_facecolor(bg)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    if major:
        ax.grid(True, color=_MAJOR, linewidth=0.5)
        ax.set_axisbelow(True)
    else:
        ax.grid(False)


def _save(fig, fileName, width, height, dpi=400):
    if fileName is not None:
        fig.set_size_inches(width, height)
        fig.savefig(fileName, dpi=dpi, bbox_inches="tight")
    return fig


def _logRrtoSE(logRr, alpha, mu, sigma):
    """R's ``logRrtoSE``: the standard error at which calibrated p equals alpha."""
    logRr = np.asarray(logRr, dtype=float)
    phi = (mu - logRr) ** 2 / qnorm(alpha / 2) ** 2 - sigma ** 2
    phi = np.where(phi < 0, 0, phi)
    return np.sqrt(phi)


def _checkWithinLimits(limits, values, label):
    """R's ``checkWithinLimits``."""
    values = np.asarray([v for v in np.atleast_1d(values) if v is not None], dtype=float)
    values = values[~np.isnan(values)]
    if values.size > 0:
        if limits[0] > values.min() or limits[1] < values.max():
            warnings.warn(f"Values are outside plotted range. "
                          f"Consider adjusting {label} parameter")


# --------------------------------------------------------------------------

def plotForest(logRr, seLogRr, names, xLabel="Relative risk", title=None,
               fileName=None):
    """Create a forest plot of effect size estimates.

    Estimates significantly different from 1 (alpha = 0.05) are marked in
    orange, others in blue.

    Parameters
    ----------
    logRr, seLogRr : array_like
        Effect estimates on the log scale and their standard errors.
    names : array_like
        Names of the drugs or outcomes.
    xLabel : str
        Label for the effect estimate axis.
    title : str, optional
        Main title for the plot.
    fileName : str, optional
        Where to save the plot.

    Returns
    -------
    matplotlib.figure.Figure
    """
    logRr = np.asarray(logRr, dtype=float)
    seLogRr = np.asarray(seLogRr, dtype=float)
    names = np.asarray(names, dtype=object)

    breaks = [0.25, 0.5, 1, 2, 4, 6, 8, 10]
    logLb = logRr + qnorm(0.025) * seLogRr
    logUb = logRr + qnorm(0.975) * seLogRr
    significant = (logLb > 0) | (logUb < 0)

    # R turns names into a factor (levels sorted) and reverses them for display
    levels = sorted({str(n) for n in names})
    order = {lev: i for i, lev in enumerate(levels)}
    y = np.array([len(levels) - 1 - order[str(n)] for n in names], dtype=float)

    fig, ax = plt.subplots(figsize=(5, 2.5 + len(logRr) * 0.8))
    _style_panel(ax, major=False)
    for b in breaks:
        ax.axvline(b, color=_GRID, lw=0.2, zorder=0)
    ax.axvline(1, color="black", lw=0.5, zorder=1)
    for i in range(len(logRr)):
        c = _COL[int(significant[i])]
        f = _COL_FILL[int(significant[i])]
        ax.plot([np.exp(logLb[i]), np.exp(logUb[i])], [y[i], y[i]], color=c, lw=1, zorder=3)
        ax.plot([np.exp(logRr[i])], [y[i]], marker="D", markersize=5,
                markerfacecolor=f, markeredgecolor=c, zorder=4)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(breaks))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter([str(b) for b in breaks]))
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlim(0.25, 10)
    ax.set_yticks(np.arange(len(levels)))
    ax.set_yticklabels(list(reversed(levels)), fontsize=5, ha="right")
    ax.set_ylim(-0.5, len(levels) - 0.5)
    ax.tick_params(axis="x", labelsize=6)
    ax.grid(True, color=_MAJOR, axis="y", lw=0.5)
    ax.set_axisbelow(True)
    if title is not None:
        ax.set_title(title, loc="center")
    fig.tight_layout()
    return _save(fig, fileName, 5, 2.5 + len(logRr) * 0.8)


def plotCalibrationEffect(logRrNegatives, seLogRrNegatives, logRrPositives=None,
                          seLogRrPositives=None, null=None, alpha=0.05,
                          xLabel="Relative risk", title=None, showCis=False,
                          showExpectedAbsoluteSystematicError=False,
                          fileName=None, xLimits=(0.25, 10), yLimits=(0, 1.5)):
    """Plot the effect of the calibration.

    Effect estimate on the x-axis, standard error on the y-axis.  Negative
    controls are blue dots, positive controls yellow diamonds.  The area below
    the dashed line indicates estimates with p < 0.05; the orange area indicates
    estimates with calibrated p < 0.05.

    Parameters
    ----------
    logRrNegatives, seLogRrNegatives : array_like
        Estimates of the negative controls and their standard errors.
    logRrPositives, seLogRrPositives : array_like, optional
        Estimates of the positive controls and their standard errors.
    null : Null or McmcNull, optional
        Fitted null distribution.  If omitted, one is fitted before plotting.
    alpha : float
        Alpha for the hypothesis test.
    xLabel : str
        Label for the effect estimate axis.
    title : str, optional
    showCis : bool
        Show 95% credible intervals for the calibrated p = alpha boundary.
        Requires an :class:`McmcNull`.
    showExpectedAbsoluteSystematicError : bool
        Annotate the plot with the expected absolute systematic error.
    fileName : str, optional
    xLimits, yLimits : sequence of 2 floats

    Returns
    -------
    matplotlib.figure.Figure
    """
    if null is None:
        if showCis:
            null = fitMcmcNull(logRrNegatives, seLogRrNegatives)
        else:
            null = fitNull(logRrNegatives, seLogRrNegatives)
    if showCis and isinstance(null, Null):
        raise ValueError("Cannot show credible intervals when using asymptotic null. "
                         "Please use 'fitMcmcNull' to fit the null")
    if len(xLimits) < 2:
        raise ValueError("xLimits must be a vector of length 2")
    if len(yLimits) < 2:
        raise ValueError("yLimits must be a vector of length 2")

    x = np.exp(np.arange(np.log(xLimits[0]), np.log(xLimits[1]) + 1e-12, 0.01))
    yLb = yUb = None
    if isinstance(null, Null):
        y = _logRrtoSE(np.log(x), alpha, null[0], null[1])
    else:
        chain = null.mcmc["chain"]
        with np.errstate(divide="ignore", invalid="ignore"):
            mat = np.array([_logRrtoSE(np.log(x), alpha, row[0], 1 / np.sqrt(row[1]))
                            for row in chain])
        ys = np.nanquantile(mat, [0.025, 0.50, 0.975], axis=0)
        y, yLb, yUb = ys[1], ys[0], ys[2]
    seTheoretical = np.abs(np.log(x)) / qnorm(1 - alpha / 2)

    breaks = [0.25, 0.5, 1] + [2 * i for i in range(1, int(np.ceil(xLimits[1] / 2.0)) + 1)]
    if xLimits[0] < 1:
        breaks = [0.125] + breaks

    _checkWithinLimits(yLimits, np.concatenate([
        np.atleast_1d(seLogRrNegatives),
        np.atleast_1d(seLogRrPositives) if seLogRrPositives is not None else np.array([])]),
        "yLimits")
    _checkWithinLimits(np.log(xLimits), np.concatenate([
        np.atleast_1d(logRrNegatives),
        np.atleast_1d(logRrPositives) if logRrPositives is not None else np.array([])]),
        "xLimits")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    _style_panel(ax)
    for b in breaks:
        ax.axvline(b, color=_GRID, lw=0.5, zorder=0)
    ax.axvline(1, color="black", lw=1, zorder=1)
    # calibrated p < alpha region
    ax.fill_between(x, 0, y, facecolor=(1, 0.5, 0, 0.5), edgecolor=(1, 0.5, 0),
                    linewidth=1, zorder=2)
    if showCis:
        ax.fill_between(x, yLb, yUb, color=(0.8, 0.2, 0.2), alpha=0.3, zorder=3)
        ax.plot(x, yLb, color=(0.8, 0.2, 0.2, 0.2), lw=1, zorder=3)
        ax.plot(x, yUb, color=(0.8, 0.2, 0.2, 0.2), lw=1, zorder=3)
    if showExpectedAbsoluteSystematicError:
        ease = computeExpectedAbsoluteSystematicError(null)
        if isinstance(null, Null):
            label = "Expected absolute systematic error = %0.2f" % ease
        else:
            label = ("Expected absolute systematic error = %0.2f (%0.2f - %0.2f)"
                     % (ease["ease"].iloc[0], ease["ciLb"].iloc[0], ease["ciUb"].iloc[0]))
        ax.text(0.26, 1.49, label, ha="left", va="top", fontsize=3.5 * 2,
                bbox=dict(boxstyle="round", fc="white", alpha=0.9), zorder=6)
    # theoretical p < alpha region
    ax.fill_between(x, 0, seTheoretical, facecolor=(0, 0, 0, 0.1),
                    edgecolor=(0, 0, 0, 0.1), zorder=4)
    ax.plot(x, seTheoretical, color=(0, 0, 0, 0.5), ls="--", lw=1, zorder=4)
    ax.plot(np.exp(np.atleast_1d(logRrNegatives)), np.atleast_1d(seLogRrNegatives),
            "o", color=(0, 0, 0.8), alpha=0.5, markersize=4, ls="none", zorder=5)
    if logRrPositives is not None:
        ax.plot(np.exp(np.atleast_1d(logRrPositives)),
                np.atleast_1d(seLogRrPositives), "D", markerfacecolor=(1, 1, 0),
                markeredgecolor="black", alpha=0.8, markersize=7, ls="none", zorder=6)
    ax.axhline(0, color="black", lw=1, zorder=1)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(breaks))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter([str(b) for b in breaks]))
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    # set limits last: assigning ticks outside the range can widen the view
    ax.set_xlim(xLimits[0], xLimits[1])
    ax.set_xlabel(xLabel, fontsize=12)
    ax.set_ylabel("Standard Error", fontsize=12)
    ax.set_ylim(yLimits[0], yLimits[1])
    ax.tick_params(labelsize=12)
    if title is not None:
        ax.set_title(title, loc="center")
    fig.tight_layout()
    return _save(fig, fileName, 6, 4.5)


def plotCalibration(logRr, seLogRr, useMcmc=False, legendPosition="right",
                    title=None, fileName=None):
    """Create a calibration plot.

    Shows the number of effects with p < alpha for every level of alpha.  The
    empirical calibration uses a leave-one-out design: the p-value of an effect
    is computed by fitting a null on all *other* negative controls.  Ideally the
    calibration line approximates the diagonal.

    Parameters
    ----------
    logRr, seLogRr : array_like
    useMcmc : bool
        Use MCMC to estimate the calibrated p-value.
    legendPosition : str
        ``"none"``, ``"left"``, ``"right"``, ``"bottom"`` or ``"top"``.
    title : str, optional
    fileName : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    from .asymptotics import _clean_estimates
    logRr, seLogRr = _clean_estimates(logRr, seLogRr)

    data = pd.DataFrame({"logRr": logRr, "SE": seLogRr})
    data["Z"] = data["logRr"] / data["SE"]
    z = data["Z"].to_numpy()
    data["P"] = 2 * np.minimum(pnorm(z), 1 - np.asarray(pnorm(z)))
    P = data["P"].to_numpy()
    data["Y"] = [np.sum(P < x) / len(data) for x in P]

    calibratedP = np.empty(len(data), dtype=float)
    for i in range(len(data)):
        keep = np.arange(len(data)) != i
        if useMcmc:
            null = fitMcmcNull(data["logRr"].to_numpy()[keep], data["SE"].to_numpy()[keep])
            calibratedP[i] = np.asarray(_as_p(null, data["logRr"].iloc[i], data["SE"].iloc[i]))
        else:
            null = fitNull(data["logRr"].to_numpy()[keep], data["SE"].to_numpy()[keep])
            calibratedP[i] = np.asarray(_as_p(null, data["logRr"].iloc[i], data["SE"].iloc[i]))
    data["calibratedP"] = calibratedP
    data = data[~data["calibratedP"].isna()].copy()
    cp = data["calibratedP"].to_numpy()
    data["AdjustedY"] = [np.sum(cp < x) / len(data) for x in cp]

    breaks = [0, 0.25, 0.5, 0.75, 1]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    _style_panel(ax)
    for b in breaks:
        ax.axvline(b, color=_GRID, lw=0.3, zorder=0)
        ax.axhline(b, color=_GRID, lw=0.3, zorder=0)
    ax.axvline(0.05, color="#888888", ls="--", lw=1, zorder=1)
    ax.plot([0, 1], [0, 1], color=_GRID, lw=0.3, zorder=1)

    for xv, yv, lab, ls in [
        (data["calibratedP"].to_numpy(), data["AdjustedY"].to_numpy(), "Empirical", "-"),
        (data["P"].to_numpy(), data["Y"].to_numpy(), "Theoretical", (0, (5, 2, 1, 2))),
    ]:
        o = np.argsort(xv)
        ax.step(xv[o], yv[o], where="post", color="black", ls=ls, lw=1, label=lab, zorder=3)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(breaks + [0.05])
    ax.set_xticklabels(["", ".25", ".50", ".75", "1", ".05"])
    ax.set_yticks(breaks)
    ax.set_yticklabels(["0", ".25", ".50", ".75", "1"])
    ax.set_xlabel(r"$\alpha$", fontsize=10)
    ax.set_ylabel(r"Fraction with p < $\alpha$", fontsize=10)
    ax.tick_params(labelsize=10)
    _legend(ax, legendPosition, title="P-value calculation")
    if title is not None:
        ax.set_title(title, loc="center")
    fig.tight_layout()
    return _save(fig, fileName, 6, 4.5)


def _as_p(null, logRr, seLogRr):
    from .asymptotics import calibrateP
    res = calibrateP(null, logRr, seLogRr, pValueOnly=True)
    return np.atleast_1d(np.asarray(res, dtype=float))[0]


def _figure_legend(fig, ax, position, title=None, order=None):
    """Figure-level legend, for the faceted plots where R draws one legend for
    the whole figure rather than one per panel."""
    if position == "none":
        return
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    if order is not None:
        byLabel = dict(zip(labels, handles))
        handles = [byLabel[lab] for lab in order if lab in byLabel]
        labels = [lab for lab in order if lab in byLabel]
    # anchor just outside the figure so the legend never lands on the panel
    # titles; bbox_inches="tight" at save time keeps it in frame
    loc, anchor = {
        "top": ("lower center", (0.5, 1.0)),
        "bottom": ("upper center", (0.5, 0.0)),
        "right": ("center left", (1.0, 0.5)),
        "left": ("center right", (0.0, 0.5)),
    }.get(position, ("lower center", (0.5, 1.0)))
    ncol = len(labels) if position in ("top", "bottom") else 1
    fig.legend(handles, labels, loc=loc, bbox_to_anchor=anchor, title=title,
               frameon=False, fontsize=10, ncol=ncol)


def _legend(ax, position, title=None):
    if position == "none":
        return
    loc = {"right": "center left", "left": "center right",
           "top": "lower center", "bottom": "upper center"}.get(position, "best")
    anchor = {"right": (1.01, 0.5), "left": (-0.01, 0.5),
              "top": (0.5, 1.01), "bottom": (0.5, -0.15)}.get(position)
    ncol = 2 if position in ("top", "bottom") else 1
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc=loc, bbox_to_anchor=anchor, title=title,
                  frameon=False, fontsize=9, ncol=ncol)


def plotCiCalibration(logRr=None, seLogRr=None, trueLogRr=None, strata=None,
                      crossValidationGroup=None, legacy=False, evaluation=None,
                      legendPosition="top", title=None, fileName=None):
    """Create a confidence interval calibration plot.

    Shows the fraction of effects within the confidence interval, using a
    leave-one-out design.  Ideally the calibration line approximates the
    diagonal.  Both theoretical and empirically calibrated intervals are shown.

    Parameters
    ----------
    logRr, seLogRr, trueLogRr : array_like
    strata : array_like, optional
    crossValidationGroup : array_like, optional
    legacy : bool
    evaluation : pandas.DataFrame, optional
        As generated by :func:`evaluateCiCalibration`.  If given, the ``logRr``,
        ``seLogRr``, ``trueLogRr``, ``strata`` and ``legacy`` arguments are
        ignored.
    legendPosition : str
    title : str, optional
    fileName : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    if evaluation is None:
        evaluation = evaluateCiCalibration(
            logRr=logRr, seLogRr=seLogRr, trueLogRr=trueLogRr, strata=strata,
            crossValidationGroup=crossValidationGroup, legacy=legacy)
    vizData = evaluation[evaluation["label"] == "Within confidence interval"]
    trueRrs = sorted(vizData["trueRr"].unique())
    breaks = [0, 0.25, 0.5, 0.75, 1]

    fig, axes = plt.subplots(1, len(trueRrs), sharey=True,
                             figsize=(1 + 2 * len(trueRrs), 3.5), squeeze=False)
    for k, trueRr in enumerate(trueRrs):
        ax = axes[0][k]
        _style_panel(ax)
        for b in breaks:
            ax.axvline(b, color=_GRID, lw=0.3, zorder=0)
            ax.axhline(b, color=_GRID, lw=0.3, zorder=0)
        ax.axvline(0.95, color="#888888", ls="--", lw=1, zorder=1)
        ax.plot([0, 1], [0, 1], color=_GRID, lw=0.3, zorder=1)
        sub = vizData[vizData["trueRr"] == trueRr]
        for kind, ls in [("Calibrated", "-"), ("Uncalibrated", (0, (5, 2, 1, 2)))]:
            s = sub[sub["Confidence interval calculation"] == kind].sort_values("ciWidth")
            ax.plot(s["ciWidth"], s["coverage"], color="black", ls=ls, lw=1,
                    label=kind, zorder=3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks(breaks + [0.95])
        ax.set_xticklabels(["0", ".25", ".50", ".75", "", ".95"], fontsize=10)
        ax.set_title(f"{trueRr:g}", fontsize=10)
        if k == 0:
            ax.set_yticks(breaks)
            ax.set_yticklabels(["0", ".25", ".50", ".75", "1"], fontsize=10)
    # one centred axis title and one figure-wide legend, as in R
    fig.supxlabel("Width of CI", fontsize=10)
    fig.supylabel("Coverage", fontsize=10)
    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    _figure_legend(fig, axes[0][0], legendPosition,
                   title="Confidence interval calculation")
    return _save(fig, fileName, 1 + 2 * len(trueRrs), 3.5)


def plotTrueAndObserved(logRr, seLogRr, trueLogRr, xLabel="Relative risk",
                        title=None, fileName=None):
    """Plot true and observed values, for example from a simulation study.

    Estimates significantly different from the true value (alpha = 0.05) are
    marked in orange, others in blue.

    Returns
    -------
    matplotlib.figure.Figure
    """
    logRr = np.asarray(logRr, dtype=float)
    seLogRr = np.asarray(seLogRr, dtype=float)
    trueLogRr = np.asarray(trueLogRr, dtype=float)

    breaks = [0.25, 0.5, 1, 2, 4, 6, 8, 10]
    data = pd.DataFrame({
        "logRr": logRr,
        "logLb95Rr": logRr + qnorm(0.025) * seLogRr,
        "logUb95Rr": logRr + qnorm(0.975) * seLogRr,
        "trueLogRr": trueLogRr,
        "trueRr": np.round(np.exp(trueLogRr), 2),
    })
    data["significant"] = ((data["logLb95Rr"] > data["trueLogRr"])
                           | (data["logUb95Rr"] < data["trueLogRr"]))
    data = data.sort_values(["trueLogRr", "logRr"]).reset_index(drop=True)

    groups = sorted(data["trueRr"].unique())
    heights = [int((data["trueRr"] == g).sum()) for g in groups]
    fig, axes = plt.subplots(len(groups), 1, figsize=(5, 7), squeeze=False,
                             gridspec_kw={"height_ratios": heights})
    for k, g in enumerate(groups):
        ax = axes[k][0]
        _style_panel(ax, major=False)
        sub = data[data["trueRr"] == g].reset_index(drop=True)
        for b in breaks:
            ax.axvline(b, color=_GRID, lw=0.2, zorder=0)
        for i in range(len(sub)):
            c = _COL[int(sub["significant"].iloc[i])]
            f = _COL_FILL[int(sub["significant"].iloc[i])]
            ax.plot([np.exp(sub["logLb95Rr"].iloc[i]), np.exp(sub["logUb95Rr"].iloc[i])],
                    [i, i], color=c, lw=1, zorder=3)
            ax.plot([np.exp(sub["logRr"].iloc[i])], [i], "o", markersize=4,
                    markerfacecolor=f, markeredgecolor=c, zorder=4)
        ax.axvline(g, color="black", lw=1, zorder=2)
        ax.set_xscale("log")
        ax.set_xlim(0.25, 10)
        ax.set_ylim(-1, len(sub))
        ax.set_yticks([])
        ax.grid(True, color=_MAJOR, lw=0.5)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(FixedLocator(breaks))
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlim(0.25, 10)
        if k == len(groups) - 1:
            ax.xaxis.set_major_formatter(
                matplotlib.ticker.FixedFormatter([str(b) for b in breaks]))
            ax.tick_params(axis="x", labelsize=6)
        else:
            ax.xaxis.set_major_formatter(NullFormatter())
    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    return _save(fig, fileName, 5, 7)


def plotMcmcTrace(mcmcNull, fileName=None):
    """Plot the MCMC trace for diagnostic purposes.

    Parameters
    ----------
    mcmcNull : McmcNull
        As generated by :func:`fitMcmcNull`.
    fileName : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    chain = mcmcNull.mcmc["chain"]
    x = np.arange(1, chain.shape[0] + 1)
    fig, axes = plt.subplots(2, 1, figsize=(5, 3.5), sharex=True)
    for ax, col, name in [(axes[0], 0, "Mean"), (axes[1], 1, "Precision")]:
        ax.plot(x, chain[:, col], color="black", alpha=0.7, lw=0.5)
        ax.set_ylabel("")
        ax.text(1.01, 0.5, name, transform=ax.transAxes, rotation=-90,
                va="center", ha="left")
    axes[1].set_xlabel("Iterations")
    fig.tight_layout()
    return _save(fig, fileName, 5, 3.5)


def plotCiCoverage(logRr=None, seLogRr=None, trueLogRr=None, strata=None,
                   crossValidationGroup=None, legacy=False, evaluation=None,
                   legendPosition="top", title=None, fileName=None):
    """Create a confidence interval coverage plot.

    Shows the fraction of effects above, within and below the confidence
    interval, for both theoretical and empirically calibrated intervals, using a
    leave-one-out design.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if evaluation is None:
        evaluation = evaluateCiCalibration(
            logRr=logRr, seLogRr=seLogRr, trueLogRr=trueLogRr, strata=strata,
            crossValidationGroup=crossValidationGroup, legacy=legacy)
    evaluation = evaluation.copy()
    evaluation["trueRr"] = "True effect size = " + evaluation["trueRr"].map(lambda v: f"{v:g}")

    trueRrs = sorted(evaluation["trueRr"].unique())
    kinds = ["Uncalibrated", "Calibrated"]
    # R reverses the label levels to (Above, Within, Below) and position_stack
    # puts the first level on top, so bottom-to-top the bands are Below,
    # Within, Above.  The fill scale is applied in level order, giving
    # Above = red, Within = grey, Below = blue.
    stack = ["Below confidence interval", "Within confidence interval",
             "Above confidence interval"]
    fillCols = {"Below confidence interval": (0, 0, 0.8),
                "Within confidence interval": (0.75, 0.75, 0.75),
                "Above confidence interval": (0.8, 0, 0)}
    breaks = [0, 0.25, 0.5, 0.75, 1]

    fig, axes = plt.subplots(len(kinds), len(trueRrs), sharex=True, sharey=True,
                             figsize=(1 + 1.8 * len(trueRrs), 5), squeeze=False)
    for r, kind in enumerate(kinds):
        for c, trueRr in enumerate(trueRrs):
            ax = axes[r][c]
            ax.set_facecolor("white")
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(length=0)
            for b in breaks:
                ax.axvline(b, color=_GRID, lw=0.3, zorder=0)
                ax.axhline(b, color=_GRID, lw=0.3, zorder=0)
            sub = evaluation[(evaluation["trueRr"] == trueRr)
                             & (evaluation["Confidence interval calculation"] == kind)]
            base = None
            for label in stack:
                s = sub[sub["label"] == label].sort_values("ciWidth")
                if len(s) == 0:
                    continue
                xv = s["ciWidth"].to_numpy()
                yv = s["coverage"].to_numpy()
                if base is None:
                    base = np.zeros_like(yv)
                ax.fill_between(xv, base, base + yv, facecolor=fillCols[label],
                                alpha=0.6, label=label if (r == 0 and c == 0) else None,
                                zorder=2)
                base = base + yv
            ax.plot([0, 1], [0.5, 1.0], color="black", lw=1, ls="--", zorder=3)
            ax.plot([0, 1], [0.5, 0.0], color="black", lw=1, ls="--", zorder=3)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks(breaks)
            ax.set_xticklabels(["0", ".25", ".50", ".75", "1"], fontsize=12)
            ax.set_yticks(breaks)
            ax.set_yticklabels(["0", ".25", ".50", ".75", "1"], fontsize=12)
            if r == 0:
                ax.set_title(trueRr, fontsize=12)
            if c == len(trueRrs) - 1:
                ax.text(1.03, 0.5, kind, transform=ax.transAxes, rotation=-90,
                        va="center", ha="left", fontsize=12)
    # R draws a single centred axis title and a figure-wide legend; per-axes
    # labels would be repeated once per column and overlap into unreadable text.
    fig.supxlabel("Width of the confidence interval", fontsize=12)
    fig.supylabel("Fraction", fontsize=12)
    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    _figure_legend(fig, axes[0][0], legendPosition, order=list(reversed(stack)))
    return _save(fig, fileName, 1 + 1.8 * len(trueRrs), 5)


def plotErrorModel(logRr, seLogRr, trueLogRr, title=None, legacy=False, fileName=None):
    """Plot the systematic error model.

    True effect size on the x-axis, the mean plus and minus the standard
    deviation on the y-axis.  Also shown are simple error models fitted at each
    true relative risk in the input.

    Returns
    -------
    matplotlib.figure.Figure
    """
    logRr = np.asarray(logRr, dtype=float)
    seLogRr = np.asarray(seLogRr, dtype=float)
    trueLogRr = np.asarray(trueLogRr, dtype=float)

    model = fitSystematicErrorModel(logRr=logRr, seLogRr=seLogRr,
                                    trueLogRr=trueLogRr, legacy=legacy)
    uniqueTrue = pd.unique(trueLogRr)
    simple = []
    for t in uniqueTrue:
        idx = trueLogRr == t
        n = fitNull(logRr[idx], seLogRr[idx])
        simple.append((np.exp(t), np.exp(n[0]), np.exp(n[0] - n[1]), np.exp(n[0] + n[1])))
    simple = np.array(simple)

    breaks = [0.25, 0.5, 1, 2, 4, 6, 8, 10]
    x = np.exp(np.arange(np.log(0.25), np.log(10) + 1e-12, 0.01))
    y = np.exp(model[0] + model[1] * np.log(x))
    if legacy:
        ymin = np.exp(np.log(y) - np.exp(model[2] + model[3] * np.log(x)))
        ymax = np.exp(np.log(y) + np.exp(model[2] + model[3] * np.log(x)))
    else:
        ymin = np.exp(np.log(y) - (model[2] + model[3] * np.abs(np.log(x))))
        ymax = np.exp(np.log(y) + (model[2] + model[3] * np.abs(np.log(x))))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    _style_panel(ax)
    for b in breaks:
        ax.axvline(b, color=_GRID, lw=0.2, zorder=0)
        ax.axhline(b, color=_GRID, lw=0.2, zorder=0)
    ax.axvline(1, color="black", lw=1, zorder=1)
    ax.axhline(1, color="black", lw=1, zorder=1)
    ax.fill_between(x, ymin, ymax, color=(0, 0, 0.8), alpha=0.3, zorder=2)
    ax.plot(x, y, color=(0, 0, 0.8), zorder=3)
    for trueRr, yv, lo, hi in simple:
        ax.plot([trueRr, trueRr], [lo, hi], color=(0, 0, 0.8), lw=1, zorder=4)
        for cap in (lo, hi):
            ax.plot([trueRr * 0.95, trueRr * 1.05], [cap, cap],
                    color=(0, 0, 0.8), lw=1, zorder=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.25, 10)
    ax.set_ylim(0.25, 10)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(FixedLocator(breaks))
        axis.set_major_formatter(matplotlib.ticker.FixedFormatter([str(b) for b in breaks]))
        axis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlim(0.25, 10)
    ax.set_ylim(0.25, 10)
    ax.set_xlabel("True effect size")
    ax.set_ylabel("Systematic error mean (plus and minus one SD)")
    if title is not None:
        ax.set_title(title, loc="center")
    fig.tight_layout()
    return _save(fig, fileName, 6, 4.5)


def _plotIsobars(ax, null, alpha, xLabel="Relative risk", seLogRrPositives=None):
    """R's internal ``plotIsobars``, drawn onto an existing axis."""
    if isinstance(null, McmcNull):
        null = np.array([null[0], 1 / np.sqrt(null[1])])
    null = np.asarray(null, dtype=float)
    x = np.exp(np.arange(np.log(0.25), np.log(10) + 1e-12, 0.01))
    y = np.abs(np.log(x)) / qnorm(1 - alpha / 2)
    thresholds = [0.05, 0.25, 0.5, 0.75]
    breaks = [0.25, 0.5, 1, 2, 4, 6, 8, 10]

    _style_panel(ax)
    for b in breaks:
        ax.axvline(b, color=(0, 0, 0, 0.2), lw=0.5, zorder=0)
    for h in np.arange(0, 5) / 4:
        ax.axhline(h, color=(0, 0, 0, 0.2), lw=0.5, zorder=0)
    ax.axvline(1, color="black", lw=1, zorder=1)
    ax.plot(x, y, color=(0, 0, 0, 0.5), ls="--", lw=1, zorder=2)

    for threshold in thresholds:
        yy = _logRrtoSE(np.log(x), threshold, null[0], null[1])
        mask = yy < y
        ax.fill_between(x[mask], yy[mask], y[mask], color=(1, 0.5, 0, 0.2), zorder=2)
        ax.plot(x, yy, color=(1, 0.5, 0), lw=1, alpha=0.5, zorder=3)
        left = x < np.exp(null[0])
        right = x > np.exp(null[0])
        for side in (left, right):
            if not np.any(side):
                continue
            xs, ys = x[side], yy[side]
            j = int(np.argmin(np.abs(ys - 0.625)))
            ax.text(xs[j], 0.625, f"{100 * threshold:g}%", color=(1, 0.5, 0),
                    ha="center", va="center",
                    bbox=dict(boxstyle="round", fc="white", ec=(1, 0.5, 0)), zorder=5)
    if seLogRrPositives is not None:
        for s in np.atleast_1d(seLogRrPositives):
            ax.axhline(s, color="black", lw=1, zorder=4)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(breaks))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter([str(b) for b in breaks]))
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlim(0.25, 10)
    ax.set_ylim(0, 1)
    ax.set_xlabel(xLabel, fontsize=12)
    ax.set_ylabel("Standard Error", fontsize=12)
    ax.tick_params(labelsize=12)


def plotExpectedType1Error(logRrNegatives, seLogRrNegatives, seLogRrPositives=None,
                           alpha=0.05, null=None, xLabel="Relative risk",
                           title=None, showCis=False, showEffectSizes=False,
                           fileName=None):
    """Plot the expected type 1 error as a function of standard error.

    The red line indicates the expected type 1 error given the estimated
    empirical null distribution if no calibration is performed.  The dashed line
    indicates the nominal expected type 1 error rate, assuming the theoretical
    null.  Standard errors of non-negative estimates are plotted on the red line
    as yellow diamonds.

    Parameters
    ----------
    logRrNegatives, seLogRrNegatives : array_like
    seLogRrPositives : array_like, optional
    alpha : float
        Nominal type 1 error.
    null : Null or McmcNull, optional
    xLabel : str
    title : str, optional
    showCis : bool
        Show 95% credible intervals; requires an :class:`McmcNull`.
    showEffectSizes : bool
        Show the expected effect sizes alongside the expected type 1 error.
    fileName : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    if null is None:
        if showCis:
            null = fitMcmcNull(logRrNegatives, seLogRrNegatives)
        else:
            null = fitNull(logRrNegatives, seLogRrNegatives)
    if showCis and isinstance(null, Null):
        raise ValueError("Cannot show credible intervals when using asymptotic null. "
                         "Please use 'fitMcmcNull' to fit the null")

    se = np.arange(0, 101) / 100
    lb = ub = None
    if isinstance(null, Null):
        mean = null[0]
        sd = np.sqrt(se ** 2 + null[1] ** 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            type1Error = (np.asarray(pnorm(qnorm(1 - alpha / 2, 0, se), mean, sd,
                                           lower_tail=False))
                          + np.asarray(pnorm(qnorm(alpha / 2, 0, se), mean, sd)))
    else:
        chain = null.mcmc["chain"]
        est = []
        with np.errstate(divide="ignore", invalid="ignore"):
            for s in se:
                mean = chain[:, 0]
                sd = np.sqrt(s ** 2 + (1 / np.sqrt(chain[:, 1])) ** 2)
                t1 = (np.asarray(pnorm(qnorm(1 - alpha / 2, 0, s), mean, sd,
                                       lower_tail=False))
                      + np.asarray(pnorm(qnorm(alpha / 2, 0, s), mean, sd)))
                est.append(np.quantile(t1, [0.025, 0.5, 0.975]))
        est = np.array(est).T
        type1Error, lb, ub = est[1], est[0], est[2]

    breaks = list(np.arange(0, 5) / 4)

    if showEffectSizes:
        fig = plt.figure(figsize=(10, 5))
        gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])
        axIso = fig.add_subplot(gs[0, 0])
        ax = fig.add_subplot(gs[0, 1])
        _plotIsobars(axIso, null, alpha, xLabel=xLabel,
                     seLogRrPositives=seLogRrPositives)
    else:
        fig, ax = plt.subplots(figsize=(5, 5))

    _style_panel(ax)
    # when showing effect sizes R flips the coordinates
    def draw_line(xv, yv, **kw):
        return ax.plot(yv, xv, **kw) if showEffectSizes else ax.plot(xv, yv, **kw)

    for b in breaks:
        (ax.axhline if showEffectSizes else ax.axvline)(b, color=(0, 0, 0, 0.2), lw=0.5, zorder=0)
        (ax.axvline if showEffectSizes else ax.axhline)(b, color=(0, 0, 0, 0.2), lw=0.5, zorder=0)
    (ax.axvline if showEffectSizes else ax.axhline)(
        alpha, color=(0, 0, 0, 0.5), ls="--", lw=1, zorder=1)
    if showCis:
        if showEffectSizes:
            ax.fill_betweenx(se, lb, ub, color=(0.8, 0.2, 0.2), alpha=0.3, zorder=2)
        else:
            ax.fill_between(se, lb, ub, color=(0.8, 0.2, 0.2), alpha=0.3, zorder=2)
        draw_line(se, lb, color=(0.8, 0.2, 0.2, 0.2), lw=1, zorder=2)
        draw_line(se, ub, color=(0.8, 0.2, 0.2, 0.2), lw=1, zorder=2)
    draw_line(se, type1Error, color=(0.8, 0, 0), lw=1, alpha=0.5, zorder=3)

    labelPos = (0.875 + showEffectSizes * 0.125, alpha + showEffectSizes * 0.04)
    ax.text(labelPos[1] if showEffectSizes else labelPos[0],
            labelPos[0] if showEffectSizes else labelPos[1],
            rf"$\alpha = {alpha}$", ha="center", va="center",
            bbox=dict(boxstyle="round", fc="white"), zorder=5)

    if seLogRrPositives is not None:
        yPos = np.array([type1Error[int(np.argmin(np.abs(x - se)))]
                         for x in np.atleast_1d(seLogRrPositives)])
        for xv, yv in zip(np.atleast_1d(seLogRrPositives), yPos):
            (ax.axhline if showEffectSizes else ax.axvline)(xv, color="black", lw=1, zorder=4)
            if showEffectSizes:
                ax.plot([yv], [xv], "D", markersize=7, markerfacecolor=(1, 1, 0),
                        markeredgecolor="black", alpha=0.8, zorder=5)
            else:
                ax.plot([xv], [yv], "D", markersize=7, markerfacecolor=(1, 1, 0),
                        markeredgecolor="black", alpha=0.8, zorder=5)

    if showEffectSizes:
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)
        ax.set_xticks(breaks)
        ax.set_xticklabels([f"{b:g}" for b in breaks], fontsize=12)
        ax.set_yticklabels([])
        ax.set_xlabel("Expected type 1 error", fontsize=12)
    else:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks(breaks)
        ax.set_yticks(breaks)
        ax.set_xticklabels([f"{b:g}" for b in breaks], fontsize=12)
        ax.set_yticklabels([f"{b:g}" for b in breaks], fontsize=12)
        ax.set_xlabel("Standard error", fontsize=12)
        ax.set_ylabel("Expected type 1 error", fontsize=12)

    if title is not None:
        fig.suptitle(title) if showEffectSizes else ax.set_title(title, loc="center")
    fig.tight_layout()
    return _save(fig, fileName, 10 if showEffectSizes else 5, 5)


def plotCiCalibrationEffect(logRr, seLogRr, trueLogRr, legacy=False, model=None,
                            xLabel="Relative risk", title=None, fileName=None):
    """Plot the effect of the confidence interval calibration.

    Effect estimate on the x-axis, standard error on the y-axis, trellised by
    true effect size.  Controls are shown as blue dots.  The area below the
    dashed line indicates estimates statistically significantly different from
    the true effect size (p < 0.05); the orange area indicates estimates with
    calibrated p < 0.05.

    Returns
    -------
    matplotlib.figure.Figure or None
        ``None`` if no non-missing estimates remain, matching R.
    """
    alpha = 0.05
    logRr = np.asarray(logRr, dtype=float)
    seLogRr = np.asarray(seLogRr, dtype=float)
    trueLogRr = np.asarray(trueLogRr, dtype=float)

    if model is None:
        model = fitSystematicErrorModel(logRr=logRr, seLogRr=seLogRr,
                                        trueLogRr=trueLogRr,
                                        estimateCovarianceMatrix=False, legacy=legacy)
    else:
        legacy = getattr(model, "names", [None, None, None])[2] == "logSdIntercept"

    d = pd.DataFrame({
        "logRr": logRr, "seLogRr": seLogRr, "trueLogRr": trueLogRr,
        "trueRr": np.exp(trueLogRr),
        "logCi95lb": logRr + qnorm(0.025) * seLogRr,
        "logCi95ub": logRr + qnorm(0.975) * seLogRr,
    })
    d = d[~d["logRr"].isna()]
    d = d[~d["seLogRr"].isna()]
    if len(d) == 0:
        return None
    d["Significant"] = (d["logCi95lb"] > d["trueLogRr"]) | (d["logCi95ub"] < d["trueLogRr"])

    grouped = d.groupby("trueRr", sort=True)["Significant"]
    dd = pd.DataFrame({"trueRr": [], "nLabel": [], "meanLabel": []})
    rows = []
    for trueRr, s in grouped:
        rows.append({
            "trueRr": trueRr,
            "nLabel": f"{len(s):,} estimates",
            "meanLabel": f"{100 * (1 - s.mean()):.1f}% of CIs includes {trueRr:g}",
        })
    dd = pd.DataFrame(rows)

    breaks = [0.1, 0.25, 0.5, 1, 2, 4, 6, 8, 10]
    x = np.arange(np.log(0.1), np.log(10) + 1e-12, 0.01)

    fig, axes = plt.subplots(1, len(dd), sharey=True, squeeze=False,
                             figsize=(2 + 3.5 * len(dd), 2.8))
    for k in range(len(dd)):
        ax = axes[0][k]
        trueRr = dd["trueRr"].iloc[k]
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
        for b in breaks:
            ax.axvline(np.log(b), color=_GRID, lw=0.5, zorder=0)
        mu = model[0] + model[1] * np.log(trueRr)
        if legacy:
            sigma = np.exp(model[2] + model[3] * np.log(trueRr))
        else:
            sigma = model[2] + model[3] * abs(np.log(trueRr))
        yCal = _logRrtoSE(x, alpha, mu, sigma)
        ax.fill_between(x, 0, yCal, facecolor=(1, 0.5, 0, 0.5), edgecolor=(1, 0.5, 0),
                        linewidth=1, zorder=1)
        # theoretical significance boundaries
        ax.plot(x, (x - np.log(trueRr)) / qnorm(0.025), color=(0, 0, 0, 0.5),
                ls="--", lw=1, zorder=2)
        ax.plot(x, (x - np.log(trueRr)) / qnorm(0.975), color=(0, 0, 0, 0.5),
                ls="--", lw=1, zorder=2)
        sub = d[d["trueRr"] == trueRr]
        ax.plot(sub["logRr"], sub["seLogRr"], "o", color=(0, 0, 0.8), alpha=0.5,
                markersize=4, ls="none", zorder=3)
        ax.axhline(0, color="black", lw=1, zorder=2)
        ax.text(np.log(0.15), 0.95, dd["nLabel"].iloc[k], ha="left", va="center",
                fontsize=7, bbox=dict(boxstyle="round", fc="white"), zorder=5)
        ax.text(np.log(0.15), 0.8, dd["meanLabel"].iloc[k], ha="left", va="center",
                fontsize=7, bbox=dict(boxstyle="round", fc="white"), zorder=5)
        ax.set_xlim(np.log(0.1), np.log(10))
        ax.set_ylim(0, 1)
        ax.set_xticks([np.log(b) for b in breaks])
        ax.set_xticklabels([str(b) for b in breaks], fontsize=10)
        ax.set_xlabel(xLabel, fontsize=10)
        ax.set_title(f"True {xLabel.lower()} = {trueRr:g}", fontsize=10)
        if k == 0:
            ax.set_ylabel("Standard Error", fontsize=10)
    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    return _save(fig, fileName, 2 + 3.5 * len(dd), 2.8)
