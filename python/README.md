# EmpiricalCalibration (Python)

A Python port of the OHDSI R package
[EmpiricalCalibration](https://github.com/OHDSI/EmpiricalCalibration) v3.1.4.

By using a set of negative control hypotheses we can estimate the empirical null
distribution of a particular observational study setup. This empirical null
distribution can be used to compute a **calibrated p-value**, which reflects the
probability of observing an estimated effect size when the null hypothesis is
true, taking both random and systematic error into account. A similar approach
calibrates **confidence intervals**, using both negative and positive controls.

* Schuemie MJ, Ryan PB, DuMouchel W, Suchard MA, Madigan D. *Interpreting
  observational studies: why empirical calibration is needed to correct
  p-values.* Statistics in Medicine 33(2):209-18, 2014.
  [doi:10.1002/sim.5925](https://doi.org/10.1002/sim.5925)
* Schuemie MJ, Hripcsak G, Ryan PB, Madigan D, Suchard MA. *Empirical confidence
  interval calibration for population-level effect estimation studies in
  observational healthcare data.* PNAS 115(11):2571-2577, 2018.
  [doi:10.1073/pnas.1708282114](https://doi.org/10.1073/pnas.1708282114)

## Install

```bash
pip install -e .
```

## Quick start

```python
import empiricalcalibration as ec

sccs = ec.datasets.sccs()
negatives = sccs[sccs.groundTruth == 0]
positive  = sccs[sccs.groundTruth == 1]

# Fit the empirical null on the negative controls
null = ec.fitNull(negatives.logRr, negatives.seLogRr)
print(null)
#> Estimated null distribution
#>
#>      Estimate
#> Mean 0.792158
#>   SD 0.283436

ec.computeTraditionalP(positive.logRr, positive.seLogRr)   # array([0.])
ec.calibrateP(null, positive.logRr, positive.seLogRr)      # array([0.83891417])

# Calibrated confidence intervals
model = ec.convertNullToErrorModel(null)
ec.calibrateConfidenceInterval(positive.logRr, positive.seLogRr, model)

# Plots return a matplotlib Figure
fig = ec.plotCalibrationEffect(negatives.logRr, negatives.seLogRr,
                               positive.logRr, positive.seLogRr)
fig.savefig("calibration.png", dpi=150)
```

## Mapping from R

Function names, argument names, argument order and defaults are unchanged, so R
code translates line by line:

| R | Python |
|---|---|
| `data(sccs)` | `ec.datasets.sccs()` |
| `fitNull(logRr, seLogRr)` | `ec.fitNull(logRr, seLogRr)` |
| `calibrateP(null, logRr, seLogRr)` | `ec.calibrateP(null, logRr, seLogRr)` |
| `null[1]`, `null[2]` | `null[0]`, `null[1]` (or `null["mean"]`, `null.mean`) |
| `attr(cv, "alpha")` | `cv.alpha` |
| `attr(model, "LB95CI")` | `model.attributes["LB95CI"]` |
| `attr(delta, "ease1")` | `delta.attrs["ease1"]` |
| `attr(null, "mcmc")$chain` | `null.mcmc["chain"]` |
| `set.seed(42)` | `ec.set_seed(42)` |
| `ggsave(...)` / `fileName=` | `fileName=` (unchanged) |

Everything exported by the R package is available:

**Null distribution & p-values** — `fitNull`, `fitNullNonNormalLl`, `fitMcmcNull`,
`calibrateP`, `computeTraditionalP`
**Confidence intervals** — `fitSystematicErrorModel`, `calibrateConfidenceInterval`,
`computeTraditionalCi`, `convertNullToErrorModel`
**Systematic error** — `computeExpectedAbsoluteSystematicError`, `compareEase`
**MaxSPRT** — `calibrateLlr`, `computeCvPoisson`, `computeCvBinomial`,
`computeCvPoissonRegression`
**Evaluation & simulation** — `evaluateCiCalibration`, `simulateControls`,
`simulateMaxSprtData`
**Plots** — `plotForest`, `plotCalibration`, `plotCalibrationEffect`,
`plotCiCalibration`, `plotCiCalibrationEffect`, `plotCiCoverage`,
`plotErrorModel`, `plotExpectedType1Error`, `plotMcmcTrace`, `plotTrueAndObserved`

## Numerical fidelity

The point of this port is that it computes the *same numbers*, not merely the
same statistics. To get there, several pieces of R were ported rather than
approximated with a SciPy equivalent:

| Piece | Why |
|---|---|
| `_roptim.nmmin` | R's Nelder-Mead (`optim` default). SciPy's stops on different criteria and lands 3-4 decimals away. |
| `_roptim.vmmin` | R's BFGS, with R's finite-difference gradient (`ndeps = 1e-3`). |
| `_roptim.optimhess` | R's finite-difference Hessian, for `estimateCovarianceMatrix=True`. |
| `_roptim.uniroot` | R's Brent `zeroin`, with `tol = eps^0.25`. |
| `_rintegrate.integrate` | QUADPACK with R's tolerances (`1.22e-4`), not SciPy's (`1.49e-8`). |
| `_rrng` | R's Mersenne-Twister, `set.seed` scrambling, inversion normals, `rpois`, `rbinom`, and `R_unif_index` rejection sampling. |
| `_rmath.dnorm/dpois/dbinom` | R's `nmath` (Loader's saddle point); SciPy differs by up to 2.4e-13. |

Verified against R 4.6.1 / EmpiricalCalibration 3.1.4 on 56 checkpoints covering
every exported function. Worst relative deviation observed, by group:

| Agreement | Functions |
|---|---|
| **Bit-identical** (0) | `fitNull`, `fitNullNonNormalLl`, `convertNullToErrorModel`, `computeTraditionalP`, `computeTraditionalCi`, `computeExpectedAbsoluteSystematicError`, `evaluateCiCalibration`, `compareEase`, all three `computeCv*` critical values, `simulateMaxSprtData`, the MCMC starting state, and the packaged datasets |
| **~1e-15** | `calibrateLlr`, `computePFromLlr`/`computeLlrFromP`, `fitMcmcNull`'s posterior summaries, `calibrateP` under an MCMC null, the raw MaxSPRT samplers |
| **~1e-13** | `calibrateConfidenceInterval` (Brent root-finding at `tol = eps^0.25`), `simulateControls` |
| **~3e-10** | `fitSystematicErrorModel` and especially its covariance matrix, where BFGS accumulates through finite-difference gradients that amplify last-ulp differences |

`ec.set_seed(n)` reproduces R's `set.seed(n)` stream, so stochastic functions
agree run-for-run rather than merely in distribution. The raw draws —
`unif_rand`, `rpois`, `rbinom` and `sample.int` — are **bit-identical** to R's.
Normal draws differ by at most 1 ulp (R's `qnorm` AS 241 rounds differently in
the last bit), and values derived as `mu + sigma * z` or `a + (b - a) * u` differ
by ~1e-14 relative because R's C compiler contracts those expressions into a
fused multiply-add on this platform while NumPy rounds each operation
separately. That is why `simulateControls` is listed at ~1e-13 rather than as
bit-identical, even though its underlying uniform stream matches exactly.

## Deliberate differences from the R package

1. **Plots use matplotlib** and return a `matplotlib.figure.Figure` instead of a
   ggplot object. Signatures, colours, breaks, labels and saved dimensions are
   unchanged. `plotExpectedType1Error(showEffectSizes=True)` returns a two-panel
   Figure where R returns a `gtable`.
2. **`cohortMethod.groundTruth` is an integer.** In the R data it is a *factor*
   with levels `"0"`/`"1"`, while `sccs` and `caseControl` store it as a number.
   Making it an integer everywhere matches the documented format and keeps
   `df.groundTruth == 1` working consistently.
3. **`title` and `evaluation` default to `None`** rather than being R "missing"
   arguments; passing `None` is the same as omitting them.
4. **Errors are Python exceptions.** R's `stop()` becomes `ValueError`, and
   `warning()` becomes `warnings.warn(...)` with the same message text.
5. **`print()` messages** (R's `message()`, e.g. "Detected data following normal
   distribution", "Selected alpha: ...") go to stdout rather than stderr.

## Performance

The `computeCv*` functions default to `sampleSize=1_000_000` and take roughly
5-8 s each, versus under a second for the R package's C++ implementation. A
vectorised path is used whenever every Poisson mean is below 10 and every
binomial `n*p` below 30 — the regimes where R's samplers consume exactly one
uniform per variate — with an exact scalar fallback otherwise, so the result
matches R either way. Lower `sampleSize` if you need speed over precision.

## Tests

```bash
python -m pytest tests -q
```

The suite is a port of the R package's `testthat` suite, plus regression tests
pinning values produced by R.

## License

Apache 2.0, as the original.
