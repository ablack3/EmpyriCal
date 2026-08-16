# Changelog

## 0.1.1

### Fixed — incorrect numerical results

- **`computeCvBinomial` returned a critical value up to ~10x too small when any
  `groupSizes` entry was non-integer.** `groupSizes` are *expected* event counts
  under the null, so non-integer values are the ordinary case. Such counts make
  `observed / count` exceed 1, both `dbinom` terms return `-inf`, and their
  difference is `nan`; the vectorised sampler then combined that with
  `np.maximum`, which propagates `nan` and destroyed the running maximum for the
  rest of that sample. The `values.max() == 0` guard could not catch it, because
  `nan == 0` is False. Nothing was raised or warned — the function simply
  returned a threshold that signals far more readily than its nominal alpha.
  All three vectorised samplers now use `np.fmax`, which drops `nan` exactly as
  R's `if (llr > maxLlr)` does, and the fast path is asserted bit-identical to
  the exact scalar path.

- **`plotCalibration` dropped estimates with `abs(logRr) > log(100)`, which R
  does not.** R's `plotCalibration` inlines four validity filters; the fifth
  belongs only to `fitNull`/`fitMcmcNull`. The extra filter changed the plotted
  points, both curves, and every leave-one-out denominator.

### Fixed — crashes on valid input

- **`computeTraditionalCi` rejected a scalar estimate against a vector of
  standard errors.** 0.1.0 accepted scalar+scalar but raised
  `ValueError: All arrays must be of the same length` for the
  one-estimate-many-precisions sweep that R recycles. Both arguments are now
  broadcast.
- **`plotTrueAndObserved` raised on scalar arguments**, the same defect in a
  different module.
- **`computeCvPoissonRegression(..., minimumEvents=0)` raised
  `ZeroDivisionError`.** `minimumEvents=0` is accepted by validation and
  disabled the guard on a division by the observed count.

### Performance

- Non-normal likelihood fitting is roughly twice as fast. The likelihood
  approximations re-read their constants out of a one-row DataFrame on every
  quadrature evaluation — 4.3 million pandas scalar lookups in one profiled fit,
  65% of its runtime. Parameters are now hoisted into a plain dict once per
  call, which is arithmetically identical (verified bit-for-bit on the custom
  and skew-normal branches). The test suite went from 468s to 387s.

### Packaging

- The 0.1.0 **sdist wrongly contained 13 R source files** from
  `extras/r-package/`: the `include` patterns were unanchored and matched any
  directory named `tests` or file named `README.md` at any depth. Now anchored
  (56 files -> 42).
- `requires = ["hatchling>=1.27"]` — the PEP 639 `license`/`license-files` keys
  need it, and older backends failed with a confusing build-backend crash.
- The version is now read from `__version__` by `[tool.hatch.version]`, so the
  installed metadata and `ec.__version__` cannot drift apart.
- The test suite no longer uses `groupby().apply(include_groups=...)`, which
  requires pandas 2.2 while the package declares `pandas>=1.4`.

### Documentation

- **The confidence-interval bound equation had its lower and upper bounds
  swapped.** Since `z = qnorm((1-w)/2)` is negative, the equation as written
  solves for the *upper* bound, not the lower.
- `plotErrorModel` was described on three pages as a trellis of per-effect-size
  panels showing control points; it draws a single pair of axes with a fitted
  mean line and a standard-deviation band. `plotCiCalibration` was described as
  plotting estimates and intervals; it plots coverage curves.
- Every "Edit this page" link on the documentation site 404'd — `repo-subdir`
  was missing, so links resolved relative to the repository root rather than
  `docs/`.

## 0.1.0

First release. Python port of the OHDSI R package
[EmpiricalCalibration](https://github.com/OHDSI/EmpiricalCalibration) 3.1.4,
verified against R on 56 checkpoints covering every exported function.
