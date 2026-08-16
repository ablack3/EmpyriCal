Before you open a pull request, please **file an issue** first and check that the
maintainer agrees it's a problem and is happy with your proposed fix. That saves
you spending time on something we might not want to merge.

Additional requirements for pull requests:

- If possible, add tests for new functionality.

- Restrict your pull request to solving the issue at hand. Do not try to
  'improve' parts of the code that are not related to the issue. If you feel
  other parts of the code need better organization, create a separate issue.

- Make sure the test suite passes before submitting:

  ```bash
  python -m pytest -q
  ```

  The slowest tests are marked `slow`; `-m "not slow"` gives a fast loop, but
  please run the full suite at least once before opening the PR.

- Make sure the linter is clean:

  ```bash
  ruff check src tests
  ```

- Target the `main` branch.

**A note on numerical fidelity.** This package is a port of the OHDSI R package
[EmpiricalCalibration](https://github.com/OHDSI/EmpiricalCalibration), and many
functions are required to reproduce R's output bit-for-bit. If you change
anything under `src/empiricalcalibration/`, say in the PR whether the numbers
move, and check the R-parity assertions in `tests/` still hold. The gold
standard and the comparison harness are in `extras/r-parity/`.
