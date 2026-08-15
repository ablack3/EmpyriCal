# R comparison harness

Regenerates the reference values used to verify that this port computes the same
numbers as the R package.

```bash
# 1. produce gold.json from R (needs R with EmpiricalCalibration 3.1.4 installed)
Rscript generate_r_gold_standard.R

# 2. compare the Python package against it
PYTHONPATH=../src python compare_to_r.py
```

`compare_to_r.py` prints one row per checkpoint with the maximum relative
deviation and how many values are bit-identical, and exits non-zero if any
checkpoint deviates by more than 1e-5. `gold.json` is the output of step 1 on
R 4.6.1 / macOS arm64 and is committed so step 2 can be run without R.
