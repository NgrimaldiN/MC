# Control variates for a GARCH model

Deliverables:

- `control_variates_garch.ipynb`: executed notebook with code, plots, and repeated-run summaries.
- `report.tex`: LaTeX source for the written support document.
- `report.pdf`: compiled PDF.
- `src/garch_cv.py`: reusable implementation of the GARCH posterior, RWM sampler, gradients, OLS controls, and Lasso-screened controls.
- `data/eurusd_frankfurter_2023_2024.csv`: local EUR/USD data used in the real-data section.
- `figures/` and `results/`: generated figures and tables used by the notebook and report.

To rebuild the notebook:

```powershell
py -3.11 scripts\create_notebook.py
$env:JUPYTER_ALLOW_INSECURE_WRITES='1'
jupyter nbconvert --to notebook --execute --inplace control_variates_garch.ipynb --ExecutePreprocessor.timeout=1200
```

To rebuild the report:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error report.tex
pdflatex -interaction=nonstopmode -halt-on-error report.tex
```

The notebook uses cached repeated estimates in `results/repeated_estimates.csv` by default.  Set `USE_CACHED_RESULTS = False` in the settings cell to rerun all repeated chains from scratch.
