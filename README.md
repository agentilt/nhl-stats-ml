# NHL Stats ML

A small, reproducible pipeline that predicts NHL player stats with regularized linear
models. Group project.

Pipeline (`src/`):

- `load_data.py`, `clean_data.py`, and `features.py` build the dataset.
- `main.py` trains OLS and Lasso models; `evaluate.py` and `infer.py` score and predict.
- Trained artifacts land in `models/` (`ols_model.pkl`, `lasso_model.pkl`, `scaler.pkl`).
- Diagnostics go to `reports/`: actual-vs-predicted, residual plots, and an OLS summary.

Run:

```bash
conda env create -f environment.yml
python src/main.py     # settings in config.yaml
pytest                 # tests
```
