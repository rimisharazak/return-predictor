# Return Predictor — E-Commerce Intelligence

A Flask web app that serves your trained "will this order be returned?" model
through a form UI matching your capstone reference design.

## 1. Add your trained model

This app does **not** retrain anything — it loads the exact model you already
trained and saved at the end of your notebook:

```python
joblib.dump(model_data, "artifacts/model_data.joblib")
```

Copy that `model_data.joblib` file into this project's `artifacts/` folder, so
you end up with:

```
return-predictor/
  artifacts/
    model_data.joblib   <-- put your file here
  app.py
  ...
```

The app expects `model_data` to be a dict containing the same keys your
notebook builds: `model`, `scaler`, `model_columns`, `numeric_features`,
`binary_flags`, `categorical_features`, `train_medians`, `metrics`,
`feature_importance`, and `threshold`. If your key names differ, either
rename them before dumping, or adjust `app.py` (`build_feature_row` and the
`/` route) to match.

## 2. Install dependencies

```bash
cd return-predictor
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

## 3. Run it

```bash
python app.py
```

Open **http://localhost:3000** in your browser.

## How it works

- `app.py` — Flask server. Loads `artifacts/model_data.joblib` once, renders
  the form with your model's name/metrics/feature importance, and exposes
  `POST /predict`.
- `templates/index.html` — the Order Details form (Product / Risk Rates /
  Seller & Logistics / Customer), plus the Model Performance and Feature
  Importance panels, plus the Prediction Result gauge.
- `static/js/script.js` — populates the Category/Sub-category/Brand
  dropdowns, submits the form as JSON to `/predict`, and animates the
  result gauge.
- `static/css/style.css` — the dark UI theme.

`/predict` rebuilds each submitted order into a single-row DataFrame using
**exactly** the same feature-engineering steps as your notebook:

1. `total_support_contacts = support_calls + chat_messages`
2. `log_price = log1p(price)`
3. Missing numeric values filled with `train_medians`
4. Numeric features scaled with your saved `StandardScaler`
5. Categorical features one-hot encoded with `pd.get_dummies`, then
   reindexed to your saved `model_columns` (any category your model never
   saw during training becomes 0, exactly like your notebook's
   `X_test.reindex(...)` step)

It then calls `model.predict_proba(X)` and returns the return-probability,
a High/Medium/Low risk label, and whether it crosses your saved
`threshold`.

## Notes

- Category → sub-category options and the 30 brand names in the dropdowns
  are hard-coded in `static/js/script.js` from what showed up in your
  training data. Edit that file if your real category list differs.
- If `artifacts/model_data.joblib` is missing, the page still loads and
  shows a clear banner telling you what to do — it won't crash.
