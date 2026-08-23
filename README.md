# Used Car Price Prediction

An end-to-end machine learning project that predicts the resale price of a used car
from its brand, model year, mileage, fuel type, engine specs, transmission, and
accident/title history — including a trained model, a Streamlit web app, and
supporting scripts.

---

## 1. Problem Statement

**Problem:** Buyers and sellers of used cars struggle to price a vehicle fairly.
Price depends on many interacting factors (brand, age, mileage, engine, accident
history, etc.), which is hard to reason about manually or with simple rules.

**Business Objective:** Build a regression model that predicts a used car's fair
market price from its listed attributes, so that:
- Sellers can price their car competitively.
- Buyers can flag listings that are overpriced or underpriced.
- A dealership or marketplace (e.g. Cars.com/CarGurus-style platform) can auto-suggest
  a price when a user creates a listing.

---

## 2. Data Collection

- **File:** `data/used_cars.csv`
- **Rows / Columns:** ~4,009 rows × 12 columns.
- **Source:** This is a real-world used-car listings dataset (the schema — brand,
  model, model_year, milage, fuel_type, engine, transmission, ext_col, int_col,
  accident, clean_title, price — matches the publicly available "Used Car Price
  Prediction Dataset" commonly hosted on Kaggle). **Please confirm/attach the exact
  source link in your submission** if your instructor requires attribution, since the
  file was provided directly rather than downloaded fresh for this project.
- **Real vs Dummy:** Real listing data (not synthetic) — prices, mileages, and
  descriptions correspond to actual used-car ads.

---

## 3. Data Understanding

| Column | Type (raw) | Notes |
|---|---|---|
| brand | object | Car manufacturer |
| model | object | 1,885 unique values (high cardinality) |
| model_year | int | Year of manufacture |
| milage | object | e.g. `"51,000 mi."` — needs cleaning |
| fuel_type | object | ~170 missing |
| engine | object | Free text, e.g. `"300.0HP 3.7L V6..."` |
| transmission | object | e.g. `"6-Speed A/T"` |
| ext_col / int_col | object | Exterior / interior color (high cardinality) |
| accident | object | ~113 missing |
| clean_title | object | ~596 missing |
| **price** | object | **Target variable**, e.g. `"$10,300"` — needs cleaning |

Target variable: **`price`** (continuous → regression problem).

---

## 4. Exploratory Data Analysis (EDA)

Covered in the notebook (`used_car_price_prediction.ipynb`):
- Summary statistics for numeric fields (post-cleaning).
- Missing value counts per column.
- Duplicate row check.
- Univariate analysis: price distribution, mileage distribution, brand frequency.
- Bivariate analysis: price vs. model year, price vs. mileage, price by fuel type.
- Correlation heatmap of numeric features.
- Every chart is paired with a one-line takeaway explaining the insight it gives.

---

## 5. Data Cleaning

- **Missing values:** `fuel_type` filled with `"Unknown"` (corrected to `"Electric"`
  where the engine description says so); `accident` and `clean_title` converted to
  binary flags where missing is treated as "no info reported / not confirmed clean".
- **Duplicates:** checked and removed if present.
- **Incorrect data types:** `price` (`"$10,300"` → `10300.0`) and `milage`
  (`"51,000 mi."` → `51000.0`) converted from text to numeric.
- **Outliers:** prices above the 99.5th percentile are capped/removed — a handful of
  listings (e.g. a $250,000 Mercedes Metris) are clear data-entry anomalies relative
  to the rest of the market.

---

## 6. Data Preprocessing

- **Feature engineering:** `car_age` (from `model_year`), `horsepower` and
  `engine_size_l` (regex-extracted from the free-text `engine` field).
- **Encoding:** One-Hot Encoding for categorical features (`brand`, `fuel_type`,
  `transmission`, `accident_reported`, `clean_title_flag`) — used because these are
  nominal (no inherent order) and the cardinality is manageable after dropping
  `model`/colors.
- **Feature scaling:** `StandardScaler` on numeric features — needed because Linear
  Regression is scale-sensitive (tree models don't strictly need it, but it doesn't
  hurt and keeps one shared pipeline).
- **Feature selection:** `model`, `ext_col`, and `int_col` are dropped — too
  high-cardinality (1,885 / 315 / 150 unique values) to one-hot encode without
  exploding the feature space; `brand` retains most of the useful signal.

---

## 7. Train-Test Split

80/20 split (`random_state=42`) via `train_test_split`.

---

## 8. Model Building

Three regression models are trained and compared:
- Linear Regression (baseline)
- Random Forest Regressor
- Gradient Boosting Regressor

---

## 9. Model Evaluation

Regression metrics: **MAE, MSE, RMSE, R² Score**, plus 5-fold cross-validation for
robustness and a predicted-vs-actual scatter plot.

Latest run (see `models/metrics.json` for the exact numbers from the saved model):

| Model | R² (test) |
|---|---|
| Linear Regression | ~0.68 |
| Gradient Boosting | ~0.80–0.83 |
| **Random Forest (tuned, saved model)** | **~0.82** |

---

## 10. Hyperparameter Tuning

`RandomizedSearchCV` (5-fold, R² scoring) is used to tune the best baseline model
(Random Forest or Gradient Boosting, whichever wins the initial comparison) over
`n_estimators`, `max_depth`, `min_samples_split`/`learning_rate`, etc.

---

## 11. Save the Model

The full pipeline (preprocessing **and** model together) is saved with `joblib` to
`models/best_model.joblib`, so predictions can be made directly on raw input rows —
no need to re-run preprocessing by hand.

---

## 12. Streamlit Application

`app.py` provides a simple UI: enter brand, year, mileage, horsepower, engine size,
fuel type, transmission, accident history, and title status → click **Predict Price**
→ see the estimated price.

Run locally:
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 13. Deployment

To deploy `app.py` publicly (e.g. on **Streamlit Community Cloud**):
1. Push this repository to GitHub (see structure below).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
3. Click "New app", select this repo/branch, and set the main file to `app.py`.
4. Ensure `requirements.txt` is at the repo root (it is) — Streamlit Cloud installs
   from it automatically.
5. Deploy. Add the resulting public URL to your submission.

*(This step must be completed by you, since it requires your own GitHub/Streamlit
account — the app and all code are ready to deploy as-is.)*

---

## 14. GitHub Repository Structure

```
used-car-price-prediction/
├── used_car_price_prediction.ipynb   # Full notebook (EDA -> model -> save)
├── train_model.py                    # Training script (cleaning -> tuning -> save)
├── predict.py                        # CLI prediction script
├── app.py                            # Streamlit application
├── data/
│   └── used_cars.csv                 # Dataset
├── models/
│   ├── best_model.joblib             # Saved trained pipeline
│   └── metrics.json                  # Saved evaluation metrics
├── requirements.txt
└── README.md
```

### How to reproduce
```bash
pip install -r requirements.txt
python train_model.py      # trains + saves models/best_model.joblib
python predict.py          # sanity-check prediction on an example car
streamlit run app.py       # launch the web UI
```

---

## Notes / Limitations

- `model` (specific trim) was dropped in favor of `brand` to avoid an unmanageable
  number of one-hot columns; a target/frequency-encoding of `model` would likely
  improve accuracy further but was left as a next step given the project scope.
- Price is right-skewed; log-transforming the target was tested informally but did
  not meaningfully change tree-model performance, so the raw target was kept for
  simplicity and interpretability of error metrics (MAE/RMSE in dollars).
- Outlier capping at the 99.5th percentile removes a small number of ultra-luxury
  listings (Bugatti, Rolls-Royce, etc.) that behave more like a separate market
  segment; the model is best trusted for mainstream-market vehicles.
