"""
train_model.py
---------------
Trains a used-car price prediction model end-to-end:
  data cleaning -> feature engineering -> preprocessing -> model training
  -> hyperparameter tuning -> evaluation -> save model with joblib.

Run:
    python train_model.py

Produces:
    models/best_model.joblib   (full sklearn Pipeline: preprocessing + model)
    models/metrics.json        (evaluation metrics for the saved model)
"""

import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = "data/used_cars.csv"
MODEL_PATH = "models/best_model.joblib"
METRICS_PATH = "models/metrics.json"
CURRENT_YEAR = 2024

NUM_FEATURES = ["milage", "car_age", "horsepower", "engine_size_l"]
CAT_FEATURES = ["brand", "fuel_type", "transmission", "accident_reported", "clean_title_flag"]


def load_and_clean(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV and clean/engineer features. Returns a model-ready DataFrame."""
    df = pd.read_csv(path)
    data = df.copy()

    # Target: strip "$" and "," -> float
    data["price"] = data["price"].replace(r"[\$,]", "", regex=True).astype(float)

    # Mileage: strip "mi." and "," -> float
    data["milage"] = data["milage"].replace(r"[^0-9]", "", regex=True).astype(float)

    # Car age from model year
    data["car_age"] = CURRENT_YEAR - data["model_year"]

    # Horsepower / engine size parsed out of free-text engine description
    data["horsepower"] = data["engine"].str.extract(r"([\d\.]+)\s*HP").astype(float)
    data["engine_size_l"] = data["engine"].str.extract(r"([\d\.]+)\s*L\b").astype(float)

    # Accident -> binary flag
    data["accident_reported"] = data["accident"].apply(
        lambda x: 1 if isinstance(x, str) and "least" in x.lower() else 0
    )

    # Clean title -> binary flag (missing treated as "No")
    data["clean_title_flag"] = data["clean_title"].apply(lambda x: 1 if x == "Yes" else 0)

    # Fuel type: fill missing, and correct known Electric-vehicle blanks
    data["fuel_type"] = data["fuel_type"].fillna("Unknown")
    data.loc[data["engine"].str.contains("Electric", case=False, na=False), "fuel_type"] = "Electric"

    # Drop invalid targets, cap extreme outliers (top 0.5%)
    data = data[data["price"] > 0]
    price_cap = data["price"].quantile(0.995)
    data = data[data["price"] <= price_cap]

    return data[NUM_FEATURES + CAT_FEATURES + ["price"]].reset_index(drop=True)


def build_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUM_FEATURES),
        ("cat", categorical_transformer, CAT_FEATURES)
    ])


def evaluate(y_true, y_pred) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mean_squared_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def main():
    model_df = load_and_clean()
    X = model_df.drop(columns="price")
    y = model_df["price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = build_preprocessor()

    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }

    # Baseline comparison (untuned) to pick which model is worth tuning
    baseline_scores = {}
    for name, model in candidates.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("regressor", model)])
        pipe.fit(X_train, y_train)
        baseline_scores[name] = r2_score(y_test, pipe.predict(X_test))
        print(f"[baseline] {name:20s} R2={baseline_scores[name]:.3f}")

    best_name = max(baseline_scores, key=baseline_scores.get)
    print(f"\nBest baseline model: {best_name}. Proceeding to hyperparameter tuning...")

    if best_name == "Random Forest":
        base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
        param_dist = {
            "regressor__n_estimators": [150, 250, 350],
            "regressor__max_depth": [8, 12, 16],
            "regressor__min_samples_split": [2, 5],
            "regressor__min_samples_leaf": [1, 2],
        }
    elif best_name == "Gradient Boosting":
        base_model = GradientBoostingRegressor(random_state=42)
        param_dist = {
            "regressor__n_estimators": [150, 250],
            "regressor__max_depth": [2, 3],
            "regressor__learning_rate": [0.05, 0.1],
            "regressor__subsample": [0.8, 1.0],
        }
    else:
        base_model = LinearRegression()
        param_dist = {}

    tuned_pipe = Pipeline([("preprocessor", preprocessor), ("regressor", base_model)])

    if param_dist:
        search = RandomizedSearchCV(
            tuned_pipe, param_distributions=param_dist,
            n_iter=6, cv=3, scoring="r2", random_state=42, n_jobs=-1
        )
        search.fit(X_train, y_train)
        best_pipe = search.best_estimator_
        print("Best params:", search.best_params_)
    else:
        best_pipe = tuned_pipe.fit(X_train, y_train)

    preds = best_pipe.predict(X_test)
    metrics = evaluate(y_test, preds)
    print("\nFinal test-set metrics:", metrics)

    cv_scores = cross_val_score(best_pipe, X, y, cv=5, scoring="r2", n_jobs=-1)
    metrics["cv_r2_mean"] = float(cv_scores.mean())
    metrics["cv_r2_std"] = float(cv_scores.std())
    metrics["model_name"] = best_name
    print(f"CV R2: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

    joblib.dump(best_pipe, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
