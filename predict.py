"""
predict.py
----------
Loads the saved model (models/best_model.joblib) and predicts the price
for a single car, either from hardcoded example values or from CLI args.

Run:
    python predict.py
"""

import sys
from pathlib import Path

import joblib
import pandas as pd

# Absolute path anchored to this file, so it works from any working directory
MODEL_PATH = Path(__file__).resolve().parent / "models" / "best_model.joblib"


def predict_price(sample: dict, model_path=MODEL_PATH) -> float:
    """
    sample keys required:
        milage (float), car_age (int), horsepower (float), engine_size_l (float),
        brand (str), fuel_type (str), transmission (str),
        accident_reported (0/1), clean_title_flag (0/1)
    """
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. Run `python train_model.py` first."
        )
    model = joblib.load(model_path)
    X = pd.DataFrame([sample])
    return float(model.predict(X)[0])


if __name__ == "__main__":
    example = {
        "milage": 30000,
        "car_age": 3,
        "horsepower": 300,
        "engine_size_l": 3.5,
        "brand": "Toyota",
        "fuel_type": "Gasoline",
        "transmission": "Automatic",
        "accident_reported": 0,
        "clean_title_flag": 1,
    }
    try:
        price = predict_price(example)
    except FileNotFoundError as exc:
        sys.exit(str(exc))
    except Exception as exc:  # e.g. model saved with an incompatible scikit-learn version
        sys.exit(f"Could not load/use the model ({type(exc).__name__}: {exc}). "
                 "Re-train it with `python train_model.py`.")
    print(f"Predicted price: ${price:,.0f}")
