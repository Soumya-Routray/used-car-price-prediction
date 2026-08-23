"""
predict.py
----------
Loads the saved model (models/best_model.joblib) and predicts the price
for a single car, either from hardcoded example values or from CLI args.

Run:
    python predict.py
"""

import joblib
import pandas as pd

MODEL_PATH = "models/best_model.joblib"


def predict_price(sample: dict, model_path: str = MODEL_PATH) -> float:
    """
    sample keys required:
        milage (float), car_age (int), horsepower (float), engine_size_l (float),
        brand (str), fuel_type (str), transmission (str),
        accident_reported (0/1), clean_title_flag (0/1)
    """
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
    price = predict_price(example)
    print(f"Predicted price: ${price:,.0f}")
