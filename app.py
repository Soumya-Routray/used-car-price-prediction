"""
app.py — Streamlit UI for the Used Car Price Prediction model.

Run locally:
    streamlit run app.py

The app loads the trained pipeline from models/best_model.joblib
(preprocessing + model bundled together, so raw inputs go straight in).
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "models/best_model.joblib"
APP_DIR = Path(__file__).resolve().parent

BRANDS = [
    'Acura', 'Alfa', 'Aston', 'Audi', 'BMW', 'Bentley', 'Bugatti', 'Buick',
    'Cadillac', 'Chevrolet', 'Chrysler', 'Dodge', 'FIAT', 'Ferrari', 'Ford',
    'GMC', 'Genesis', 'Honda', 'Hummer', 'Hyundai', 'INFINITI', 'Jaguar',
    'Jeep', 'Karma', 'Kia', 'Lamborghini', 'Land', 'Lexus', 'Lincoln',
    'Lotus', 'Lucid', 'MINI', 'Maserati', 'Maybach', 'Mazda', 'McLaren',
    'Mercedes-Benz', 'Mercury', 'Mitsubishi', 'Nissan', 'Plymouth',
    'Polestar', 'Pontiac', 'Porsche', 'RAM', 'Rivian', 'Rolls-Royce',
    'Saab', 'Saturn', 'Scion', 'Subaru', 'Suzuki', 'Tesla', 'Toyota',
    'Volkswagen', 'Volvo', 'smart'
]
FUEL_TYPES = ['Gasoline', 'Diesel', 'Hybrid', 'Plug-In Hybrid', 'E85 Flex Fuel', 'Electric', 'Unknown']
TRANSMISSIONS = [
    'Automatic', 'A/T', 'M/T', 'Manual', 'CVT Transmission',
    '6-Speed A/T', '7-Speed A/T', '8-Speed A/T', '9-Speed A/T', '10-Speed A/T',
    'Transmission w/Dual Shift Mode'
]


def find_model_file():
    """Look for the model next to app.py, in the working dir, then anywhere in the repo."""
    candidates = [APP_DIR / MODEL_PATH, Path(MODEL_PATH)]
    for path in candidates:
        if path.is_file():
            return path

    # Fallback: search the repo for the file by name (handles folder/case differences)
    for pattern in ("best_model.joblib", "*.joblib"):
        for root in (APP_DIR, Path.cwd()):
            matches = sorted(root.rglob(pattern))
            if matches:
                return matches[0]
    return None


@st.cache_resource
def load_model():
    model_file = find_model_file()
    if model_file is None:
        return None
    return joblib.load(model_file)


def main():
    st.set_page_config(page_title="Used Car Price Predictor", page_icon="🚗", layout="centered")

    st.title("🚗 Used Car Price Predictor")

    model = load_model()

    if model is None:
        st.error("Model file not found. Make sure models/best_model.joblib is committed to your repo.")
        st.write("App folder:", str(APP_DIR))
        st.write("Files in app folder:", sorted(p.name for p in APP_DIR.iterdir()))
        models_dir = APP_DIR / "models"
        if models_dir.exists():
            st.write("Files in models/:", sorted(p.name for p in models_dir.iterdir()))
        st.stop()

    with st.container(border=True):
        st.subheader("🔧 Vehicle Details")
        col1, col2 = st.columns(2)

        with col1:
            brand = st.selectbox("Brand", sorted(BRANDS))
            model_year = st.number_input("Model Year", min_value=1990, max_value=2025, value=2019, step=1)
            milage = st.number_input("Mileage (miles)", min_value=0, max_value=400000, value=40000, step=1000)
            horsepower = st.number_input("Horsepower", min_value=50, max_value=1200, value=250, step=10)
            engine_size_l = st.number_input("Engine Size (Liters)", min_value=0.5, max_value=8.0, value=2.0, step=0.1)

        with col2:
            fuel_type = st.selectbox("Fuel Type", FUEL_TYPES)
            transmission = st.selectbox("Transmission", TRANSMISSIONS)
            accident = st.radio("Accident Reported?", ["No", "Yes"], horizontal=True)
            clean_title = st.radio("Clean Title?", ["Yes", "No"], horizontal=True)

    st.divider()

    if st.button("Predict Price", type="primary", use_container_width=True):
        car_age = 2024 - model_year
        sample = pd.DataFrame([{
            "milage": milage,
            "car_age": car_age,
            "horsepower": horsepower,
            "engine_size_l": engine_size_l,
            "brand": brand,
            "fuel_type": fuel_type,
            "transmission": transmission,
            "accident_reported": 1 if accident == "Yes" else 0,
            "clean_title_flag": 1 if clean_title == "Yes" else 0,
        }])

        prediction = model.predict(sample)[0]
        st.metric(label="Estimated Price", value=f"${prediction:,.0f}")
        st.success(f"Estimated Price: ${prediction:,.0f}")


if __name__ == "__main__":
    main()
