"""
app.py — Streamlit UI for the Used Car Price Prediction model.

Run locally:
    streamlit run app.py

The app loads the trained pipeline from models/best_model.joblib
(preprocessing + model bundled together, so raw inputs go straight in).
The path is anchored to this file, so it works from any working directory
(local Windows/Linux and Streamlit Community Cloud). The model is loaded once
and cached; it is only trained if the user explicitly clicks the button shown
when the model file is missing or unreadable.
"""

import joblib
import pandas as pd
import streamlit as st

from train_model import MODEL_PATH  # single source of truth for the absolute path

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


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return joblib.load(MODEL_PATH)


def offer_training(reason: str):
    """Show a friendly error plus a one-click (manual) training option, then stop."""
    st.error(reason)
    st.info(
        "Locally, run `python train_model.py` and restart the app. "
        "On a hosted app you can train it once here (can take several minutes); "
        "the model is saved to disk and reused on later runs."
    )
    if st.button("Train model now"):
        trained = False
        try:
            import train_model
            with st.spinner("Training model... this can take several minutes."):
                train_model.main()
            trained = True
        except Exception as exc:
            st.error(f"Training failed: {exc}")
        if trained:  # outside try/except: st.rerun() works by raising a control exception
            load_model.clear()
            st.rerun()
    st.stop()


def main():
    st.set_page_config(page_title="Used Car Price Predictor", page_icon="🚗", layout="centered")
    st.title("🚗 Used Car Price Predictor")
    st.write(
        "Enter a car's details below and get an estimated resale price from a "
        "model trained on ~4,000 real used-car listings."
    )

    if not MODEL_PATH.is_file():
        offer_training(f"Trained model not found at `{MODEL_PATH}`.")
    try:
        model = load_model()
    except Exception as exc:  # corrupt file or scikit-learn version mismatch
        offer_training(
            f"The saved model could not be loaded ({type(exc).__name__}: {exc}). "
            "It was probably saved with a different scikit-learn version."
        )

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

    if st.button("Predict Price", type="primary"):
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
        st.success(f"### Estimated Price: ${prediction:,.0f}")
        st.caption(
            "This is an estimate from a machine learning model trained on historical "
            "listings — actual market prices may vary."
        )


if __name__ == "__main__":
    main()
