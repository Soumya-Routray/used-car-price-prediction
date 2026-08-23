"""
app.py — Streamlit UI for the Used Car Price Prediction model.

Run locally:
    streamlit run app.py

The app loads the trained pipeline from models/best_model.joblib
(preprocessing + model bundled together, so raw inputs go straight in).
"""

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "models/best_model.joblib"

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


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def inject_css():
    st.markdown(
        """
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap" rel="stylesheet">
        <style>
        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at 15% 0%, rgba(99,102,241,0.18) 0%, transparent 45%),
                radial-gradient(circle at 85% 15%, rgba(236,72,153,0.14) 0%, transparent 45%),
                radial-gradient(circle at 50% 100%, rgba(16,185,129,0.10) 0%, transparent 50%),
                #0b0f1a;
        }

       
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title="Used Car Price Predictor", page_icon="🚗", layout="centered")
    inject_css()

    st.markdown(
        """
        <div class="hero">
            <div class="hero-badge">🚗</div>
            <div class="hero-text">
                <h1>Used Car Price Predictor</h1>
                <p>Instant resale estimates from a model trained on ~4,000 real listings.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("ℹ️ About")
        st.write(
            "This tool uses a machine learning pipeline (preprocessing + regression "
            "model) trained on historical used-car listing data to estimate resale "
            "prices."
        )
        st.markdown("---")
        st.write("**How it works:**")
        st.write("1. Fill in the car's specs\n2. Click **Predict Price**\n3. Get an instant estimate")
        st.markdown("---")
        st.caption("Estimates are for informational purposes only and are not a guarantee of actual sale price.")

    model = load_model()

    with st.container(border=True):
        st.markdown(
            '<div class="glass-title">🔧 VEHICLE SPECS<div class="line"></div></div>',
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="field-label">🏷️ Brand</div>', unsafe_allow_html=True)
            brand = st.selectbox("Brand", sorted(BRANDS), label_visibility="collapsed")

            st.markdown('<div class="field-label">📅 Model Year</div>', unsafe_allow_html=True)
            model_year = st.number_input(
                "Model Year", min_value=1990, max_value=2025, value=2019, step=1,
                label_visibility="collapsed",
            )

            st.markdown('<div class="field-label">🛣️ Mileage (miles)</div>', unsafe_allow_html=True)
            milage = st.number_input(
                "Mileage (miles)", min_value=0, max_value=400000, value=40000, step=1000,
                label_visibility="collapsed",
            )

            st.markdown('<div class="field-label">🐎 Horsepower</div>', unsafe_allow_html=True)
            horsepower = st.number_input(
                "Horsepower", min_value=50, max_value=1200, value=250, step=10,
                label_visibility="collapsed",
            )

            st.markdown('<div class="field-label">🔧 Engine Size (Liters)</div>', unsafe_allow_html=True)
            engine_size_l = st.number_input(
                "Engine Size (Liters)", min_value=0.5, max_value=8.0, value=2.0, step=0.1,
                label_visibility="collapsed",
            )

        with col2:
            st.markdown('<div class="field-label">⛽ Fuel Type</div>', unsafe_allow_html=True)
            fuel_type = st.selectbox("Fuel Type", FUEL_TYPES, label_visibility="collapsed")

            st.markdown('<div class="field-label">⚙️ Transmission</div>', unsafe_allow_html=True)
            transmission = st.selectbox("Transmission", TRANSMISSIONS, label_visibility="collapsed")

            st.markdown('<div class="field-label">⚠️ Accident Reported?</div>', unsafe_allow_html=True)
            try:
                accident = st.segmented_control(
                    "Accident Reported?", ["No", "Yes"], default="No",
                    label_visibility="collapsed", key="accident_toggle",
                )
            except AttributeError:
                accident = st.radio(
                    "Accident Reported?", ["No", "Yes"], horizontal=True,
                    label_visibility="collapsed",
                )

            st.markdown('<div class="field-label">📄 Clean Title?</div>', unsafe_allow_html=True)
            try:
                clean_title = st.segmented_control(
                    "Clean Title?", ["Yes", "No"], default="Yes",
                    label_visibility="collapsed", key="clean_title_toggle",
                )
            except AttributeError:
                clean_title = st.radio(
                    "Clean Title?", ["Yes", "No"], horizontal=True,
                    label_visibility="collapsed",
                )

    accident = accident or "No"
    clean_title = clean_title or "Yes"

    st.write("")
    predict_clicked = st.button("✨ Predict Price", type="primary")

    if predict_clicked:
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

        with st.spinner("Crunching the numbers..."):
            prediction = model.predict(sample)[0]

        st.markdown(
            f"""
            <div class="result-wrap">
                <div class="label">Estimated Price</div>
                <div class="price">${prediction:,.0f}</div>
                <div class="sub">Based on a {brand} · {model_year} · {milage:,.0f} mi</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            "This is an estimate from a machine learning model trained on historical "
            "listings — actual market prices may vary."
        )


if __name__ == "__main__":
    main()