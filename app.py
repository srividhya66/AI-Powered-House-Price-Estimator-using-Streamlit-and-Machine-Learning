# app.py
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("🏠 House Price Predictor")

# Load model
model = joblib.load("model.pkl")

# Load locations
@st.cache_data
def load_locations():
    df = pd.read_csv("data/housing.csv")
    return sorted(df['location'].dropna().unique())

location = st.selectbox("Select Location", load_locations())
total_sqft = st.number_input("Total Square Feet", min_value=300, step=50)
bath = st.selectbox("Number of Bathrooms", [1, 2, 3, 4, 5])

if st.button("Predict Price"):
    input_df = pd.DataFrame({
        'location': [location],
        'total_sqft': [total_sqft],
        'bath': [bath]
    })

    try:
        prediction = model.predict(input_df)[0]
        st.success(f"🏷 Estimated House Price: ₹ {prediction:.2f} Lakhs")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
