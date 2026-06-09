import streamlit as st
import pandas as pd
import joblib
import os

# ---------------------------
# Base directory
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------
# Load model safely
# ---------------------------
MODEL_PATH = os.path.join(BASE_DIR, "random_forest_model.pkl")
model = joblib.load(MODEL_PATH)

# ---------------------------
# Load dataset safely
# ---------------------------
DATA_PATH = os.path.join(BASE_DIR, "data.csv")
df1 = pd.read_csv(DATA_PATH)


def preprocess_data(data):
    threshold = 1000
    data = data.copy()
    data['Safe'] = data['TOTAL IPC CRIMES'].apply(
        lambda x: 1 if x < threshold else 0
    )
    return data


def fetch_district_options(data):
    return sorted(data['DISTRICT'].unique())


def predict_safety(district, year):
    df = preprocess_data(df1)

    district_data = df[
        (df['DISTRICT'] == district) & (df['YEAR'] == year)
    ][[
        'MURDER', 'ATTEMPT TO MURDER', 'RAPE',
        'KIDNAPPING & ABDUCTION',
        'KIDNAPPING AND ABDUCTION OF WOMEN AND GIRLS',
        'DACOITY', 'ROBBERY', 'THEFT',
        'AUTO THEFT', 'RIOTS', 'CHEATING',
        'COUNTERFIETING', 'TOTAL IPC CRIMES'
    ]]

    if district_data.empty:
        return "Data not found"

    prediction = model.predict(district_data)
    return "Safe" if prediction[0] == 1 else "Unsafe"


def main():
    st.title("District Safety Predictor")

    district = st.selectbox("Select District", fetch_district_options(df1))
    year = st.slider("Select Year", 2001, 2012, 2001)

    if st.button("Predict"):
        result = predict_safety(district, year)
        st.success(f"{district} in {year} is **{result}**")


if __name__ == "__main__":
    main()
