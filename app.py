import streamlit as st
import pandas as pd
import joblib

# Load the trained model
# Update with the path to your trained model
model = joblib.load('random_forest_model.pkl')

# Load your dataset
# Update with the path to your dataset file

df1 = pd.read_csv(
    r'C:\Users\hp\Downloads\01_District_wise_crimes_committed_IPC_2001_2012.csv')
# Function to preprocess data


def preprocess_data(data):
    # Preprocess the data as required
    threshold = 1000
    data['Safe'] = data['TOTAL IPC CRIMES'].apply(
        lambda x: 1 if x < threshold else 0)
    return data

# Function to fetch district options


def fetch_district_options(data):
    return data['DISTRICT'].unique().tolist()

# Function to predict safety


def predict_safety(district, year):
    # Preprocess data
    df = preprocess_data(df1)
    # Filter data for the given district and year
    district_data = df[(df['DISTRICT'] == district) & (df['YEAR'] == year)][['MURDER', 'ATTEMPT TO MURDER', 'RAPE',
                                                                             'KIDNAPPING & ABDUCTION',
                                                                             'KIDNAPPING AND ABDUCTION OF WOMEN AND GIRLS',
                                                                             'DACOITY', 'ROBBERY', 'THEFT',
                                                                             'AUTO THEFT', 'RIOTS', 'CHEATING',
                                                                             'COUNTERFIETING', 'TOTAL IPC CRIMES']]
    # Check if data for the given district and year exists
    if len(district_data) == 0:
        return "Data not found for the given district and year."
    else:
        # Predict safety
        prediction = model.predict(district_data)
        return "Safe" if prediction[0] == 1 else "Unsafe"

# Streamlit web application


def main():
    st.title("District Safety Predictor")

    # Fetch district options
    district_options = fetch_district_options(df1)

    # Input widgets
    district = st.selectbox("Select District", options=district_options)
    year = st.slider("Select Year", min_value=2001,
                     max_value=2012, value=2001, step=1)

    # Predict safety
    if st.button("Predict"):
        result = predict_safety(district, year)
        st.write(
            f"The district '{district}' in the year {year} is predicted to be {result}.")


if __name__ == "__main__":
    main()
