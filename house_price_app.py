import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained model
with open('house_price_model.joblib', 'rb') as file:
    model = pickle.load(file)

# Create the Streamlit app
st.title('House Price Prediction Model')

# Input fields for features
st.header('Enter House Features')
square_footage = st.number_input('Square Footage', min_value=500, max_value=10000, value=1500)
bedrooms = st.number_input('Number of Bedrooms', min_value=1, max_value=10, value=3)
bathrooms = st.number_input('Number of Bathrooms', min_value=1, max_value=5, value=2)
year_built = st.number_input('Year Built', min_value=1900, max_value=2023, value=2000)
location = st.selectbox('Location', ['Urban', 'Suburban', 'Rural'])

# Convert location to numerical value
location_mapping = {'Urban': 0, 'Suburban': 1, 'Rural': 2}
location_encoded = location_mapping[location]

# Create a feature array
features = np.array([[square_footage, bedrooms, bathrooms, year_built, location_encoded]])

# Make prediction
if st.button('Predict Price'):
    prediction = model.predict(features)
    st.success(f'Predicted House Price: ${prediction[0]:,.2f}')
