
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained model and scaler
# Ensure these files are in the same directory as your app.py or provide full paths
model = joblib.load('tuned_decision_tree_regressor_model.joblib')
scaler = joblib.load('scaler.joblib')

st.title('Hemolysis Index Prediction App')
st.write('Enter the feature values to predict the Log of Hemolysis Index.')

# Define the expected feature names in the correct order
# These must match X.columns from your notebook
feature_names = [
    'Flow Rate (ml/min)', 'Inlet Diameter (mm)', 'Vol  Avg Shear Rate (1/s)',
    'Max Wall Shear Rate (1/s)', '95th % Wall Shear Rate (1/s)',
    'Throat Avg Shear Rate (1/s)', 'Vol. Avg Exposure Time (s)',
    'Max Exposure Time (s)', 'Throat Exposure Time (s)',
    'Hydraulic Diameter (mm)', 'Device Type Encoded'
]

# Create input widgets for each feature
input_data = {}
for feature in feature_names:
    if feature == 'Device Type Encoded':
        input_data[feature] = st.number_input(f'Enter {feature} (0-4 for device types)', min_value=0, max_value=4, value=0, step=1, key=feature)
    elif feature == 'Inlet Diameter (mm)':
        input_data[feature] = st.number_input(f'Enter {feature}', value=1.6, format="%.1f", key=feature)
    else:
        # Set a reasonable default value and format for float inputs
        input_data[feature] = st.number_input(f'Enter {feature}', value=0.0, format="%.6f", key=feature)

if st.button('Predict Log Hemolysis Index'):
    # Convert input data to a DataFrame
    input_df = pd.DataFrame([input_data])

    # Ensure the order of columns in input_df matches the order the scaler was fitted on
    input_df = input_df[feature_names]

    # Scale the input data using the loaded scaler
    input_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_scaled)

    st.success(f'Predicted Log of Hemolysis Index: {prediction[0]:.4f}')
