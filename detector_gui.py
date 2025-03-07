import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Cache the dataset loading so it doesn't reload on every change.
@st.cache_data
def load_data(data_file):
    data = pd.read_csv(data_file)
    return data

# Cache the model training to avoid retraining on every interaction.
@st.cache_resource
def train_model(data):
    # Define the features to use (numeric columns)
    features = ['Thermal_Brightness', 'Glare_Level', 'Surrounding_Brightness',
                'Thermal_Contrast', 'Thermal_Max', 'Thermal_Min', 'Thermal_Std',
                'Ambient_Temperature']
    
    # Check if required columns exist
    for col in features + ['Pest_Present']:
        if col not in data.columns:
            st.error(f"Column '{col}' not found in dataset.")
            st.stop()
    
    X = data[features]
    y = data['Pest_Present']
    
    # Split data to evaluate performance (optional)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a logistic regression model.
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Calculate and return model accuracy for feedback.
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc, features

st.title("Pest Detection Predictor with a Larger Dataset")

st.sidebar.header("1. Upload Dataset")
data_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])

if data_file is not None:
    data = load_data(data_file)
    st.write("### Sample of your dataset")
    st.dataframe(data.head())
    
    st.sidebar.header("2. Model Training")
    model, acc, features = train_model(data)
    st.sidebar.write(f"Trained model accuracy (on held-out data): {acc*100:.2f}%")
    
    st.header("Make a Prediction")
    st.write("Adjust the parameters below to simulate a new sample for prediction.")
    
    input_data = {}
    for feature in features:
        # Automatically set slider limits based on your dataset values.
        min_val = float(data[feature].min())
        max_val = float(data[feature].max())
        mean_val = float(data[feature].mean())
        input_data[feature] = st.slider(feature, min_value=min_val, max_value=max_val, value=mean_val)
    
    if st.button("Predict"):
        # Convert input data to a DataFrame
        input_df = pd.DataFrame([input_data])
        # Get the probability of pest detection (probability for class 1)
        probability = model.predict_proba(input_df)[0, 1]
        st.subheader("Prediction Result")
        st.write(f"**Probability of pest detected:** {probability*100:.2f}%")
        st.write(f"**Probability of no pest detected:** {(1 - probability)*100:.2f}%")
else:
    st.write("Please upload a CSV dataset with the required columns to continue.")
