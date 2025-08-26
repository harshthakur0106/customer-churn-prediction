import streamlit as st
import pickle
import pandas as pd

# Load model and feature names
with open("customer_churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
feature_names = model_data["features_names"]

# Load encoders
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

st.title("📊 Customer Churn Prediction")
st.write("Fill in customer details to predict churn and see probability.")

# Collect user input dynamically
st.sidebar.header("Customer Details")

input_data = {}
for feature in feature_names:
    if feature in encoders:  # Categorical feature
        options = list(encoders[feature].classes_)
        input_data[feature] = st.sidebar.selectbox(f"{feature}", options)
    else:  # Numeric feature
        default_val = 0.0
        if feature.lower() in ["tenure", "seniorcitizen"]:
            default_val = 0
        elif feature.lower() in ["monthlycharges", "totalcharges"]:
            default_val = 50.0
        input_data[feature] = st.sidebar.number_input(f"{feature}", value=default_val)

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Prediction button
if st.button("Predict Churn"):
    try:
        # Apply encoders for categorical columns
        for col, encoder in encoders.items():
            input_df[col] = encoder.transform(input_df[col])

        # Ensure columns in correct order
        input_df = input_df[feature_names]

        # Make prediction
        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        st.subheader("Prediction Result")
        st.write(f"**Churn:** {'Yes' if prediction == 1 else 'No'}")
        st.write(f"**Probability:** {prob:.2%}")

    except Exception as e:
        st.error(f"Error: {e}")
