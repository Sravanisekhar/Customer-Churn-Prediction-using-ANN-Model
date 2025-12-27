import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

# ===============================
# Page Configuration
# ===============================
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

# ===============================
# Load Model & Encoders
# ===============================
model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('one_hot_encoder_geo.pkl', 'rb') as file:
    one_hot_encoder_geo = pickle.load(file)

with open('scalar.pkl', 'rb') as file:
    scalar = pickle.load(file)

# ===============================
# App Title
# ===============================
st.markdown(
    "<h1 style='text-align: center;'>📉 Customer Churn Prediction</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center; color: gray;'>Predict whether a customer will leave the bank</p>",
    unsafe_allow_html=True
)

st.divider()

# ===============================
# Sidebar Inputs
# ===============================
st.sidebar.header("🧾 Customer Information")

geography = st.sidebar.selectbox(
    "Geography",
    one_hot_encoder_geo.categories_[0]
)

gender = st.sidebar.selectbox(
    "Gender",
    label_encoder_gender.classes_
)

age = st.sidebar.slider("Age", 18, 92, 35)
tenure = st.sidebar.slider("Tenure (Years)", 0, 10, 5)
credit_score = st.sidebar.number_input("Credit Score", 300, 900, 650)
balance = st.sidebar.number_input("Account Balance", value=0.0)
estimated_salary = st.sidebar.number_input("Estimated Salary", value=50000.0)
num_of_products = st.sidebar.slider("Number of Products", 1, 4, 1)
has_cr_card = st.sidebar.selectbox("Has Credit Card", [0, 1])
is_active_member = st.sidebar.selectbox("Is Active Member", [0, 1])

# ===============================
# Predict Button
# ===============================
st.markdown("### 🔍 Prediction Result")

predict_button = st.button("🚀 Predict Churn")

if predict_button:

    # Prepare input data
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    # One-hot encode Geography
    geo_encoded = one_hot_encoder_geo.transform([[geography]])
    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=one_hot_encoder_geo.get_feature_names_out(['Geography'])
    )

    # Combine data
    input_data = pd.concat(
        [input_data.reset_index(drop=True), geo_encoded_df],
        axis=1
    )

    # Scale input
    input_data_scaled = scalar.transform(input_data)

    # Predict
    prediction = model.predict(input_data_scaled)
    prediction_prob = prediction[0][0]

    # Display result
    st.divider()

    if prediction_prob > 0.5:
        st.error(
            f"⚠️ **Customer is likely to churn**  \n\n"
            f"📊 **Churn Probability:** `{prediction_prob:.2%}`"
        )
    else:
        st.success(
            f"✅ **Customer is not likely to churn**  \n\n"
            f"📊 **Churn Probability:** `{prediction_prob:.2%}`"
        )
