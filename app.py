import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

#Loading the model
model = tf.keras.models.load_model("model.h5")

#Loading the Scalers and encoders
with open('label_encoder_gender.pkl', 'rb') as File:
    label_encoder_gender = pickle.load(File)
with open('onehot_encoder_geo.pkl', 'rb') as File:
    onehot_encoder_geo = pickle.load(File)
with open('scaler.pkl', 'rb') as File:
    scaler = pickle.load(File)

# Streamlit app
st.title("Customer Churn Prediction")

# User inputs
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input("Balance")
Credit_Score = st.number_input("Credit Score")
Estimated_Salary = st.number_input("Estimated Salary")
Tenure = st.slider("Number of Tenure", 0, 10)
Num_of_Products = st.slider('Number of Products', 1, 4)
Has_Cr_Card = st.selectbox("Has creditCard", [0, 1])
Is_active_Member = st.selectbox("Is a Active member", [0, 1])

#Preparing the input Data
input_data = pd.DataFrame({
    'CreditScore': [Credit_Score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [Tenure],
    'Balance': [balance],
    'NumOfProducts': [Num_of_Products],
    'HasCrCard': [Has_Cr_Card],
    'IsActiveMember': [Is_active_Member],
    'EstimatedSalary': [Estimated_Salary]
})

# One-hot Encode for geography
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

#Combining one-hot encoded value with the input data Frame
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis = 1)

# Calculate button
if st.button("Calculate"):

    # Scaling the data
    input_data_scaled = scaler.transform(input_data)

    # Prediction Churn
    Prediction = model.predict(input_data_scaled)
    Prediction_proba = Prediction[0][0]

    st.write(f"### Churn Probability: {Prediction_proba:.2%}")

    if Prediction_proba > 0.5:
        st.success("The customer is likely to churn.")
    else:
        st.success("The customer is not likely to churn.")