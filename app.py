import streamlit as st
import pandas as pd
import pickle

# Load the trained model
import os
import pickle

# Path relative to this file
current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, "heart_svm.pkl")

with open(model_path, "rb") as file:
    model = pickle.load(file)


st.title("❤️ Heart Disease Prediction App")

st.write("Enter patient details below and predict the risk of heart disease.")

# User inputs
age = st.number_input("Age", 18, 100, 30)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Cholesterol Level", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", [0, 1])
restecg = st.selectbox("Resting ECG (0–2)", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", 70, 220, 150)
exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 10.0, 1.0)
slope = st.selectbox("Slope of Peak Exercise ST Segment (0–2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0–4)", [0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia (1 = Normal, 2 = Fixed Defect, 3 = Reversable Defect)", [1, 2, 3])

# Predict button
if st.button("🔮 Predict"):
    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                                exang, oldpeak, slope, ca, thal]],
                              columns=['age','sex','cp','trestbps','chol','fbs','restecg',
                                       'thalach','exang','oldpeak','slope','ca','thal'])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error(" The patient is likely to have Heart Disease.")
    else:
        st.success(" The patient is unlikely to have Heart Disease.")
