import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

st.title("🧠 Paralysis Disease Prediction App")

# Sample Dataset
data = {
    "Headache": [1, 0, 1, 0, 1, 1, 0, 0],
    "Dizziness": [1, 0, 1, 0, 1, 0, 1, 0],
    "Vision_Problem": [1, 0, 1, 0, 1, 0, 0, 0],
    "Speech_Difficulty": [1, 0, 1, 0, 1, 0, 0, 0],
    "Weakness": [1, 0, 1, 0, 1, 1, 0, 0],
    "Numbness": [1, 0, 1, 0, 1, 1, 0, 0],
    "Confusion": [1, 0, 1, 0, 0, 1, 0, 0],
    "Diagnosis": ["Yes", "No", "Yes", "No", "Yes", "Yes", "No", "No"]
}
df = pd.DataFrame(data)

# Model training
X = df.drop("Diagnosis", axis=1)
y = LabelEncoder().fit_transform(df["Diagnosis"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Input fields
st.header("Enter Your Symptoms:")
headache = st.selectbox("Headache?", [0, 1])
dizziness = st.selectbox("Dizziness?", [0, 1])
vision = st.selectbox("Vision Problem?", [0, 1])
speech = st.selectbox("Speech Difficulty?", [0, 1])
weakness = st.selectbox("Weakness?", [0, 1])
numbness = st.selectbox("Numbness?", [0, 1])
confusion = st.selectbox("Confusion?", [0, 1])

# Prediction
if st.button("Predict"):
    input_data = np.array([[headache, dizziness, vision, speech, weakness, numbness, confusion]])
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.error("⚠ High chance of Paralysis symptoms! Please consult a doctor.")
    else:
        st.success("✅ No major signs of Paralysis. You seem safe.")