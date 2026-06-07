import streamlit as st
import pandas as pd
import joblib

# Load the saved model and preprocessing tools
try:
    model = joblib.load('diabetes_model_tuned.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
except Exception as e:
    st.error(f"Error loading model files: {e}. Please ensure .pkl files are in the same folder.")

st.title("🏥 Diabetes Risk Prediction App")
st.write("Enter the patient's medical and demographic details to predict the risk of diabetes.")

# Create input columns for a cleaner UI
col1, col2 = st.columns(2)

with col1:
    year = st.number_input("Year", min_value=2015, max_value=2025, value=2020)
    gender = st.selectbox("Gender", ['Female', 'Male'])
    age = st.number_input("Age", min_value=0.0, max_value=120.0, value=30.0)
    location = st.text_input("Location (State)", "Alabama")
    hypertension = st.selectbox("Hypertension (0 = No, 1 = Yes)", [0, 1])
    heart_disease = st.selectbox("Heart Disease (0 = No, 1 = Yes)", [0, 1])

with col2:
    smoking_history = st.selectbox("Smoking History", ['never', 'current', 'former', 'ever', 'not current', 'No Info'])
    bmi = st.number_input("BMI", min_value=10.0, max_value=100.0, value=27.3)
    hba1c = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0, value=5.5)
    glucose = st.number_input("Blood Glucose Level", min_value=50, max_value=350, value=138)

st.write("---")
st.write("**Race Information**")
r_col1, r_col2, r_col3 = st.columns(3)
with r_col1:
    race_aa = st.checkbox("African American")
    race_asian = st.checkbox("Asian")
with r_col2:
    race_caucasian = st.checkbox("Caucasian")
    race_hispanic = st.checkbox("Hispanic")
with r_col3:
    race_other = st.checkbox("Other")

# Prediction Button
if st.button("Predict Diabetes Risk"):
    try:
        encoded_gender = label_encoders['gender'].transform([gender])[0]
        if location in label_encoders['location'].classes_:
            encoded_location = label_encoders['location'].transform([location])[0]
        else:
            encoded_location = 0 
            
        encoded_smoking = label_encoders['smoking_history'].transform([smoking_history])[0]
        
        input_data = pd.DataFrame([[
            year, encoded_gender, age, encoded_location,
            int(race_aa), int(race_asian), int(race_caucasian), int(race_hispanic), int(race_other),
            hypertension, heart_disease, encoded_smoking, bmi, hba1c, glucose
        ]], columns=['year', 'gender', 'age', 'location', 'race:AfricanAmerican', 'race:Asian', 
                     'race:Caucasian', 'race:Hispanic', 'race:Other', 'hypertension', 
                     'heart_disease', 'smoking_history', 'bmi', 'hbA1c_level', 'blood_glucose_level'])
        
        # Structure the input as a dataframe (ensure DOUBLE brackets [[ ]] are used)
        input_data = pd.DataFrame([[
            year, encoded_gender, age, encoded_location,
            int(race_aa), int(race_asian), int(race_caucasian), int(race_hispanic), int(race_other),
            hypertension, heart_disease, encoded_smoking, bmi, hba1c, glucose, bmi_cat
        ]], columns=['year', 'gender', 'age', 'location', 'race:AfricanAmerican', 'race:Asian', 
                     'race:Caucasian', 'race:Hispanic', 'race:Other', 'hypertension', 
                     'heart_disease', 'smoking_history', 'bmi', 'hbA1c_level', 'blood_glucose_level', 'BMI_Category_Encoded'])
        
        # Scale the 2D dataframe
        input_scaled = scaler.transform(input_data)
        
        # Predict using the 2D scaled data, THEN extract the single result with [0]
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        
        st.subheader("Prediction Result:")
        if prediction == 1:
            st.error(f"⚠️ High Risk of Diabetes detected. (Probability: {probability:.2%})")
        else:
            st.success(f"✅ Low Risk of Diabetes detected. (Probability: {probability:.2%})")
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        
