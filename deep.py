import streamlit as st 
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_model():     # Load the trained model, scaler, and label encoder
    with open("student_lr_final_model.pkl", 'rb') as file:
        model, scaler, le = pickle.load(file)
    return model, scaler, le

def preprocesssing_input_data(data, scaler, le):
    # Ensure LabelEncoder is fitted with expected values
    if set(le.classes_) != {"Yes", "No"}:
        le.fit(["Yes", "No"])
    
    # Transform categorical value safely
    if data['Extracurricular Activities'] in le.classes_:
        data['Extracurricular Activities'] = le.transform([data['Extracurricular Activities']])[0]
    else:
        raise ValueError(f"Unexpected value '{data['Extracurricular Activities']}' found! Expected: {le.classes_}")
    
    df = pd.DataFrame([data])
    df_transformed = scaler.transform(df)
    return df_transformed

def predict_data(data):
    model, scaler, le = load_model() 
    processed_data = preprocesssing_input_data(data, scaler, le)  
    prediction = model.predict(processed_data)
    return prediction

def main():
    st.title("Student Performance Prediction")
    st.write("Enter your data to get a prediction for your performance")
    
    hour_studied = st.number_input("Hours Studied", min_value=1, max_value=10, value=5)
    previous_score = st.number_input("Previous Score", min_value=40, max_value=100, value=70)
    extra = st.selectbox("Extracurricular Activity", ['Yes', 'No'])
    sleeping_hr = st.number_input("Sleeping Hours", min_value=4, max_value=10, value=7)
    num_solved = st.number_input("Number of Question Papers Solved", min_value=0, max_value=10, value=5)
    
    if st.button("Predict Your Score"):
        user_data = {
            "Hours Studied": hour_studied,
            "Previous Scores": previous_score,
            "Extracurricular Activities": extra,
            "Sleep Hours": sleeping_hr,
            "Sample Question Papers Practiced": num_solved
        }
        try:
            prediction = predict_data(user_data)
            st.success(f"Your predicted score is {prediction}")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
if __name__ == "__main__":  
    main()
