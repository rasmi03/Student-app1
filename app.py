import streamlit as st 
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler,LabelEncoder


def load_model():     # to read physical file read  and binary
    with  open(r"D:\FSDS Euron\student_perf_app\student_final_linear_model12.pkl",'rb') as file:
        model,scaler,le=pickle.load(file) # u can change variables name
    return model,scaler,le

def preprocesssing_input_data(data, scaler, le):  # in dataset we do some preproccesing before creating model due to presence of categorical and also do ND
    data['Extracurricular Activities']= le.transform([data['Extracurricular Activities']])[0]  # [0] means convert it  from 0th index
    df = pd.DataFrame([data]) # then it pass into dataframe
    df_transformed = scaler.transform(df)  # then it pass into scaler means ND
    return df_transformed

def predict_data(data): # pass user data
    model,scaler,le = load_model() # call our model i.e load_model which returns model ,scaler,le
    processed_data = preprocesssing_input_data(data,scaler,le)  # call preprocesssing_input_data function 
    prediction = model.predict(processed_data) #pass preprocessed data into model for prediction 
    return prediction

def main():  # here main function call preict_data function and predict_data calls rest two function use by someone pass UAI data
    st.title("student performnce perdiction")
    st.write("enter your data to get a prediction for your performance")
    hour_studied=st.number_input("Hours studied",min_value = 1, max_value = 10 , value = 5)
    previous_score=st.number_input("previous score",min_value = 40, max_value = 100 , value = 70)
    Extra=st.selectbox("Extracurricular Activities",['Yes','No'])
    sleeping_hr=st.number_input("sleeping hours",min_value = 4, max_value = 10 , value = 7)
    Nnumber_of_solved=st.number_input("number of question paper solved",min_value = 0, max_value = 10 , value = 5)
    if st.button("predict_your_score") :
        user_data={
            "Extracurricular Activities":Extra,
            "Hours Studied":hour_studied,
            "Previous Scores":previous_score,
            "Sleep Hours":sleeping_hr,
            "Sample Question Papers Practiced": Nnumber_of_solved

        }
        prediction=predict_data(user_data)
        st.success(f"your  prediction result is {prediction}")
   
    
if __name__ == "__main__":  # calling main function
    main()
    

