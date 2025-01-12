import pickle 
import streamlit as st
import numpy as np
import tensorflow as tf
import sklearn
import pandas as pd


@st.cache_data
def load_pkl_files():
    model_path = "./model_weights/Ann_Model.h5"
    label_encoder_path = "./model_weights/label_encoder_gender.pkl"
    col_transformer_path = "./model_weights/column_transformer.pkl"
    standard_scaler_path = "./model_weights/standard_scaler.pkl"
    #load pkl files
    with open(label_encoder_path, "rb") as file:
        label_encoder = pickle.load(file)
    with open(col_transformer_path, "rb") as file:
        c_transformer = pickle.load(file)
    with open(standard_scaler_path, "rb") as file:
        stand_scaler= pickle.load(file)
    #load the model
    model = tf.keras.models.load_model(model_path)
    my_dict = {}
    my_dict["label_encoder"] = label_encoder
    my_dict["c_transformer"] =  c_transformer
    my_dict["stand_scaler"]= stand_scaler
    my_dict["model"]= model
    return my_dict

def predict(my_dict):
    st.title("Customer Churn Classification")
    geography = st.selectbox("Geography",["France","Germany","Spain"])
    gender = st.selectbox("Gender",my_dict["label_encoder"].classes_)
    age = st.slider("Age",18,92)
    balance = st.number_input("Balance")
    credit_score = st.number_input("Credit Score")
    estimated_salary = st.number_input("Estimated Salary")
    tenure = st.slider("Tenure",0,10)
    num_products = st.slider("Num of Products",1,4)
    has_cred_card = st.selectbox("Has Credit Card",[0,1])
    is_active_member = st.selectbox("Is Active Member",[0,1])
    input_data = {
    'CreditScore' : [credit_score],
     'Geography' :[geography] ,
    'Gender' : [gender],
    'Age' : [age],
    'Tenure' : [tenure],
    'Balance' : [balance],
    'NumOfProducts' : [num_products],
    'HasCrCard' : [has_cred_card],
    'IsActiveMember' : [is_active_member],
    'EstimatedSalary' : [estimated_salary]
    }
    process_data(my_dict,input_data)
def process_data(my_dict,input_data):
    input_data = pd.DataFrame(input_data)
    input_data["Gender"] = my_dict["label_encoder"].transform(input_data["Gender"])
    transformed_df = my_dict["c_transformer"].transform(input_data.values)
    scaled_df = my_dict['stand_scaler'].transform(transformed_df)
    predicted_probab = my_dict["model"].predict(scaled_df)
    prob = round(predicted_probab[0][0],2) * 100
    st.write(f"Probability: {prob:.2f} %")
    if predicted_probab > 0.5:
        st.write("Customer is likely to leave the bank")
    else:
        st.write("Customer is likely to stay in the bank!")


def main():
    my_dict = load_pkl_files()
    predict(my_dict)

main()
