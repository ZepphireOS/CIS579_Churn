# Importing stuff
import streamlit as st
import pandas as pd
from joblib import load

# Load the model
model = load('random_forest_churn_model_smote.pkl')

# Quick message to verify the app is working
st.write("The Weeknd is da GOAT")

# Load dataset to get column info
df_wo_churn = pd.read_csv('/media/jai/Projects/projects/ai-churn/CIS579_Churn/data/bs_eda_wo_index.csv')
df = df_wo_churn.drop(columns=['Churn'])
bin_fields = df.select_dtypes(include='bool').columns.to_list()
num_fields = df.drop(columns=bin_fields).columns.to_list()

predictor = []

# Collecting binary field inputs
for field in bin_fields[:3]:
    predictor.append(st.selectbox(f"{field}:", ("Yes", "No")) == "Yes")

# First numeric input
predictor.append(st.number_input(num_fields[0]))

# More binary fields
for field in bin_fields[3:12]:
    predictor.append(st.selectbox(f"{field}:", ("Yes", "No")) == "Yes")

# More numeric inputs
for field in num_fields[1:3]:
    predictor.append(st.number_input(field))

# Remaining binary inputs
for field in bin_fields[12:22]:
    predictor.append(st.selectbox(f"{field}:", ("Yes", "No")) == "Yes")

# Only predict if the button is clicked
if st.button("Predict Churn"):
    if len(predictor) == len(df.columns):
        input_df = pd.DataFrame([predictor], columns=df.columns.to_list())
        prediction = model.predict(input_df)
        if prediction == 1:
            print_pred = "The Customer will Churn."
        elif prediction == 0:
            print_pred = "The customer will stay."
        st.success(print_pred)
    else:
        st.error(f"Inputs incomplete ({len(predictor)}/{len(df.columns)} fields filled). Please fill out all fields.")