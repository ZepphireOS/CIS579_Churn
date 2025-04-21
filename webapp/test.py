# # Importing shit
# import streamlit as st
# import pandas as pd
# from joblib import load

# #Loading the model
# model = load('random_forest_churn_model_smote.pkl')

# # Just a test to see if the webapp prints stuff accurately
# st.write("The Weeknd is da GOAT")

# # Getting the column names from the dataset directly instead of hardcoding
# # df_wo_churn = pd.read_csv('./data/bs_eda_wo_index.csv')
# df_wo_churn = pd.read_csv('/media/jai/Projects/projects/ai-churn/CIS579_Churn/data/bs_eda_wo_index.csv')
# df = df_wo_churn.drop(columns=['Churn'])
# bin_fields = df.select_dtypes(include='bool').columns.to_list()
# num_fields = df.drop(columns=bin_fields).columns.to_list()

# predictor = []

# # bins = {}
# for field in bin_fields[:3]:
#     # predictor[field] = bool(st.selectbox(f"{field}:", ("Yes", "No")))
#     predictor.append(bool(st.selectbox(f"{field}:", ("Yes", "No"))))

# # predictor[3] = st.number_input(num_fields[0])
# predictor.append(st.number_input(num_fields[0]))

# for field in bin_fields[3:12]:
#     # predictor[field] = bool(st.selectbox(f"{field}:", ("Yes", "No")))
#     predictor.append(bool(st.selectbox(f"{field}:", ("Yes", "No"))))

# for field in num_fields[1:3]:
#     # predictor[field] = st.number_input(num_fields[field])
#     predictor.append(st.number_input(field))

# for field in bin_fields[12:22]:
#     # predictor[field] = bool(st.selectbox(f"{field}:", ("Yes", "No")))
#     predictor.append(bool(st.selectbox(f"{field}:", ("Yes", "No"))))


# # prediction = float(txts["Monthly Charges"]) + float(txts["Total Charges"])
# # st.write("The churn score is: ", prediction)
# # st.write("The churn score is: ", prediction)
# columnss=df.columns.to_list()
# print(columnss)
# # Turn into a DataFrame
# input_df = pd.DataFrame([predictor], columns=columnss)

# # Remove the target column if it's present
# # input_df = input_df.drop(columns=['Churn'])

# # Predict
# # prediction = model.predict(input_df)
# # proba = model.predict_proba(input_df)

# prediction = model.predict(input_df)

# print("Prediction (0 = No Churn, 1 = Churn):", prediction[0])
# st.write("Prediction: ", prediction[0])
# # print("Probabilities:", proba[0])

# Importing stuff
import streamlit as st
import pandas as pd
from joblib import load

# Loading the model
model = load('random_forest_churn_model_smote.pkl')

# Just a test to see if the webapp prints stuff accurately
st.write("The Weeknd is da GOAT")

# Load dataset to get column info
df_wo_churn = pd.read_csv('/media/jai/Projects/projects/ai-churn/CIS579_Churn/data/bs_eda_wo_index.csv')
df = df_wo_churn.drop(columns=['Churn'])
bin_fields = df.select_dtypes(include='bool').columns.to_list()
num_fields = df.drop(columns=bin_fields).columns.to_list()

predictor = []

for field in bin_fields[:3]:
    predictor.append(st.selectbox(f"{field}:", ("Yes", "No")) == "Yes")


predictor.append(st.number_input(num_fields[0]))


for field in bin_fields[3:12]:
    predictor.append(st.selectbox(f"{field}:", ("Yes", "No")) == "Yes")


for field in num_fields[1:3]:
    predictor.append(st.number_input(field))


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
