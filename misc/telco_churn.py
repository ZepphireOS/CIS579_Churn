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

option = st.selectbox(
    "Internet Service",
    ("DSL", "Fiber Optic", "No"),
)
if option == "DSL":
    predictor.extend([True, False, False])
elif option == 'Fiber Optic':
    predictor.extend([False, True, False])
elif option == 'No':
    predictor.extend([False, False, True])

option = st.selectbox(
    "Contract Type",
    ("Month-to-Month", "One Year", "Two Year"),
)

if option == "Month-to-Month":
    predictor.extend([True, False, False])
elif option == 'One Year':
    predictor.extend([False, True, False])
elif option == 'Two Year':
    predictor.extend([False, False, True])

option = st.selectbox(
    "Payment Method",
    ("Bank Transfer (Automatic)", "Credit Card (Automatic)", "Electronic Check", "Mailed Check"),
)

if option == "Bank Transfer (Automatic)":
    predictor.extend([True, False, False, False])
elif option == 'Credit Card (Automatic)':
    predictor.extend([False, True, False, False])
elif option == 'Electronic Check':
    predictor.extend([False, False, True, False])
elif option == 'Mailed Check':
    predictor.extend([False, False, False, True])


# Remaining binary inputs
# for field in bin_fields[12:22]:
#     predictor.append(st.selectbox(f"{field}:", ("Yes", "No")) == "Yes")

# Only predict if the button is clicked
import torch
import torch.nn as nn

class ChurnModel(nn.Module):
    def __init__(self, input_dim):
        super(ChurnModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(128, 1)

        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p = 0.3)

    def forward(self, x):
        x = self.dropout(self.relu(self.bn1(self.fc1(x))))
        x = self.fc3(x)
        return x

# Tell PyTorch it's safe to unpickle ChurnModel
torch.serialization.add_safe_globals([ChurnModel])

# ✅ Now load the full model directly
ann_model = torch.load(
    "/media/jai/Projects/projects/ai-churn/CIS579_Churn/bharath/temp_model.pth",
    map_location=torch.device('cpu'),
    weights_only=False
)
model.eval()

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