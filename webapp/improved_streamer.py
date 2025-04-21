import streamlit as st
import pandas as pd
import predictors

df_wo_churn = pd.read_csv('/media/jai/Projects/projects/ai-churn/CIS579_Churn/data/bs_eda_wo_index.csv')
df = df_wo_churn.drop(columns=['Churn'])
bin_fields = df.select_dtypes(include='bool').columns.to_list()
num_fields = df.drop(columns=bin_fields).columns.to_list()

def inputter():
    predictor = []

    col1, col2, col3 = st.columns(3)

    for i, field in enumerate(bin_fields[:3]):
        with [col1, col2, col3][i % 3]:
            predictor.append(st.selectbox(f"{field}:", ("Yes", "No")) == "Yes")

    with col1:
        predictor.append(st.number_input(num_fields[0] + " in years", step=1.0))

    for i, field in enumerate(bin_fields[3:12]):
        with [col1, col2, col3][i % 3]:
            predictor.append(st.selectbox(f"{field}:", ("Yes", "No")) == "Yes")

    for i, field in enumerate(num_fields[1:3]):
        with [col1, col2, col3][i % 3]:
            predictor.append(st.number_input(field+" in USD", step=1.0))

    with col3:
        option = st.selectbox("Internet Service", ("DSL", "Fiber Optic", "No"))
    if option == "DSL":
        predictor.extend([True, False, False])
    elif option == 'Fiber Optic':
        predictor.extend([False, True, False])
    elif option == 'No':
        predictor.extend([False, False, True])

    with col2:
        option = st.selectbox("Contract Type", ("Month-to-Month", "One Year", "Two Year"))
    if option == "Month-to-Month":
        predictor.extend([True, False, False])
    elif option == 'One Year':
        predictor.extend([False, True, False])
    elif option == 'Two Year':
        predictor.extend([False, False, True])

    with col3:
        option = st.selectbox("Payment Method", ("Bank Transfer (Automatic)", "Credit Card (Automatic)", "Electronic Check", "Mailed Check"))
    if option == "Bank Transfer (Automatic)":
        predictor.extend([True, False, False, False])
    elif option == 'Credit Card (Automatic)':
        predictor.extend([False, True, False, False])
    elif option == 'Electronic Check':
        predictor.extend([False, False, True, False])
    elif option == 'Mailed Check':
        predictor.extend([False, False, False, True])

    return predictor

def submitter(predictor):
    with st.container():
        col1, col2 = st.columns([1,1000])

        with col2:
            if st.button("Predict Churn"):
                if len(predictor) == len(df.columns):
                    input_df = pd.DataFrame([predictor], columns=df.columns.to_list())
                    prediction = predictors.predict_all(input_df)
                    prediction_conf = sum(prediction) * 100 / 3

                    if prediction_conf > 50:
                        print_pred = "The Customer will Churn.  \n Confidence: " + str(round(prediction_conf))   + "%"
                        st.warning(print_pred)
                    else:
                        print_pred = "The Customer will Stay.  \n Confidence: " + str(100 - round(prediction_conf))  + "%"
                        st.success(print_pred)


                    pred_txt = list(map(lambda x: "Churning" if x == 1 else "Not churning", prediction))
                    pred_df = pd.DataFrame([pred_txt], columns=['XG Classifier', 'Random Forest Classifier', 'Artificial Neural Network'])
                    st.table(pred_df)
                else:
                    st.error(f"Inputs incomplete ({len(predictor)}/{len(df.columns)} fields filled). Please fill out all fields.")

if __name__ == '__main__':
    st.set_page_config(layout="wide")
    col_left, col_right = st.columns([0.2,1])
    with col_left:
        st.image("a51.png", width = 160)
    with col_right:
        st.title("Area 51's ")
        st.title("Customer Churn Predictor")

    col_left, col_spacer, col_right = st.columns([2, 0.5, 3])

    with col_left:
        st.header("Customer Details")
        user_input = inputter()

    with col_right:
        st.header("Prediction Results")
        submitter(user_input)

