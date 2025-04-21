import streamlit as st
import pandas as pd
import predictors

def inputter():
    predictor = []
    for field in bin_fields[:3]:
        predictor.append(st.selectbox(f"{field}:", ("Yes", "No")) == "Yes")

    predictor.append(st.number_input(num_fields[0]))

    for field in bin_fields[3:12]:
        predictor.append(st.selectbox(f"{field}:", ("Yes", "No")) == "Yes")

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
    
    return predictor

def submitter(predictor):
    if st.button("Predict Churn"): #transfer this to the other file (main or maybe any other file that has the other stuff)
        if len(predictor) == len(df.columns):
            input_df = pd.DataFrame([predictor], columns=df.columns.to_list())
            prediction = predictors.predict_all(input_df)
            prediction_conf = sum(prediction)*100/3
            if prediction_conf > 50:
                # print_pred = "The Customer will Churn.  \n Confidence: " + str(prediction_conf) + "  \nXG Classifier: " + str(prediction[0]) + "  \nRandom Forest Classifier: " + str(prediction[1]) + "  \nANN: " + str(prediction[2])
                print_pred = "The Customer will Churn.  \n Confidence: " + str(prediction_conf)
            elif prediction_conf < 50:
                # print_pred = "The Customer will stay.  \n Confidence: " + str(100-prediction_conf) + "  \nIndividual results:  \nXG Classifier: " + str(prediction[0]) + "  \nRandom Forest Classifier: " + str(prediction[1]) + "  \nANN: " + str(prediction[2])
                print_pred = "The Customer will stay.  \n Confidence: " + str(100-prediction_conf)
            st.success(print_pred)
            pred_txt = list(map(lambda x: "Churning" if x == 1 else "Not churning", prediction))
            pred_df = pd.DataFrame([pred_txt], columns=['XG Classifier', 'Random Forest Classifier', 'Artificial Neural Network'])
            st.table(pred_df)
        else:
            st.error(f"Inputs incomplete ({len(predictor)}/{len(df.columns)} fields filled). Please fill out all fields.")

if __name__ == '__main__':
    st.title("Customer Churn Predictor")
    df_wo_churn = pd.read_csv('/media/jai/Projects/projects/ai-churn/CIS579_Churn/data/bs_eda_wo_index.csv')
    df = df_wo_churn.drop(columns=['Churn'])
    bin_fields = df.select_dtypes(include='bool').columns.to_list()
    num_fields = df.drop(columns=bin_fields).columns.to_list()
    user_input = inputter()
    submitter(user_input)