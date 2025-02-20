import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import pandas as pd
import os

# Title of the app
st.title("UPI Fraud Detection App")

# Load the models and scaler
@st.cache_resource  # Cache for better performance
def load_models_and_scaler():
    model_paths = {
    "XGBoost": "models/2xgboost.pkl",
    "Decision Tree": "models/2decisionTree.pkl",
    "Random Forest": "models/2RandForest.pkl",
    "Gradient Boosting": "models/2GradientBoosting.pkl"
    }
    scaler_path = "models/scalerUnd.pkl"


    models = {}
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
    else:
        st.error("Scaler file not found. Please check the file paths.")
        return None, None

    for model_name, path in model_paths.items():
        if os.path.exists(path):
            with open(path, 'rb') as model_file:
                models[model_name] = pickle.load(model_file)
        else:
            st.error(f"Model file {model_name} not found. Please check the file paths.")
    
    return models, scaler

models, scaler = load_models_and_scaler()

# Transaction type mapping
transaction_types = {
    "CASH_IN": 0,
    "CASH_OUT": 1,
    "DEBIT": 2,
    "PAYMENT": 3,
    "TRANSFER": 4
}

# Horizontal navigation menu
selected = option_menu(
    menu_title=None,
    options=[ "Batch Prediction", "Single Prediction"],
    default_index=0,
    orientation="horizontal",
)

# Single Prediction Page
if selected == "Single Prediction":
    st.header("Single Transaction Prediction")

    # Input fields for user data
    st.subheader("Enter Transaction Details")
    
    # Use columns to organize inputs
    col1, col2 = st.columns(2)
    
    with col1:
        transaction_type = st.selectbox("Type", list(transaction_types.keys()))
        amount = st.number_input("Amount", min_value=0.0, value=181.0)
        oldbalanceOrg = st.number_input("Old Balance Org", min_value=0.0, value=181.0)
    
    with col2:
        oldbalanceDest = st.number_input("Old Balance Dest", min_value=0.0, value=0.0)
        newbalanceDest = st.number_input("New Balance Dest", min_value=0.0, value=0.0)
    
    # Create a DataFrame from user input
    input_data = {
        'type': transaction_types[transaction_type],
        'amount': amount,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': oldbalanceOrg - amount,
        'oldbalanceDest': oldbalanceDest,
        'newbalanceDest': newbalanceDest,
        'isFlaggedFraud': 0,
    }

    input_df = pd.DataFrame([input_data])
    
    # Display user input
    # st.subheader("User Input")
    # st.write(input_df)

    # Make a prediction
    if models and scaler:
        input_data_scaled = scaler.transform(input_df)
        prediction = models["XGBoost"].predict(input_data_scaled)[0]
        prediction_prob = models["XGBoost"].predict_proba(input_data_scaled)[0][1]

        st.subheader("Prediction")
        if prediction == 1:
            st.error(f"The transaction is **fraudulent** with a probability of {prediction_prob:.2f}.")
        else:
            st.success(f"The transaction is **not fraudulent** with a probability of {1 - prediction_prob:.2f}.")
    else:
        st.warning("Models or scaler not loaded. Please check the file paths.")

# Batch Prediction Page
elif selected == "Batch Prediction":
    st.header("Batch Transaction Prediction")
    st.subheader("Upload a CSV File for Batch Predictions")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    print("Models:", models)

    model_choice = st.selectbox("Choose a Model", list(models.keys()), index=0)

    if uploaded_file is not None:
        batch_data = pd.read_csv(uploaded_file)
        required_columns = ['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud']
        
        if all(column in batch_data.columns for column in required_columns):
            type_mapping = {'PAYMENT': 3, 'TRANSFER': 4, 'CASH_OUT': 1, 'DEBIT': 2, 'CASH_IN': 0}
            batch_data['type'] = batch_data['type'].map(type_mapping)
            batch_data_scaled = scaler.transform(batch_data[required_columns])
            batch_predictions = models[model_choice].predict(batch_data_scaled)
            batch_predictions_proba = models[model_choice].predict_proba(batch_data_scaled)[:, 1]
            
            batch_data['Prediction'] = batch_predictions
            batch_data['Prediction_Probability'] = batch_predictions_proba
            
            st.subheader("Batch Predictions")
            st.write(batch_data)
            
            st.download_button(
                label="Download Predictions",
                data=batch_data.to_csv(index=False).encode('utf-8'),
                file_name='batch_predictions.csv',
                mime='text/csv'
            )
        else:
            st.error(f"The uploaded file must contain the following columns: {required_columns}")
