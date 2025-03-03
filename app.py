import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import json

# Title of the app
st.title("UPI Fraud Detection")

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
    options=["Batch Prediction", "Single Prediction", "Model Performance"],
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
        oldbalanceOrg = st.number_input("Sender Balance Before", min_value=0.0, value=181.0)
    
    with col2:
        oldbalanceDest = st.number_input("Reciever Balance Before", min_value=0.0, value=0.0)
        newbalanceDest = st.number_input("Reciever Balance After", min_value=0.0, value=0.0)
    
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
    
    # Define fraud conditions
    is_fraud = False
    fraud_reason = ""

    # Case 1: If transaction type is payment and amount is greater than sender's balance
    if transaction_type == "PAYMENT" and amount > oldbalanceOrg:
        is_fraud = True
        fraud_reason = "Transaction amount exceeds sender's balance."

    # Case 2: If receiver's balance after transaction is less than before transaction
    if newbalanceDest < oldbalanceDest:
        is_fraud = True
        fraud_reason = "Receiver's balance after transaction is less than before."

    # Case 3: If sender's balance after transaction is negative
    if input_data['newbalanceOrig'] < 0:
        is_fraud = True
        fraud_reason = "Sender's balance after transaction is negative."

    # Case 4: If the amount is unusually high (e.g., greater than a threshold)
    if amount > 1000000:  # Example threshold
        is_fraud = True
        fraud_reason = "Transaction amount is unusually high."

    # Make a prediction or use fraud conditions
    if is_fraud:
        st.subheader("Prediction")
        st.error(f"The transaction is **fraudulent** with a probability of 1.0")
    elif models and scaler:
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

elif selected == "Model Performance":
    st.header("Model Performance")
    
    # Load precomputed metrics
    with open("model_metrics.json", "r") as f:
        model_metrics = json.load(f)
    
    # Dropdown to select a model
    model_choice = st.selectbox("Choose a Model", list(model_metrics.keys()), index=0)
    
    # Retrieve metrics for the selected model
    metrics = model_metrics[model_choice]
    
    # Display Evaluation Metrics
    st.subheader("Evaluation Metrics")
    st.write(f"Accuracy: {metrics['accuracy']:.2f}")
    st.write(f"Precision: {metrics['precision']:.2f}")
    st.write(f"Recall: {metrics['recall']:.2f}")
    st.write(f"F1-Score: {metrics['f1']:.2f}")
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = metrics['confusion_matrix']
    fig, ax = plt.subplots()
    ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(len(cm)):
        for j in range(len(cm[i])): 
            ax.text(x=j, y=i, s=cm[i][j], va='center', ha='center')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(fig)
    
    # ROC Curve
    st.subheader("ROC Curve")
    fpr = metrics['roc_curve']['fpr']
    tpr = metrics['roc_curve']['tpr']
    roc_auc = metrics['roc_curve']['roc_auc']
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    st.pyplot(fig)