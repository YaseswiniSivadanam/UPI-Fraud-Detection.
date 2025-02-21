import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

# Load your test dataset
test_data_path = "data/data.csv"  # Replace with your test data path
test_data = pd.read_csv(test_data_path)

# Define required columns
required_columns = ['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud']

# Map transaction types to numerical values
type_mapping = {'PAYMENT': 3, 'TRANSFER': 4, 'CASH_OUT': 1, 'DEBIT': 2, 'CASH_IN': 0}
test_data['type'] = test_data['type'].map(type_mapping)

# Separate features and target
X_test = test_data[required_columns]
y_test = test_data['isFlaggedFraud']

# Load your models and scaler
models = {
    "XGBoost": pickle.load(open("models/2xgboost.pkl", "rb")),
    "Decision Tree": pickle.load(open("models/2decisionTree.pkl", "rb")),
    "Random Forest": pickle.load(open("models/2RandForest.pkl", "rb")),
    "Gradient Boosting": pickle.load(open("models/2GradientBoosting.pkl", "rb"))
}
scaler = pickle.load(open("models/scalerUnd.pkl", "rb"))

# Scale the test data
X_test_scaled = scaler.transform(X_test)

# Precompute metrics for each model
model_metrics = {}

for model_name, model in models.items():
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Store metrics
    model_metrics[model_name] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm.tolist(),  # Convert numpy array to list for JSON serialization
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "roc_auc": roc_auc
        }
    }

# Save the precomputed metrics to a file
import json
with open("model_metrics.json", "w") as f:
    json.dump(model_metrics, f, indent=4)

print("Metrics precomputed and saved to model_metrics.json")