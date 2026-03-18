# This file is a basline.py file compare neural network against logistic regression and evalute performance improvement. 

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from src.data_preprocessing import preprocess_data

def run_baseline():
    X_train, X_val, y_train, y_val, y_test = preprocess_data()

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Predict on validation set
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1]  # Get predicted probabilities for the positive class

    # Evaluate performance
    acc = accuracy_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred) # Calculate F1 score
    roc_auc = roc_auc_score(y_val, y_val_proba) # Calculate ROC AUC score

    print(f"Baseline Logistic Regression - Accuracy: {acc:.4f}, F1 Score: {f1:.4f}, ROC AUC: {roc_auc:.4f}")

if __name__ == "__main__":
    run_baseline() # Run the baseline evaluation when the script is executed directly