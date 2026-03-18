# this is evaluate.py file which will handle the evaluation of the trained neural network model on the test set. It will load the best model saved during training, make predictions on the test data, and calculate performance metrics such as accuracy, F1 score, and ROC AUC to assess the model's performance.

import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve

from src.data_preprocessing import preprocess_data
from src.model import HeartNet

def evaluate_model():
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data() # Load preprocessed data

    model = HeartNet(input_dim=X_train.shape[1]) # Initialize the HeartNet model with the number of input features
    model.load_state_dict(torch.load("models/heart_net.pth")) # Load the best

    model.eval() # Set the model to evaluation mode

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32) # Convert test features to a PyTorch tensor

    with torch.no_grad(): # Disable gradient calculation for evaluation
        outputs = model(X_test_tensor) # Forward pass to get model outputs
        probs = torch.sigmoid(outputs).numpy().flatten() # Apply sigmoid to get predicted probabilities and convert to NumPy array
        y_pred = (probs >= 0.5).astype(int) # Convert probabilities to binary predictions using a threshold of 0.5

    # Evaluate performance
    acc = accuracy_score(y_test, y_pred) # Calculate accuracy
    f1 = f1_score(y_test, y_pred) # Calculate F1 score
    roc_auc = roc_auc_score(y_test, probs) # Calculate ROC AUC score using predicted probabilities

    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_test, probs)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    plt.savefig("results/plots/roc_curve.png")
    plt.close()

    print("Neural Network Evaluation on Test Set - Accuracy: {:.4f}, F1 Score: {:.4f}, ROC AUC: {:.4f}".format(acc, f1, roc_auc)) # Print the evaluation results

if __name__ == "__main__":
    evaluate_model() # Run the evaluation process when the script is executed directly