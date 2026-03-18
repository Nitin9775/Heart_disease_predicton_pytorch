# This is train.py file which will handle the training loop for the neural network model. It will include loading the preprocessed data, creating dataloaders, defining the loss function and optimizer, and iterating through epochs to train the model while monitoring performance on the validation set.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data_preprocessing import preprocess_data
from src.dataset import HeartDiseaseDataset
from src.model import HeartNet

def train_model():
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data() # Load preprocessed data

    #create datasets and dataloaders
    train_dataset = HeartDiseaseDataset(X_train, y_train)
    val_dataset = HeartDiseaseDataset(X_val, y_val)

    #dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) # Create dataloader for training data with batch size of 32 and shuffling
    val_loader = DataLoader(val_dataset, batch_size=32) # Create dataloader for validation data with batch size of 32 (no shuffling)

    # Initialize the model, loss function, and optimizer
    model = HeartNet(input_dim=X_train.shape[1]) # Initialize the HeartNet model with the number of input features
    criterion = nn.BCEWithLogitsLoss() # Use binary cross-entropy loss with logits for binary classification
    optimizer = optim.Adam(model.parameters(), lr=0.001) # Use Adam optimizer with a learning rate of 0.001

    best_val_loss = float("inf") # Initialize best validation loss to infinity for tracking the best model
    epochs = 50 # 50 epochs for training

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train() # Set the model to training mode
        train_loss = 0.0

        
        for X_batch, y_batch in train_loader: # Iterate through batches of training data
            optimizer.zero_grad() # Zero the gradients
            outputs = model(X_batch) # Forward pass to get model outputs
            loss = criterion(outputs, y_batch) # Calculate the loss
            loss.backward() # Backward pass to compute gradients
            optimizer.step() # Update model parameters
            train_loss += loss.item() # Accumulate the training loss

        # Validation phase
        model.eval() # Set the model to evaluation mode
        val_loss = 0.0

        with torch.no_grad(): # Disable gradient calculation for validation
            for X_batch, y_batch in val_loader: # Iterate through batches of validation data
                outputs = model(X_batch) # Forward pass to get model outputs
                loss = criterion(outputs, y_batch) # Calculate the loss
                val_loss += loss.item() # Accumulate the validation loss

        # Print average losses for the epoch
        train_loss /= len(train_loader) # Average training loss over all batches
        val_loss /= len(val_loader) # Average validation loss over all batches
       
        train_losses.append(train_loss) # Append the average training loss for this epoch to the list
        val_losses.append(val_loss) # Append the average validation loss for this epoch to the list
       
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        # Save the best model based on validation loss
        if val_loss < best_val_loss: # Check if the current validation loss is better than the best one
            best_val_loss = val_loss # Update the best validation loss
            torch.save(model.state_dict(), "models/heart_net.pth") # Save the model state dictionary to a file
            print("Best model saved with validation loss: {:.4f}".format(best_val_loss)) # Print a message indicating that the best model has been saved

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()

    plt.savefig("results/plots/loss_curve.png") # Save the loss curve plot to a file
    plt.close()

if __name__ == "__main__":
    train_model() # Run the training process when the script is executed directly