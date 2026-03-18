# This is model.py file which will define the architecture of the neural network for heart disease prediction. It will include layers, activation functions, and the forward pass method to compute the output from the input features. The model will be designed to handle the specific characteristics of the heart disease dataset and optimize for binary classification.

import torch.nn as nn

class HeartNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),  # First hidden layer with 32 neurons. I can also use 64 or 128 neurons, but I will start with 32 to keep it simple and avoid overfitting.
            nn.ReLU(),                # Activation function for the first hidden layer
            nn.Dropout(0.2),             # Dropout layer to prevent overfitting (20% dropout rate)
            nn.Linear(32, 16),         # Second hidden layer with 16 neurons
            nn.ReLU(),                # Activation function for the second hidden layer
            nn.Dropout(0.2),             # Dropout layer to prevent overfitting (20% dropout rate)
            nn.Linear(16, 1),          # Output layer with 1 neuron for binary classification
           
        )
    def forward(self, x):
        return self.model(x)  # Pass the input through the model and return the output
