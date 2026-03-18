# This is dataset.py file which will convert my data into Pytorch tensors and create dataloaders for training, validation, and testing. It will also handle batching and shuffling of the data during training.

import torch
from torch.utils.data import Dataset

class HeartDiseaseDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32) # Convert features to PyTorch tensor
        self.y = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)   # Convert target variable to PyTorch tensor and this view(-1, 1) is used to reshape the target variable into a column vector, which is necessary for binary classification tasks where the output layer has a single neuron.

    def __len__(self):
        return len(self.X)  # Return the number of samples in the dataset

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]  # Return a single sample (features and target) at the given index
