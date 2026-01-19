import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm

# Assuming previous CustomImageDataset and data loading code remains the same
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt
import os
import pandas as pd
from torchvision.io import read_image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_metrics(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device).float(), y.to(device)
            pred = model(X)
            
            # Convert one-hot encoded labels back to class indices
            labels = torch.argmax(y, dim=1)
            preds = pred.argmax(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm
    }

def plot_confusion_matrix(cm, classes=range(10)):
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, 
                yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    wandb.log({"confusion_matrix": wandb.Image(plt)})


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):

        img_csv = self.img_labels.iloc[idx, 1:].values
        image = img_csv.reshape(28, 28)

        label = self.img_labels.iloc[idx, 0]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
training_data = CustomImageDataset(
    annotations_file = "/home/walke/college/cv/ass1/archive/train.csv",
    transform=ToTensor(),
    target_transform = Lambda(lambda y: torch.zeros(
    10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
)

test_data = CustomImageDataset(
    annotations_file = "/home/walke/college/cv/ass1/archive/test.csv",
    transform=ToTensor(),
    target_transform = Lambda(lambda y: torch.zeros(
    10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
)

train_dataset, val_dataset = torch.utils.data.random_split(training_data, [int(0.75*len(training_data)), int(0.25*len(training_data))])


from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

def create_model(config):
    class NeuralNetwork(nn.Module):
        def __init__(self, activation_fn):
            super().__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(28*28, config.hidden_size1),
                activation_fn(),
                nn.Linear(config.hidden_size1, config.hidden_size2),
                activation_fn(),
                nn.Linear(config.hidden_size2, 10),
            )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits

    # Map activation function name to actual function
    activation_map = {
        'relu': nn.ReLU,
        'leaky_relu': nn.LeakyReLU,
        'elu': nn.ELU
    }
    
    activation_fn = activation_map.get(config.activation, nn.ReLU)
    return NeuralNetwork(activation_fn).to(device)

def train_and_validate(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        # Access all hyperparameters through wandb.config
        config = wandb.config

        # Create model, loss function, and optimizer
        model = create_model(config)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

        # Training loop
        for epoch in range(config.epochs):
            # Training phase
            model.train()
            train_loss = 0
            for batch, (X, y) in enumerate(train_dataloader):
                X, y = X.to(device).float(), y.to(device)
                
                # Forward pass
                pred = model(X)
                loss = loss_fn(pred, y)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()

            # Validation phase
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for X, y in val_dataloader:
                    X, y = X.to(device).float(), y.to(device)
                    pred = model(X)
                    loss = loss_fn(pred, y)
                    
                    val_loss += loss.item()
                    correct += (pred.argmax(1) == torch.argmax(y, dim=1)).sum().item()
                    total += y.size(0)

            # Log metrics to wandb
            wandb.log({
                'train_loss': train_loss / len(train_dataloader),
                'val_loss': val_loss / len(val_dataloader),
                'val_accuracy': 100. * correct / total
            })
            test_metrics = calculate_metrics(model, test_dataloader, device)
            wandb.log({
                'test_accuracy': test_metrics['accuracy'],
                'test_precision': test_metrics['precision'],
                'test_recall': test_metrics['recall'],
                'test_f1_score': test_metrics['f1_score']
            })
            plot_confusion_matrix(test_metrics['confusion_matrix'])

def main():
    # Sweep configuration
    sweep_config = {
        'method': 'grid',  # Random search
        'metric': {
            'name': 'val_accuracy',
            'goal': 'maximize'
        },
        'parameters': {
            'learning_rate': {
                'values': [1e-3]
            },
            'hidden_size1': {
                'values': [512]
            },
            'hidden_size2': {
                'values': [128]
            },
            'activation': {
                'values': ['relu', 'leaky_relu', 'elu']
            },
            'epochs': {
                'value': 15
            }
        }
    }

    # Initialize wandb sweep
    sweep_id = wandb.sweep(sweep_config, project="mnist-hyperparameter-tuning")

    # Start the sweep
    wandb.agent(sweep_id, train_and_validate)  # Limit to 10 runs to be kind to your laptop

# Make sure to install wandb first with: pip install wandb
if __name__ == "__main__":
    main()