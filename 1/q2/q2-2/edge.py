import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm


import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt
import os
import pandas as pd
from torchvision.io import read_image
import cv2 as cv
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



test_img_labels = pd.read_csv("/home/walke/college/cv/ass1/archive/test.csv")
train_img_labels = pd.read_csv("/home/walke/college/cv/ass1/archive/train.csv")

for index in range(test_img_labels.shape[0]):
    img_csv = np.matrix(test_img_labels.iloc[index, 1:].values, dtype=np.uint8).reshape(28, 28)
    edges = cv.Canny(img_csv, 100, 200)
    test_img_labels.iloc[index, 1:] = edges.flatten()

test_img_labels.to_csv("edges_test.csv", index=False)

for index in range(train_img_labels.shape[0]):
    img_csv = np.matrix(train_img_labels.iloc[index, 1:].values, dtype=np.uint8).reshape(28, 28)
    edges = cv.Canny(img_csv, 100, 200)
    train_img_labels.iloc[index, 1:] = edges.flatten()

train_img_labels.to_csv("edges_train.csv", index=False)

class MLP2ImageDataset(Dataset):
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
    

training_data_edge = MLP2ImageDataset(
    annotations_file = "edges_train.csv",
    transform=ToTensor(),
    target_transform = Lambda(lambda y: torch.zeros(
    10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
)

test_data_edge = MLP2ImageDataset(
    annotations_file = "edges_test.csv",
    transform=ToTensor(),
    target_transform = Lambda(lambda y: torch.zeros(
    10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
)

train_dataset_edge, val_dataset_edge = torch.utils.data.random_split(training_data_edge, [int(0.75*len(training_data_edge)), int(0.25*len(training_data_edge))])

from torch.utils.data import DataLoader
train_dataloader_edge = DataLoader(train_dataset_edge, batch_size=64, shuffle=True)
val_dataloader_edge = DataLoader(val_dataset_edge, batch_size=64, shuffle=True)
test_dataloader_edge = DataLoader(test_data_edge, batch_size=64, shuffle=True)

batch_size=64

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
    with wandb.init(config=config):
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
            for batch, (X, y) in enumerate(train_dataloader_edge):
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
                for X, y in val_dataloader_edge:
                    X, y = X.to(device).float(), y.to(device)
                    pred = model(X)
                    loss = loss_fn(pred, y)
                    
                    val_loss += loss.item()
                    correct += (pred.argmax(1) == torch.argmax(y, dim=1)).sum().item()
                    total += y.size(0)

            # Log metrics to wandb
            wandb.log({
                'train_loss': train_loss / len(train_dataloader_edge),
                'val_loss': val_loss / len(val_dataloader_edge),
                'val_accuracy': 100. * correct / total
            })            
            test_metrics = calculate_metrics(model, test_dataloader_edge, device)
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
        'method': 'grid',
        'metric': {
            'name': 'val_accuracy',
            'goal': 'maximize'
        },
        'parameters': {
            'learning_rate': {
                'values': [1e-3]
            },
            'hidden_size1': {
                'values': [256]
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
    sweep_id = wandb.sweep(sweep_config, project="mnist-edge-hyperparameter-tuning")

    # Start the sweep
    wandb.agent(sweep_id, train_and_validate)

if __name__ == "__main__":
    main()