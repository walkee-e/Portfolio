import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import numpy as np
import pandas as pd
from torchvision.transforms import ToTensor, Lambda

# Reuse existing dataset and dataloader setup

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

new_test_data = [] 

for index in range(test_img_labels.shape[0]):
    img_csv = np.matrix(test_img_labels.iloc[index, 1:].values, dtype=np.uint8).reshape(28, 28)
    label = test_img_labels.iloc[index, 0]


    winSize = (20,20)
    blockSize = (10,10)
    blockStride = (5,5)
    cellSize = (10,10)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    signedGradients = True
 
    hog = cv.HOGDescriptor(winSize,blockSize,blockStride,
    cellSize,nbins,derivAperture,
    winSigma,histogramNormType,L2HysThreshold,
    gammaCorrection,nlevels, signedGradients)

    descriptor = hog.compute(img_csv)

    row = np.concatenate(([label], descriptor))
    new_test_data.append(row)

column_names = ["label"] + [f"feature_{i+1}" for i in range(len(new_test_data[0]) - 1)]
new_df = pd.DataFrame(new_test_data, columns=column_names)


new_df.to_csv("hog_test.csv", index=False)

train_img_labels = pd.read_csv("/home/walke/college/cv/ass1/archive/train.csv")

new_train_data = [] 

for index in range(train_img_labels.shape[0]):
    img_csv = np.matrix(train_img_labels.iloc[index, 1:].values, dtype=np.uint8).reshape(28, 28)
    label = train_img_labels.iloc[index, 0]


    winSize = (20,20)
    blockSize = (10,10)
    blockStride = (5,5)
    cellSize = (10,10)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    signedGradients = True
 
    hog = cv.HOGDescriptor(winSize,blockSize,blockStride,
    cellSize,nbins,derivAperture,
    winSigma,histogramNormType,L2HysThreshold,
    gammaCorrection,nlevels, signedGradients)

    descriptor = hog.compute(img_csv)

    row = np.concatenate(([label], descriptor))
    new_train_data.append(row)

column_names = ["label"] + [f"feature_{i+1}" for i in range(len(new_train_data[0]) - 1)]
new_df = pd.DataFrame(new_train_data, columns=column_names)


new_df.to_csv("hog_train.csv", index=False)


class hogImageDataset(torch.utils.data.Dataset):
    def __init__(self, annotations_file, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_csv = self.img_labels.iloc[idx, 1:].values
        image = img_csv.reshape(9, 9)
        label = self.img_labels.iloc[idx, 0]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(int(label))
        return image, label

training_data_hog = hogImageDataset(
    annotations_file = "hog_train.csv",
    transform=ToTensor(),
    target_transform = Lambda(lambda y: torch.zeros(
    10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
)

test_data_hog = hogImageDataset(
    annotations_file = "hog_test.csv",
    transform=ToTensor(),
    target_transform = Lambda(lambda y: torch.zeros(
    10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
)

train_dataset_hog, val_dataset_hog = torch.utils.data.random_split(training_data_hog, [int(0.75*len(training_data_hog)), int(0.25*len(training_data_hog))])

from torch.utils.data import DataLoader
train_dataloader_hog = DataLoader(train_dataset_hog, batch_size=64, shuffle=True)
val_dataloader_hog = DataLoader(val_dataset_hog, batch_size=64, shuffle=True)
test_dataloader_hog = DataLoader(test_data_hog, batch_size=64, shuffle=True)

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



# Neural Network with configurable activation
class NeuralNetwork_hog(nn.Module):
    def __init__(self, activation='relu', hidden_layers=2, layer_size=512):
        super().__init__()
        self.flatten = nn.Flatten()
        
        # Dynamically create layers based on configuration
        layers = []
        prev_size = 9*9
        
        for _ in range(hidden_layers):
            layers.append(nn.Linear(prev_size, layer_size))
            
            # Select activation function
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU())
            elif activation == 'elu':
                layers.append(nn.ELU())
            
            prev_size = layer_size
        
        layers.append(nn.Linear(prev_size, 10))
        
        self.linear_stack = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        return self.linear_stack(x)

# Training function
def train(config=None):
    # Initialize wandb
    with wandb.init(config=config):
        # Access hyperparameters through wandb.config
        config = wandb.config
        
        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load datasets
        training_data_hog = hogImageDataset(
            annotations_file="hog_train.csv",
            transform=ToTensor(),
            target_transform=Lambda(lambda y: torch.zeros(
                10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
        )
        
        # Split into train and validation
        train_size = int(0.75 * len(training_data_hog))
        val_size = len(training_data_hog) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(training_data_hog, [train_size, val_size])
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        
        # Initialize model
        model = NeuralNetwork_hog(
            activation=config.activation, 
            hidden_layers=config.hidden_layers, 
            layer_size=config.layer_size
        ).to(device)
        
        # Loss and optimizer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        
        # Training loop
        for epoch in range(15):  # Limit epochs to prevent long runs
            model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device).float(), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = loss_fn(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()

                
            
            # Validation
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device).float(), batch_y.to(device)
                    outputs = model(batch_x)
                    loss = loss_fn(outputs, batch_y)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    _, true_labels = torch.max(batch_y, 1)
                    total += true_labels.size(0)
                    correct += (predicted == true_labels).sum().item()
            
            # Log metrics to wandb
            wandb.log({
                "train_loss": train_loss / len(train_loader),
                "val_loss": val_loss / len(val_loader),
                "val_accuracy": 100 * correct / total
            })
            test_metrics = calculate_metrics(model, test_dataloader_hog, device)
            wandb.log({
                'test_accuracy': test_metrics['accuracy'],
                'test_precision': test_metrics['precision'],
                'test_recall': test_metrics['recall'],
                'test_f1_score': test_metrics['f1_score']
            })
            plot_confusion_matrix(test_metrics['confusion_matrix'])

# Sweep configuration
sweep_config = {
    'method': 'grid',  # random search
    'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'learning_rate': {
            'values': [1e-3]
        },
        'batch_size': {
            'values': [64]
        },
        'activation': {
            'values': ['relu', 'leaky_relu', 'elu']
        },
        'hidden_layers': {
            'values': [2]
        },
        'layer_size': {
            'values': [256]
        }
    }
}

# Run the sweep
def main():
    # Initialize wandb project
    wandb.login()  # Make sure to set WANDB_API_KEY environment variable
    sweep_id = wandb.sweep(sweep_config, project="hog-mnist-tuning")
    wandb.agent(sweep_id, train)  # Limit to 20 runs to be kind to your laptop

if __name__ == "__main__":
    main()