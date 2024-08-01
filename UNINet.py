import os  # Import the os module for interacting with the operating system
import numpy as np  # Import numpy for numerical operations
import pandas as pd  # Import pandas for data manipulation and analysis
from tqdm import tqdm  # Import tqdm for displaying progress bars
from argparse import ArgumentParser  # Import ArgumentParser for command-line option parsing
from torchvision import models  # Import models from torchvision
import torch  # Import torch library
import torch.nn as nn  # Import the neural network module from PyTorch
import torch.nn.functional as F  # Import functional interface from PyTorch's neural network module
import torchvision  # Import torchvision for image processing utilities
from torchvision import transforms  # Import transforms for data augmentation and preprocessing
import torchvision.transforms as T  # Import alias T for transforms
from torch import optim  # Import optimizers from PyTorch
import open_clip  # Import open_clip for loading the CLIP model
import pytorch_lightning as pl  # Import PyTorch Lightning for organizing PyTorch code
import timm  # Import timm for pretrained models
from PIL import Image  # Import Image from PIL for image processing
from torchmetrics.functional import auroc  # Import auroc for calculating the Area Under the Receiver Operating Characteristic Curve

# Define a PyTorch Lightning module for the UNINet Model
class UNINetModel(pl.LightningModule):
    def __init__(self, num_classes=2, batch_size=64, optim_lr=0.0001):
        super().__init__()  # Initialize the superclass
        self.num_classes = num_classes  # Number of classes for classification
        self.optim_lr = optim_lr  # Learning rate for the optimizer
        self.batch_size = batch_size  # Batch size for training

        self.predictions = []  # Initialize list to store predictions
        self.targets = []  # Initialize list to store targets

        self.train_step_preds = []  # Initialize list to store training step predictions
        self.train_step_trgts = []  # Initialize list to store training step targets
        self.val_step_preds = []  # Initialize list to store validation step predictions
        self.val_step_trgts = []  # Initialize list to store validation step targets
        self.train_loss = []  # Initialize list to store training losses
        self.val_loss = []  # Initialize list to store validation losses

        # Create the UNINet model with pretrained weights
        self.model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)
        self.image_embed_size = 1024  # Size of the image embedding
        self.fc = nn.Linear(self.image_embed_size, num_classes)  # Fully connected layer for classification

        # Print statements for debugging
        print("model created")
        print(self.device)

    # Forward pass of the model
    def forward(self, x):
        x = self.model(x)  # Pass input through the model
        out = self.fc(x)  # Pass the output through the fully connected layer
        return out

    # Compute loss using cross-entropy
    def compute_loss(self, y, yp):
        return F.cross_entropy(y, yp)

    # Configure the optimizer
    def configure_optimizers(self):
        # Use SGD optimizer with momentum and weight decay
        optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9, nesterov=True, weight_decay=0.0001)
        return optimizer

    # Process a batch of data
    def process_batch(self, batch):
        img = batch[0].to(self.device)  # Move images to the appropriate device
        lab = batch[1].to(self.device)  # Move labels to the appropriate device
        out = self.forward(img)  # Forward pass
        prd = torch.softmax(out, dim=1)  # Apply softmax to get predictions
        loss = self.compute_loss(prd, lab)  # Compute loss
        return loss, prd, lab

    # Training step
    def training_step(self, batch, batch_idx):
        loss, prd, lab = self.process_batch(batch)  # Process the batch
        self.train_step_preds.append(prd)  # Append predictions for this step
        self.train_step_trgts.append(lab)  # Append targets for this step
        self.log('train_loss', loss, batch_size=self.batch_size)  # Log the training loss
        ''' Commented out code for logging additional metrics
        batch_ratio = len(np.where(lab.cpu().numpy() == 1)[0]) / len(np.where(lab.cpu().numpy() == 0)[0])
        self.log('batch_ratio', batch_ratio, batch_size=self.batch_size)
        grid = torchvision.utils.make_grid(batch['image'][0:4, ...], nrow=2, normalize=True)
        self.logger.experiment.add_image('images', grid, self.global_step)'''
        return loss

    # End of training epoch
    def on_train_epoch_end(self):
        all_preds = torch.cat(self.train_step_preds, dim=0)  # Concatenate all predictions
        all_trgts = torch.cat(self.train_step_trgts, dim=0)  # Concatenate all targets
        auc = auroc(all_preds, all_trgts, num_classes=self.num_classes, average='macro', task='multiclass')  # Compute AUROC
        self.log('train_auc', auc, batch_size=len(all_preds))  # Log the training AUROC
        self.train_step_preds.clear()  # Clear the predictions list for the next epoch
        self.train_step_trgts.clear()  # Clear the targets list for the next epoch

    # Validation step
    def validation_step(self, batch, batch_idx):
        loss, prd, lab = self.process_batch(batch)  # Process the batch
        self.val_step_preds.append(prd)  # Append predictions for this step
        self.val_step_trgts.append(lab)  # Append targets for this step
        self.log('val_loss', loss, batch_size=self.batch_size)  # Log the validation loss

    # End of validation epoch
    def on_validation_epoch_end(self):
        all_preds = torch.cat(self.val_step_preds, dim=0)  # Concatenate all predictions
        all_trgts = torch.cat(self.val_step_trgts, dim=0)  # Concatenate all targets
        auc = auroc(all_preds, all_trgts, num_classes=self.num_classes, average='macro', task='multiclass')  # Compute AUROC
        self.log('val_auc', auc, batch_size=len(all_preds))  # Log the validation AUROC
        self.val_step_preds.clear()  # Clear the predictions list for the next epoch
        self.val_step_trgts.clear()  # Clear the targets list for the next epoch

    # Start of the test phase
    def on_test_start(self):
        self.predictions = []  # Initialize list to store predictions
        self.targets = []  # Initialize list to store targets

    # Test step
    def test_step(self, batch, batch_idx):
        _, prd, lab = self.process_batch(batch)  # Process the batch (ignore the loss)
        self.predictions.append(prd)  # Append predictions for this step
        self.targets.append(lab.squeeze())  # Append targets for this step
