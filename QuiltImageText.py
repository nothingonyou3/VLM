# Import the os module for interacting with the operating system
import os

# Import numpy for numerical operations
import numpy as np

# Import pandas for data manipulation and analysis
import pandas as pd

# Import tqdm for displaying progress bars
from tqdm import tqdm

# Import ArgumentParser for command-line option parsing
from argparse import ArgumentParser

# Import models from torchvision
from torchvision import models

# Import torch library
import torch

# Import the neural network module from PyTorch
import torch.nn as nn

# Import functional interface from PyTorch's neural network module
import torch.nn.functional as F

# Import torchvision for image processing utilities
import torchvision

# Import transforms for data augmentation and preprocessing
from torchvision import transforms

# Import alias T for transforms
import torchvision.transforms as T

# Import AdamW optimizer from PyTorch
from torch.optim import AdamW

# Import open_clip for loading the CLIP model
import open_clip

# Import PyTorch Lightning for organizing PyTorch code
import pytorch_lightning as pl

# Import Image from PIL for image processing
from PIL import Image

# Import auroc for calculating the Area Under the Receiver Operating Characteristic Curve
from torchmetrics.functional import auroc

# Define a PyTorch Lightning module for the Quilt Image Text Encoder Model
class QuiltImageTextEncoderModel(pl.LightningModule):
    def __init__(self, num_classes=2, batch_size=64, optim_lr=0.0001):
        # Initialize the superclass
        super().__init__()
        # Number of classes for classification
        self.num_classes = num_classes
        # Learning rate for the optimizer
        self.optim_lr = optim_lr
        # Batch size for training
        self.batch_size = batch_size

        # Initialize lists to store predictions and targets
        self.predictions = []
        self.targets = []

        # Initialize lists to store training and validation step predictions and targets
        self.train_step_preds = []
        self.train_step_trgts = []
        self.val_step_preds = []
        self.val_step_trgts = []
        # Initialize lists to store training and validation losses
        self.train_loss = []
        self.val_loss = []

        # Create the CLIP model and transforms for training and validation
        self.model, self.preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms('hf-hub:wisdomik/QuiltNet-B-32')
        # Size of the text embedding
        self.text_embed_size = 512
        # Size of the image embedding
        self.image_embed_size = 512
        # Fully connected layer for classification
        self.fc = nn.Linear(self.text_embed_size + self.image_embed_size, num_classes)

        # Print statements for debugging
        print("model created")
        print(self.device)

    # Forward pass of the model
    def forward(self, image_preprocessed, text_inputs):
        # Encode the image using the CLIP model
        image_embeds = self.model.encode_image(image_preprocessed)
        # Encode the text using the CLIP model
        text_embeds = self.model.encode_text(text_inputs.squeeze(1))
        #print(image_embeds.shape, text_embeds.shape)
        # Concatenate image and text embeddings
        x = torch.cat((image_embeds, text_embeds), 1)
        # Pass the concatenated embeddings through the fully connected layer
        out = self.fc(x)
        return out

    # Compute loss using cross-entropy
    def compute_loss(self, y, yp):
        return F.cross_entropy(y, yp)

    # Configure the optimizer
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.optim_lr)
        return optimizer

    # Process a batch of data
    def process_batch(self, batch):
        # Unpack the batch into images, texts, and labels
        img, txt, lab = batch
        # Squeeze and move images to the appropriate device
        img = img.squeeze(1).to(self.device)
        # Convert texts to tensor and move to the appropriate device
        txt = torch.tensor(txt).to(self.device)
        # Move labels to the appropriate device
        lab = lab.to(self.device)
        # Forward pass
        out = self.forward(img, txt)
        # Apply softmax to get predictions
        prd = torch.softmax(out, dim=1)
        # Compute loss
        loss = self.compute_loss(prd, lab)
        return loss, prd, lab

    # Training step
    def training_step(self, batch, batch_idx):
        # Process the batch
        loss, prd, lab = self.process_batch(batch)
        # Append predictions and targets for this step
        self.train_step_preds.append(prd)
        self.train_step_trgts.append(lab)
        # Log the training loss
        self.log('train_loss', loss, batch_size=self.batch_size)
        ''' Commented out code for logging additional metrics
        batch_ratio = len(np.where(lab.cpu().numpy() == 1)[0]) / len(np.where(lab.cpu().numpy() == 0)[0])
        self.log('batch_ratio', batch_ratio, batch_size=self.batch_size)
        grid = torchvision.utils.make_grid(batch['image'][0:4, ...], nrow=2, normalize=True)
        self.logger.experiment.add_image('images', grid, self.global_step)'''
        return loss

    # End of training epoch
    def on_train_epoch_end(self):
        # Concatenate all predictions and targets
        all_preds = torch.cat(self.train_step_preds, dim=0)
        all_trgts = torch.cat(self.train_step_trgts, dim=0)
        # Compute AUROC for the training epoch
        auc = auroc(all_preds, all_trgts, num_classes=self.num_classes, average='macro', task='multiclass')
        # Log the training AUROC
        self.log('train_auc', auc, batch_size=len(all_preds))
        # Clear the lists for the next epoch
        self.train_step_preds.clear()
        self.train_step_trgts.clear()

    # Validation step
    def validation_step(self, batch, batch_idx):
        # Process the batch
        loss, prd, lab = self.process_batch(batch)
        # Append predictions and targets for this step
        self.val_step_preds.append(prd)
        self.val_step_trgts.append(lab)
        # Log the validation loss
        self.log('val_loss', loss, batch_size=self.batch_size)

    # End of validation epoch
    def on_validation_epoch_end(self):
        # Concatenate all predictions and targets
        all_preds = torch.cat(self.val_step_preds, dim=0)
        all_trgts = torch.cat(self.val_step_trgts, dim=0)
        # Compute AUROC for the validation epoch
        auc = auroc(all_preds, all_trgts, num_classes=self.num_classes, average='macro', task='multiclass')
        # Log the validation AUROC
        self.log('val_auc', auc, batch_size=len(all_preds))
        # Clear the lists for the next epoch
        self.val_step_preds.clear()
        self.val_step_trgts.clear()

    # Start of the test phase
    def on_test_start(self):
        # Initialize lists to store predictions and targets
        self.predictions = []
        self.targets = []

    # Test step
    def test_step(self, batch, batch_idx):
        # Process the batch (ignore the loss)
        _, prd, lab = self.process_batch(batch)
        # Append predictions and targets for this step
        self.predictions.append(prd)
        self.targets.append(lab.squeeze())
