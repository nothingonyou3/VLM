import os  # Import the os module for interacting with the operating system
import numpy as np  # Import numpy for numerical operations
import pandas as pd  # Import pandas for data manipulation and analysis
from tqdm import tqdm  # Import tqdm for displaying progress bars
from argparse import ArgumentParser  # Import ArgumentParser for command-line argument parsing
from torchvision import models  # Import models from torchvision for pretrained models
import torch  # Import PyTorch for tensor operations and neural networks
import torch.nn as nn  # Import the neural network module from PyTorch
import torch.nn.functional as F  # Import functional interface from PyTorch's neural network module
import torchvision  # Import torchvision for image processing utilities
from torchvision import transforms  # Import transforms for data augmentation and preprocessing
import torchvision.transforms as T  # Import alias T for transforms
from torch import optim  # Import optimizers from PyTorch
import open_clip  # Import open_clip for loading CLIP models
import pytorch_lightning as pl  # Import PyTorch Lightning for simplified model training
import timm  # Import timm for pretrained models
from PIL import Image  # Import Image from PIL for image processing
from torchmetrics.functional import auroc  # Import auroc for computing Area Under the Receiver Operating Characteristic Curve

# Define a PyTorch Lightning module for the UNINet Text Model
class UNINetTextModel(pl.LightningModule):
    def __init__(self, num_classes=2, batch_size=64, optim_lr=0.0001):
        super().__init__()  # Initialize the parent class (pl.LightningModule)
        self.num_classes = num_classes  # Set the number of output classes
        self.optim_lr = optim_lr  # Set the learning rate for the optimizer
        self.batch_size = batch_size  # Set the batch size

        self.predictions = []  # Initialize list to store model predictions
        self.targets = []  # Initialize list to store true labels

        self.train_step_preds = []  # Initialize list to store predictions during training
        self.train_step_trgts = []  # Initialize list to store true labels during training
        self.val_step_preds = []  # Initialize list to store predictions during validation
        self.val_step_trgts = []  # Initialize list to store true labels during validation
        self.train_loss = []  # Initialize list to store loss values during training
        self.val_loss = []  # Initialize list to store loss values during validation

        # Create the CLIP model and preprocessing transforms
        self.model, self.preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms('hf-hub:wisdomik/QuiltNet-B-32')
        # Create a TIMM model for image encoding
        self.model_image = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)
        self.text_embed_size = 512  # Size of the text embedding
        self.image_embed_size = 1024  # Size of the image embedding
        self.fc = nn.Linear(self.image_embed_size, num_classes)  # Fully connected layer for classification
        self.multihead_attn = nn.MultiheadAttention(self.image_embed_size, 1)  # Multi-head attention layer

        print("model created")  # Print message indicating that the model is created
        print(self.device)  # Print the device (CPU or GPU) the model is on

    # Forward pass of the model
    def forward(self, x, text_inputs):
        text_embeds = self.model.encode_text(text_inputs.squeeze(1))  # Encode text inputs
        image_embeds = self.model_image(x)  # Encode image inputs using TIMM model
        # Apply multi-head attention to combine image and text embeddings
        attn_output, attn_output_weights = self.multihead_attn(image_embeds, text_embeds, text_embeds)  # Query, key, value
        # Use attention output as input to the fully connected layer
        x = attn_output
        out = self.fc(x)  # Get final output from the fully connected layer
        return out

    # Compute loss using cross-entropy
    def compute_loss(self, y, yp):
        return F.cross_entropy(y, yp)

    # Configure the optimizer
    def configure_optimizers(self):
        # Use SGD optimizer with momentum, Nesterov acceleration, and weight decay
        optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9, nesterov=True, weight_decay=0.0001)
        return optimizer

    # Process a batch of data
    def process_batch(self, batch):
        img, txt, lab = batch  # Unpack the batch into images, text, and labels
        img = img.squeeze(1).to(self.device)  # Remove singleton dimension and move images to the device
        txt = torch.tensor(txt).to(self.device)  # Convert text to tensor and move to the device
        lab = lab.to(self.device)  # Move labels to the device
        out = self.forward(img, txt)  # Forward pass
        prd = torch.softmax(out, dim=1)  # Apply softmax to get predictions
        loss = self.compute_loss(prd, lab)  # Compute loss
        return loss, prd, lab

    # Training step
    def training_step(self, batch, batch_idx):
        loss, prd, lab = self.process_batch(batch)  # Process the batch
        self.train_step_preds.append(prd)  # Append predictions for this step
        self.train_step_trgts.append(lab)  # Append targets for this step
        self.log('train_loss', loss, batch_size=self.batch_size)  # Log training loss
        ''' Commented out code for additional logging
        batch_ratio = len(np.where(lab.cpu().numpy() == 1)[0]) / len(np.where(lab.cpu().numpy() == 0)[0])
        self.log('batch_ratio', batch_ratio, batch_size=self.batch_size)
        grid = torchvision.utils.make_grid(batch['image'][0:4, ...], nrow=2, normalize=True)
        self.logger.experiment.add_image('images', grid, self.global_step)'''
        return loss

    # End of training epoch
    def on_train_epoch_end(self):
        all_preds = torch.cat(self.train_step_preds, dim=0)  # Concatenate all training step predictions
        all_trgts = torch.cat(self.train_step_trgts, dim=0)  # Concatenate all training step targets
        auc = auroc(all_preds, all_trgts, num_classes=self.num_classes, average='macro', task='multiclass')  # Compute AUROC
        self.log('train_auc', auc, batch_size=len(all_preds))  # Log training AUROC
        self.train_step_preds.clear()  # Clear predictions list for the next epoch
        self.train_step_trgts.clear()  # Clear targets list for the next epoch

    # Validation step
    def validation_step(self, batch, batch_idx):
        loss, prd, lab = self.process_batch(batch)  # Process the batch
        self.val_step_preds.append(prd)  # Append predictions for this step
        self.val_step_trgts.append(lab)  # Append targets for this step
        self.log('val_loss', loss, batch_size=self.batch_size)  # Log validation loss

    # End of validation epoch
    def on_validation_epoch_end(self):
        all_preds = torch.cat(self.val_step_preds, dim=0)  # Concatenate all validation step predictions
        all_trgts = torch.cat(self.val_step_trgts, dim=0)  # Concatenate all validation step targets
        auc = auroc(all_preds, all_trgts, num_classes=self.num_classes, average='macro', task='multiclass')  # Compute AUROC
        self.log('val_auc', auc, batch_size=len(all_preds))  # Log validation AUROC
        self.val_step_preds.clear()  # Clear predictions list for the next epoch
        self.val_step_trgts.clear()  # Clear targets list for the next epoch

    # Start of the test phase
    def on_test_start(self):
        self.predictions = []  # Initialize list to store test predictions
        self.targets = []  # Initialize list to store test targets

    # Test step
    def test_step(self, batch, batch_idx):
        _, prd, lab = self.process_batch(batch)  # Process the batch (ignore the loss)
        self.predictions.append(prd)  # Append predictions for this step
        self.targets.append(lab.squeeze())  # Append targets for this step
