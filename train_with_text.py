# Import PCAMTDataModule from a custom module
from Data.PCAMTDataLoader import PCAMTDataModule

# Import os module for interacting with the operating system
import os

# Import numpy for numerical operations
import numpy as np

# Import pandas for data manipulation and analysis
import pandas as pd

# Import tqdm for displaying progress bars
from tqdm import tqdm

# Import ArgumentParser for command-line option parsing
from argparse import ArgumentParser

# Import shuffle for shuffling datasets
from sklearn.utils import shuffle

# Import PyTorch Lightning for organizing PyTorch code
import pytorch_lightning as pl

# Import torch library
import torch

# Import ModelCheckpoint and EarlyStopping callbacks from PyTorch Lightning
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Import TensorBoardLogger for logging training metrics
from pytorch_lightning.loggers import TensorBoardLogger

# Import auroc for calculating the Area Under the Receiver Operating Characteristic Curve
from torchmetrics.functional import auroc

# Import custom models
from Models.QuiltImageText import QuiltImageTextEncoderModel
from Models.ResNetText import ResNetTextModel

# Function to save predictions and calculate AUROC
def save_predictions(model, output_fname, num_classes):
    # Concatenate predictions and targets from the model
    prds = torch.cat(model.predictions, dim=0)
    trgs = torch.cat(model.targets, dim=0)

    # Calculate AUROC for the predictions
    auc = auroc(prds, trgs, num_classes=num_classes, average='macro', task='multiclass')

    # Print AUROC for the test set
    print('AUROC (test)')
    print(auc)

    # Create column names for the DataFrame
    cols_names = ['class_' + str(i) for i in range(0, num_classes)]

    # Create a DataFrame with predictions and targets
    df = pd.DataFrame(data=prds.cpu().numpy(), columns=cols_names)
    df['target'] = trgs.cpu().numpy()

    # Save the DataFrame to a CSV file
    df.to_csv(output_fname, index=False)

# Number of training epochs
epochs = 20

# Set device to "gpu" if CUDA is available, otherwise use CPU
device = "gpu" if torch.cuda.is_available() else "cpu"

# Set the seed for reproducibility
pl.seed_everything(42, workers=True)

# Number of classes in the dataset
num_classes = 2

# Batch size and number of workers for data loading
batch_size, num_workers = 64, 30

# Define the model to use (ResNetTextModel)
Net = ResNetTextModel

'''Loading Data'''
# Load data using PCAMTDataModule
data = PCAMTDataModule(batch_size, num_workers)

# Output directory for saving model checkpoints and predictions
output_base_dir = 'output'
output_name = 'train_with_text_resnet_att_aug_'
output_dir = os.path.join(output_base_dir, output_name)

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

'''Creating the model'''
# Create the model using ResNetTextModel
model = Net(num_classes, batch_size)

# Print statements for debugging
print('=============================================================')
print('Training...')
print(device)

# Define a checkpoint callback to save the best model based on validation loss
checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode='min')

# Define an early stopping callback to stop training if validation loss doesn't improve
early_stop_callback = EarlyStopping(
    monitor="val_loss",
    min_delta=0.001,
    patience=10,  # Number of validation epochs with no improvement
    verbose=False,
    mode="min",
)

# Create a PyTorch Lightning trainer
trainer = pl.Trainer(
    callbacks=[checkpoint_callback],  # List of callbacks
    log_every_n_steps=5,  # Log metrics every 5 steps
    max_epochs=epochs,  # Maximum number of epochs
    accelerator=device,  # Use GPU or CPU based on availability
    devices=1,  # Number of devices to use
    logger=TensorBoardLogger(output_base_dir, name=output_name),  # Logger for TensorBoard
)

# Disable default HP metric logging
trainer.logger._default_hp_metric = False

# Train the model using the trainer
trainer.fit(model, data)

# Load the best model from the checkpoint
model = Net.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

# Print the path to the best model checkpoint
print(trainer.checkpoint_callback.best_model_path)

# Evaluate the model on the test set
print(trainer.test(model=model, datamodule=data))

# Save predictions and calculate AUROC
save_predictions(model, os.path.join(output_dir, 'predictions.csv'), num_classes)
