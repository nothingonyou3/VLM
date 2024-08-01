# Import necessary libraries
from torchvision import datasets  # For accessing torchvision predefined datasets
from torchvision.transforms import ToTensor  # To transform images to tensors
from torch.utils.data import DataLoader  # For loading data in batches
import pytorch_lightning as pl  # For using PyTorch Lightning
import open_clip  # For using the CLIP model and its transformations
import torch  # For tensor operations and PyTorch functions
from torchvision.transforms import v2  # For accessing transformations in the new version of torchvision

# Define a custom class that extends PyTorch Lightning's LightningDataModule
class PCAMDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers):
        super().__init__()  # Call the constructor of the base class
        # Create the CLIP model and preprocessing transformations for training and validation
        self.model, self.preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms('hf-hub:wisdomik/QuiltNet-B-32')

        # Define transformations for the training data
        transform = v2.Compose([
                self.preprocess_val.transforms[0],  # Preprocessing transformation (first)
                self.preprocess_val.transforms[1],  # Preprocessing transformation (second)
                v2.RandomHorizontalFlip(p=0.5),  # Random horizontal flip transformation
                v2.RandomVerticalFlip(p=0.5),  # Random vertical flip transformation
                v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.05),  # Color jitter transformation
                self.preprocess_val.transforms[2],  # Preprocessing transformation (third)
                self.preprocess_val.transforms[3],  # Preprocessing transformation (fourth)
                self.preprocess_val.transforms[4],  # Preprocessing transformation (fifth)
        ])

        # Load the training data with the defined transformations
        self.training_data = datasets.PCAM(
            root="Data/Data",  # Data path
            download=True,  # Download data if not already present
            transform=transform  # Apply transformations to the data
        )

        # Load the test data with the validation transformations
        self.test_data = datasets.PCAM(
            root="Data/Data",  # Data path
            download=True,  # Download data if not already present
            transform=self.preprocess_val,  # Apply validation transformations
            split="test"  # Specify the test data split
        )

        # Load the validation data with the validation transformations
        self.valid_data = datasets.PCAM(
            root="Data/Data",  # Data path
            download=True,  # Download data if not already present
            transform=self.preprocess_val,  # Apply validation transformations
            split="val"  # Specify the validation data split
        )

        # Set the batch size and number of workers for data loading
        self.batch_size = batch_size
        self.num_workers = num_workers

    # Method for the training dataloader
    def train_dataloader(self):
        return DataLoader(dataset=self.training_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    # Method for the validation dataloader
    def val_dataloader(self):
        return DataLoader(dataset=self.valid_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    # Method for the test dataloader
    def test_dataloader(self):
        return DataLoader(dataset=self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
