# Import datasets and transformations from torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

# Import Dataset and DataLoader from torch.utils.data
from torch.utils.data import Dataset, DataLoader

# Import pytorch_lightning for organizing PyTorch code
import pytorch_lightning as pl

# Import open_clip library
import open_clip

# Import torch library
import torch

# Import pandas for handling data frames
import pandas as pd

# Import json for handling JSON data
import json

# Import additional transformations from torchvision
from torchvision import transforms

# Import numpy for numerical operations
import numpy as np

# Import new version of transformations from torchvision
from torchvision.transforms import v2

# Set device to "cuda" if GPU is available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define a custom dataset class inheriting from Dataset
class PachDataset(Dataset):

    def __init__(self, split):
        """
        Arguments:
            split (string): Dataset split, either 'train', 'val', or 'test'.
        """
        # Initialize the model and its transformations using open_clip
        self.model, self.preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms('hf-hub:wisdomik/QuiltNet-B-32')
        self.model.to(device)  # Move the model to the specified device (GPU/CPU)
        self.tokenizer = open_clip.get_tokenizer('hf-hub:wisdomik/QuiltNet-B-32')  # Initialize tokenizer

        # Set the transformation to the validation preprocessing
        transform = self.preprocess_val

        # If the split is 'train', apply additional data augmentation
        if split == "train":
            transform = v2.Compose([
                self.preprocess_val.transforms[0],  # Resize
                self.preprocess_val.transforms[1],  # CenterCrop
                v2.RandomHorizontalFlip(p=0.5),  # Random horizontal flip
                v2.RandomVerticalFlip(p=0.5),  # Random vertical flip
                v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.05),  # Color jitter
                self.preprocess_val.transforms[2],  # ToTensor
                self.preprocess_val.transforms[3],  # Normalize
                self.preprocess_val.transforms[4],  # Final transformation
            ])

        # Load the PCAM dataset
        self.data = datasets.PCAM(
            root="Data/Data",  # Root directory for the dataset
            download=True,  # Download the dataset if not already present
            transform=transform,  # Apply the defined transformations
            split=split  # Dataset split (train/val/test)
        )

        # Optional suffix for the sample file
        suffix = ""
        if False:
            suffix = "_v2"

        # Load text embeddings and captions from a JSON file
        with open("./sample" + suffix + ".json", "r") as f:
            text_data = json.load(f)
        self.embed_tensor = torch.tensor(text_data['Txtemb']).to(device)  # Convert text embeddings to tensor
        self.texts = text_data['Caption']  # Load captions

        # Load precomputed text IDs
        self.text_ids = np.load("./" + split + suffix + ".npy")

    def __len__(self):
        # Return the length of the dataset
        return self.data.__len__()

    def get_image_embedding(self, image):
        # Convert tensor to PIL image
        image = transforms.ToPILImage()(image).convert("RGB")

        # Preprocess the image and add a batch dimension
        image_preprocessed = self.preprocess_val(image).unsqueeze(0)

        # Compute image embeddings without computing gradients
        with torch.no_grad():
            image_embedding = self.model.encode_image(image_preprocessed.to(device))
            image_embedding /= image_embedding.norm(dim=-1, keepdim=True)  # Normalize embeddings

        return image_embedding, image_preprocessed

    def get_texts(self):
        # List to store text indices
        txt_list = []

        # Iterate through the dataset
        for idx in range(len(self.data)):
            img, label = self.data.__getitem__(idx)  # Get image and label
            image_embedding, image_preprocessed = self.get_image_embedding(img)  # Get image embedding
            dot_prod = (image_embedding * self.embed_tensor).sum(dim=1)  # Compute dot product with text embeddings
            txtidx = dot_prod.cpu().argmax().item()  # Get index of maximum dot product
            txt_list.append(txtidx)  # Append to text index list

        return txt_list

    def __getitem__(self, idx):
        # Convert tensor index to list if needed
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get image and label
        img, label = self.data.__getitem__(idx)

        # Tokenize the corresponding text
        txt = self.tokenizer(self.texts[self.text_ids[idx]])

        return (img, txt, label)

# Define a PyTorch Lightning data module
class PCAMTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers):
        super().__init__()

        # Initialize datasets for training, validation, and testing
        self.training_data = PachDataset("train")
        self.test_data = PachDataset("test")
        self.valid_data = PachDataset("val")

        # Set batch size and number of workers for data loading
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        # Return DataLoader for training data
        return DataLoader(dataset=self.training_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        # Return DataLoader for validation data
        return DataLoader(dataset=self.valid_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        # Return DataLoader for test data
        return DataLoader(dataset=self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
