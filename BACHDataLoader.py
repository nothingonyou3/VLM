from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import open_clip
import torch
from torchvision.transforms import v2
from eva.vision.data import datasets
from eva.vision.data.transforms.common import ResizeAndCrop

class BACHDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers, download = True):
        super().__init__()
        self.training_data = datasets.BACH(
                                root="Data/Data/Bach",
                                split="train",
                                download = download,
                                image_transforms = ResizeAndCrop(size = 224, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
                            )
        self.training_data.prepare_data()
        self.training_data.configure()
        
        self.valid_data = datasets.BACH(
                                root="Data/Data/Bach",
                                split="val",
                                download = download,
                                image_transforms = ResizeAndCrop(size = 224, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
                            )
        
        self.valid_data.prepare_data()
        self.valid_data.configure()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(dataset=self.training_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(dataset=self.valid_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)