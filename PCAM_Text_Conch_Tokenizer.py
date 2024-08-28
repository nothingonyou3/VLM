from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import open_clip
import torch
import pandas as pd
import json
from torchvision import transforms
import numpy as np
from torchvision.transforms import v2
from eva.vision.data.transforms.common import ResizeAndCrop
from conch.open_clip_custom import create_model_from_pretrained
from conch.open_clip_custom import tokenize, get_tokenizer
checkpoint_path = 'counch_checkpoint/pytorch_model.bin'
device = "cuda" if torch.cuda.is_available() else "cpu"
class PachDataset(Dataset):

    def __init__(self, split):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.model, self.preprocess_val = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=checkpoint_path)
        self.model = self.model.to(device)
        _ = self.model.eval()
        self.tokenizer = get_tokenizer()
        transform = self.preprocess_val#v2.Compose([ToTensor(),])
        if split == "train":
            transform = v2.Compose([
                self.preprocess_val.transforms[0],
                self.preprocess_val.transforms[1],
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.ColorJitter(brightness = 0.5, contrast = 0.5, saturation = 0.5, hue = 0.05),
                self.preprocess_val.transforms[2],
                self.preprocess_val.transforms[3],
                self.preprocess_val.transforms[4],])
        self.data =  datasets.PCAM(
                root="Data/Data",
                download=True,
                transform= transform,
                split = split
            )
        suffix = ""
        if False:
            suffix = "_v2"
        with open("./sample.json", "r") as f: 
            text_data =  json.load(f)
        self.embed_tensor = torch.tensor(text_data['Txtemb']).to(device)#torch.Size([1, 512])
        self.texts = text_data['Caption']
        '''self.image_embeddings = []
        for idx in range(len(self.data)):
            img, label = self.data.__getitem__(idx)
            image_embedding, _ = self.get_image_embedding(img)
            self.image_embeddings.append(image_embedding)'''

        self.text_ids = np.load("./"+split+suffix+".npy")#self.get_texts()

    def __len__(self):
        return self.data.__len__()

    def get_image_embedding(self, image):
        # Converti il tensore in immagine PIL
        #image = transforms.ToPILImage()(img_tensor).convert("RGB")
        image = transforms.ToPILImage()(image).convert("RGB")
        image_preprocessed = self.preprocess_val(image).unsqueeze(0)
        with torch.no_grad():
            image_embedding = self.model.encode_image(image_preprocessed.to(device))
            image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
        return image_embedding, image_preprocessed


    def get_texts(self):
        txt_list = []
        for idx in range(len(self.data)):
            img, label = self.data.__getitem__(idx)
            image_embedding, image_preprocessed = self.get_image_embedding(img)
            dot_prod = (image_embedding*self.embed_tensor).sum(dim = 1)
            txtidx = dot_prod.cpu().argmax().item()#self.tokenizer(self.texts[])
            txt_list.append(txtidx)
        return txt_list
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img, label = self.data.__getitem__(idx)
        txt = self.tokenizer(self.texts[self.text_ids[idx]])
        '''dot_prod = (self.image_embeddings[idx]*self.embed_tensor).sum(dim = 1)
        txt = self.tokenizer(self.texts[dot_prod.cpu().argmax().item()])'''
        print(txt)
        return (img, txt, label)#self.tokenizer()

class PCAMTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers):
        super().__init__()
        
        self.training_data = PachDataset("train")
        self.test_data = PachDataset("test")
        self.valid_data = PachDataset("val")

        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(dataset=self.training_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(dataset=self.valid_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)