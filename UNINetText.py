import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser
from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torchvision.transforms as T
from torch import optim
import open_clip
import pytorch_lightning as pl
import open_clip
from PIL import Image
from torchmetrics.functional import auroc
import timm

class UNINetTextModel(pl.LightningModule):
    def __init__(self, num_classes = 2, batch_size = 64, optim_lr = 0.0001):
        super().__init__()
        self.num_classes = num_classes        
        self.optim_lr = optim_lr
        self.batch_size = batch_size

        self.predictions = []
        self.targets = []

        self.train_step_preds = []
        self.train_step_trgts = []
        self.val_step_preds = []
        self.val_step_trgts = []
        self.train_loss = []
        self.val_loss = []

        self.model, self.preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms('hf-hub:wisdomik/QuiltNet-B-32')
        self.model_image = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)
        self.text_embed_size = 512
        self.image_embed_size = 1024
        self.fc = nn.Linear(self.image_embed_size, num_classes)
        self.multihead_attn = nn.MultiheadAttention(self.image_embed_size, 1)

        print("model created")
        print(self.device)
        

    
    def forward(self, x, text_inputs):
        text_embeds = self.model.encode_text(text_inputs.squeeze(1))
        image_embeds = self.model_image(x)
        attn_output, attn_output_weights = self.multihead_attn(image_embeds, text_embeds, text_embeds)#query, key, value
        #x = torch.cat((image_embeds, text_embeds), 1)
        x = attn_output
        out = self.fc(x)
        return out
    
    def compute_loss(self, y, yp):
        return F.cross_entropy(y, yp)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9, nesterov = True, weight_decay = 0.0001)
        return optimizer

    def process_batch(self, batch):
        img, txt, lab = batch
        img = img.squeeze(1).to(self.device)
        txt = torch.tensor(txt).to(self.device)
        lab = lab.to(self.device)
        out = self.forward(img, txt)
        prd = torch.softmax(out, dim=1)
        loss = self.compute_loss(prd, lab)
        return loss, prd, lab

    def training_step(self, batch, batch_idx):
        loss, prd, lab = self.process_batch(batch)
        self.train_step_preds.append(prd)
        self.train_step_trgts.append(lab)
        self.log('train_loss', loss, batch_size=self.batch_size)        
        '''batch_ratio = len(np.where(lab.cpu().numpy() == 1)[0]) / len(np.where(lab.cpu().numpy() == 0)[0])
        self.log('batch_ratio', batch_ratio, batch_size=self.batch_size)                        
        grid = torchvision.utils.make_grid(batch['image'][0:4, ...], nrow=2, normalize=True)
        self.logger.experiment.add_image('images', grid, self.global_step)'''
        return loss

    def on_train_epoch_end(self):
        all_preds = torch.cat(self.train_step_preds, dim=0)
        all_trgts = torch.cat(self.train_step_trgts, dim=0)
        auc = auroc(all_preds, all_trgts, num_classes=self.num_classes, average='macro', task='multiclass')
        self.log('train_auc', auc, batch_size=len(all_preds))
        self.train_step_preds.clear()
        self.train_step_trgts.clear()

    def validation_step(self, batch, batch_idx):
        loss, prd, lab = self.process_batch(batch)
        self.val_step_preds.append(prd)
        self.val_step_trgts.append(lab)
        self.log('val_loss', loss, batch_size=self.batch_size)

    def on_validation_epoch_end(self):
        all_preds = torch.cat(self.val_step_preds, dim=0)
        all_trgts = torch.cat(self.val_step_trgts, dim=0)
        auc = auroc(all_preds, all_trgts, num_classes=self.num_classes, average='macro', task='multiclass')
        self.log('val_auc', auc, batch_size=len(all_preds))
        self.val_step_preds.clear()
        self.val_step_trgts.clear()

    def on_test_start(self):
        self.predictions = []
        self.targets = []

    def test_step(self, batch, batch_idx):
        _, prd, lab = self.process_batch(batch)        
        self.predictions.append(prd)
        self.targets.append(lab.squeeze())

        
