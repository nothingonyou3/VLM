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

from PIL import Image
from torchmetrics.functional import auroc
from torchmetrics.classification import BinaryAccuracy
from torchmetrics.classification import MulticlassAccuracy
from torch.optim.lr_scheduler import CosineAnnealingLR

class ResNetModel(pl.LightningModule):
    def __init__(self, num_classes = 4, batch_size = 64, lr=0.001, momentum=0.9, nesterov = True, weight_decay = 0.0001, bbfroze = True):
        super().__init__()
        self.num_classes = num_classes      
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self.batch_size = batch_size

        self.predictions = []
        self.targets = []

        self.train_step_preds = []
        self.train_step_trgts = []
        self.val_step_preds = []
        self.val_step_trgts = []
        self.train_loss = []
        self.val_loss = []
        if num_classes == 2:
            self.metric = BinaryAccuracy()
        else:
            self.metric = MulticlassAccuracy(num_classes=num_classes)

        self.resnet50 = models.resnet50(pretrained = True)
        '''if bbfroze:
            for param in self.resnet50.parameters():
                param.requires_grad = False'''
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, self.num_classes)

        print("model created")
        print(self.device)
        

    
    def forward(self, x):
        out = self.resnet50(x)
        return out
    
    def compute_loss(self, y, yp):
        return F.cross_entropy(y, yp)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr = self.lr, momentum = self.momentum, nesterov = self.nesterov, weight_decay = self.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max = 12500)
        return {"optimizer": optimizer}#, "lr_scheduler": scheduler

    def process_batch(self, batch):
        img = batch[0].to(self.device)
        lab = batch[1].to(self.device)
        out = self.forward(img)
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
        if self.num_classes == 2:
            all_preds = all_preds.max(dim = 1).values
        acc = self.metric(all_preds, all_trgts)
        self.log('train_acc', acc, batch_size=len(all_preds))
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
        if self.num_classes == 2:
            all_preds = all_preds.max(dim = 1).values
        acc = self.metric(all_preds, all_trgts)
        self.log('val_acc', acc, batch_size=len(all_preds))
        self.val_step_preds.clear()
        self.val_step_trgts.clear()

    def on_test_start(self):
        self.predictions = []
        self.targets = []

    def test_step(self, batch, batch_idx):
        _, prd, lab = self.process_batch(batch)        
        self.predictions.append(prd)
        self.targets.append(lab.squeeze())

        
