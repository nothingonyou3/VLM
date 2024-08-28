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
from torchmetrics.classification import BinaryAccuracy
from torchmetrics.classification import MulticlassAccuracy
from torch.optim.lr_scheduler import CosineAnnealingLR

class UNINetTextModel(pl.LightningModule):
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

        self.model, self.preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms('hf-hub:wisdomik/QuiltNet-B-32')
        
        self.model_image = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)
        if bbfroze:
            for param in self.model_image.parameters():
                param.requires_grad = False
        if True:
            PATH = '/home/banafsh7/projects/def-hadi87/7/output/Bach_text_only_filtered/version_0/checkpoints/epoch=17-step=90.ckpt'
            checkpoint = torch.load(PATH)
            checkpoint = {key.replace("model.",""): checkpoint['state_dict'][key] for key in checkpoint['state_dict'].keys() if "model" in key}
            self.model.load_state_dict(checkpoint)
            for param in self.model.parameters():
                param.requires_grad = False
        self.text_embed_size = 512
        self.image_embed_size = 1024
        #self.fc0 = nn.Linear(self.text_embed_size, self.image_embed_size)
        #self.fc0 = nn.Linear(self.image_embed_size, self.image_embed_size)
        self.fc = nn.Linear(self.image_embed_size, num_classes)
        self.multihead_attn = nn.MultiheadAttention(embed_dim = 1024, kdim = 512, vdim = 512, num_heads=1, dropout = 0.1)#nn.MultiheadAttention(self.image_embed_size, 1)
        
        '''
        self.model, self.preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms('hf-hub:wisdomik/QuiltNet-B-32')
        self.model_image = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)
        self.text_embed_size = 512
        self.model_image.blocks[-1].mlp.fc2 = nn.Linear(in_features=4096, out_features=self.text_embed_size, bias=True)
        self.model_image.norm = nn.LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        self.model_image.blocks[-1].ls2.gamma = nn.Parameter(1e-5 * torch.ones(512))
        self.fc = nn.Linear(self.text_embed_size, num_classes)
        self.multihead_attn = nn.MultiheadAttention(self.text_embed_size, 1)
        '''

        print("model created")
        print(self.device)
        

    
    def forward(self, x, text_inputs, residual = True):
        text_embeds = self.model.encode_text(text_inputs.squeeze(1))
        image_embeds = self.model_image(x)
        #print(image_embeds.shape, text_embeds.shape)
        attn_output, attn_output_weights = self.multihead_attn(image_embeds, text_embeds, text_embeds)#query, key, value
        #x = torch.cat((image_embeds, text_embeds), 1)
        if residual:
            x = attn_output#self.fc0(attn_output)
            x = x + image_embeds
        else:
            x = attn_output
        out = self.fc(x)
        return out
    
    def compute_loss(self, y, yp):
        return F.cross_entropy(y, yp)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr = self.lr, momentum = self.momentum, nesterov = self.nesterov, weight_decay = self.weight_decay)
        #optimizer = optim.Adam(self.parameters(), lr = self.lr, weight_decay = self.weight_decay)
        #scheduler = CosineAnnealingLR(optimizer, T_max = 12500, eta_min = 0.000001)
        return {"optimizer": optimizer}#, "lr_scheduler": scheduler

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
        acc = self.metric(all_preds.argmax(1), all_trgts)
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
        acc = self.metric(all_preds.argmax(1), all_trgts)
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

        
