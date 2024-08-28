from Data.PCAMDataLoader import PCAMDataModule
from Data.BACHDataLoader import BACHDataModule
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser
from sklearn.utils import shuffle
import pytorch_lightning as pl
import torch
from lightning.pytorch.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.functional import auroc
from Models.QuiltImage import QuiltImageEncoderModel
from Models.ResNet import ResNetModel
from Models.UNINet import UNINetModel
from Models.DINOL14Net import DINOL14NetModel
from Models.VITSNet import VITSNetModel
from Models.VITBNet import VITBNetModel
data = "bach"
model = "VITB"
pl.seed_everything(0, workers=True)
num_workers = 30
device = "gpu" if torch.cuda.is_available() else "cpu"
models = {'ResNet': ResNetModel, 'UNI': UNINetModel, "DINOL14": DINOL14NetModel, 'VITS': VITSNetModel, 'VITB': VITBNetModel}
output_name = model+'_'+data + '_v0_8'
Net = models[model]
print(output_name)
if data == "pcam":
    max_steps = 12500
    num_classes = 2
    batch_size = 64
    DataModule = PCAMDataModule
    lr=0.001#0.01
    momentum=0.9
    nesterov = True
    weight_decay = 0.0001#0
    bbfroze = True
    min_delta = 0 
    patience = 9
    monitor="train_loss"
    mode='min'
    #BATCH_SIZE = 4096, PREDICT_BATCH_SIZE = 64, N_RUNS =  5, BCEWithLogitsLoss, 
elif data == "bach":
    max_steps = 2500
    num_classes = 4
    batch_size = 256
    DataModule = BACHDataModule
    min_delta = 0 
    patience = 400
    bbfroze = True
    momentum=0.9
    nesterov = True
    weight_decay = 0.0
    lr = 0.001
    monitor="train_loss"
    mode='min'
    #BATCH_SIZE = 256, PREDICT_BATCH_SIZE = 64, N_RUNS =  5
elif data == "crc":
    max_steps = 12500
    num_classes = 9
    batch_size = 64 #256
    DataModule = CRCDataModule
    min_delta = 0 
    patience = 24
    bbfroze = True
    momentum=0.9
    nesterov = True
    weight_decay = 0.0
    lr = 0.01
    monitor="train_loss"
    mode='min'
    #BATCH_SIZE = 4096, PREDICT_BATCH_SIZE = 64, N_RUNS =  5

def save_predictions(model, output_fname, num_classes):
    prds = torch.cat(model.predictions, dim=0)
    trgs = torch.cat(model.targets, dim=0)

    auc = auroc(prds, trgs, num_classes=num_classes, average='macro', task='multiclass')

    print('AUROC (test)')
    print(auc)

    cols_names = ['class_' + str(i) for i in range(0, num_classes)]

    df = pd.DataFrame(data=prds.cpu().numpy(), columns=cols_names)    
    df['target'] = trgs.cpu().numpy()
    df.to_csv(output_fname, index=False)

'''Loading Data'''
data = DataModule(batch_size, num_workers)
output_base_dir = 'output'
output_dir = os.path.join(output_base_dir,output_name)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
'''Creating the model'''
model = Net(num_classes, batch_size, lr, momentum, nesterov, weight_decay, bbfroze)

print('=============================================================')
print('Training...')

checkpoint_callback = ModelCheckpoint(monitor=monitor, mode=mode)
early_stop_callback = EarlyStopping(
        monitor=monitor,
        min_delta=min_delta,
        patience=patience,  # NOTE no. val epochs, not train epochs
        verbose=False,
        mode=mode,
    )
trainer = pl.Trainer(
        callbacks=[checkpoint_callback, early_stop_callback],#
        log_every_n_steps=1,
        max_steps=max_steps,
        accelerator=device,
        devices=1,
        logger=[CSVLogger("logs", name=output_name)],
    )
trainer.logger._default_hp_metric = False
trainer.fit(model, data)
print(trainer.checkpoint_callback.best_model_path)
model = Net.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
print(trainer.test(model=model, datamodule=data))
save_predictions(model, os.path.join(output_dir, 'predictions.csv'), num_classes)