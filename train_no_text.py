from Data.PCAMDataLoader import PCAMDataModule
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser
from sklearn.utils import shuffle
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.functional import auroc
from Models.QuiltImage import QuiltImageEncoderModel
from Models.ResNet import ResNetModel
from Models.UNINet import UNINetModel
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
epochs = 20
device = "gpu" if torch.cuda.is_available() else "cpu"
pl.seed_everything(42, workers=True)
num_classes = 2
batch_size, num_workers = 64, 10
'''Loading Data'''
data = PCAMDataModule(batch_size, num_workers)
output_base_dir = 'output'
output_name = 'train_no_text_UNI_aug'
output_dir = os.path.join(output_base_dir,output_name)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
'''Creating the model'''
Net = UNINetModel
model = Net(num_classes, batch_size)

print('=============================================================')
print('Training...')

checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode='min')
early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.001,
        patience=10,  # NOTE no. val epochs, not train epochs
        verbose=False,
        mode="min",
    )
trainer = pl.Trainer(
        callbacks=[checkpoint_callback],#, early_stop_callback
        log_every_n_steps=5,
        max_epochs=epochs,
        accelerator=device,
        devices=1,
        logger=TensorBoardLogger(output_base_dir, name=output_name),
    )
trainer.logger._default_hp_metric = False
trainer.fit(model, data)
model = Net.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
print(trainer.checkpoint_callback.best_model_path)
print(trainer.test(model=model, datamodule=data))
save_predictions(model, os.path.join(output_dir, 'predictions.csv'), 2)