from Data.PCAMTDataLoader import PCAMTDataModule #questa riga verra sostituita in base al testo da valutare basic(PCAM_Text), Banafsheh (Text_) e mia (Text_filtered)
from Data.BACHTDataLoader import BACHTDataModule
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
from Models.TextNet import TextNetModel
#from Models.CONCH import CONCH_QuiltTextEncoderModel
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
#from Models.TextNetPMB import QuiltTextEncoderModel
#quando devi fare i test con PMB sostituisci con TextNetPMB come import


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
    l = []
    for i in range(num_classes):
        l.append(df['class_'+ str(i)])
    preds = np.stack(l).transpose()
    targets = np.array(df['target'])
    print("balanced accuracy, F1 score:")
    print(balanced_accuracy_score(targets, preds.argmax(1)), f1_score(targets, preds.argmax(1), average='micro'))




epochs = 20
device = "gpu" if torch.cuda.is_available() else "cpu"
pl.seed_everything(0, workers=True)
batch_size, num_workers = 64, 10


'''Loading Data'''
data_name = "pcam"
if data_name == "pcam":
    num_classes = 2
    DataModule = PCAMTDataModule
else:
    num_classes = 4
    DataModule = BACHTDataModule
data = DataModule(batch_size, num_workers) # Define a PyTorch Lightning data module
output_base_dir = 'output'
output_name = data_name + '_text_only_GPT_0'
output_dir = os.path.join(output_base_dir,output_name)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


'''Creating the model'''
Net = TextNetModel
model = Net(num_classes, batch_size)

print('=============================================================')
print('Training...')

checkpoint_callback = ModelCheckpoint(monitor="train_loss", mode='min')
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
model = Net.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, num_classes = num_classes)
print(trainer.checkpoint_callback.best_model_path)
print(trainer.test(model=model, datamodule=data))
save_predictions(model, os.path.join(output_dir, 'predictions.csv'), num_classes)