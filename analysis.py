import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import top_k_accuracy_score
data = pd.read_csv('./output/pcam_text_only_GPT_0/predictions.csv')
l = []
for i in range(2):
    l.append(data['class_'+ str(i)])
preds = np.stack(l).transpose()
targets = np.array(data['target'])
print(balanced_accuracy_score(targets, preds.argmax(1)), f1_score(targets, preds.argmax(1), average='micro'))
tn, fp, fn, tp = confusion_matrix(targets, preds.argmax(1)).ravel()
print(tp, fp)
print(fn, tn)
fpr, tpr, _ = roc_curve(targets, preds[:,1])
roc_auc = auc(fpr, tpr)
print(roc_auc)