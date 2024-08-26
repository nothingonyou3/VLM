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
from eva.vision.data import datasets
from eva.vision.data.transforms.common import ResizeAndCrop
import matplotlib.pyplot as plt

import matplotlib.cm as cm
import re
device = "cuda" if torch.cuda.is_available() else "cpu"

#pcam too

class BachDataset(Dataset):

    def __init__(self, split):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.model, self.preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms('hf-hub:wisdomik/QuiltNet-B-32')
        self.model.to(device)
        self.tokenizer = open_clip.get_tokenizer('hf-hub:wisdomik/QuiltNet-B-32')
        self.data =  datasets.BACH(
                                root="Data/Data/Bach",
                                split=split,
                                download = True,
                                image_transforms = ResizeAndCrop(size = 224, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
                            )
        
        self.data.prepare_data()
        self.data.configure()
        with open("./sample.json", "r") as f: 
            text_data =  json.load(f)
        self.embed_tensor = torch.tensor(text_data['Txtemb']).to(device)#torch.Size([1, 512])
        self.texts = text_data['Caption']
        index = [i for i in range(len(self.texts))] #Banafsheh text if not(re.search('neural network', self.texts[i], re.IGNORECASE) or re.search('machine learning', self.texts[i], re.IGNORECASE) or re.search('classification network', self.texts[i], re.IGNORECASE))
        self.texts = [self.texts[i] for i in index]
        embed = [text_data['Txtemb'][i] for i in index]
        self.embed_tensor = torch.tensor(embed).to(device)#torch.Size([1, 512])
        '''self.image_embeddings = []
        for idx in range(len(self.data)):
            img, label = self.data.__getitem__(idx)
            image_embedding, _ = self.get_image_embedding(img)
            self.image_embeddings.append(image_embedding)'''

        self.text_ids = self.get_texts()#np.load("./"+split+"_bach_conch.npy")
        
        
        
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


    def get_best(self):
        txt_list = []
        value_list = []
        for idx in range(len(self.data)):
            img, label = self.data.__getitem__(idx)
            image_embedding, image_preprocessed = self.get_image_embedding(img)
            dot_prod = (image_embedding*self.embed_tensor).sum(dim = 1)
            value, txtidx = dot_prod.cpu().topk(10)
            txt_list.append(txtidx)
            value_list.append(value)
        return txt_list, value_list
    
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
        return (img, txt, label)#self.tokenizer()
    
    
    def get_max(self):
        # Inizializza una lista vuota per contenere i valori massimi della dot product associati alle immagini
        max_dot_prod_list = []
        
        # Itera su tutti gli elementi del dataset 'self.data'
        for idx in range(len(self.data)):
            # Recupera l'immagine e l'etichetta dal dataset alla posizione 'idx'
            img, label = self.data.__getitem__(idx)
            
            # Calcola l'embedding e il preprocessing dell'immagine
            image_embedding, image_preprocessed = self.get_image_embedding(img)
            
            # Calcola il prodotto scalare tra l'embedding dell'immagine e gli embeddings testuali (self.embed_tensor)
            dot_prod = (image_embedding * self.embed_tensor).sum(dim=1)
            
            # Trova il massimo valore della dot product
            max_dot_prod = dot_prod.cpu().max().item()  # Converte il tensore nel massimo valore scalare
            
            # Aggiunge il massimo valore della dot product alla lista dei risultati
            max_dot_prod_list.append(max_dot_prod)
        
        # Restituisce la lista dei valori massimi della dot product associati a ciascuna immagine
        return max_dot_prod_list
    
    
training_data = BachDataset("train")
#print(training_data.get_max())

# Supponiamo che max_dot_prod_list contenga i valori che hai calcolato
max_dot_prod_list = training_data.get_max()

# Conta le frequenze di ogni valore nella lista
#values, counts = np.unique(max_dot_prod_list, return_counts=True)
print(max_dot_prod_list)
# Salva il grafico come file immagine (es. "grafico_probabilita.png")
plt.hist(max_dot_prod_list, density=True, bins=100)  
plt.savefig('./grafico_probabilita_filtered_1_Bach.png', dpi=300)

# Chiudi la figura per evitare di sovraccaricare la memoria
plt.close()
'''
# Calcola le probabilità
total_count = len(max_dot_prod_list)
probabilities = (counts / total_count) * 100

# Crea una mappa di colori
cmap = plt.get_cmap('tab10')  # Usa plt.get_cmap() per ottenere la colormap
colors = [cmap(i / len(values)) for i in range(len(values))]  # Genera colori unici

# Crea il grafico
plt.figure(figsize=(12, 6))  # Imposta la dimensione della figura
bars = plt.bar(values, probabilities, color=colors, width=0.5, alpha=0.7)

# Etichette degli assi
plt.xlabel('Valori in max_dot_prod_list', fontsize=12)
plt.ylabel('Probabilità (%)', fontsize=12)

# Titolo del grafico
plt.title('Probabilità dei valori in max_dot_prod_list', fontsize=15)

# Aggiunge etichette sopra ogni barra per visualizzare le probabilità
for bar, prob in zip(bars, probabilities):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{prob:.2f}%', ha='center', va='bottom')

# Salva il grafico come file immagine (es. "grafico_probabilita.png")
plt.savefig('./grafico_probabilitaVS4.png', dpi=300, bbox_inches='tight')

# Chiudi la figura per evitare di sovraccaricare la memoria
plt.close()
'''