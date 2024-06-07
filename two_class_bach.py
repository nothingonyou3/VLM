#Il file dovra acquisire le immagini microscopy_ground_truth.csv e crearne uno secondario in cui
# al psoto di normal e in situ verra messo no cancer mentre al posto di InSitu e Invasive sara' messo cancer

#per evitare perdite di dati tutte le manipolazioni verranno fatte su un file di destinazione A e poi B

#Le stringhe da controllare sono Normal, Benign, InSitu, Invasive --» totale immagini 400. La colonna su cui operare e' la seconda del csv e non presenta titolo.



#In questo primo step mi limito alla creazione del nuovo csv


import pandas as pd

# Leggere il file CSV "a.csv"
df = pd.read_csv('/home/giuliavanzato/Desktop/Quilt_complete/ICIAR2018_BACH_Challenge/ICIAR2018_BACH_Challenge/Photos/microscopy_ground_truth.csv')

# Definire le sostituzioni
sostituzioni = {
    "Normal": "no",
    "Benign": "no",
    "InSitu": "cancer",
    "Invasive": "cancer"
}

# Applicare le sostituzioni nella seconda colonna
df.iloc[:, 1] = df.iloc[:, 1].replace(sostituzioni)

# Salvare il risultato in un nuovo file CSV "two_class_bach.csv"
df.to_csv('two_class_bach.csv', index=False)
