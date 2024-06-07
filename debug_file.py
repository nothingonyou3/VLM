
#Questo codice si occupa di stampare il nome delle colonne del mio csv di partenza e l indice associato ad esse

import csv

def print_csv_column_indices(file_path):
    try:
        with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            headers = next(reader)  # Legge la prima riga come intestazione
            for index, column_name in enumerate(headers):
                print(f"Indice: {index}, Colonna: {column_name}")
    except FileNotFoundError:
        print(f"Il file {file_path} non è stato trovato.")
    except Exception as e:
        print(f"Si è verificato un errore: {e}")

# Esempio di utilizzo
file_path = '/home/giuliavanzato/Desktop/Quilt_complete/quilt_1M_lookup.csv'
print_csv_column_indices(file_path)
