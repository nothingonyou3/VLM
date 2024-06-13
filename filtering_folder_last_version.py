import csv
import os
import shutil

"""
In questo codice ci occupiamo di filtrare la cartella e le immagini di destinazione in base a la cartella images pt 2:)

Devi gestire le sottocartelle di ogni sovracartella
"""

#Ti convoene modificarlo e fare degli esperimenti differenti conuna cartella nuova in maniera tale da essere sicura che funzioni

def copy_filtered_images(csv_file_path, source_dir, destination_dir):
    try:
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)
        
        with open(csv_file_path, mode='r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            image_paths = [row['image_path'] for row in reader]
            
            for image_name in image_paths:
                source_path = os.path.join(source_dir, image_name)
                destination_path = os.path.join(destination_dir, image_name)
                
                if os.path.exists(source_path):
                    shutil.copy2(source_path, destination_path)
                    print(f"Copiato: {image_name}")
                    print("FILE MATCHATO E SPOSTATO!")
                else:
                    print(f"File non trovato: {image_name}")
                    
        print("Copia completata.")
    
    except FileNotFoundError:
        print(f"Il file CSV {csv_file_path} non è stato trovato.")
    except Exception as e:
        print(f"Si è verificato un errore: {e}")

# Esempio di utilizzo



output_file_path = '/home/giuliavanzato/Desktop/Quilt_complete/FILTRATO_OGGI.csv'
keyword = 'Breast'


source_dir_1 = '/home/giuliavanzato/Desktop/Quilt_complete/images_part_1/quilt_1m'
source_dir_2 = '/home/giuliavanzato/Desktop/Quilt_complete/images_part_2/quilt_1m'
source_dir_3 = '/home/giuliavanzato/Desktop/Quilt_complete/images_part_3/quilt_1m'
source_dir_4 = '/home/giuliavanzato/Desktop/Quilt_complete/images_part_4/quilt_1m'
source_dir_5 = '/home/giuliavanzato/Desktop/Quilt_complete/images_part_5/quilt_1m'
source_dir_6 = '/home/giuliavanzato/Desktop/Quilt_complete/images_part_6/quilt_1m'
source_dir_7 = '/home/giuliavanzato/Desktop/Quilt_complete/images_part_7/quilt_1m'
source_dir_8 = '/home/giuliavanzato/Desktop/Quilt_complete/images_part_8/quilt_1m'
source_dir_9 = '/home/giuliavanzato/Desktop/Quilt_complete/images_part_9/quilt_1m'
source_dir_10 = '/home/giuliavanzato/Desktop/Quilt_complete/images_part_10/quilt_1m'

destination_dir = '/home/giuliavanzato/Desktop/Quilt_complete/unique_quilt_before_filtering'

print("Inizio salvataggio immagini CARTELLA 1")
copy_filtered_images(output_file_path, source_dir_1 , destination_dir)

print("Inizio salvataggio immagini CARTELLA 2")
copy_filtered_images(output_file_path, source_dir_2 , destination_dir) #----------------

print("Inizio salvataggio immagini CARTELLA 3")
copy_filtered_images(output_file_path, source_dir_3 , destination_dir)

print("Inizio salvataggio immagini CARTELLA 4")
copy_filtered_images(output_file_path, source_dir_4 , destination_dir)

print("Inizio salvataggio immagini CARTELLA 5")
copy_filtered_images(output_file_path, source_dir_5 , destination_dir)

print("Inizio salvataggio immagini CARTELLA 6")
copy_filtered_images(output_file_path, source_dir_6 , destination_dir)

print("Inizio salvataggio immagini CARTELLA 7")
copy_filtered_images(output_file_path, source_dir_7 , destination_dir)

print("Inizio salvataggio immagini CARTELLA 8")
copy_filtered_images(output_file_path, source_dir_8 , destination_dir)

print("Inizio salvataggio immagini CARTELLA 9")
copy_filtered_images(output_file_path, source_dir_9, destination_dir)

print("Inizio salvataggio immagini CARTELLA 10")
copy_filtered_images(output_file_path, source_dir_10, destination_dir)
